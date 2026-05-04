#!/usr/bin/env bash
# scripts/run_lora.sh
#
# Overnight LoRA pipeline:
#   1. Verify llm_server is up + holdout split exists
#   2. Distill 1800 train scenarios via the LLM teacher (uses GT to construct CoT)
#       -> traces/train_traces.jsonl
#   3. Stop llm_server to free GPU 0
#   4. LoRA fine-tune the base model on the traces
#       -> training/checkpoints/run_v1/best_lora/
#   5. Restart llm_server with --lora attached
#   6. Re-run submit_now on holdout to validate LoRA helps
#       (decision gate: revert to baseline if it doesn't)
#   7. Run on Phase 1 test set + convert to Zindi format
#
# Resumable: skips finished steps. Idempotent.
#
# Run in background, log to file, disconnect:
#     nohup bash scripts/run_lora.sh > eval/logs/run_lora.log 2>&1 &
#     echo "lora pid=$!"
#     tail -f eval/logs/run_lora.log
#
# Total time: ~8-10 hours on RTX 8000 x2.

set -uo pipefail
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found"; exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
LLM_PORT="${LLM_PORT:-8001}"
TRAIN_FILE="${TRAIN_FILE:-data/local_split/train_1800.json}"
HOLDOUT_FILE="${HOLDOUT_FILE:-data/local_split/holdout_200.json}"
TRACES="${TRACES:-traces/train_traces.jsonl}"
LORA_OUT_PARENT="${LORA_OUT_PARENT:-training/checkpoints/run_v1}"
LORA_DIR="$LORA_OUT_PARENT/best_lora"
LLM_LOG="${LLM_LOG:-eval/logs/run_all/llm_server.log}"
SKIP_DISTILL="${SKIP_DISTILL:-0}"
SKIP_FINETUNE="${SKIP_FINETUNE:-0}"

mkdir -p traces "$LORA_OUT_PARENT" eval/logs/run_all eval/results

c_blue()  { printf "\033[1;34m%s\033[0m\n" "$1"; }
c_green() { printf "\033[1;32m%s\033[0m\n" "$1"; }
c_yel()   { printf "\033[1;33m%s\033[0m\n" "$1"; }
c_red()   { printf "\033[1;31m%s\033[0m\n" "$1"; }
step() { echo; c_blue "============================================================"; c_blue "  $1"; c_blue "============================================================"; }

wait_for_health() {
    local timeout_s="${1:-1200}"
    local deadline=$(( $(date +%s) + timeout_s ))
    while ! curl -sf "http://localhost:$LLM_PORT/health" 2>/dev/null | grep -q '"status":"ok"'; do
        if [ "$(date +%s)" -gt "$deadline" ]; then
            c_red "  llm_server never became healthy"
            tail -n 60 "$LLM_LOG" >&2 || true
            return 1
        fi
        printf "."
        sleep 5
    done
    echo " ready"
    return 0
}

start_llm_server() {
    local extra_args=("$@")
    pkill -f "scripts/llm_server.py" 2>/dev/null || true
    sleep 5
    nohup python scripts/llm_server.py \
        --model "$MODEL_NAME" --port "$LLM_PORT" "${extra_args[@]}" \
        > "$LLM_LOG" 2>&1 &
    echo "  pid=$!  log=$LLM_LOG"
    wait_for_health 1200 || return 1
    return 0
}

# -------- preflight: holdout split must exist
step "PREFLIGHT — holdout split"
if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$HOLDOUT_FILE" ]; then
    echo "  building holdout..."
    python -c "
import json, random, os
from collections import defaultdict
random.seed(42)
t = json.load(open('data/Phase_1/train.json'))
buckets = defaultdict(list)
for s in t:
    k = (s.get('tag','single-answer'),
         (s.get('context',{}).get('wireless_network_information') or {}).get('num_base_stations','4'))
    buckets[k].append(s)
train, hold = [], []
for k, sc in buckets.items():
    random.shuffle(sc)
    n = max(1, len(sc)*200//2000)
    hold.extend(sc[:n]); train.extend(sc[n:])
os.makedirs('data/local_split', exist_ok=True)
json.dump(train, open('data/local_split/train_1800.json','w'))
json.dump(hold,  open('data/local_split/holdout_200.json','w'))
print(f'  train={len(train)} holdout={len(hold)}')
"
fi
c_green "  $TRAIN_FILE / $HOLDOUT_FILE"

# -------- preflight: llm_server up
step "PREFLIGHT — llm_server health"
if curl -sf "http://localhost:$LLM_PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
    c_green "  already healthy"
else
    c_yel "  not running — starting base model"
    start_llm_server || { c_red "failed to start llm_server"; exit 1; }
fi

# -------- step 1: distill
step "STEP 1 — Distill ${TRAIN_FILE} -> ${TRACES}"
if [ "$SKIP_DISTILL" = "1" ]; then
    c_yel "  SKIP_DISTILL=1"
elif [ -f "$TRACES" ] && [ "$(wc -l < "$TRACES")" -ge 1000 ]; then
    c_yel "  $TRACES already has $(wc -l < "$TRACES") traces — skipping distill"
else
    BEFORE=0; [ -f "$TRACES" ] && BEFORE=$(wc -l < "$TRACES")
    echo "  starting distillation (resumable). Expect ~6 hours for 1800 scenarios."
    python scripts/distill.py \
        --train_file "$TRAIN_FILE" \
        --output    "$TRACES" \
        --model_url "http://localhost:$LLM_PORT/v1" \
        --model_name "$MODEL_NAME" || { c_red "distill failed"; exit 1; }
    AFTER=$(wc -l < "$TRACES")
    c_green "  traces: $BEFORE -> $AFTER (+$((AFTER - BEFORE)))"
    if [ "$AFTER" -lt 200 ]; then
        c_red "  only $AFTER traces — fine-tune unlikely to help. Aborting."
        exit 1
    fi
fi

# -------- step 2: free GPU + LoRA fine-tune
step "STEP 2 — LoRA fine-tune (~2-3 hours)"
if [ "$SKIP_FINETUNE" = "1" ]; then
    c_yel "  SKIP_FINETUNE=1"
elif [ -d "$LORA_DIR" ] && [ -f "$LORA_DIR/adapter_config.json" ]; then
    c_yel "  $LORA_DIR already exists — skipping fine-tune"
else
    echo "  stopping llm_server to free GPU 0"
    pkill -f "scripts/llm_server.py" 2>/dev/null || true
    sleep 8
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python scripts/finetune.py \
            --traces "$TRACES" \
            --output_dir "$LORA_OUT_PARENT" \
            --base_model "$MODEL_NAME" || { c_red "finetune failed"; exit 1; }
    if [ ! -d "$LORA_DIR" ] || [ ! -f "$LORA_DIR/adapter_config.json" ]; then
        c_red "  best_lora not produced at $LORA_DIR"
        exit 1
    fi
    c_green "  saved to $LORA_DIR"
fi

# -------- step 3: restart llm_server with LoRA
step "STEP 3 — Restart llm_server with --lora $LORA_DIR"
start_llm_server --lora "$LORA_DIR" || { c_red "lora server failed"; exit 1; }

# -------- step 4: holdout score with LoRA (and RAG if KB exists)
step "STEP 4 — Holdout score (LoRA"
RAG_FLAGS=""
if [ -f knowledge/processed/embeddings.npy ] && [ -f knowledge/processed/chunks.json ]; then
    RAG_FLAGS="--use_rag --rag_k 3 --max_tokens 128"
    c_yel "  KB found — testing LoRA + RAG"
else
    RAG_FLAGS="--max_tokens 128"
    c_yel "  no KB — testing LoRA only"
fi

rm -rf eval/results/holdout_lora
python scripts/submit_now.py \
    --test_file "$HOLDOUT_FILE" \
    --out_dir   eval/results/holdout_lora \
    $RAG_FLAGS 2>&1 | tee eval/results/holdout_lora.log || true

SCORE_LORA=$(grep -m1 "mean   :" eval/results/holdout_lora.log 2>/dev/null \
             | grep -oE "[0-9]\.[0-9]+" | head -1)
SCORE_BASE=""
if [ -f eval/results/holdout_base.log ]; then
    SCORE_BASE=$(grep -m1 "mean   :" eval/results/holdout_base.log 2>/dev/null \
                 | grep -oE "[0-9]\.[0-9]+" | head -1)
fi
c_green "  LoRA holdout score : ${SCORE_LORA:-?}"
[ -n "$SCORE_BASE" ] && c_green "  baseline score (vs): ${SCORE_BASE}"

# -------- step 5: decision + final run
USE_LORA_FOR_FINAL=1
if [ -n "$SCORE_BASE" ] && [ -n "$SCORE_LORA" ]; then
    if python -c "import sys; sys.exit(0 if float('$SCORE_LORA') >= float('$SCORE_BASE') else 1)"; then
        c_green "  LoRA helps -> using for final"
    else
        USE_LORA_FOR_FINAL=0
        c_red "  LoRA HURTS holdout (delta=$(python -c "print(round(float('$SCORE_LORA')-float('$SCORE_BASE'),4))")). Reverting to base for final."
        pkill -f "scripts/llm_server.py" 2>/dev/null || true
        sleep 5
        start_llm_server || { c_red "base server restart failed"; exit 1; }
    fi
fi

step "STEP 5 — Final run on data/Phase_1/test.json"
rm -rf eval/results/final_lora
python scripts/submit_now.py \
    --test_file data/Phase_1/test.json \
    --out_dir   eval/results/final_lora \
    $RAG_FLAGS || { c_red "final run failed"; exit 1; }

# -------- step 6: convert to Zindi format
step "STEP 6 — Convert to Zindi format (ID,Track A,Track B)"
for v in v1_raw v2_multi_recall v3_insurance; do
    python -c "
import pandas as pd
df = pd.read_csv('eval/results/final_lora/result_${v}.csv', dtype=str).fillna('')
df = df.rename(columns={'scenario_id':'ID','answers':'Track A'})
df['Track B'] = ''
df = df[['ID','Track A','Track B']]
df.to_csv('eval/results/final_lora/result_${v}_zindi.csv', index=False)
print('  wrote ${v}_zindi:', len(df), 'rows')"
done

# -------- summary
step "DONE"
[ -n "$SCORE_BASE" ] && echo "Holdout (no LoRA)       : $SCORE_BASE"
[ -n "$SCORE_LORA" ] && echo "Holdout (LoRA${RAG_FLAGS:+ + RAG}) : $SCORE_LORA"
echo "LoRA used in final      : $([ "$USE_LORA_FOR_FINAL" = "1" ] && echo YES || echo NO)"
echo
c_green "Upload these to Zindi:"
ls -la eval/results/final_lora/result_*_zindi.csv 2>/dev/null
