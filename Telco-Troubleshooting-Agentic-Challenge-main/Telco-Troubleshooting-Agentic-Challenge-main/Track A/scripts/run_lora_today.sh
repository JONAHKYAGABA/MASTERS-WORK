#!/usr/bin/env bash
# scripts/run_lora_today.sh
#
# ONE-COMMAND LoRA fine-tune pipeline for Qwen3.5-35B-A3B on a single 24 GB GPU.
#
# Phases (each skipped if its output already exists):
#   A. venv + deps + env
#   B. Ensure llm_server is up (needed for distillation)
#   C. Build stratified 1800/200 holdout split
#   D. Distill traces from train fold (LLM teacher with GT answer)
#         -> traces/train_traces.jsonl   (resumable)
#   E. Stop llm_server (free the GPU)
#   F. LoRA fine-tune on traces
#         -> training/checkpoints/run_v1/best_lora/
#   G. Restart llm_server with --lora attached
#   H. Score LoRA on holdout vs baseline
#   I. Run agentic agent on Phase 1 test (with LoRA + tools)
#   J. Convert outputs to Zindi format
#
# Usage:
#     cd /workspace/MASTERS-WORK/"Track A"
#     export HF_TOKEN=hf_YOUR_TOKEN
#     nohup bash scripts/run_lora_today.sh > eval/logs/run_all/lora.log 2>&1 &
#     echo "pid=$!"
#     tail -f eval/logs/run_all/lora.log
#
# Total runtime ~6-9 hours on RTX A5000:
#   distill ~4-6h, finetune ~1-2h, test run ~1h.

set -uo pipefail
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# -------- config
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
LLM_PORT="${LLM_PORT:-8001}"
TOOL_PORT="${TOOL_PORT:-7860}"
TEST_FILE="${TEST_FILE:-data/Phase_1/test.json}"
TRAIN_FOLD="${TRAIN_FOLD:-data/local_split/train_1800.json}"
HOLDOUT="${HOLDOUT:-data/local_split/holdout_200.json}"
TRACES="${TRACES:-traces/train_traces.jsonl}"
LORA_PARENT="${LORA_PARENT:-training/checkpoints/run_v1}"
LORA_DIR="$LORA_PARENT/best_lora"
LLM_LOG="eval/logs/run_all/llm_server.log"
TOOL_LOG="eval/logs/run_all/tool_server.log"
SKIP_DISTILL="${SKIP_DISTILL:-0}"
SKIP_FINETUNE="${SKIP_FINETUNE:-0}"
SKIP_FINAL="${SKIP_FINAL:-0}"
DISTILL_LIMIT="${DISTILL_LIMIT:-1800}"

# Fine-tune hyperparams (defaults tuned for 24 GB A5000)
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_MAX_SEQ_LEN="${LORA_MAX_SEQ_LEN:-2048}"
LORA_NUM_EPOCHS="${LORA_NUM_EPOCHS:-1}"
LORA_BATCH_SIZE="${LORA_BATCH_SIZE:-1}"
LORA_GRAD_ACCUM="${LORA_GRAD_ACCUM:-16}"
LORA_LR="${LORA_LR:-1e-4}"

mkdir -p eval/logs/run_all eval/results traces "$LORA_PARENT" data/local_split

c_blue()  { printf "\033[1;34m%s\033[0m\n" "$1"; }
c_green() { printf "\033[1;32m%s\033[0m\n" "$1"; }
c_yel()   { printf "\033[1;33m%s\033[0m\n" "$1"; }
c_red()   { printf "\033[1;31m%s\033[0m\n" "$1"; }
step()    { echo; c_blue "========================================================"; c_blue "  $1"; c_blue "========================================================"; }

wait_health() {
    local deadline=$(( $(date +%s) + ${2:-900} ))
    while ! curl -sf "$1" 2>/dev/null | grep -q '"status":"ok"'; do
        [ "$(date +%s)" -gt "$deadline" ] && return 1
        printf "."; sleep 5
    done
    echo " ready"; return 0
}

zindi_convert() {
    python -c "
import pandas as pd
df = pd.read_csv('$1', dtype=str).fillna('')
df = df.rename(columns={'scenario_id':'ID','answers':'Track A'})
df['Track B'] = ''
df = df[['ID','Track A','Track B']]
df.to_csv('$2', index=False)
print('  ${2##*/}:', len(df), 'rows')"
}

start_llm_server() {
    pkill -f "scripts/llm_server.py" 2>/dev/null || true
    sleep 5
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup python scripts/llm_server.py --model "$MODEL_NAME" --port "$LLM_PORT" "$@" \
        > "$LLM_LOG" 2>&1 &
    echo "  llm pid=$!"
    wait_health "http://localhost:$LLM_PORT/health" 900 || return 1
    return 0
}

# -------- A. venv + deps
step "A. venv + dependencies"
if [ ! -d ".venv" ]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
if ! python -c "import torch, transformers, peft, bitsandbytes, datasets" >/dev/null 2>&1; then
    pip install --upgrade -q pip wheel setuptools
    pip install -q -r requirements.txt
    pip install -q "openai>=1.50.0" httpx requests python-dateutil tqdm \
        "torch>=2.4.0" "transformers>=4.45.0" "accelerate>=1.0.0" \
        "bitsandbytes>=0.44.0" "peft>=0.13.0" "datasets>=3.0.0" \
        safetensors sentencepiece protobuf "huggingface_hub>=0.25" hf_transfer pandas \
        "uvicorn[standard]" python-multipart "fastapi>=0.110"
fi

# -------- B. llm_server up (for distill)
step "B. Ensure llm_server is healthy (needed for distillation)"
if curl -sf "http://localhost:$LLM_PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
    c_yel "  already healthy"
else
    start_llm_server || { c_red "llm_server failed"; tail -n 40 "$LLM_LOG" >&2; exit 1; }
fi

# -------- C. holdout split
step "C. Build 1800/200 stratified holdout"
if [ -f "$TRAIN_FOLD" ] && [ -f "$HOLDOUT" ]; then
    c_yel "  exists — skipping"
else
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
json.dump(train, open('$TRAIN_FOLD','w'))
json.dump(hold,  open('$HOLDOUT','w'))
print(f'  train={len(train)} holdout={len(hold)}')"
fi

# -------- D. distill
step "D. Distill traces (~4-6 hours on A5000)"
if [ "$SKIP_DISTILL" = "1" ]; then
    c_yel "  SKIP_DISTILL=1"
elif [ -f "$TRACES" ] && [ "$(wc -l < "$TRACES")" -ge 800 ]; then
    c_yel "  $TRACES has $(wc -l < "$TRACES") traces — skipping"
else
    BEFORE=0; [ -f "$TRACES" ] && BEFORE=$(wc -l < "$TRACES")
    python scripts/distill.py \
        --train_file "$TRAIN_FOLD" \
        --output    "$TRACES" \
        --model_url "http://localhost:$LLM_PORT/v1" \
        --model_name "$MODEL_NAME" \
        --max_samples "$DISTILL_LIMIT" || { c_red "distill failed"; exit 1; }
    AFTER=$(wc -l < "$TRACES")
    c_green "  traces: $BEFORE -> $AFTER (+$((AFTER - BEFORE)))"
    if [ "$AFTER" -lt 200 ]; then
        c_red "  only $AFTER traces — fine-tune unlikely to help; aborting"
        exit 1
    fi
fi

# -------- E. stop llm_server before fine-tune
step "E. Stop llm_server to free GPU for training"
pkill -f "scripts/llm_server.py" 2>/dev/null || true
sleep 8
# Wait for the GPU memory to actually release
for _ in 1 2 3 4 5; do
    if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; then break; fi
    sleep 3
done
nvidia-smi --query-gpu=memory.free --format=csv,noheader

# -------- F. LoRA fine-tune
step "F. LoRA fine-tune (~1-2 hours on A5000)"
if [ "$SKIP_FINETUNE" = "1" ]; then
    c_yel "  SKIP_FINETUNE=1"
elif [ -d "$LORA_DIR" ] && [ -f "$LORA_DIR/adapter_config.json" ]; then
    c_yel "  $LORA_DIR exists — skipping"
else
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python scripts/finetune.py \
            --traces "$TRACES" \
            --output_dir "$LORA_PARENT" \
            --base_model "$MODEL_NAME" \
            --lora_r "$LORA_R" \
            --lora_alpha "$LORA_ALPHA" \
            --lora_dropout "$LORA_DROPOUT" \
            --max_seq_length "$LORA_MAX_SEQ_LEN" \
            --epochs "$LORA_NUM_EPOCHS" \
            --per_device_batch_size "$LORA_BATCH_SIZE" \
            --grad_accum "$LORA_GRAD_ACCUM" \
            --lr "$LORA_LR" \
        || { c_red "fine-tune failed (try smaller LORA_MAX_SEQ_LEN=1024 LORA_R=4)"; exit 1; }
    if [ ! -d "$LORA_DIR" ] || [ ! -f "$LORA_DIR/adapter_config.json" ]; then
        c_red "  no adapter at $LORA_DIR"
        exit 1
    fi
    c_green "  saved $LORA_DIR"
fi

# -------- G. restart llm_server with --lora
step "G. Restart llm_server with LoRA"
start_llm_server --lora "$LORA_DIR" || { c_red "lora server failed"; exit 1; }

# -------- H. holdout with LoRA
step "H. Score LoRA on holdout"
if [ ! -f eval/results/holdout_lora.log ] || \
   [ ! -f eval/results/holdout_lora/result.csv ] || \
   [ "$(wc -l < eval/results/holdout_lora/result.csv 2>/dev/null || echo 0)" -lt 200 ]; then
    rm -rf eval/results/holdout_lora
    # Bring up server.py for tools if needed
    if ! curl -sf "http://localhost:$TOOL_PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
        DATA_SPLIT=test nohup python server.py > "$TOOL_LOG" 2>&1 &
        wait_health "http://localhost:$TOOL_PORT/health" 60 || c_yel "  tool server failed, continuing w/o tools"
    fi
    python scripts/agentic_agent.py \
        --test_file "$HOLDOUT" \
        --out_dir   eval/results/holdout_lora \
        --llm_url   "http://localhost:$LLM_PORT" \
        --tool_url  "http://localhost:$TOOL_PORT" \
        --max_tokens 384 \
        --max_tool_calls 2 \
        --scenario_timeout_s 120 2>&1 | tee eval/results/holdout_lora.log
fi
SCORE_LORA=$(grep -m1 "mean   :" eval/results/holdout_lora.log 2>/dev/null | grep -oE "[0-9]\.[0-9]+" | head -1)
SCORE_BASE=""
[ -f eval/results/agentic_holdout.log ] && \
    SCORE_BASE=$(grep -m1 "mean   :" eval/results/agentic_holdout.log 2>/dev/null | grep -oE "[0-9]\.[0-9]+" | head -1)
c_green "  LoRA holdout    : ${SCORE_LORA:-?}"
[ -n "$SCORE_BASE" ] && c_green "  baseline holdout: $SCORE_BASE"

# -------- I. final run with LoRA on Phase 1 test
if [ "$SKIP_FINAL" = "1" ]; then
    c_yel "I. SKIP_FINAL=1 — not running on test set"
else
    step "I. Final run on $TEST_FILE with LoRA + tools (~1 hour)"
    if [ -f eval/results/lora_final/result.csv ] && \
       [ "$(wc -l < eval/results/lora_final/result.csv)" -ge 500 ]; then
        c_yel "  exists — skipping"
    else
        rm -rf eval/results/lora_final
        python scripts/agentic_agent.py \
            --test_file "$TEST_FILE" \
            --out_dir   eval/results/lora_final \
            --llm_url   "http://localhost:$LLM_PORT" \
            --tool_url  "http://localhost:$TOOL_PORT" \
            --max_tokens 384 \
            --max_tool_calls 2 \
            --scenario_timeout_s 120 2>&1 | tee eval/results/lora_final.log
    fi

    step "J. Convert to Zindi format"
    for v in v1_raw v2_multi_recall v3_insurance; do
        if [ -f "eval/results/lora_final/result_${v}.csv" ]; then
            zindi_convert "eval/results/lora_final/result_${v}.csv" \
                          "eval/results/lora_final/result_${v}_zindi.csv"
        fi
    done
fi

# -------- summary
step "DONE"
echo "Holdout scores:"
[ -n "$SCORE_BASE" ] && echo "  baseline agentic : $SCORE_BASE"
[ -n "$SCORE_LORA" ] && echo "  LoRA + agentic   : $SCORE_LORA"
echo
c_green "LoRA adapter      : $LORA_DIR"
c_green "Distilled traces  : $TRACES ($(wc -l < "$TRACES" 2>/dev/null || echo 0))"
echo
c_green "Submission candidates:"
ls -la eval/results/lora_final/result_*_zindi.csv 2>/dev/null
