#!/usr/bin/env bash
# scripts/run_today.sh
#
# ONE script for the entire Track A submission pipeline. Resumable —
# re-running picks up where the previous run left off.
#
#   1. Holdout build (1800/200 stratified)
#   2. Baseline holdout score (LLM, no RAG, no LoRA)
#   3. Build RAG knowledge base (scrape ShareTechNote, embed)
#   4. RAG holdout score (LLM + RAG)
#   5. Distill 1800 train scenarios -> traces
#   6. LoRA fine-tune
#   7. LoRA holdout score (LoRA + RAG if available)
#   8. Decide best config (none / RAG / LoRA / LoRA+RAG)
#   9. Final run on data/Phase_1/test.json with the winning config
#  10. Convert to Zindi format (ID,Track A,Track B), no external template needed
#
# Usage:
#     bash scripts/run_today.sh                # full run
#     SKIP_LORA=1 bash scripts/run_today.sh    # everything except LoRA (fast)
#     SKIP_RAG=1  bash scripts/run_today.sh    # everything except RAG
#     SKIP_HOLDOUT=1 bash scripts/run_today.sh # straight to final run
#
# Run in background so it survives disconnects:
#     nohup bash scripts/run_today.sh > eval/logs/run_today.log 2>&1 &
#     echo "pid=$!"
#     tail -f eval/logs/run_today.log
#
# Total runtime: ~10-12 hours full (mostly distillation + fine-tune).
# Without LoRA: ~2 hours.

set -uo pipefail
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found at $PROJECT_DIR/.venv"
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# -------- config (override via env vars)
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
LLM_PORT="${LLM_PORT:-8001}"
TEST_FILE="${TEST_FILE:-data/Phase_1/test.json}"
TRAIN_FILE_FOLD="${TRAIN_FILE_FOLD:-data/local_split/train_1800.json}"
HOLDOUT_FILE="${HOLDOUT_FILE:-data/local_split/holdout_200.json}"
TRACES="${TRACES:-traces/train_traces.jsonl}"
LORA_PARENT="${LORA_PARENT:-training/checkpoints/run_v1}"
LORA_DIR="$LORA_PARENT/best_lora"
LLM_LOG="${LLM_LOG:-eval/logs/run_all/llm_server.log}"

SKIP_HOLDOUT="${SKIP_HOLDOUT:-0}"
SKIP_RAG="${SKIP_RAG:-0}"
SKIP_LORA="${SKIP_LORA:-0}"
SKIP_DISTILL="${SKIP_DISTILL:-0}"
SKIP_FINETUNE="${SKIP_FINETUNE:-0}"

mkdir -p data/local_split traces "$LORA_PARENT" eval/logs/run_all eval/results

# -------- ui
c_blue()  { printf "\033[1;34m%s\033[0m\n" "$1"; }
c_green() { printf "\033[1;32m%s\033[0m\n" "$1"; }
c_yel()   { printf "\033[1;33m%s\033[0m\n" "$1"; }
c_red()   { printf "\033[1;31m%s\033[0m\n" "$1"; }
step()    { echo; c_blue "========================================================"; c_blue "  $1"; c_blue "========================================================"; }

wait_for_health() {
    local timeout_s="${1:-1200}"
    local deadline=$(( $(date +%s) + timeout_s ))
    while ! curl -sf "http://localhost:$LLM_PORT/health" 2>/dev/null | grep -q '"status":"ok"'; do
        if [ "$(date +%s)" -gt "$deadline" ]; then
            c_red "  llm_server never became healthy"
            tail -n 60 "$LLM_LOG" >&2 || true
            return 1
        fi
        printf "."; sleep 5
    done
    echo " ready"; return 0
}

start_llm_server() {
    pkill -f "scripts/llm_server.py" 2>/dev/null || true
    sleep 5
    nohup python scripts/llm_server.py --model "$MODEL_NAME" --port "$LLM_PORT" "$@" \
        > "$LLM_LOG" 2>&1 &
    local pid=$!
    echo "  llm_server pid=$pid (log=$LLM_LOG)"
    wait_for_health 1500
}

ensure_llm_server() {
    # Args: extra args to pass to llm_server (e.g. --lora <path>)
    local want_lora="${1:-}"
    local current_id=""
    if curl -sf "http://localhost:$LLM_PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
        current_id=$(curl -s "http://localhost:$LLM_PORT/health" | python -c 'import sys,json;print(json.load(sys.stdin).get("model",""))' 2>/dev/null || echo "")
        if [ -z "$want_lora" ] && [[ "$current_id" != *"+lora"* ]]; then
            return 0
        fi
        if [ -n "$want_lora" ] && [[ "$current_id" == *"+lora"* ]]; then
            return 0
        fi
        c_yel "  current server is '$current_id', restarting to match request"
    else
        c_yel "  llm_server not running, starting"
    fi
    if [ -n "$want_lora" ]; then
        start_llm_server --lora "$want_lora"
    else
        start_llm_server
    fi
}

extract_score() {
    # $1 = log file. Picks the line "  mean   : 0.NNNN" written by submit_now.py
    grep -m1 "mean   :" "$1" 2>/dev/null | grep -oE "[0-9]\.[0-9]+" | head -1
}

write_zindi_csv() {
    # $1 = our_csv (scenario_id, answers). $2 = output zindi_csv (ID, Track A, Track B).
    python -c "
import pandas as pd
df = pd.read_csv('$1', dtype=str).fillna('')
df = df.rename(columns={'scenario_id':'ID','answers':'Track A'})
df['Track B'] = ''
df = df[['ID','Track A','Track B']]
df.to_csv('$2', index=False)
print('  ${2##*/}:', len(df), 'rows')"
}

# -------- preflight: holdout split
step "PREFLIGHT — holdout split (1800/200 stratified)"
if [ -f "$TRAIN_FILE_FOLD" ] && [ -f "$HOLDOUT_FILE" ]; then
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
json.dump(train, open('$TRAIN_FILE_FOLD','w'))
json.dump(hold,  open('$HOLDOUT_FILE','w'))
print(f'  train={len(train)} holdout={len(hold)}')
" || { c_red "split failed"; exit 1; }
fi

# -------- preflight: llm_server (base, no LoRA initially)
step "PREFLIGHT — llm_server (base model)"
ensure_llm_server "" || { c_red "llm_server unavailable"; exit 1; }

# -------- step 1: baseline holdout
SCORE_BASE=""
if [ "$SKIP_HOLDOUT" = "1" ]; then
    c_yel "STEP 1 — skipping (SKIP_HOLDOUT=1)"
else
    step "STEP 1 — Baseline holdout (LLM only)"
    if [ -f eval/results/holdout_base.log ] && [ -f eval/results/holdout_base/result.csv ] && \
       [ "$(wc -l < eval/results/holdout_base/result.csv)" -ge 200 ]; then
        c_yel "  exists — reading score"
    else
        rm -rf eval/results/holdout_base
        python scripts/submit_now.py \
            --test_file "$HOLDOUT_FILE" \
            --out_dir   eval/results/holdout_base \
            --max_tokens 64 2>&1 | tee eval/results/holdout_base.log || true
    fi
    SCORE_BASE=$(extract_score eval/results/holdout_base.log)
    c_green "  baseline score: ${SCORE_BASE:-?}"
fi

# -------- step 2: build RAG KB
RAG_AVAILABLE=0
if [ "$SKIP_RAG" = "1" ]; then
    c_yel "STEP 2 — skipping RAG (SKIP_RAG=1)"
else
    step "STEP 2 — Build RAG knowledge base"
    if [ -f knowledge/processed/embeddings.npy ] && [ -f knowledge/processed/chunks.json ]; then
        c_yel "  KB exists ($(python -c "import json;print(len(json.load(open('knowledge/processed/chunks.json'))))") chunks)"
        RAG_AVAILABLE=1
    else
        echo "  installing deps"
        pip install -q requests trafilatura sentence-transformers numpy 2>&1 | tail -2 || true
        if python scripts/scrape_5g_kb.py && python scripts/build_kb_index.py; then
            RAG_AVAILABLE=1
            c_green "  KB ready"
        else
            c_red "  KB build failed; continuing without RAG"
        fi
    fi
fi

# -------- step 3: holdout WITH RAG
SCORE_RAG=""
if [ "$SKIP_HOLDOUT" != "1" ] && [ "$RAG_AVAILABLE" = "1" ]; then
    step "STEP 3 — Holdout LLM + RAG"
    if [ -f eval/results/holdout_rag.log ] && [ -f eval/results/holdout_rag/result.csv ] && \
       [ "$(wc -l < eval/results/holdout_rag/result.csv)" -ge 200 ]; then
        c_yel "  exists — reading score"
    else
        rm -rf eval/results/holdout_rag
        python scripts/submit_now.py \
            --test_file "$HOLDOUT_FILE" \
            --out_dir   eval/results/holdout_rag \
            --max_tokens 128 --use_rag --rag_k 3 2>&1 | tee eval/results/holdout_rag.log || true
    fi
    SCORE_RAG=$(extract_score eval/results/holdout_rag.log)
    c_green "  LLM+RAG score: ${SCORE_RAG:-?}"
fi

# -------- step 4: distill
DISTILL_OK=0
if [ "$SKIP_LORA" = "1" ] || [ "$SKIP_DISTILL" = "1" ]; then
    c_yel "STEP 4 — skipping distill"
else
    step "STEP 4 — Distill 1800 train scenarios -> traces (~6 hours)"
    if [ -f "$TRACES" ] && [ "$(wc -l < "$TRACES")" -ge 800 ]; then
        c_yel "  $TRACES has $(wc -l < "$TRACES") traces — skipping"
        DISTILL_OK=1
    else
        BEFORE=0; [ -f "$TRACES" ] && BEFORE=$(wc -l < "$TRACES")
        if python scripts/distill.py \
            --train_file "$TRAIN_FILE_FOLD" \
            --output    "$TRACES" \
            --model_url "http://localhost:$LLM_PORT/v1" \
            --model_name "$MODEL_NAME"; then
            AFTER=$(wc -l < "$TRACES")
            c_green "  traces: $BEFORE -> $AFTER"
            [ "$AFTER" -ge 200 ] && DISTILL_OK=1 || c_red "  too few traces ($AFTER) — skipping fine-tune"
        else
            c_red "  distill failed"
        fi
    fi
fi

# -------- step 5: LoRA fine-tune
LORA_AVAILABLE=0
if [ -d "$LORA_DIR" ] && [ -f "$LORA_DIR/adapter_config.json" ]; then
    c_yel "STEP 5 — LoRA already exists at $LORA_DIR — skipping fine-tune"
    LORA_AVAILABLE=1
elif [ "$SKIP_LORA" = "1" ] || [ "$SKIP_FINETUNE" = "1" ]; then
    c_yel "STEP 5 — skipping fine-tune"
elif [ "$DISTILL_OK" = "1" ]; then
    step "STEP 5 — LoRA fine-tune (~2-3 hours)"
    echo "  stopping llm_server to free GPU 0"
    pkill -f "scripts/llm_server.py" 2>/dev/null || true
    sleep 8
    if CUDA_VISIBLE_DEVICES=0 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       python scripts/finetune.py \
           --traces "$TRACES" \
           --output_dir "$LORA_PARENT" \
           --base_model "$MODEL_NAME"; then
        if [ -d "$LORA_DIR" ] && [ -f "$LORA_DIR/adapter_config.json" ]; then
            LORA_AVAILABLE=1
            c_green "  saved $LORA_DIR"
        else
            c_red "  fine-tune produced no adapter"
        fi
    else
        c_red "  fine-tune failed"
    fi
fi

# -------- step 6: holdout WITH LoRA (and RAG if available)
SCORE_LORA=""
if [ "$LORA_AVAILABLE" = "1" ]; then
    step "STEP 6 — Holdout with LoRA $([ "$RAG_AVAILABLE" = 1 ] && echo "+ RAG")"
    ensure_llm_server "$LORA_DIR" || { c_red "lora server failed"; LORA_AVAILABLE=0; }
    if [ "$LORA_AVAILABLE" = "1" ]; then
        if [ -f eval/results/holdout_lora.log ] && [ -f eval/results/holdout_lora/result.csv ] && \
           [ "$(wc -l < eval/results/holdout_lora/result.csv)" -ge 200 ]; then
            c_yel "  exists — reading score"
        else
            rm -rf eval/results/holdout_lora
            RAG_FLAGS=""
            [ "$RAG_AVAILABLE" = "1" ] && RAG_FLAGS="--use_rag --rag_k 3"
            python scripts/submit_now.py \
                --test_file "$HOLDOUT_FILE" \
                --out_dir   eval/results/holdout_lora \
                --max_tokens 128 $RAG_FLAGS 2>&1 | tee eval/results/holdout_lora.log || true
        fi
        SCORE_LORA=$(extract_score eval/results/holdout_lora.log)
        c_green "  LoRA score: ${SCORE_LORA:-?}"
    fi
fi

# -------- step 7: pick best config
step "STEP 7 — Decide best config"
BEST_LABEL="baseline"
BEST_SCORE="$SCORE_BASE"
USE_RAG_FINAL=0
USE_LORA_FINAL=0

choose() {
    local label="$1" score="$2" rag="$3" lora="$4"
    [ -z "$score" ] && return
    if [ -z "$BEST_SCORE" ] || python -c "import sys; sys.exit(0 if float('$score') > float('$BEST_SCORE') else 1)"; then
        BEST_LABEL="$label"; BEST_SCORE="$score"; USE_RAG_FINAL="$rag"; USE_LORA_FINAL="$lora"
    fi
}
choose "baseline"   "$SCORE_BASE" 0 0
choose "RAG"        "$SCORE_RAG"  1 0
[ "$LORA_AVAILABLE" = "1" ] && choose "LoRA${RAG_AVAILABLE:+ +RAG}" "$SCORE_LORA" "$RAG_AVAILABLE" 1

c_green "  baseline       : ${SCORE_BASE:-?}"
c_green "  +RAG           : ${SCORE_RAG:-?}"
c_green "  +LoRA(+RAG)    : ${SCORE_LORA:-?}"
c_green "  WINNER -> $BEST_LABEL  (score=${BEST_SCORE:-?})"

# Per-config final output dir so we never overwrite previous-run results
case "$BEST_LABEL" in
    baseline) FINAL_DIR="eval/results/final_baseline" ;;
    RAG)      FINAL_DIR="eval/results/final_rag" ;;
    *LoRA*)   FINAL_DIR="eval/results/final_lora_rag" ;;
    *)        FINAL_DIR="eval/results/final_baseline" ;;
esac
echo "  final output dir: $FINAL_DIR"

# One-time migration: if the user has eval/results/final/ from before
# (which was always baseline), keep it in place under final_baseline
if [ -d eval/results/final ] && [ ! -d "$FINAL_DIR" ] && [ "$BEST_LABEL" = "baseline" ]; then
    cp -r eval/results/final "$FINAL_DIR"
    c_yel "  migrated existing eval/results/final -> $FINAL_DIR"
fi

# -------- step 8: final run (skip if outputs for this config already exist)
step "STEP 8 — Final run on $TEST_FILE (config=$BEST_LABEL)"
if [ -f "$FINAL_DIR/result_v1_raw.csv" ] && \
   [ "$(wc -l < "$FINAL_DIR/result_v1_raw.csv" 2>/dev/null || echo 0)" -ge 500 ]; then
    c_yel "  $FINAL_DIR/result_v1_raw.csv already complete (>=500 rows) — SKIPPING final LLM run"
else
    # Restart server with the right config (only when actually re-running)
    if [ "$USE_LORA_FINAL" = "1" ]; then
        ensure_llm_server "$LORA_DIR" || { c_red "lora server unavailable for final"; exit 1; }
    else
        ensure_llm_server "" || { c_red "base server unavailable for final"; exit 1; }
    fi
    RAG_FLAGS=""
    TOK_FLAG="--max_tokens 64"
    if [ "$USE_RAG_FINAL" = "1" ]; then
        RAG_FLAGS="--use_rag --rag_k 3"
        TOK_FLAG="--max_tokens 128"
    fi
    python scripts/submit_now.py \
        --test_file "$TEST_FILE" \
        --out_dir   "$FINAL_DIR" \
        $TOK_FLAG $RAG_FLAGS || { c_red "final run failed"; exit 1; }
fi

# -------- step 9: convert to Zindi format (skip per-file if zindi already exists)
step "STEP 9 — Convert to Zindi format (ID,Track A,Track B)"
for v in v1_raw v2_multi_recall v3_insurance; do
    OUT="$FINAL_DIR/result_${v}_zindi.csv"
    if [ -f "$OUT" ] && [ "$(wc -l < "$OUT" 2>/dev/null || echo 0)" -ge 500 ]; then
        c_yel "  ${v}_zindi exists — skipping"
    else
        write_zindi_csv "$FINAL_DIR/result_${v}.csv" "$OUT"
    fi
done
# Also write a heuristic-only safety-net submission
step "BONUS — heuristic-only safety-net submission"
if [ -f eval/results/heuristic_only/heuristic_zindi.csv ] && \
   [ "$(wc -l < eval/results/heuristic_only/heuristic_zindi.csv 2>/dev/null || echo 0)" -ge 500 ]; then
    c_yel "  heuristic_zindi.csv exists — skipping"
else
    if [ -f eval/results/heuristic_only/result_v1_raw.csv ] && \
       [ "$(wc -l < eval/results/heuristic_only/result_v1_raw.csv 2>/dev/null || echo 0)" -ge 500 ]; then
        c_yel "  heuristic result.csv exists — converting only"
    else
        rm -rf eval/results/heuristic_only
        python scripts/submit_now.py \
            --test_file "$TEST_FILE" \
            --out_dir   eval/results/heuristic_only \
            --no_llm 2>&1 | tail -8 || true
    fi
    write_zindi_csv \
        eval/results/heuristic_only/result_v1_raw.csv \
        eval/results/heuristic_only/heuristic_zindi.csv
fi

# -------- summary
step "DONE"
echo "Holdout scores:"
echo "  baseline       : ${SCORE_BASE:-?}"
echo "  +RAG           : ${SCORE_RAG:-?}"
echo "  +LoRA(+RAG)    : ${SCORE_LORA:-?}"
echo "Final config used: $BEST_LABEL  (LoRA=$USE_LORA_FINAL  RAG=$USE_RAG_FINAL)"
echo
c_green "Upload ANY 3 of these to Zindi (best of 3 counted):"
ls -la eval/results/final/result_*_zindi.csv 2>/dev/null
ls -la eval/results/heuristic_only/heuristic_zindi.csv 2>/dev/null
echo
c_green "Recommended trio:"
echo "  1) $FINAL_DIR/result_v1_raw_zindi.csv         # winning config ($BEST_LABEL)"
echo "  2) $FINAL_DIR/result_v2_multi_recall_zindi.csv # multi-recall variant"
echo "  3) eval/results/heuristic_only/heuristic_zindi.csv    # safety net (~26.8% on train)"
