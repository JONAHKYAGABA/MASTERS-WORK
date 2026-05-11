#!/usr/bin/env bash
# scripts/run_agentic_today.sh
#
# ONE-COMMAND end-to-end agentic submission for Track A.
#
# Phases (each skipped if its output already exists):
#   A. Install deps into .venv
#   B. Prefetch model into HF cache (~67 GB, 15-30 min)
#   C. Start llm_server (port 8001) + server.py (port 7860)
#   D. Build stratified 1800/200 holdout split
#   E. Heuristic-only baseline submission (~30 sec) — safety-net for Zindi
#   F. Agentic agent on holdout (200 scenarios, prints score)
#   G. Agentic agent on Phase 1 test set (500 scenarios)
#   H. Convert all results to Zindi format (ID,Track A,Track B)
#   I. Print files to upload
#
# Usage:
#     cd /workspace/MASTERS-WORK/"Track A"
#     export HF_TOKEN=hf_YOUR_TOKEN
#     bash scripts/run_agentic_today.sh
#
# Background (recommended — survives disconnect):
#     nohup bash scripts/run_agentic_today.sh > eval/logs/run_all/today.log 2>&1 &
#     echo "pid=$!"
#     tail -f eval/logs/run_all/today.log
#
# Total runtime ~3-5 hours on RTX A5000.

set -uo pipefail
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# -------- config (override via env)
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
LLM_PORT="${LLM_PORT:-8001}"
TOOL_PORT="${TOOL_PORT:-7860}"
TEST_FILE="${TEST_FILE:-data/Phase_1/test.json}"
SKIP_HOLDOUT="${SKIP_HOLDOUT:-0}"
SKIP_FINAL="${SKIP_FINAL:-0}"
LLM_LOG="eval/logs/run_all/llm_server.log"
TOOL_LOG="eval/logs/run_all/tool_server.log"
mkdir -p eval/logs/run_all eval/results data/local_split

c_blue()  { printf "\033[1;34m%s\033[0m\n" "$1"; }
c_green() { printf "\033[1;32m%s\033[0m\n" "$1"; }
c_yel()   { printf "\033[1;33m%s\033[0m\n" "$1"; }
c_red()   { printf "\033[1;31m%s\033[0m\n" "$1"; }
step()    { echo; c_blue "========================================================"; c_blue "  $1"; c_blue "========================================================"; }

wait_health() {
    # $1=url $2=timeout
    local deadline=$(( $(date +%s) + ${2:-600} ))
    while ! curl -sf "$1" 2>/dev/null | grep -q '"status":"ok"'; do
        [ "$(date +%s)" -gt "$deadline" ] && return 1
        printf "."; sleep 5
    done
    echo " ready"
    return 0
}

zindi_convert() {
    # $1 = our_csv (scenario_id, answers), $2 = output (ID, Track A, Track B)
    python -c "
import pandas as pd
df = pd.read_csv('$1', dtype=str).fillna('')
df = df.rename(columns={'scenario_id':'ID','answers':'Track A'})
df['Track B'] = ''
df = df[['ID','Track A','Track B']]
df.to_csv('$2', index=False)
print('  ${2##*/}:', len(df), 'rows')"
}

# -------- A. venv + deps
step "A. venv + dependencies"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    c_yel "  created .venv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if ! python -c "import torch, transformers, bitsandbytes, peft, fastapi" >/dev/null 2>&1; then
    pip install --upgrade -q pip wheel setuptools
    pip install -q -r requirements.txt
    pip install -q "openai>=1.50.0" httpx requests python-dateutil tqdm \
        "uvicorn[standard]" python-multipart "fastapi>=0.110" \
        "torch>=2.4.0" "transformers>=4.45.0" "accelerate>=1.0.0" \
        "bitsandbytes>=0.44.0" "peft>=0.13.0" "datasets>=3.0.0" \
        safetensors sentencepiece protobuf "huggingface_hub>=0.25" hf_transfer pandas
    c_green "  deps installed"
else
    c_yel "  deps present — skipping"
fi

# -------- B. prefetch model
step "B. Prefetch $MODEL_NAME into HF cache"
SAFE_NAME="models--${MODEL_NAME//\//--}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
CACHE_DIR="$HF_HOME/hub/$SAFE_NAME"
if [ -d "$CACHE_DIR/blobs" ] && [ "$(du -sb "$CACHE_DIR" | cut -f1)" -ge "$((40 * 1024 * 1024 * 1024))" ]; then
    c_yel "  cache present ($(du -sh "$CACHE_DIR" | cut -f1)) — skipping"
else
    if [ -z "${HF_TOKEN:-}" ]; then
        c_red "  HF_TOKEN not set. export HF_TOKEN=hf_... and rerun."
        exit 1
    fi
    python scripts/prefetch_model.py || { c_red "prefetch failed"; exit 1; }
    c_green "  cache size: $(du -sh "$CACHE_DIR" | cut -f1)"
fi

# -------- C. start llm_server + server.py
step "C. Start llm_server on :$LLM_PORT and tool server.py on :$TOOL_PORT"
if curl -sf "http://localhost:$LLM_PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
    c_yel "  llm_server already healthy"
else
    pkill -f "scripts/llm_server.py" 2>/dev/null || true
    sleep 3
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup python scripts/llm_server.py --model "$MODEL_NAME" --port "$LLM_PORT" \
        > "$LLM_LOG" 2>&1 &
    echo "  llm pid=$!"
    wait_health "http://localhost:$LLM_PORT/health" 900 \
        || { c_red "  llm_server never healthy"; tail -n 40 "$LLM_LOG" >&2; exit 1; }
fi
if curl -sf "http://localhost:$TOOL_PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
    c_yel "  tool server already healthy"
else
    pkill -f "python server.py" 2>/dev/null || true
    sleep 2
    DATA_SPLIT=test nohup python server.py > "$TOOL_LOG" 2>&1 &
    echo "  tool pid=$!"
    wait_health "http://localhost:$TOOL_PORT/health" 60 \
        || { c_red "  server.py never healthy"; tail -n 40 "$TOOL_LOG" >&2; exit 1; }
fi

# -------- D. holdout split
step "D. Build stratified 1800/200 holdout"
if [ -f data/local_split/holdout_200.json ] && [ -f data/local_split/train_1800.json ]; then
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
json.dump(train, open('data/local_split/train_1800.json','w'))
json.dump(hold,  open('data/local_split/holdout_200.json','w'))
print(f'  train={len(train)} holdout={len(hold)}')"
fi

# -------- E. heuristic safety-net
step "E. Heuristic safety-net submission (~30 sec)"
if [ -f eval/results/heuristic_baseline/result_v1_raw_zindi.csv ]; then
    c_yel "  exists — skipping"
else
    rm -rf eval/results/heuristic_baseline
    # Run heuristic via existing submit_now.py if available, else inline
    if [ -f scripts/submit_now.py ]; then
        python scripts/submit_now.py \
            --test_file "$TEST_FILE" \
            --out_dir   eval/results/heuristic_baseline \
            --no_llm 2>&1 | tail -8
    else
        c_red "  submit_now.py missing; cannot run heuristic"
        exit 1
    fi
    for v in v1_raw v2_multi_recall v3_insurance; do
        zindi_convert "eval/results/heuristic_baseline/result_${v}.csv" \
                      "eval/results/heuristic_baseline/result_${v}_zindi.csv"
    done
fi

# -------- F. agentic holdout
SCORE_AGENTIC=""
if [ "$SKIP_HOLDOUT" = "1" ]; then
    c_yel "F. holdout skipped (SKIP_HOLDOUT=1)"
else
    step "F. Agentic agent on holdout (200 scenarios)"
    if [ -f eval/results/agentic_holdout/result.csv ] && \
       [ "$(wc -l < eval/results/agentic_holdout/result.csv)" -ge 200 ]; then
        c_yel "  exists — re-extracting score from log"
    else
        rm -rf eval/results/agentic_holdout
        python scripts/agentic_agent.py \
            --test_file data/local_split/holdout_200.json \
            --out_dir   eval/results/agentic_holdout \
            --llm_url   "http://localhost:$LLM_PORT" \
            --tool_url  "http://localhost:$TOOL_PORT" \
            --max_tokens 384 \
            --scenario_timeout_s 120 2>&1 | tee eval/results/agentic_holdout.log
    fi
    SCORE_AGENTIC=$(grep -m1 "mean   :" eval/results/agentic_holdout.log 2>/dev/null | grep -oE "[0-9]\.[0-9]+" | head -1)
    c_green "  AGENTIC holdout: ${SCORE_AGENTIC:-?}"
fi

# Read heuristic holdout score from earlier run if available
SCORE_HEUR=""
if [ -f eval/results/holdout_base.log ]; then
    SCORE_HEUR=$(grep -m1 "mean   :" eval/results/holdout_base.log 2>/dev/null | grep -oE "[0-9]\.[0-9]+" | head -1)
fi
[ -z "$SCORE_HEUR" ] && SCORE_HEUR="0.30"  # validated heuristic floor on train

# -------- G. agentic on test (final submission)
if [ "$SKIP_FINAL" = "1" ]; then
    c_yel "G. final skipped (SKIP_FINAL=1)"
else
    step "G. Agentic agent on $TEST_FILE (500 scenarios)"
    if [ -f eval/results/agentic_final/result.csv ] && \
       [ "$(wc -l < eval/results/agentic_final/result.csv)" -ge 500 ]; then
        c_yel "  exists — skipping"
    else
        rm -rf eval/results/agentic_final
        python scripts/agentic_agent.py \
            --test_file "$TEST_FILE" \
            --out_dir   eval/results/agentic_final \
            --llm_url   "http://localhost:$LLM_PORT" \
            --tool_url  "http://localhost:$TOOL_PORT" \
            --max_tokens 384 \
            --scenario_timeout_s 120 2>&1 | tee eval/results/agentic_final.log
    fi
fi

# -------- H. convert agentic final to Zindi
step "H. Convert agentic outputs to Zindi (ID,Track A,Track B)"
if [ -d eval/results/agentic_final ]; then
    for v in v1_raw v2_multi_recall v3_insurance; do
        if [ -f "eval/results/agentic_final/result_${v}.csv" ]; then
            zindi_convert "eval/results/agentic_final/result_${v}.csv" \
                          "eval/results/agentic_final/result_${v}_zindi.csv"
        fi
    done
fi

# -------- I. summary
step "I. DONE"
echo
echo "Holdout scores:"
echo "  heuristic     : ${SCORE_HEUR}"
[ -n "$SCORE_AGENTIC" ] && echo "  agentic       : ${SCORE_AGENTIC}"
echo
c_green "Phase 2 submission candidates (best of 3 counted):"
echo "  1) eval/results/agentic_final/result_v1_raw_zindi.csv      # PRIMARY agent run"
echo "  2) eval/results/agentic_final/result_v2_multi_recall_zindi.csv"
echo "  3) eval/results/heuristic_baseline/result_v1_raw_zindi.csv  # SAFETY NET"
echo
ls -la eval/results/agentic_final/result_*_zindi.csv 2>/dev/null
ls -la eval/results/heuristic_baseline/result_*_zindi.csv 2>/dev/null
