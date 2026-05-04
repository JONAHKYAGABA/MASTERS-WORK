#!/usr/bin/env bash
# scripts/run_today.sh
#
# ONE COMMAND for today's submission cycle:
#   1. Build stratified 1800/200 holdout
#   2. Score baseline LLM (no RAG) on holdout
#   3. Install RAG deps, scrape ShareTechNote, build embedding index
#   4. Score LLM + RAG on holdout (decision gate: keep RAG only if it helps)
#   5. Run final pipeline on Phase 1 test set
#   6. Convert all 3 outputs to Zindi format
#   7. Print which CSVs to upload
#
# Idempotent: re-running skips steps whose outputs already exist.
#
# Run from the Track A directory:
#     bash scripts/run_today.sh
#
# Optional overrides:
#     TEST_FILE=data/Phase_1/test.json   bash scripts/run_today.sh
#     SKIP_RAG=1                         bash scripts/run_today.sh   # baseline only
#     SKIP_HOLDOUT=1                     bash scripts/run_today.sh   # straight to test set

set -uo pipefail   # NOTE: not -e — we want graceful fallback if RAG scrape fails

# -------- env
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found in $PROJECT_DIR. Run setup first."
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

TEST_FILE="${TEST_FILE:-data/Phase_1/test.json}"
SKIP_RAG="${SKIP_RAG:-0}"
SKIP_HOLDOUT="${SKIP_HOLDOUT:-0}"
LLM_PORT="${LLM_PORT:-8001}"
SAMPLE_TEMPLATE="${SAMPLE_TEMPLATE:-../submission/Phase_1/result.csv}"

c_blue()  { printf "\033[1;34m%s\033[0m\n" "$1"; }
c_green() { printf "\033[1;32m%s\033[0m\n" "$1"; }
c_yel()   { printf "\033[1;33m%s\033[0m\n" "$1"; }
c_red()   { printf "\033[1;31m%s\033[0m\n" "$1"; }

step() { echo; c_blue "============================================================"; c_blue "  $1"; c_blue "============================================================"; }

# -------- preflight: llm_server
step "PREFLIGHT — check llm_server health on port $LLM_PORT"
if curl -sf "http://localhost:$LLM_PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
    c_green "  llm_server healthy"
else
    c_red "  llm_server NOT running. Start it first:"
    echo
    echo "    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\"
    echo "      nohup python scripts/llm_server.py --model Qwen/Qwen3.5-35B-A3B --port $LLM_PORT \\"
    echo "      > eval/logs/run_all/llm_server.log 2>&1 &"
    echo
    echo "  Then re-run this script."
    exit 1
fi

# -------- step 1: stratified holdout
step "STEP 1 — Build stratified 1800/200 holdout"
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
print(f'  train={len(train)}  holdout={len(hold)}')
" || { c_red "holdout build failed"; exit 1; }
fi

# -------- step 2: holdout baseline (no RAG)
SCORE_BASE=""
if [ "$SKIP_HOLDOUT" = "1" ]; then
    c_yel "skipping holdout (SKIP_HOLDOUT=1)"
else
    step "STEP 2 — Holdout baseline (LLM, no RAG)"
    if [ -f eval/results/holdout_base/result.csv ] && \
       [ "$(wc -l < eval/results/holdout_base/result.csv 2>/dev/null || echo 0)" -ge 200 ]; then
        c_yel "  exists — skipping"
    else
        rm -rf eval/results/holdout_base
        python scripts/submit_now.py \
            --test_file data/local_split/holdout_200.json \
            --out_dir   eval/results/holdout_base \
            --max_tokens 64 2>&1 | tee eval/results/holdout_base.log || true
    fi
    SCORE_BASE=$(grep -m1 "mean   :" eval/results/holdout_base.log 2>/dev/null \
                 | grep -oE "[0-9]\.[0-9]+" | head -1)
    c_green "  baseline holdout score: ${SCORE_BASE:-?}"
fi

# -------- step 3: build RAG KB
USE_RAG=0
if [ "$SKIP_RAG" = "1" ]; then
    c_yel "skipping RAG (SKIP_RAG=1)"
else
    step "STEP 3 — Build RAG knowledge base"
    if [ -f knowledge/processed/embeddings.npy ] && [ -f knowledge/processed/chunks.json ]; then
        c_yel "  KB exists — skipping scrape + index"
        USE_RAG=1
    else
        echo "  installing RAG deps (one-time)"
        pip install -q requests trafilatura sentence-transformers numpy 2>&1 | tail -2 || true
        echo "  scraping ShareTechNote..."
        if python scripts/scrape_5g_kb.py; then
            echo "  indexing chunks..."
            if python scripts/build_kb_index.py; then
                USE_RAG=1
                c_green "  KB ready: $(wc -l < knowledge/processed/chunks.json 2>/dev/null || echo 0) chunks"
            else
                c_red "  index build failed; continuing without RAG"
            fi
        else
            c_red "  scrape failed; continuing without RAG"
        fi
    fi
fi

# -------- step 4: holdout WITH RAG
SCORE_RAG=""
RAG_DECISION=0
if [ "$SKIP_HOLDOUT" != "1" ] && [ "$USE_RAG" = "1" ]; then
    step "STEP 4 — Holdout LLM + RAG"
    rm -rf eval/results/holdout_rag
    python scripts/submit_now.py \
        --test_file data/local_split/holdout_200.json \
        --out_dir   eval/results/holdout_rag \
        --max_tokens 128 --use_rag --rag_k 3 2>&1 | tee eval/results/holdout_rag.log || true
    SCORE_RAG=$(grep -m1 "mean   :" eval/results/holdout_rag.log 2>/dev/null \
                | grep -oE "[0-9]\.[0-9]+" | head -1)
    c_green "  RAG holdout score: ${SCORE_RAG:-?}"

    if [ -n "$SCORE_BASE" ] && [ -n "$SCORE_RAG" ]; then
        if python -c "import sys; sys.exit(0 if float('$SCORE_RAG') > float('$SCORE_BASE') + 0.005 else 1)"; then
            RAG_DECISION=1
            c_green "  RAG helps (+$(python -c "print(round(float('$SCORE_RAG')-float('$SCORE_BASE'),4))")) — using for final"
        else
            c_yel  "  RAG does not help (delta=$(python -c "print(round(float('$SCORE_RAG')-float('$SCORE_BASE'),4))")) — final run will skip it"
        fi
    fi
fi

# -------- step 5: final run on test set
step "STEP 5 — Final run on $TEST_FILE"
EXTRA_FLAGS=""
if [ "$RAG_DECISION" = "1" ]; then
    EXTRA_FLAGS="--use_rag --rag_k 3 --max_tokens 128"
    c_green "  using RAG"
else
    EXTRA_FLAGS="--max_tokens 64"
    c_yel "  not using RAG"
fi

rm -rf eval/results/final
python scripts/submit_now.py \
    --test_file "$TEST_FILE" \
    --out_dir   eval/results/final \
    $EXTRA_FLAGS || { c_red "final run failed"; exit 1; }

# -------- step 6: convert all 3 to Zindi format
step "STEP 6 — Convert outputs to Zindi format (ID,Track A,Track B)"
if [ ! -f "$SAMPLE_TEMPLATE" ]; then
    c_red "  sample template not found at $SAMPLE_TEMPLATE — skipping convert"
else
    for v in v1_raw v2_multi_recall v3_insurance; do
        python scripts/convert_to_zindi_format.py \
            --our_csv    "eval/results/final/result_${v}.csv" \
            --sample_csv "$SAMPLE_TEMPLATE" \
            --out        "eval/results/final/result_${v}_zindi.csv" \
            --track A || c_red "  convert ${v} failed"
    done
fi

# -------- summary
step "DONE — UPLOAD THESE FILES TO ZINDI"
ls -la eval/results/final/result_*_zindi.csv 2>/dev/null || true
echo
[ -n "$SCORE_BASE" ]    && echo "Holdout (LLM only)   : $SCORE_BASE"
[ -n "$SCORE_RAG" ]     && echo "Holdout (LLM + RAG)  : $SCORE_RAG"
echo "RAG used in final    : $([ "$RAG_DECISION" = "1" ] && echo YES || echo NO)"
echo
c_green "Upload these three to Zindi (best of 3 counted):"
echo "  eval/results/final/result_v1_raw_zindi.csv"
echo "  eval/results/final/result_v2_multi_recall_zindi.csv"
echo "  eval/results/final/result_v3_insurance_zindi.csv"
