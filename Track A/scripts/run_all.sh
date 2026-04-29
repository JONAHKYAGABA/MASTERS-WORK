#!/usr/bin/env bash
# scripts/run_all.sh
#
# ONE SCRIPT END-TO-END. Visible progress at every step.
#
# Stages (each prints a clear banner and an elapsed-time line on completion):
#   1.  venv + deps
#   2.  HuggingFace cache pin + train/test data check
#   3.  Prefetch base model (with progress bar; resumes if interrupted)
#   4.  Start llm_server in background, tail its log live until /health=ok
#   5.  Start locked server.py with DATA_SPLIT=test
#   6.  Run agent on 500 test scenarios → Phase 1 result.csv
#   7.  Self-distillation on 2000 train scenarios → traces/train_traces.jsonl
#   8.  LoRA fine-tune on the corpus → training/checkpoints/run_v1/best_lora/
#   9.  Restart llm_server with LoRA, run agent on 500 test scenarios → Phase 2 result.csv
#   10. Build 3 candidate submission files
#
# Single command (foreground; you see everything live):
#     bash scripts/run_all.sh
#
# Or detached + still-watchable:
#     nohup bash scripts/run_all.sh > run_all.log 2>&1 &
#     tail -f run_all.log
#
# Resumable: re-running skips work that already produced its output.
#
# Useful overrides:
#   N_TEST=500            (number of test scenarios; reduce for a fast sanity run)
#   N_TRAIN_DISTILL=2000  (cap distillation training scenarios)
#   MODEL_NAME=Qwen/...   (override base model)
#   SKIP_DISTILL=1        (skip stage 7)
#   SKIP_FINETUNE=1       (skip stage 8 — Phase 2 will reuse base model)
#   FAST=1                (skip slow optional steps; for quick smoke test)

set -euo pipefail

# ============================================================
#                     CONFIG + ENV VARS
# ============================================================
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"  # 3x faster downloads
export TOKENIZERS_PARALLELISM=false
export SEED="${SEED:-42}"
export PYTHONHASHSEED="$SEED"
mkdir -p "$HF_HOME/hub"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
LLM_PORT="${LLM_PORT:-8001}"
SERVER_PORT="${SERVER_PORT:-7860}"
N_TEST="${N_TEST:-500}"
N_TRAIN_DISTILL="${N_TRAIN_DISTILL:-2000}"
DATA_SPLIT_RUN="${DATA_SPLIT_RUN:-test}"

PHASE1_DIR="eval/results/phase1_test"
PHASE2_DIR="eval/results/phase2_test"
TRACES="traces/train_traces.jsonl"
LORA_DIR="training/checkpoints/run_v1/best_lora"
LOG_DIR="eval/logs/run_all"
LLM_LOG="$LOG_DIR/llm_server.log"
SERVER_LOG="$LOG_DIR/server.log"
mkdir -p "$PHASE1_DIR" "$PHASE2_DIR" "$(dirname "$TRACES")" \
    "$(dirname "$LORA_DIR")" "$LOG_DIR"

LLM_PID=""
SERVER_PID=""
TAIL_PID=""

# ============================================================
#                       UI HELPERS
# ============================================================
TOTAL_STEPS=10
STEP_NUM=0
PIPELINE_START=$(date +%s)
STEP_START=0
CURRENT_STAGE="init"

c_blue()   { printf "\033[1;34m%s\033[0m" "$1"; }
c_green()  { printf "\033[1;32m%s\033[0m" "$1"; }
c_yellow() { printf "\033[1;33m%s\033[0m" "$1"; }
c_red()    { printf "\033[1;31m%s\033[0m" "$1"; }

step() {
    STEP_NUM=$((STEP_NUM + 1))
    CURRENT_STAGE="$1"
    STEP_START=$(date +%s)
    echo
    echo "$(c_blue "============================================================")"
    echo "$(c_blue "  STEP $STEP_NUM/$TOTAL_STEPS  ─  $1")"
    echo "$(c_blue "============================================================")"
}

step_done() {
    local elapsed=$(( $(date +%s) - STEP_START ))
    local total=$(( $(date +%s) - PIPELINE_START ))
    echo "$(c_green "  ✓ done in ${elapsed}s")  (pipeline elapsed ${total}s)"
}

step_skip() {
    echo "$(c_yellow "  → skipped: $1")"
}

fail() {
    echo
    echo "$(c_red "FAIL at step $STEP_NUM ($CURRENT_STAGE): $*")" >&2
    exit 1
}

cleanup() {
    set +e
    [ -n "$TAIL_PID"   ] && kill "$TAIL_PID"   2>/dev/null || true
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
    [ -n "$LLM_PID"    ] && kill "$LLM_PID"    2>/dev/null || true
}
trap cleanup EXIT

# Wait for an HTTP /health = ok with live progress.
wait_for_health() {
    local url="$1"
    local timeout_s="${2:-1800}"
    local log_file="${3:-}"
    local label="${4:-server}"
    local deadline; deadline=$(( $(date +%s) + timeout_s ))
    local last_log_size=0
    local now
    while true; do
        if curl -sf "$url" 2>/dev/null | grep -q '"status":"ok"'; then
            echo "  $(c_green "[$label] ready")"
            return 0
        fi
        now=$(date +%s)
        if [ "$now" -gt "$deadline" ]; then
            echo "  $(c_red "[$label] timeout after ${timeout_s}s")"
            return 1
        fi
        # Print whatever's new in the log since last poll.
        if [ -n "$log_file" ] && [ -f "$log_file" ]; then
            local size; size=$(stat -c%s "$log_file" 2>/dev/null || echo 0)
            if [ "$size" -gt "$last_log_size" ]; then
                tail -c +"$((last_log_size + 1))" "$log_file" 2>/dev/null \
                    | grep -E "Loading|Fetching|Downloading|loaded|cache|GPU|MISS|HIT|Error|error|Traceback" \
                    | tail -n 5 | sed 's/^/  | /'
                last_log_size="$size"
            fi
        fi
        # Cache disk growth (visible during download)
        if [ -d "$HF_HOME/hub/models--${MODEL_NAME//\//--}" ]; then
            local sz; sz=$(du -sh "$HF_HOME/hub/models--${MODEL_NAME//\//--}" 2>/dev/null | cut -f1)
            printf "  ... waiting [$label]  cache=%s  elapsed=%ds\r" "${sz:-0}" $(( now - PIPELINE_START ))
        else
            printf "  ... waiting [$label]  elapsed=%ds\r" $(( now - PIPELINE_START ))
        fi
        sleep 8
    done
}

# ============================================================
#  STEP 1  ─  Python venv + dependencies
# ============================================================
step "venv + dependencies"
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "$PYTHON_BIN not on PATH"
if [ ! -d ".venv" ]; then
    echo "  creating .venv"
    "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "  python   : $(python --version 2>&1)"
echo "  venv     : $VIRTUAL_ENV"
echo "  HF_HOME  : $HF_HOME"
echo "  MODEL    : $MODEL_NAME"

if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    step_skip "SKIP_INSTALL=1"
else
    echo "  installing/updating pip + wheel"
    python -m pip install -q --upgrade pip wheel setuptools
    echo "  installing locked requirements + agent extras (1-2 min)"
    pip install -q -r requirements.txt
    pip install -q "openai>=1.50.0" httpx requests python-dateutil tqdm \
        "uvicorn[standard]" python-multipart "fastapi>=0.110"
    echo "  installing torch / transformers / accelerate / bitsandbytes / peft (slow first time, 5-10 min)"
    pip install -q "torch>=2.4.0" "transformers>=4.45.0" "accelerate>=1.0.0" \
        "bitsandbytes>=0.44.0" "peft>=0.13.0" \
        "datasets>=3.0.0" safetensors sentencepiece protobuf \
        "huggingface_hub>=0.25" hf_transfer
    step_done
fi

# ============================================================
#  STEP 2  ─  data check
# ============================================================
step "Verify train/test data"
TRAIN_JSON="data/Phase_1/train.json"
TEST_JSON="data/Phase_1/test.json"
ok_size() {
    local f="$1" min="${2:-1000000}"
    [ -f "$f" ] || return 1
    local s; s=$(stat -c%s "$f" 2>/dev/null || echo 0)
    [ "$s" -ge "$min" ]
}
ok_size "$TRAIN_JSON" || {
    if command -v git-lfs >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git lfs install --skip-repo 2>/dev/null || true
        git lfs pull 2>/dev/null || true
    fi
    [ -n "${TRAIN_URL:-}" ] && curl -fL --retry 3 -o "$TRAIN_JSON" "$TRAIN_URL" || true
}
ok_size "$TRAIN_JSON" || fail "$TRAIN_JSON missing or LFS pointer"
ok_size "$TEST_JSON"  || fail "$TEST_JSON missing"
echo "  train: $(stat -c%s "$TRAIN_JSON" | numfmt --to=iec) bytes  (data/Phase_1/train.json)"
echo "  test : $(stat -c%s "$TEST_JSON"  | numfmt --to=iec) bytes  (data/Phase_1/test.json)"
step_done

# ============================================================
#  STEP 3  ─  prefetch model
# ============================================================
step "Prefetch base model into HF cache"
SAFE_NAME="models--${MODEL_NAME//\//--}"
CACHE_DIR="$HF_HOME/hub/$SAFE_NAME"
if [ -d "$CACHE_DIR" ]; then
    echo "  cache dir exists: $CACHE_DIR"
    du -sh "$CACHE_DIR" || true
fi
echo "  hf_transfer enabled: $HF_HUB_ENABLE_HF_TRANSFER"
echo "  starting download via huggingface_hub.snapshot_download (with resume + retries)..."
# Use Python API directly — sidesteps huggingface-cli vs hf binary churn.
if MODEL_NAME="$MODEL_NAME" python scripts/prefetch_model.py; then
    AFTER_SIZE=$(du -sb "$CACHE_DIR" 2>/dev/null | cut -f1 || echo 0)
    AFTER_HUMAN=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1 || echo 0)
    NUM_BLOBS=$(ls "$CACHE_DIR/blobs" 2>/dev/null | wc -l || echo 0)
    echo "  $(c_green "cache size:") $AFTER_HUMAN  ($NUM_BLOBS blobs)"
    # Sanity: 35B / 30B-A3B in fp16 should be in the 60-75 GB range.
    if [ "$AFTER_SIZE" -lt $((40 * 1024 * 1024 * 1024)) ]; then
        echo "  $(c_yellow "WARNING: cache is only $AFTER_HUMAN, expected 60-75 GB.")"
        echo "  $(c_yellow "         Re-run scripts/prefetch_model.py until size stabilizes near full.")"
    fi
else
    fail "model prefetch failed. Try a different MODEL_NAME (e.g. Qwen/Qwen3-30B-A3B), check network, or rerun."
fi
# After successful download we can safely run offline.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
echo "  HF_HUB_OFFLINE set to $HF_HUB_OFFLINE for the rest of this run (no further network calls)"
step_done

# ============================================================
#  STEP 4  ─  start llm_server
# ============================================================
step "Start llm_server.py (loads $MODEL_NAME in 4-bit, port $LLM_PORT)"
if curl -sf "http://localhost:$LLM_PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
    echo "  llm_server already healthy at port $LLM_PORT"
else
    extra_args=()
    if [ -d "$LORA_DIR" ] && [ -f "$LORA_DIR/adapter_config.json" ]; then
        echo "  found existing LoRA at $LORA_DIR — attaching"
        extra_args+=(--lora "$LORA_DIR")
    fi
    nohup python scripts/llm_server.py \
        --model "$MODEL_NAME" --port "$LLM_PORT" "${extra_args[@]}" \
        > "$LLM_LOG" 2>&1 &
    LLM_PID=$!
    echo "  llm_server pid=$LLM_PID  log=$LLM_LOG"
    echo "  expected wait: 3-10 min (cache hit) or 15-20 min (cache miss)"
    wait_for_health "http://localhost:$LLM_PORT/health" 1800 "$LLM_LOG" "llm" \
        || { tail -n 80 "$LLM_LOG" >&2; fail "llm_server never became healthy"; }
fi
step_done

# ============================================================
#  STEP 5  ─  start server.py
# ============================================================
step "Start locked server.py (DATA_SPLIT=$DATA_SPLIT_RUN)"
pkill -f "python server.py" 2>/dev/null || true
sleep 2
DATA_SPLIT="$DATA_SPLIT_RUN" nohup python server.py > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "  server.py pid=$SERVER_PID  log=$SERVER_LOG"
wait_for_health "http://localhost:$SERVER_PORT/health" 60 "$SERVER_LOG" "server" \
    || { tail -n 60 "$SERVER_LOG" >&2; fail "server.py never became healthy"; }
step_done

# ============================================================
#  STEP 6  ─  Phase 1 test run
# ============================================================
step "Run agent on $N_TEST $DATA_SPLIT_RUN scenarios → Phase 1 result.csv"
if [ -f "$PHASE1_DIR/result.csv" ] && [ "$(wc -l < "$PHASE1_DIR/result.csv")" -ge "$N_TEST" ]; then
    step_skip "$PHASE1_DIR/result.csv already complete"
else
    python main.py \
        --server_url "http://localhost:$SERVER_PORT" \
        --model_url "http://localhost:$LLM_PORT/v1" \
        --model_name "$MODEL_NAME" \
        --max_samples "$N_TEST" \
        --num_attempts 1 \
        --max_iterations 2 \
        --save_dir "$PHASE1_DIR" \
        --log_file "$PHASE1_DIR/agent.log" \
        --quiet 2>&1 | grep -E "Scenario:|RUN_SUMMARY|ERROR|FAIL" || true
    step_done
fi
echo
echo "  $(c_green ">>> Phase 1 candidate file:") $PHASE1_DIR/result.csv"
echo "  $(c_yellow ">>> Upload to Zindi NOW; pipeline continues with distillation.")"

# ============================================================
#  STEP 7  ─  self-distillation
# ============================================================
step "Self-distillation on training scenarios → $TRACES"
if [ "${SKIP_DISTILL:-0}" = "1" ] || [ "${FAST:-0}" = "1" ]; then
    step_skip "${SKIP_DISTILL:+SKIP_DISTILL=1} ${FAST:+FAST=1}"
else
    BEFORE=0; [ -f "$TRACES" ] && BEFORE=$(wc -l < "$TRACES")
    python scripts/distill.py \
        --train_file "$TRAIN_JSON" \
        --output "$TRACES" \
        --model_url "http://localhost:$LLM_PORT/v1" \
        --model_name "$MODEL_NAME" \
        --max_samples "$N_TRAIN_DISTILL"
    AFTER=$(wc -l < "$TRACES")
    echo "  corpus: $BEFORE → $AFTER traces (+$((AFTER - BEFORE)))"
    [ "$AFTER" -ge 100 ] || fail "only $AFTER traces accepted; cannot fine-tune meaningfully."
    step_done
fi

# ============================================================
#  STEP 8  ─  LoRA fine-tune
# ============================================================
step "LoRA fine-tune Qwen on the corpus → $LORA_DIR"
if [ "${SKIP_FINETUNE:-0}" = "1" ] || [ "${FAST:-0}" = "1" ]; then
    step_skip "${SKIP_FINETUNE:+SKIP_FINETUNE=1} ${FAST:+FAST=1}"
elif [ -d "$LORA_DIR" ] && [ -f "$LORA_DIR/adapter_config.json" ]; then
    step_skip "$LORA_DIR already exists"
else
    echo "  stopping llm_server to free GPU memory for training"
    [ -n "$LLM_PID" ] && kill "$LLM_PID" 2>/dev/null || true
    pkill -f "scripts/llm_server.py" 2>/dev/null || true
    sleep 5
    LLM_PID=""

    CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
        --traces "$TRACES" \
        --output_dir "$(dirname "$LORA_DIR")" \
        --base_model "$MODEL_NAME"
    [ -d "$LORA_DIR" ] || fail "best_lora directory not produced"
    step_done

    echo "  restarting llm_server with LoRA attached"
    nohup python scripts/llm_server.py \
        --model "$MODEL_NAME" --port "$LLM_PORT" --lora "$LORA_DIR" \
        > "$LLM_LOG" 2>&1 &
    LLM_PID=$!
    wait_for_health "http://localhost:$LLM_PORT/health" 1200 "$LLM_LOG" "llm+lora" \
        || { tail -n 80 "$LLM_LOG" >&2; fail "llm_server with LoRA never became healthy"; }
fi

# Make sure llm_server is up for stage 9 even if we skipped 8.
if ! curl -sf "http://localhost:$LLM_PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
    extra_args=()
    [ -d "$LORA_DIR" ] && [ -f "$LORA_DIR/adapter_config.json" ] && extra_args+=(--lora "$LORA_DIR")
    nohup python scripts/llm_server.py \
        --model "$MODEL_NAME" --port "$LLM_PORT" "${extra_args[@]}" \
        > "$LLM_LOG" 2>&1 &
    LLM_PID=$!
    wait_for_health "http://localhost:$LLM_PORT/health" 1200 "$LLM_LOG" "llm" \
        || fail "llm_server never came back up before Phase 2 run"
fi

# Make sure server.py is up for stage 9.
if ! curl -sf "http://localhost:$SERVER_PORT/health" >/dev/null 2>&1; then
    DATA_SPLIT="$DATA_SPLIT_RUN" nohup python server.py > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    wait_for_health "http://localhost:$SERVER_PORT/health" 60 "$SERVER_LOG" "server" \
        || fail "server.py never came back up"
fi

# ============================================================
#  STEP 9  ─  Phase 2 test run
# ============================================================
step "Run fine-tuned agent on $N_TEST scenarios → Phase 2 result.csv"
if [ -f "$PHASE2_DIR/result.csv" ] && [ "$(wc -l < "$PHASE2_DIR/result.csv")" -ge "$N_TEST" ]; then
    step_skip "$PHASE2_DIR/result.csv already complete"
else
    python main.py \
        --server_url "http://localhost:$SERVER_PORT" \
        --model_url "http://localhost:$LLM_PORT/v1" \
        --model_name "$MODEL_NAME" \
        --max_samples "$N_TEST" \
        --num_attempts 1 \
        --max_iterations 2 \
        --save_dir "$PHASE2_DIR" \
        --log_file "$PHASE2_DIR/agent.log" \
        --quiet 2>&1 | grep -E "Scenario:|RUN_SUMMARY|ERROR|FAIL" || true
    step_done
fi

# ============================================================
#  STEP 10 ─  build 3 candidate submissions
# ============================================================
step "Build 3 candidate submission files (1 run × 3 leaderboard tries)"
python scripts/build_phase2_submissions.py \
    --completions "$PHASE2_DIR/completions.jsonl" \
    --test_file "$TEST_JSON" \
    --out_dir "$PHASE2_DIR" || echo "  WARN: variant build failed; raw result.csv is still valid."
step_done

# ============================================================
#                   FINAL REPORT
# ============================================================
TOTAL_ELAPSED=$(( $(date +%s) - PIPELINE_START ))
echo
echo "$(c_green "============================================================")"
echo "$(c_green "  RUN_ALL COMPLETE in ${TOTAL_ELAPSED}s")"
echo "$(c_green "============================================================")"
echo
echo "Phase 1 candidate    : $PHASE1_DIR/result.csv"
[ -f "$TRACES"   ] && echo "Distilled corpus     : $TRACES ($(wc -l < "$TRACES") traces)"
[ -d "$LORA_DIR" ] && echo "Fine-tuned LoRA      : $LORA_DIR"
echo "Phase 2 raw          : $PHASE2_DIR/result.csv"
[ -f "$PHASE2_DIR/result_v1_raw.csv"          ] && echo "Phase 2 v1 (raw)     : $PHASE2_DIR/result_v1_raw.csv"
[ -f "$PHASE2_DIR/result_v2_multi_recall.csv" ] && echo "Phase 2 v2 (multi)   : $PHASE2_DIR/result_v2_multi_recall.csv"
[ -f "$PHASE2_DIR/result_v3_insurance.csv"    ] && echo "Phase 2 v3 (ins.)    : $PHASE2_DIR/result_v3_insurance.csv"
echo
echo "Logs in $LOG_DIR/  ;  agent log in $PHASE2_DIR/agent.log"
echo "Upload v1, v2, v3 to Zindi as your 3 Phase 2 best-of-3 submissions."
