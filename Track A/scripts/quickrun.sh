#!/usr/bin/env bash
# scripts/quickrun.sh
#
# ONE COMMAND, end-to-end. No vLLM. No conda. No manual model download.
# Uses transformers + bitsandbytes 4-bit (works on RTX 8000 / Turing).
#
#   bash scripts/quickrun.sh
#
# Optional overrides:
#   N_SCENARIOS=50            (default 5)
#   MODEL_NAME=Qwen/Qwen3-30B-A3B   (override base model)
#   MODEL_URL=https://...     (use external endpoint, skip local llm_server)
#   AGENT_API_KEY=...         (only if using a remote MODEL_URL with auth)
#   SKIP_INSTALL=1            (reuse existing .venv)
#   SKIP_LLM=1                (don't (re)start the LLM server)
#   SKIP_SERVER=1             (don't (re)start the locked server.py)
#   TRAIN_URL=https://...     (curl fallback for train.json)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ---- HuggingFace cache pinning ----
# Reuse the same cache across runs so the model is downloaded ONCE.
# Override HF_HOME if you want the cache somewhere else (e.g. on a bigger disk).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-0}"
# Set HF_HUB_OFFLINE=1 in your shell to forbid any HF Hub network call after
# the model is cached. Useful as a paranoid "no redownload" guarantee.
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME/hub"

step() { printf "\n=== %s ===\n" "$*"; }
fail() { printf "ERROR: %s\n" "$*" >&2; exit 1; }

N_SCENARIOS="${N_SCENARIOS:-5}"
EXP_NAME="${EXP_NAME:-quickrun_${N_SCENARIOS}}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
LLM_PORT="${LLM_PORT:-8001}"
SERVER_PORT="${SERVER_PORT:-7860}"
DATA_SPLIT="${DATA_SPLIT:-train}"
SAVE_DIR="eval/results/$EXP_NAME"
LOG_DIR="eval/logs/$EXP_NAME"
mkdir -p "$SAVE_DIR" "$LOG_DIR"
LLM_LOG="$LOG_DIR/llm_server.log"
SERVER_LOG="$LOG_DIR/server.log"
AGENT_LOG="$LOG_DIR/agent.log"
PIDFILE_LLM="$LOG_DIR/llm.pid"
PIDFILE_SERVER="$LOG_DIR/server.pid"

LLM_PID=""
SERVER_PID=""

cleanup() {
    set +e
    if [ -n "$LLM_PID" ] && kill -0 "$LLM_PID" 2>/dev/null; then
        kill "$LLM_PID" 2>/dev/null
        sleep 1
        kill -9 "$LLM_PID" 2>/dev/null || true
    fi
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        sleep 1
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ---------- 1. venv ----------
step "[1/8] Python venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "$PYTHON_BIN not on PATH"
if [ ! -d ".venv" ]; then
    "$PYTHON_BIN" -m venv .venv
    echo "Created .venv ($($PYTHON_BIN --version 2>&1))"
else
    echo "Reusing existing .venv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "active: $(python --version 2>&1) at $(command -v python)"

# ---------- 2. install ----------
step "[2/8] Install dependencies"
if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    echo "SKIP_INSTALL=1 → not touching pip"
else
    python -m pip install -q --upgrade pip wheel setuptools
    pip install -q -r requirements.txt
    pip install -q \
        "openai>=1.50.0" httpx requests python-dateutil tqdm \
        "uvicorn[standard]" python-multipart \
        "fastapi>=0.110"
    # The heavy bits — needed by llm_server.py
    pip install -q \
        "torch>=2.4.0" \
        "transformers>=4.45.0" \
        "accelerate>=1.0.0" \
        "bitsandbytes>=0.44.0" \
        safetensors sentencepiece protobuf
    echo "deps OK"
fi

# ---------- 3. data ----------
step "[3/8] Verify training data"
TRAIN_JSON="data/Phase_1/train.json"
TEST_JSON="data/Phase_1/test.json"

ok_size() {
    local f="$1" min="${2:-1000000}"
    [ -f "$f" ] || return 1
    local s
    s=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
    [ "$s" -ge "$min" ]
}

if ! ok_size "$TRAIN_JSON"; then
    echo "$TRAIN_JSON missing/small. Trying recovery..."
    if command -v git-lfs >/dev/null 2>&1 \
       && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git lfs install --skip-repo 2>/dev/null || true
        git lfs pull 2>/dev/null || true
    fi
    if ! ok_size "$TRAIN_JSON" && [ -n "${TRAIN_URL:-}" ]; then
        mkdir -p "$(dirname "$TRAIN_JSON")"
        curl -fL --retry 3 -o "$TRAIN_JSON" "$TRAIN_URL"
    fi
fi
ok_size "$TRAIN_JSON" || fail "$TRAIN_JSON missing or LFS pointer. Set TRAIN_URL=... or place file manually."
echo "$TRAIN_JSON OK"
ok_size "$TEST_JSON" && echo "$TEST_JSON OK" || echo "$TEST_JSON missing (only needed for Phase 1 submission)"

# ---------- 4. start LLM server (or accept external) ----------
step "[4/8] LLM server"
if [ -n "${MODEL_URL:-}" ]; then
    echo "MODEL_URL provided externally: $MODEL_URL — not starting local llm_server.py"
elif [ "${SKIP_LLM:-0}" = "1" ]; then
    MODEL_URL="${MODEL_URL:-http://localhost:$LLM_PORT/v1}"
    echo "SKIP_LLM=1 → assuming server at $MODEL_URL"
else
    MODEL_URL="http://localhost:$LLM_PORT/v1"
    if curl -sf "http://localhost:$LLM_PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
        echo "llm_server already healthy at port $LLM_PORT"
    else
        # Make sure the port is not occupied by something else
        if curl -sf "http://localhost:$LLM_PORT/health" >/dev/null 2>&1 \
           || ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${LLM_PORT}$"; then
            fail "Port $LLM_PORT is in use by something other than our llm_server. Set LLM_PORT=<free port> and retry."
        fi
        echo "Starting llm_server.py (model=$MODEL_NAME, port=$LLM_PORT)"
        echo "First load takes ~10-20 min on RTX 8000 (downloads weights, quantizes, loads on GPUs)"
        MODEL_NAME="$MODEL_NAME" LLM_PORT="$LLM_PORT" \
            nohup python scripts/llm_server.py \
                --model "$MODEL_NAME" --port "$LLM_PORT" \
                > "$LLM_LOG" 2>&1 &
        LLM_PID=$!
        echo "$LLM_PID" > "$PIDFILE_LLM"
        echo "llm_server pid=$LLM_PID  log=$LLM_LOG"
    fi
fi

# ---------- 5. wait for LLM ready ----------
step "[5/8] Wait for LLM /health = ok"
DEADLINE=$(( $(date +%s) + 1800 ))   # 30 min
while true; do
    BODY=$(curl -sf "${MODEL_URL%/v1}/health" 2>/dev/null || echo '')
    if echo "$BODY" | grep -q '"status":"ok"'; then
        echo "LLM ready: $BODY"
        break
    fi
    # If we own the process and it died, fail fast.
    if [ -n "$LLM_PID" ] && ! kill -0 "$LLM_PID" 2>/dev/null; then
        echo "llm_server died. Last 80 lines:"
        tail -n 80 "$LLM_LOG" || true
        exit 1
    fi
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        echo "LLM did not become ready within 30 min. Last 80 lines:"
        [ -n "$LLM_PID" ] && tail -n 80 "$LLM_LOG" || true
        exit 1
    fi
    sleep 10
    # Ping the model loading progress
    if [ -n "$LLM_PID" ]; then
        last=$(tail -n 1 "$LLM_LOG" 2>/dev/null || true)
        [ -n "$last" ] && echo "  ... $last"
    fi
done

# ---------- 6. start server.py ----------
step "[6/8] Start locked server.py (DATA_SPLIT=$DATA_SPLIT)"
if [ "${SKIP_SERVER:-0}" = "1" ] || curl -sf "http://localhost:$SERVER_PORT/health" >/dev/null 2>&1; then
    echo "server.py already running"
else
    DATA_SPLIT="$DATA_SPLIT" nohup python server.py > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$PIDFILE_SERVER"
    echo "server.py pid=$SERVER_PID  log=$SERVER_LOG"
    DEADLINE=$(( $(date +%s) + 120 ))
    while true; do
        if curl -sf "http://localhost:$SERVER_PORT/health" >/dev/null 2>&1; then
            echo "server.py ready"; break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "server.py died. Last 60 lines:"
            tail -n 60 "$SERVER_LOG" || true
            exit 1
        fi
        if [ "$(date +%s)" -gt "$DEADLINE" ]; then
            echo "server.py /health timeout. Last 60 lines:"
            tail -n 60 "$SERVER_LOG" || true
            exit 1
        fi
        sleep 2
    done
fi

# ---------- 7. run agent ----------
step "[7/8] Run agent on $N_SCENARIOS scenarios"
START_T=$(date +%s)
python main.py \
    --server_url "http://localhost:$SERVER_PORT" \
    --model_url "$MODEL_URL" \
    --model_name "$MODEL_NAME" \
    --max_samples "$N_SCENARIOS" \
    --num_attempts "${NUM_ATTEMPTS:-1}" \
    --max_iterations "${MAX_ITER:-2}" \
    --save_dir "$SAVE_DIR" \
    --log_file "$AGENT_LOG" \
    --quiet | tee -a "$AGENT_LOG"
END_T=$(date +%s)
echo "agent wall-clock: $((END_T - START_T))s for $N_SCENARIOS scenarios"

# ---------- 8. score ----------
step "[8/8] Score"
python scripts/score_results.py \
    --results_dir "$SAVE_DIR" \
    --train_file "$TRAIN_JSON" \
    --target_acc "${TARGET_ACC:-0.35}" \
    --target_latency_s "${TARGET_LATENCY_S:-180}" || true

echo
echo "========================================"
echo "DONE."
echo "  result.csv         : $SAVE_DIR/result.csv"
echo "  completions.jsonl  : $SAVE_DIR/completions.jsonl"
echo "  metrics.json       : $SAVE_DIR/metrics.json"
echo "  SUMMARY.md         : $SAVE_DIR/SUMMARY.md"
echo "  logs               : $LOG_DIR/"
echo "========================================"
