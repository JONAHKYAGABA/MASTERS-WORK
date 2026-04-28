#!/usr/bin/env bash
# scripts/quickrun.sh
#
# Fast end-to-end runner using a Python venv (NO conda).
# Defaults: 5 scenarios on the train split, scoring vs ground truth.
#
# Single command from the project root (the dir containing server.py):
#     bash scripts/quickrun.sh
#
# Override with env vars:
#     N_SCENARIOS=50 bash scripts/quickrun.sh
#     MODEL_URL=https://my-endpoint/v1 MODEL_NAME=Qwen/... bash scripts/quickrun.sh
#     TRAIN_URL=https://example.com/train.json bash scripts/quickrun.sh
#     PYTHON_BIN=python3.10 bash scripts/quickrun.sh
#     SKIP_INSTALL=1 bash scripts/quickrun.sh        # reuse existing .venv
#     SKIP_SERVER=1 bash scripts/quickrun.sh         # server already running

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

step()  { printf "\n=== %s ===\n" "$*"; }
fail()  { printf "ERROR: %s\n" "$*" >&2; exit 1; }

step "Project: $PROJECT_DIR"

# ---------- 1. venv ----------
step "[1/7] Python venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "$PYTHON_BIN not on PATH (set PYTHON_BIN)"
if [ ! -d ".venv" ]; then
    "$PYTHON_BIN" -m venv .venv
    echo "Created .venv with $($PYTHON_BIN --version 2>&1)"
else
    echo "Reusing existing .venv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "active: $(python --version 2>&1) at $(command -v python)"

# ---------- 2. deps ----------
step "[2/7] Install dependencies"
if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    echo "SKIP_INSTALL=1 → not touching pip"
else
    python -m pip install -q --upgrade pip wheel setuptools
    # Locked server-side deps
    pip install -q -r requirements.txt
    # Agent + server runtime extras (these aren't in the locked file)
    pip install -q "openai>=1.50.0" httpx requests python-dateutil tqdm \
                  "uvicorn[standard]" python-multipart
    echo "deps installed"
fi

# ---------- 3. data ----------
step "[3/7] Verify training data"
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
    echo "$TRAIN_JSON missing or too small (likely an LFS pointer). Trying recovery..."
    # 3a. git-lfs pull (works only if this is a real git clone with LFS configured)
    if command -v git-lfs >/dev/null 2>&1 \
       && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git lfs install --skip-repo 2>/dev/null || true
        git lfs pull 2>/dev/null || true
    fi
    # 3b. TRAIN_URL fallback (point at a public mirror you have)
    if ! ok_size "$TRAIN_JSON" && [ -n "${TRAIN_URL:-}" ]; then
        mkdir -p "$(dirname "$TRAIN_JSON")"
        echo "Downloading from TRAIN_URL..."
        curl -fL --retry 3 -o "$TRAIN_JSON" "$TRAIN_URL"
    fi
fi

if ! ok_size "$TRAIN_JSON"; then
    cat <<EOF >&2

train.json is missing or too small to be the real 24 MB file.

Pick ONE of these and re-run:

  1. Re-clone the repo with git-lfs installed:
        git lfs install
        git clone <repo-url>
        cd <repo>/Track\ A
        bash scripts/quickrun.sh

  2. Download train.json manually from the Zindi data tab and place it at:
        $PROJECT_DIR/$TRAIN_JSON

  3. Set TRAIN_URL to a direct download URL (raw GitHub or signed S3):
        TRAIN_URL=https://... bash scripts/quickrun.sh

EOF
    exit 1
fi
echo "$TRAIN_JSON OK ($(wc -c < "$TRAIN_JSON" | tr -d ' ') bytes)"
ok_size "$TEST_JSON" && echo "$TEST_JSON OK" || echo "$TEST_JSON missing (only needed for Phase 1 submission)"

# ---------- 4. LLM endpoint ----------
step "[4/7] LLM endpoint check"
MODEL_URL="${MODEL_URL:-http://localhost:8000/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
echo "MODEL_URL=$MODEL_URL"
echo "MODEL_NAME=$MODEL_NAME"

if ! curl -sf "$MODEL_URL/models" >/dev/null 2>&1; then
    cat <<EOF >&2

No OpenAI-compatible endpoint at $MODEL_URL.

Start vLLM in another terminal (RTX 8000-friendly flags):

    python -m vllm.entrypoints.openai.api_server \\
      --model $MODEL_NAME \\
      --port 8000 \\
      --dtype float16 \\
      --tensor-parallel-size 2 \\
      --gpu-memory-utilization 0.92 \\
      --max-model-len 16384 \\
      --enforce-eager \\
      --quantization bitsandbytes \\
      --load-format bitsandbytes \\
      --enable-auto-tool-choice \\
      --tool-call-parser hermes \\
      --trust-remote-code

(vLLM and its CUDA stack are NOT installed by this script — they're 10+ GB.
 Install once with:  pip install vllm)

OR set MODEL_URL to any OpenAI-compatible endpoint that serves Qwen3.5-35B-A3B
and rerun:  MODEL_URL=https://your-endpoint/v1 bash scripts/quickrun.sh
EOF
    exit 1
fi
echo "endpoint healthy"

# ---------- 5. server.py ----------
step "[5/7] Start locked server.py"
DATA_SPLIT="${DATA_SPLIT:-train}"
SERVER_LOG="$PROJECT_DIR/server.log"
SERVER_PID=""

cleanup_server() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup_server EXIT

if [ "${SKIP_SERVER:-0}" = "1" ] || curl -sf http://localhost:7860/health >/dev/null 2>&1; then
    echo "server.py already running (or SKIP_SERVER=1)"
else
    DATA_SPLIT="$DATA_SPLIT" nohup python server.py > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    echo "server.py pid=$SERVER_PID  log=$SERVER_LOG"
    for i in $(seq 1 30); do
        sleep 2
        if curl -sf http://localhost:7860/health >/dev/null 2>&1; then
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "server.py exited. Last 60 lines of log:" >&2
            tail -60 "$SERVER_LOG" >&2 || true
            exit 1
        fi
    done
    if ! curl -sf http://localhost:7860/health >/dev/null 2>&1; then
        echo "server.py never reached /health. Last 60 lines:" >&2
        tail -60 "$SERVER_LOG" >&2 || true
        exit 1
    fi
fi

# ---------- 6. agent ----------
step "[6/7] Run agent"
N_SCENARIOS="${N_SCENARIOS:-5}"
EXP_NAME="${EXP_NAME:-quickrun_${N_SCENARIOS}}"
SAVE_DIR="eval/results/$EXP_NAME"
mkdir -p "$SAVE_DIR"

START_T=$(date +%s)
python main.py \
    --server_url http://localhost:7860 \
    --model_url "$MODEL_URL" \
    --model_name "$MODEL_NAME" \
    --max_samples "$N_SCENARIOS" \
    --num_attempts "${NUM_ATTEMPTS:-1}" \
    --max_iterations "${MAX_ITER:-2}" \
    --save_dir "$SAVE_DIR" \
    --log_file "$SAVE_DIR/agent.log" \
    --quiet
END_T=$(date +%s)
echo "agent wall-clock: $((END_T - START_T))s for $N_SCENARIOS scenarios"

# ---------- 7. score ----------
step "[7/7] Score"
python scripts/score_results.py \
    --results_dir "$SAVE_DIR" \
    --train_file "$TRAIN_JSON" \
    --target_acc "${TARGET_ACC:-0.35}" \
    --target_latency_s "${TARGET_LATENCY_S:-90}"

echo
echo "========================================"
echo "DONE."
echo "  result.csv         : $SAVE_DIR/result.csv"
echo "  completions.jsonl  : $SAVE_DIR/completions.jsonl"
echo "  metrics.json       : $SAVE_DIR/metrics.json"
echo "  SUMMARY.md         : $SAVE_DIR/SUMMARY.md"
echo "========================================"
