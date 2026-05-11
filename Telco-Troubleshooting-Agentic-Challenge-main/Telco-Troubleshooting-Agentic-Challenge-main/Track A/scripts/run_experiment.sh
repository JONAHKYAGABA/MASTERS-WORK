#!/usr/bin/env bash
# scripts/run_experiment.sh
#
# ONE conclusive experiment for Stage B (prompted-only ceiling).
# Boots vLLM + locked server.py, runs the refactored agent on N train
# scenarios with ground-truth scoring, writes SUMMARY.md, tears everything
# down. Idempotent: re-running resumes from completions.jsonl.
#
# Single command:
#   bash scripts/run_experiment.sh
#
# Override defaults via env vars:
#   N_SCENARIOS=50    EXP_NAME=stage_b_50    bash scripts/run_experiment.sh
#   QUANTIZATION=awq                       (use AWQ instead of bitsandbytes)
#   MODEL_ID=Qwen/Qwen3-30B-A3B            (override base model)
#   SKIP_VLLM=1                            (assume vLLM already running)
#   SKIP_SERVER=1                          (assume server.py already running)

set -euo pipefail

# ------------------------- CONFIG -------------------------
N_SCENARIOS="${N_SCENARIOS:-50}"
EXP_NAME="${EXP_NAME:-stage_b_${N_SCENARIOS}}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-35B-A3B}"
QUANTIZATION="${QUANTIZATION:-bitsandbytes}"   # bitsandbytes | awq | gptq
DATA_SPLIT="${DATA_SPLIT:-train}"
VLLM_PORT="${VLLM_PORT:-8000}"
SERVER_PORT="${SERVER_PORT:-7860}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_ITER="${MAX_ITER:-2}"
NUM_ATTEMPTS="${NUM_ATTEMPTS:-1}"
TARGET_ACC="${TARGET_ACC:-0.35}"
TARGET_LATENCY_S="${TARGET_LATENCY_S:-90}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SAVE_DIR="$PROJECT_DIR/eval/results/$EXP_NAME"
LOG_DIR="$PROJECT_DIR/eval/logs/$EXP_NAME"
TRAIN_FILE="$PROJECT_DIR/data/Phase_1/train.json"

mkdir -p "$SAVE_DIR" "$LOG_DIR"
VLLM_LOG="$LOG_DIR/vllm.log"
SERVER_LOG="$LOG_DIR/server.log"
AGENT_LOG="$LOG_DIR/agent.log"
PIDFILE_VLLM="$LOG_DIR/vllm.pid"
PIDFILE_SERVER="$LOG_DIR/server.pid"

# Force Turing-friendly attention backend.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"

step() { echo; echo "================ $1 ================"; }

cleanup() {
    set +e
    if [ -f "$PIDFILE_VLLM" ]; then
        kill "$(cat "$PIDFILE_VLLM")" 2>/dev/null || true
        rm -f "$PIDFILE_VLLM"
    fi
    if [ -f "$PIDFILE_SERVER" ]; then
        kill "$(cat "$PIDFILE_SERVER")" 2>/dev/null || true
        rm -f "$PIDFILE_SERVER"
    fi
}
trap cleanup EXIT

cd "$PROJECT_DIR"

# ------------------------- STEP 1: preflight -------------------------
step "[1/6] Preflight"

# conda env
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "telco" ]; then
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate telco
    else
        echo "ERROR: conda not on PATH and 'telco' env not active" >&2
        exit 1
    fi
fi
echo "conda env  : $CONDA_DEFAULT_ENV"
echo "python     : $(python --version 2>&1)"
echo "model      : $MODEL_ID"
echo "quant      : $QUANTIZATION (TP=$TENSOR_PARALLEL)"
echo "scenarios  : $N_SCENARIOS  (split=$DATA_SPLIT)"
echo "save_dir   : $SAVE_DIR"

# Locked-files check
if [ -x "$PROJECT_DIR/scripts/verify_locks.sh" ]; then
    "$PROJECT_DIR/scripts/verify_locks.sh"
fi

# Train data check
if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: $TRAIN_FILE missing. Run scripts/setup_environment.sh first." >&2
    exit 1
fi
SIZE=$(stat -c%s "$TRAIN_FILE" 2>/dev/null || stat -f%z "$TRAIN_FILE")
[ "$SIZE" -gt 1000000 ] || { echo "ERROR: train.json too small ($SIZE bytes)"; exit 1; }

# ------------------------- STEP 2: start vLLM -------------------------
step "[2/6] Start vLLM (background)"

if [ "${SKIP_VLLM:-0}" = "1" ]; then
    echo "SKIP_VLLM=1 → assuming vLLM already serving at port $VLLM_PORT"
elif curl -sf "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
    echo "vLLM already healthy at port $VLLM_PORT"
else
    VLLM_CMD=(
        python -m vllm.entrypoints.openai.api_server
        --model "$MODEL_ID"
        --port "$VLLM_PORT"
        --dtype float16
        --tensor-parallel-size "$TENSOR_PARALLEL"
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --max-model-len "$MAX_MODEL_LEN"
        --enforce-eager
        --enable-auto-tool-choice
        --tool-call-parser hermes
        --trust-remote-code
    )
    case "$QUANTIZATION" in
        bitsandbytes)
            VLLM_CMD+=(--quantization bitsandbytes --load-format bitsandbytes)
            ;;
        awq|awq_marlin)
            VLLM_CMD+=(--quantization "$QUANTIZATION")
            ;;
        gptq|gptq_marlin)
            VLLM_CMD+=(--quantization "$QUANTIZATION")
            ;;
        none|"")
            ;;
        *)
            VLLM_CMD+=(--quantization "$QUANTIZATION")
            ;;
    esac
    echo "Launching: ${VLLM_CMD[*]}"
    nohup "${VLLM_CMD[@]}" > "$VLLM_LOG" 2>&1 &
    echo $! > "$PIDFILE_VLLM"
    echo "vLLM pid=$(cat "$PIDFILE_VLLM")  log=$VLLM_LOG"
fi

# Wait for vLLM
echo "Waiting for vLLM /v1/models (may take 5-10 min on RTX 8000)..."
DEADLINE=$(( $(date +%s) + 1200 ))
while true; do
    if curl -sf "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
        echo "vLLM ready."
        break
    fi
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        echo "ERROR: vLLM did not become ready within 20 min. Tail of log:"
        tail -n 80 "$VLLM_LOG" || true
        exit 1
    fi
    if [ -f "$PIDFILE_VLLM" ] && ! kill -0 "$(cat "$PIDFILE_VLLM")" 2>/dev/null; then
        echo "ERROR: vLLM process exited. Tail of log:"
        tail -n 120 "$VLLM_LOG" || true
        exit 1
    fi
    sleep 5
done

# ------------------------- STEP 3: start server.py -------------------------
step "[3/6] Start locked server.py (background, DATA_SPLIT=$DATA_SPLIT)"

if [ "${SKIP_SERVER:-0}" = "1" ]; then
    echo "SKIP_SERVER=1 → assuming server.py already at port $SERVER_PORT"
elif curl -sf "http://localhost:$SERVER_PORT/health" >/dev/null 2>&1; then
    echo "server.py already healthy at port $SERVER_PORT"
else
    DATA_SPLIT="$DATA_SPLIT" nohup python "$PROJECT_DIR/server.py" \
        > "$SERVER_LOG" 2>&1 &
    echo $! > "$PIDFILE_SERVER"
    echo "server pid=$(cat "$PIDFILE_SERVER")  log=$SERVER_LOG"
fi

echo "Waiting for server /health..."
DEADLINE=$(( $(date +%s) + 120 ))
while true; do
    if curl -sf "http://localhost:$SERVER_PORT/health" >/dev/null 2>&1; then
        echo "server.py ready."
        break
    fi
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        echo "ERROR: server.py not ready within 2 min. Tail of log:"
        tail -n 80 "$SERVER_LOG" || true
        exit 1
    fi
    if [ -f "$PIDFILE_SERVER" ] && ! kill -0 "$(cat "$PIDFILE_SERVER")" 2>/dev/null; then
        echo "ERROR: server.py process exited. Tail of log:"
        tail -n 120 "$SERVER_LOG" || true
        exit 1
    fi
    sleep 2
done

# ------------------------- STEP 4: run the agent -------------------------
step "[4/6] Run agent on $N_SCENARIOS scenarios"

START_T=$(date +%s)
python "$PROJECT_DIR/main.py" \
    --server_url "http://localhost:$SERVER_PORT" \
    --model_url "http://localhost:$VLLM_PORT/v1" \
    --model_name "$MODEL_ID" \
    --max_samples "$N_SCENARIOS" \
    --num_attempts "$NUM_ATTEMPTS" \
    --max_iterations "$MAX_ITER" \
    --save_dir "$SAVE_DIR" \
    --log_file "$AGENT_LOG" \
    --quiet \
    | tee -a "$AGENT_LOG"
END_T=$(date +%s)
WALL=$(( END_T - START_T ))
echo "Agent run wall-clock: ${WALL}s"

# ------------------------- STEP 5: score -------------------------
step "[5/6] Score against ground truth"

python "$PROJECT_DIR/scripts/score_results.py" \
    --results_dir "$SAVE_DIR" \
    --train_file "$TRAIN_FILE" \
    --target_acc "$TARGET_ACC" \
    --target_latency_s "$TARGET_LATENCY_S" \
    | tee -a "$AGENT_LOG"
SCORE_RC=${PIPESTATUS[0]}

# ------------------------- STEP 6: report -------------------------
step "[6/6] Report"

echo "Artifacts:"
echo "  $SAVE_DIR/result.csv         (Zindi-format submission)"
echo "  $SAVE_DIR/completions.jsonl  (per-scenario detail)"
echo "  $SAVE_DIR/metrics.json       (machine-readable summary)"
echo "  $SAVE_DIR/SUMMARY.md         (human-readable verdict)"
echo "  $LOG_DIR/                    (vllm.log, server.log, agent.log)"
echo
if [ "$SCORE_RC" -eq 0 ]; then
    echo "EXPERIMENT VERDICT: GO (acc + latency thresholds met)"
elif [ "$SCORE_RC" -eq 2 ]; then
    echo "EXPERIMENT VERDICT: NO-GO or PARTIAL (see SUMMARY.md)"
else
    echo "Scoring exited with code $SCORE_RC"
fi
echo
echo "Tail of SUMMARY.md:"
sed -n '1,30p' "$SAVE_DIR/SUMMARY.md" || true
