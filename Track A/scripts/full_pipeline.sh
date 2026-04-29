#!/usr/bin/env bash
# scripts/full_pipeline.sh
#
# End-to-end Stage B → C → D → E in one shot. Run after `bash scripts/quickrun.sh`
# has succeeded (i.e. the LLM is loaded and a tiny experiment has scored > 0).
#
# Phases produced:
#   B: eval/results/phase1_test/result.csv  (upload to Zindi for Phase 1 score)
#   C: traces/train_traces.jsonl            (self-distilled corpus)
#   D: training/checkpoints/run_v1/best_lora/  (fine-tuned LoRA weights)
#   E: eval/results/phase2_test/result.csv  (upload as Phase 2 submission)
#
# Total wall-clock on 2x RTX 8000: 1.5 to 3 days. Resumable: each stage skips
# work that has already been done.
#
# Single command:
#     nohup bash scripts/full_pipeline.sh > pipeline.log 2>&1 &
#     tail -f pipeline.log

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate

# ---- HuggingFace cache pinning ----
# Reuse the same cache across runs so the model never redownloads.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME/hub"

# Determinism for Zindi code review.
export SEED="${SEED:-42}"
export PYTHONHASHSEED="$SEED"

STAGE="init"
trap 'echo "PIPELINE FAILED at stage=$STAGE" >&2' ERR

step() { printf "\n##### %s #####\n" "$*"; }

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
LLM_PORT="${LLM_PORT:-8001}"
SERVER_PORT="${SERVER_PORT:-7860}"
MODEL_URL="http://localhost:$LLM_PORT/v1"

PHASE1_DIR="eval/results/phase1_test"
PHASE2_DIR="eval/results/phase2_test"
TRACES="traces/train_traces.jsonl"
LORA_DIR="training/checkpoints/run_v1/best_lora"
LLM_LOG="eval/logs/llm_server.log"
SERVER_LOG="server.log"

mkdir -p "$PHASE1_DIR" "$PHASE2_DIR" "$(dirname "$TRACES")" "$(dirname "$LORA_DIR")" eval/logs

# ----- helpers -----

wait_llm_ready() {
    local timeout_s="${1:-1800}"
    local deadline
    deadline=$(( $(date +%s) + timeout_s ))
    while true; do
        if curl -sf "http://localhost:$LLM_PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
            echo "  llm_server ready"
            return 0
        fi
        if [ "$(date +%s)" -gt "$deadline" ]; then
            echo "  llm_server health check timed out" >&2
            return 1
        fi
        sleep 10
    done
}

wait_server_ready() {
    local timeout_s="${1:-120}"
    local deadline
    deadline=$(( $(date +%s) + timeout_s ))
    while true; do
        if curl -sf "http://localhost:$SERVER_PORT/health" >/dev/null 2>&1; then
            echo "  server.py ready"
            return 0
        fi
        if [ "$(date +%s)" -gt "$deadline" ]; then
            echo "  server.py health check timed out" >&2
            return 1
        fi
        sleep 2
    done
}

restart_server_test() {
    pkill -f "python server.py" 2>/dev/null || true
    sleep 2
    DATA_SPLIT=test nohup python server.py > "$SERVER_LOG" 2>&1 &
    echo "  server.py restarted with DATA_SPLIT=test (pid=$!)"
    wait_server_ready 60
}

# ============================================================
STAGE="precheck"
step "[0] Precheck"
wait_llm_ready 60 || {
    echo "llm_server is not running on port $LLM_PORT." >&2
    echo "Start it first:  bash scripts/quickrun.sh" >&2
    exit 1
}
restart_server_test

# ============================================================
STAGE="B-phase1-test"
step "[B] Run agent on the FULL 500-scenario test split"
if [ -f "$PHASE1_DIR/result.csv" ] && [ "$(wc -l < "$PHASE1_DIR/result.csv")" -ge 500 ]; then
    echo "  $PHASE1_DIR/result.csv already complete; skipping."
else
    python main.py \
        --server_url "http://localhost:$SERVER_PORT" \
        --model_url "$MODEL_URL" \
        --model_name "$MODEL_NAME" \
        --max_samples 500 \
        --num_attempts 1 \
        --max_iterations 2 \
        --save_dir "$PHASE1_DIR" \
        --log_file "$PHASE1_DIR/agent.log" \
        --quiet
fi
echo
echo ">>> PHASE 1 SUBMISSION READY: $PHASE1_DIR/result.csv"
echo ">>> Upload to Zindi NOW. Pipeline continues with distillation."
echo

# ============================================================
STAGE="C-distill"
step "[C] Self-distillation (build training corpus)"
TRACE_COUNT_BEFORE=0
[ -f "$TRACES" ] && TRACE_COUNT_BEFORE=$(wc -l < "$TRACES")
python scripts/distill.py \
    --train_file data/Phase_1/train.json \
    --output "$TRACES" \
    --model_url "$MODEL_URL" \
    --model_name "$MODEL_NAME"
TRACE_COUNT=$(wc -l < "$TRACES")
echo "  traces in corpus: $TRACE_COUNT (was $TRACE_COUNT_BEFORE)"
if [ "$TRACE_COUNT" -lt 100 ]; then
    echo "FATAL: only $TRACE_COUNT traces accepted; cannot fine-tune. Investigate distill_summary.md." >&2
    exit 1
fi

# ============================================================
STAGE="D-finetune"
step "[D] LoRA fine-tune"
if [ -d "$LORA_DIR" ] && [ -f "$LORA_DIR/adapter_config.json" ]; then
    echo "  $LORA_DIR already present; skipping training."
else
    # Free GPU memory: stop the LLM server, train, then restart with LoRA.
    echo "  stopping llm_server.py to free GPU memory for training"
    pkill -f "scripts/llm_server.py" 2>/dev/null || true
    sleep 5

    CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \
        --traces "$TRACES" \
        --output_dir "$(dirname "$LORA_DIR")" \
        --base_model "$MODEL_NAME"

    [ -d "$LORA_DIR" ] || { echo "FATAL: best_lora not produced" >&2; exit 1; }

    # Restart llm_server WITH the LoRA adapter
    mkdir -p eval/logs
    LLM_PORT="$LLM_PORT" MODEL_NAME="$MODEL_NAME" LORA_PATH="$LORA_DIR" \
        nohup python scripts/llm_server.py \
            --model "$MODEL_NAME" \
            --port "$LLM_PORT" \
            --lora "$LORA_DIR" \
            > "$LLM_LOG" 2>&1 &
    echo "  llm_server restarted with LoRA (pid=$!)"
    wait_llm_ready 1800
fi

# Make sure llm_server is still up (in case it was killed earlier and we skipped training)
if ! curl -sf "http://localhost:$LLM_PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then
    LLM_PORT="$LLM_PORT" MODEL_NAME="$MODEL_NAME" LORA_PATH="$LORA_DIR" \
        nohup python scripts/llm_server.py \
            --model "$MODEL_NAME" \
            --port "$LLM_PORT" \
            --lora "$LORA_DIR" \
            > "$LLM_LOG" 2>&1 &
    echo "  llm_server (re)started with LoRA (pid=$!)"
    wait_llm_ready 1800
fi
restart_server_test  # in case server.py died during training

# ============================================================
STAGE="E-phase2-test"
step "[E] Run fine-tuned agent on the 500-scenario test split"
if [ -f "$PHASE2_DIR/result.csv" ] && [ "$(wc -l < "$PHASE2_DIR/result.csv")" -ge 500 ]; then
    echo "  $PHASE2_DIR/result.csv already complete; skipping."
else
    python main.py \
        --server_url "http://localhost:$SERVER_PORT" \
        --model_url "$MODEL_URL" \
        --model_name "$MODEL_NAME" \
        --max_samples 500 \
        --num_attempts 1 \
        --max_iterations 2 \
        --save_dir "$PHASE2_DIR" \
        --log_file "$PHASE2_DIR/agent.log" \
        --quiet
fi

STAGE="E-build-3-submissions"
step "[E2] Build the 3 Phase 2 candidate submissions from one execution run"
python scripts/build_phase2_submissions.py \
    --completions "$PHASE2_DIR/completions.jsonl" \
    --test_file data/Phase_1/test.json \
    --out_dir "$PHASE2_DIR" || echo "WARN: variant build failed; raw $PHASE2_DIR/result.csv is still valid."

trap - ERR
echo
echo "########################################################"
echo "PIPELINE COMPLETE."
echo
echo "Phase 1 (optional)   : $PHASE1_DIR/result.csv"
echo "Distilled corpus     : $TRACES ($(wc -l < "$TRACES") traces)"
echo "Best LoRA            : $LORA_DIR"
echo "Phase 2 raw run      : $PHASE2_DIR/result.csv"
echo "Phase 2 candidate v1 : $PHASE2_DIR/result_v1_raw.csv          ← submit"
echo "Phase 2 candidate v2 : $PHASE2_DIR/result_v2_multi_recall.csv ← submit"
echo "Phase 2 candidate v3 : $PHASE2_DIR/result_v3_insurance.csv    ← submit"
echo
echo "Upload all three v1/v2/v3 files to Zindi as your 3 Phase 2 submissions."
echo "########################################################"
