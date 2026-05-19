#!/usr/bin/env bash
# ============================================================================
# Resilient training launcher for marconi.
#   - nohup'd so SSH disconnect / power loss survival
#   - auto-resume from latest checkpoint on rerun
#   - wandb + Hugging Face Hub logging with token loading from ~/.env
#   - mid-epoch checkpoint saves (so power loss costs <save_steps not 1 epoch)
#
# Usage (first time):
#   bash scripts/launch_resilient.sh smoke      # ~50 samples sanity run
#   bash scripts/launch_resilient.sh pretrain   # full pretrain
#   bash scripts/launch_resilient.sh finetune   # full finetune
#
# After a crash / power loss / SSH death, JUST RERUN THE SAME COMMAND.
# --auto_resume picks up from the latest checkpoint and continues the same
# wandb run.
#
# Expected ~/.env format (chmod 600 ~/.env):
#   export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#   export WANDB_ENTITY=your-wandb-user-or-team   # optional
# ============================================================================

set -eo pipefail

MODE="${1:-smoke}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# --- 1. Load secrets ---
if [[ -f "$HOME/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$HOME/.env"
    set +a
    echo "[env] loaded ~/.env"
else
    echo "[env] WARNING: ~/.env not found. wandb/HF push will be skipped."
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[env] WARNING: WANDB_API_KEY not set — wandb will run in disabled/offline mode."
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[env] WARNING: HF_TOKEN not set — push to Hugging Face Hub will be skipped."
fi

# --- 2. Activate venv ---
if [[ -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# --- 3. Mode-specific config + flags ---
case "$MODE" in
    smoke)
        CONFIG="configs/pretrain_config.yaml"
        PHASE="pretrain"
        OUTPUT_DIR="./checkpoints/smoke"
        EXTRA_ARGS=(
            --max_samples 50
            --epochs 1
            --batch_size 1
            --save_steps 10
            --skip_data_check
        )
        ;;
    pretrain)
        CONFIG="configs/pretrain_config.yaml"
        PHASE="pretrain"
        OUTPUT_DIR="./checkpoints/pretrain"
        EXTRA_ARGS=(
            --save_steps 500
            --push_every_save
        )
        ;;
    finetune)
        CONFIG="configs/finetune_config.yaml"
        PHASE="finetune"
        OUTPUT_DIR="./checkpoints/finetune"
        EXTRA_ARGS=(
            --save_steps 200
            --push_every_save
        )
        ;;
    *)
        echo "Unknown mode: $MODE (expected: smoke | pretrain | finetune)"
        exit 1
        ;;
esac

mkdir -p logs "$OUTPUT_DIR"

# --- 4. Build full command ---
LOG="logs/${MODE}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="logs/${MODE}.pid"

# Avoid double-starting: if a previous run is still alive, refuse.
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[abort] $MODE already running (PID=$(cat "$PID_FILE")). Stop it first:"
    echo "        kill \$(cat $PID_FILE)"
    exit 1
fi

CMD=(
    env
    PYTHONPATH="$PROJECT_DIR"
    WANDB_API_KEY="${WANDB_API_KEY:-}"
    HF_TOKEN="${HF_TOKEN:-}"
    WANDB_ENTITY="${WANDB_ENTITY:-}"
    python -u train_mimic_cxr.py
    --config "$CONFIG"
    --phase "$PHASE"
    --mimic_cxr_path data/mimic-cxr-jpg
    --mimic_qa_path data/mimic-ext-cxr-qba
    --output_dir "$OUTPUT_DIR"
    --auto_resume
    "${EXTRA_ARGS[@]}"
)

echo "[launch] mode=$MODE"
echo "[launch] log=$LOG"
echo "[launch] cmd: ${CMD[*]}"
echo

nohup "${CMD[@]}" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo "✓ started PID=$PID"
echo "  tail:  tail -f $LOG"
echo "  status: ps -p $PID -o pid,etime,pcpu,pmem,cmd"
echo "  stop:  kill \$(cat $PID_FILE)"
echo "  rerun after crash: bash scripts/launch_resilient.sh $MODE   # auto-resumes"
