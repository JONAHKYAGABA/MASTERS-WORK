#!/usr/bin/env bash
# ============================================================================
# Resilient training launcher for marconi.
#   - nohup'd so SSH disconnect / power loss survival
#   - auto-resume from latest checkpoint on rerun
#   - wandb + Hugging Face Hub logging with token loading from ~/.env
#   - mid-epoch checkpoint saves (so power loss costs <save_steps not 1 epoch)
#   - 4-stage curriculum: sg_only → alignment → pretrain → finetune
#   - 1 or 2 GPU via GPUS env var (uses torchrun + DDP for 2 GPUs)
#
# Usage:
#   GPUS=1 bash scripts/launch_resilient.sh <mode>
#   GPUS=2 bash scripts/launch_resilient.sh <mode>
#
# Modes (in curriculum order):
#   sg_only_smoke    — Stage 1 smoke (50 samples)
#   alignment_smoke  — Stage 2 smoke (50 samples)
#   pretrain_smoke   — Stage 3 smoke (50 samples)
#   finetune_smoke   — Stage 4 smoke (10 samples)
#
#   sg_only          — Stage 1 full (~20 epochs)
#   alignment        — Stage 2 full (~10 epochs)
#   pretrain         — Stage 3 full (~30-50 epochs)
#   finetune         — Stage 4 full (~15 epochs)
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

MODE="${1:-pretrain_smoke}"
GPUS="${GPUS:-1}"
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
    echo "[env] WARNING: WANDB_API_KEY not set — wandb will run offline/disabled."
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[env] WARNING: HF_TOKEN not set — Hugging Face push will be skipped."
fi

# --- 2. Activate venv ---
if [[ -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# --- 3. Mode → phase / config / sample size / save cadence ---
# All stages share the same two config files: stages 1-3 use pretrain_config,
# stage 4 uses finetune_config. The --phase flag re-routes the model's
# training_mode (freezing the right components per the PDF).
SHARED_PRETRAIN_CFG="configs/pretrain_config.yaml"
SHARED_FINETUNE_CFG="configs/finetune_config.yaml"

case "$MODE" in
    sg_only_smoke)
        CONFIG="$SHARED_PRETRAIN_CFG"; PHASE="sg_only"
        OUTPUT_DIR="./checkpoints/stage1_sg_only_smoke"
        EXTRA_ARGS=(--max_samples 50 --epochs 1 --batch_size 1 --save_steps 10 --skip_data_check)
        ;;
    alignment_smoke)
        CONFIG="$SHARED_PRETRAIN_CFG"; PHASE="alignment"
        OUTPUT_DIR="./checkpoints/stage2_alignment_smoke"
        EXTRA_ARGS=(--max_samples 50 --epochs 1 --batch_size 1 --save_steps 10 --skip_data_check)
        ;;
    pretrain_smoke)
        CONFIG="$SHARED_PRETRAIN_CFG"; PHASE="pretrain"
        OUTPUT_DIR="./checkpoints/stage3_pretrain_smoke"
        EXTRA_ARGS=(--max_samples 50 --epochs 1 --batch_size 1 --save_steps 10 --skip_data_check)
        ;;
    finetune_smoke)
        CONFIG="$SHARED_FINETUNE_CFG"; PHASE="finetune"
        OUTPUT_DIR="./checkpoints/stage4_finetune_smoke"
        EXTRA_ARGS=(--max_samples 10 --epochs 1 --batch_size 1 --save_steps 2 --skip_data_check)
        ;;

    sg_only)
        CONFIG="$SHARED_PRETRAIN_CFG"; PHASE="sg_only"
        OUTPUT_DIR="./checkpoints/stage1_sg_only"
        EXTRA_ARGS=(--epochs 20 --save_steps 500 --push_every_save)
        ;;
    alignment)
        CONFIG="$SHARED_PRETRAIN_CFG"; PHASE="alignment"
        OUTPUT_DIR="./checkpoints/stage2_alignment"
        EXTRA_ARGS=(--epochs 10 --save_steps 500 --push_every_save)
        ;;
    pretrain)
        CONFIG="$SHARED_PRETRAIN_CFG"; PHASE="pretrain"
        OUTPUT_DIR="./checkpoints/stage3_pretrain"
        EXTRA_ARGS=(--save_steps 500 --push_every_save)
        ;;
    finetune)
        CONFIG="$SHARED_FINETUNE_CFG"; PHASE="finetune"
        OUTPUT_DIR="./checkpoints/stage4_finetune"
        EXTRA_ARGS=(--save_steps 200 --push_every_save)
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Expected: sg_only_smoke | alignment_smoke | pretrain_smoke | finetune_smoke"
        echo "          sg_only | alignment | pretrain | finetune"
        exit 1
        ;;
esac

# --- 4. Multi-GPU plumbing ---
#   DEEPSPEED=1            → deepspeed launcher + --use_deepspeed (ZeRO)
#   GPUS=1 + DEEPSPEED!=1  → plain python (single process)
#   GPUS=2 + DEEPSPEED!=1  → torchrun --nproc_per_node=2 with --use_ddp
USE_DEEPSPEED="${DEEPSPEED:-0}"
if [[ "$USE_DEEPSPEED" == "1" ]]; then
    if ! python -c "import deepspeed" 2>/dev/null; then
        echo "[abort] DEEPSPEED=1 but 'deepspeed' is not installed. Install with: pip install deepspeed"
        exit 1
    fi
    LAUNCHER=(deepspeed --num_gpus="$GPUS")
    DIST_FLAGS=(--use_deepspeed)
    echo "[gpu] DeepSpeed mode: $GPUS GPUs via deepspeed launcher (ZeRO)"
elif [[ "$GPUS" -gt 1 ]]; then
    LAUNCHER=(torchrun --nproc_per_node="$GPUS")
    DIST_FLAGS=(--use_ddp)
    echo "[gpu] multi-GPU mode: $GPUS GPUs via torchrun + DDP"
else
    LAUNCHER=(python -u)
    DIST_FLAGS=()
    echo "[gpu] single-GPU mode (GPUS=1)"
fi

mkdir -p logs "$OUTPUT_DIR"

# --- 5. Build full command ---
LOG="logs/${MODE}_g${GPUS}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="logs/${MODE}.pid"

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
    "${LAUNCHER[@]}"
    train_mimic_cxr.py
    --config "$CONFIG"
    --phase "$PHASE"
    --mimic_cxr_path data/mimic-cxr-jpg
    --mimic_qa_path data/mimic-ext-cxr-qba
    --output_dir "$OUTPUT_DIR"
    --auto_resume
    "${DIST_FLAGS[@]}"
    "${EXTRA_ARGS[@]}"
)

echo "[launch] mode=$MODE phase=$PHASE gpus=$GPUS"
echo "[launch] config=$CONFIG output=$OUTPUT_DIR"
echo "[launch] log=$LOG"
echo "[launch] cmd: ${CMD[*]}"
echo

nohup "${CMD[@]}" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo "✓ started PID=$PID"
echo "  tail:   tail -f $LOG"
echo "  status: ps -p $PID -o pid,etime,pcpu,pmem,cmd"
echo "  gpus:   watch -n 2 nvidia-smi"
echo "  stop:   kill \$(cat $PID_FILE)"
echo "  rerun after crash: GPUS=$GPUS bash scripts/launch_resilient.sh $MODE   # auto-resumes"
