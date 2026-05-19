#!/usr/bin/env bash
# ============================================================================
# Full 4-stage curriculum runner — ONE command, end-to-end.
#
#   Stage 1: sg_only    (train SG generator with GT)
#   Stage 2: alignment  (train SG encoder/projector with GT)
#   Stage 3: pretrain   (full multi-task with generated SGs)
#   Stage 4: finetune   (task-specific with generated SGs)
#
# Each stage:
#   - Runs in FOREGROUND with live progress (tee'd to a per-stage log)
#   - Picks up the previous stage's BEST checkpoint as starting weights
#     via --load_weights_only (model weights port, fresh optimizer)
#   - Survives within-stage power loss via --auto_resume (mid-epoch saves)
#   - Pushes to wandb + HF Hub (private) every checkpoint
#
# Usage:
#   GPUS=2 bash scripts/run_curriculum.sh smoke   # 50/50/50/10 samples
#   GPUS=2 bash scripts/run_curriculum.sh full    # full datasets
#
# Power-loss / SSH-death survival:
#   nohup bash scripts/run_curriculum.sh smoke > curriculum.log 2>&1 &
#   tail -f curriculum.log
#   (rerun the exact same command after a crash — within-stage auto-resume +
#    cross-stage checkpoint detection skip already-completed stages)
#
# Debugging failures:
#   The script EXITS NON-ZERO on the first stage failure with a banner
#   pointing at the log file. To re-attempt a single stage:
#       GPUS=2 bash scripts/launch_resilient.sh <stage>_smoke
# ============================================================================

set -eo pipefail

MODE="${1:-smoke}"
GPUS="${GPUS:-1}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

case "$MODE" in
    smoke|full) ;;
    *) echo "Unknown mode: $MODE (expected: smoke | full)"; exit 1 ;;
esac

# --- ANSI colors (foreground only — strip when piping to file via tee) ---
if [[ -t 1 ]]; then
    C_BANNER='\033[1;36m'; C_OK='\033[1;32m'; C_FAIL='\033[1;31m'
    C_INFO='\033[0;34m';   C_DIM='\033[2m';     C_RESET='\033[0m'
else
    C_BANNER=''; C_OK=''; C_FAIL=''; C_INFO=''; C_DIM=''; C_RESET=''
fi

banner() {
    local title="$1"
    echo
    echo -e "${C_BANNER}╔══════════════════════════════════════════════════════════════════╗${C_RESET}"
    printf "${C_BANNER}║  %-64s║${C_RESET}\n" "$title"
    echo -e "${C_BANNER}╚══════════════════════════════════════════════════════════════════╝${C_RESET}"
}

info()  { echo -e "${C_INFO}[info]${C_RESET}  $*"; }
ok()    { echo -e "${C_OK}[ ok ]${C_RESET}  $*"; }
fail()  { echo -e "${C_FAIL}[FAIL]${C_RESET}  $*" >&2; }

# --- Load secrets ---
if [[ -f "$HOME/.env" ]]; then
    set -a; source "$HOME/.env"; set +a
    info "loaded ~/.env"
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then info "WANDB_API_KEY not set — wandb offline/disabled"; fi
if [[ -z "${HF_TOKEN:-}" ]];        then info "HF_TOKEN not set — HF push skipped"; fi

# --- Activate venv ---
if [[ -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# --- Multi-GPU plumbing ---
if [[ "$GPUS" -gt 1 ]]; then
    LAUNCHER=(torchrun --nproc_per_node="$GPUS")
    DDP_FLAGS=(--use_ddp)
    info "GPUs: $GPUS via torchrun + DDP"
else
    LAUNCHER=(python -u)
    DDP_FLAGS=()
    info "GPUs: 1 (single process)"
fi

# --- Stage config table ---
# Format: PHASE | CONFIG | OUTPUT_DIR | EXTRA_ARGS
declare -a STAGE_PHASE=(sg_only alignment pretrain finetune)
declare -a STAGE_CFG=(
    "configs/pretrain_config.yaml"
    "configs/pretrain_config.yaml"
    "configs/pretrain_config.yaml"
    "configs/finetune_config.yaml"
)
declare -a STAGE_OUTDIR=(
    "./checkpoints/stage1_sg_only"
    "./checkpoints/stage2_alignment"
    "./checkpoints/stage3_pretrain"
    "./checkpoints/stage4_finetune"
)
if [[ "$MODE" == "smoke" ]]; then
    declare -a STAGE_EXTRA=(
        "--max_samples 50 --epochs 1 --batch_size 1 --save_steps 10 --skip_data_check"
        "--max_samples 50 --epochs 1 --batch_size 1 --save_steps 10 --skip_data_check"
        "--max_samples 50 --epochs 1 --batch_size 1 --save_steps 10 --skip_data_check"
        "--max_samples 20 --epochs 1 --batch_size 1 --save_steps 5  --skip_data_check"
        # NOTE: finetune uses --max_samples 20 (not 10) because the trainer
        # auto-divides val by 10 — max_samples=10 → val=1 → metric crash.
        # Set this to your real finetune count; for "10 samples" of TRAIN
        # use --max_samples 11 or override the val-split fraction in the
        # trainer.
    )
else
    declare -a STAGE_EXTRA=(
        "--epochs 20 --save_steps 500 --push_every_save"
        "--epochs 10 --save_steps 500 --push_every_save"
        "--epochs 40 --save_steps 500 --push_every_save"
        "--epochs 15 --save_steps 200 --push_every_save"
    )
fi

mkdir -p logs
CURRICULUM_START=$(date +%s)
banner "Curriculum: mode=$MODE  gpus=$GPUS  stages=4"

# --- Find latest checkpoint dir for cross-stage handoff ---
latest_ckpt_dir_for() {
    local stage_outdir="$1"
    local pointer="$stage_outdir/latest_checkpoint.txt"
    if [[ -f "$pointer" ]]; then
        local name
        name=$(<"$pointer")
        name="${name//[$'\r\n ']}"
        if [[ -d "$stage_outdir/$name" ]]; then
            echo "$stage_outdir/$name"
            return 0
        fi
    fi
    # Fallback: best_model dir
    if [[ -d "$stage_outdir/best_model" ]]; then
        echo "$stage_outdir/best_model"
        return 0
    fi
    return 1
}

# --- Is a stage already complete? Detected via training_metadata.json on
#     a "final" checkpoint (max global_step). Crude but effective.
stage_already_complete() {
    local stage_outdir="$1"
    # Heuristic: if best_model exists AND the script has been run to
    # completion before (final-save banner left a .done marker), skip.
    [[ -f "$stage_outdir/.curriculum_done" ]]
}

mark_stage_done() {
    local stage_outdir="$1"
    touch "$stage_outdir/.curriculum_done"
}

# --- Per-stage runner ---
run_stage() {
    local idx="$1"
    local phase="${STAGE_PHASE[$idx]}"
    local cfg="${STAGE_CFG[$idx]}"
    local outdir="${STAGE_OUTDIR[$idx]}"
    local extra="${STAGE_EXTRA[$idx]}"
    local stage_num=$((idx + 1))
    local stage_name="Stage ${stage_num}: ${phase}"

    if stage_already_complete "$outdir"; then
        ok "$stage_name — already completed (found ${outdir}/.curriculum_done), skipping."
        return 0
    fi

    mkdir -p "$outdir" logs

    # --- Cross-stage weight transfer: pick up previous stage's checkpoint ---
    local resume_flags=(--auto_resume)
    if [[ "$idx" -gt 0 ]]; then
        local prev_outdir="${STAGE_OUTDIR[$((idx - 1))]}"
        local prev_ckpt
        if prev_ckpt=$(latest_ckpt_dir_for "$prev_outdir"); then
            resume_flags+=(--resume_from_checkpoint "$prev_ckpt" --load_weights_only)
            info "$stage_name will load weights from previous stage: $prev_ckpt"
        else
            fail "$stage_name expected a checkpoint in $prev_outdir but none found."
            fail "    Did the previous stage finish? Check: $prev_outdir"
            return 1
        fi
    fi

    local log="logs/curriculum_${MODE}_stage${stage_num}_${phase}_$(date +%Y%m%d_%H%M%S).log"

    banner "$stage_name  →  $outdir"
    info "config:    $cfg"
    info "extras:    $extra"
    info "log:       $log"
    info "resume:    ${resume_flags[*]}"
    echo

    # --- Build + execute command (streams live to terminal AND tee'd log) ---
    local stage_start=$(date +%s)
    set +e
    env \
        PYTHONPATH="$PROJECT_DIR" \
        WANDB_API_KEY="${WANDB_API_KEY:-}" \
        HF_TOKEN="${HF_TOKEN:-}" \
        WANDB_ENTITY="${WANDB_ENTITY:-}" \
        "${LAUNCHER[@]}" \
            train_mimic_cxr.py \
            --config "$cfg" \
            --phase "$phase" \
            --mimic_cxr_path data/mimic-cxr-jpg \
            --mimic_qa_path data/mimic-ext-cxr-qba \
            --output_dir "$outdir" \
            "${DDP_FLAGS[@]}" \
            "${resume_flags[@]}" \
            $extra \
            2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e

    local stage_end=$(date +%s)
    local elapsed=$(( stage_end - stage_start ))

    if [[ $rc -ne 0 ]]; then
        fail "$stage_name failed (exit=$rc) after ${elapsed}s"
        fail "    log: $log"
        fail "    last 30 lines:"
        tail -n 30 "$log" | sed 's/^/      /'
        fail "    to retry this stage in isolation:"
        fail "      GPUS=$GPUS bash scripts/launch_resilient.sh ${phase}_smoke"
        return $rc
    fi

    mark_stage_done "$outdir"
    ok "$stage_name complete in ${elapsed}s — checkpoint dir: $outdir"
    return 0
}

# --- Run all 4 stages, halting on first failure ---
for i in 0 1 2 3; do
    if ! run_stage "$i"; then
        fail "Curriculum halted at Stage $((i + 1))"
        exit 1
    fi
done

# --- Final summary ---
CURRICULUM_END=$(date +%s)
TOTAL=$(( CURRICULUM_END - CURRICULUM_START ))
banner "Curriculum complete in ${TOTAL}s (~$(( TOTAL / 60 )) min)"
for i in 0 1 2 3; do
    ckpt=$(latest_ckpt_dir_for "${STAGE_OUTDIR[$i]}" 2>/dev/null || echo "<none>")
    ok "Stage $((i + 1)) (${STAGE_PHASE[$i]}): ${ckpt}"
done
echo
info "Use the final checkpoint for inference / visualization:"
info "  PYTHONPATH=\$PWD python scripts/predict_and_visualize.py \\"
info "      --checkpoint ${STAGE_OUTDIR[3]}/best_model \\"
info "      --image <path-to-xray.jpg> \\"
info "      --output_dir predictions/"
