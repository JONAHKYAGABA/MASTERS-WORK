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
    smoke|budget|full) ;;
    *) echo "Unknown mode: $MODE (expected: smoke | budget | full)"; exit 1 ;;
esac

# ============================================================================
# SELF-DAEMONIZATION
# ----------------------------------------------------------------------------
# Default: run in BACKGROUND so SSH disconnect / terminal close does NOT kill
# the curriculum. Re-execs this script under setsid + nohup, detaches from
# the controlling terminal, redirects all output to a master log, writes a
# PID file, prints control commands, and exits the parent.
#
# Override with FG=1 to run in foreground (useful if you want to watch the
# smoke output live and Ctrl-C it).
#
# Stop the background curriculum:
#   kill -- -$(cat logs/curriculum.pid)    # kills the whole process group
# ============================================================================
FG="${FG:-0}"
if [[ "$FG" != "1" && "${CURRICULUM_DAEMON:-0}" != "1" ]]; then
    mkdir -p logs
    MASTER_LOG="logs/curriculum_${MODE}_$(date +%Y%m%d_%H%M%S).log"
    PID_FILE="logs/curriculum.pid"

    # Refuse to start if another curriculum is already running.
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "[abort] curriculum already running (PID=$(cat "$PID_FILE"))"
        echo "        Stop it first:   kill -- -\$(cat $PID_FILE)"
        echo "        Or watch it:     tail -f \$(ls -t logs/curriculum_*.log | head -1)"
        exit 1
    fi

    # Re-exec ourselves fully detached. setsid → new session (immune to
    # controlling-terminal SIGHUP), nohup → belt-and-suspenders, </dev/null
    # closes stdin so background reads don't error, > redirects all output.
    CURRICULUM_DAEMON=1 \
        setsid nohup "$BASH" "$0" "$@" </dev/null > "$MASTER_LOG" 2>&1 &
    PID=$!
    echo "$PID" > "$PID_FILE"

    echo
    echo "════════════════════════════════════════════════════════════════════"
    echo "  ✓ Curriculum launched in background — survives SSH disconnect"
    echo "════════════════════════════════════════════════════════════════════"
    echo "  PID:    $PID"
    echo "  log:    $MASTER_LOG"
    echo
    echo "  watch live:   tail -f $MASTER_LOG"
    echo "  check alive:  ps -p $PID -o pid,etime,pcpu,pmem,cmd"
    echo "  stop run:     kill -- -$PID         # kills entire process group"
    echo "                # or: kill -- -\$(cat $PID_FILE)"
    echo "  GPU usage:    watch -n 2 nvidia-smi"
    echo
    echo "  Re-run after crash: just run this command again (auto-resume)."
    echo "════════════════════════════════════════════════════════════════════"
    exit 0
fi

# We get here only inside the daemonized child OR when FG=1 was passed.
if [[ "${CURRICULUM_DAEMON:-0}" == "1" ]]; then
    # Print a header in the log file so tail -f users see what they're watching.
    echo "════════════════════════════════════════════════════════════════════"
    echo "  Curriculum daemon — mode=$MODE gpus=$GPUS pid=$$ started $(date)"
    echo "════════════════════════════════════════════════════════════════════"
fi

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
# DEEPSPEED=1 → use DeepSpeed launcher + --use_deepspeed (ZeRO-2 auto-config)
# Otherwise → torchrun + --use_ddp (vanilla DDP). DeepSpeed gives lower memory
# (ZeRO-2 shards optimizer state) but adds a build dependency. For smoke,
# DDP is fine and avoids the deepspeed compile step.
USE_DEEPSPEED="${DEEPSPEED:-0}"

if [[ "$USE_DEEPSPEED" == "1" ]]; then
    if ! python -c "import deepspeed" 2>/dev/null; then
        fail "DEEPSPEED=1 but 'deepspeed' is not installed."
        fail "  Install with: pip install deepspeed"
        exit 1
    fi
    LAUNCHER=(deepspeed --num_gpus="$GPUS")
    DIST_FLAGS=(--use_deepspeed)
    info "GPUs: $GPUS via DeepSpeed (ZeRO)"
elif [[ "$GPUS" -gt 1 ]]; then
    LAUNCHER=(torchrun --nproc_per_node="$GPUS")
    DIST_FLAGS=(--use_ddp)
    info "GPUs: $GPUS via torchrun + DDP"
else
    LAUNCHER=(python -u)
    DIST_FLAGS=()
    info "GPUs: 1 (single process)"
fi

# --- Phase-specific QA paths (QBA pre-built exports) ---
# Pretrain stages use B_frontal (31M, grade B+, broader/noisier).
# Finetune uses A_frontal (7.5M, grade A+, clean).
# Falls back to the raw qa/ dir if exports aren't extracted.
QA_ROOT="data/mimic-ext-cxr-qba"
QA_PRETRAIN="$QA_ROOT/exports/B_frontal"
QA_FINETUNE="$QA_ROOT/exports/A_frontal"
if [[ ! -d "$QA_PRETRAIN/qa" ]]; then
    info "B_frontal not extracted, falling back to raw QA path for pretrain stages"
    QA_PRETRAIN="$QA_ROOT"
fi
if [[ ! -d "$QA_FINETUNE/qa" ]]; then
    info "A_frontal not extracted, falling back to raw QA path for finetune stage"
    QA_FINETUNE="$QA_ROOT"
fi
info "Pretrain QA path: $QA_PRETRAIN"
info "Finetune QA path: $QA_FINETUNE"

# --- Stage config table ---
# Format: PHASE | CONFIG | OUTPUT_DIR | QA_PATH | EXTRA_ARGS
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
declare -a STAGE_QA=(
    "$QA_PRETRAIN"   # Stage 1: train SG generator — use broad data
    "$QA_PRETRAIN"   # Stage 2: alignment on broad data
    "$QA_PRETRAIN"   # Stage 3: pretrain on broad data
    "$QA_FINETUNE"   # Stage 4: finetune on clean A-grade data
)
# --- Pick Qwen model size: smoke=3B (fast download, fits easily),
#     full=7B (production). Override with QWEN_MODEL=Qwen/...
if [[ "$MODE" == "smoke" ]]; then
    QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
else
    QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
fi
info "Qwen model: $QWEN_MODEL"

if [[ "$MODE" == "smoke" ]]; then
    # Smoke mode: DO NOT --push_every_save (eats HF free-tier private storage
    # quota with throwaway checkpoints). Smoke runs save locally only.
    declare -a STAGE_EXTRA=(
        "--qwen_model_id $QWEN_MODEL --max_samples 50 --epochs 1 --batch_size 1 --save_steps 10 --skip_data_check --disable_wandb"
        "--qwen_model_id $QWEN_MODEL --max_samples 50 --epochs 1 --batch_size 1 --save_steps 10 --skip_data_check --disable_wandb"
        "--qwen_model_id $QWEN_MODEL --max_samples 50 --epochs 1 --batch_size 1 --save_steps 10 --skip_data_check --disable_wandb"
        "--qwen_model_id $QWEN_MODEL --max_samples 20 --epochs 1 --batch_size 1 --save_steps 5  --skip_data_check --disable_wandb --quality_grade all"
        # NOTE: finetune uses --max_samples 20 (not 10) because the trainer
        # auto-divides val by 10 — max_samples=10 → val=1 → metric crash.
        # NOTE: --quality_grade all on smoke because the default 'A' filter
        # rejects ~100% of QBA samples (your dataset may not have A-grade
        # questions, or the filter is over-strict). Real finetune should
        # use 'A' if you have A_frontal data, else 'B' or 'all'.
        # NOTE: --disable_wandb on smoke avoids cluttering wandb with 4 tiny
        # throwaway runs per pipeline test. Full mode logs to wandb.
    )
elif [[ "$MODE" == "budget" ]]; then
    # "good enough" model on 2× RTX 8000 in ~3 days using q1M subsets.
    # batch=2 grad_accum=4 → effective batch 16, safe for Qwen 3B @ 448px.
    declare -a STAGE_EXTRA=(
        "--qwen_model_id $QWEN_MODEL --max_samples 200000 --epochs 3 --batch_size 2 --save_steps 500 --push_every_save"
        "--qwen_model_id $QWEN_MODEL --max_samples 200000 --epochs 2 --batch_size 2 --save_steps 500 --push_every_save"
        "--qwen_model_id $QWEN_MODEL --max_samples 1000000 --epochs 2 --batch_size 2 --save_steps 1000 --push_every_save"
        "--qwen_model_id $QWEN_MODEL --max_samples 1000000 --epochs 3 --batch_size 2 --save_steps 1000 --push_every_save"
    )
else
    # "full" = paper-spec curriculum. Takes ~80 days on 2× RTX 8000.
    # If you actually have that time budget, use this. Otherwise prefer 'budget'.
    declare -a STAGE_EXTRA=(
        "--qwen_model_id $QWEN_MODEL --epochs 20 --save_steps 500 --push_every_save"
        "--qwen_model_id $QWEN_MODEL --epochs 10 --save_steps 500 --push_every_save"
        "--qwen_model_id $QWEN_MODEL --epochs 40 --save_steps 500 --push_every_save"
        "--qwen_model_id $QWEN_MODEL --epochs 15 --save_steps 200 --push_every_save"
    )
fi

mkdir -p logs
CURRICULUM_START=$(date +%s)
banner "Curriculum: mode=$MODE  gpus=$GPUS  stages=4"

# --- Pre-download all HF models BEFORE launching DDP ---
# Without this, both DDP ranks call from_pretrained() at the same time and
# tqdm bars interleave into unreadable garbage that looks like a hang.
# Skip if SKIP_PREDOWNLOAD=1 is set (useful when re-running after a
# completed download).
if [[ "${SKIP_PREDOWNLOAD:-0}" != "1" ]]; then
    banner "Pre-downloading HF models (clean single-process progress)"
    python scripts/predownload_models.py --model "$QWEN_MODEL"
    ok "Models cached locally — DDP ranks will load instantly from now on."
fi

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
        QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-200704}" \
        QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-65536}" \
        SKIP_VISION_PATH_CHECK="${SKIP_VISION_PATH_CHECK:-1}" \
        PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
        "${LAUNCHER[@]}" \
            train_mimic_cxr.py \
            --config "$cfg" \
            --phase "$phase" \
            --mimic_cxr_path data/mimic-cxr-jpg \
            --mimic_qa_path "${STAGE_QA[$idx]}" \
            --output_dir "$outdir" \
            "${DIST_FLAGS[@]}" \
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
