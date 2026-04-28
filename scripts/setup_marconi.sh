#!/usr/bin/env bash
# ============================================================================
# SSG-VQA-Net v2 — one-shot environment + smoke-test setup for marconi
# ============================================================================
# Layout assumed on the server:
#   $HOME/JONAHMASTERS/
#       MASTERS-WORK/                          <-- this project
#       jpgdataset/physionet.org/files/
#           mimic-cxr-jpg/2.1.0/files/p10..p19 <-- MIMIC-CXR-JPG images + CSVs
#
# Usage (defaults are tailored to marconi; override with env vars):
#   cd ~/JONAHMASTERS/MASTERS-WORK
#   bash scripts/setup_marconi.sh
#
#   # skip steps:
#   SKIP_ENV=1     bash scripts/setup_marconi.sh   # don't create conda env
#   SKIP_DEPS=1    bash scripts/setup_marconi.sh   # don't pip install
#   SKIP_SMOKE=1   bash scripts/setup_marconi.sh   # don't run smoke test
#   ONLY_SMOKE=1   bash scripts/setup_marconi.sh   # run only the smoke test
#
#   # override paths / model:
#   ENV_NAME=myenv  PY_VERSION=3.11  bash scripts/setup_marconi.sh
#   SMOKE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct  bash scripts/setup_marconi.sh
# ============================================================================

set -euo pipefail

# ---------------- configurable knobs ---------------------------------------
ENV_NAME="${ENV_NAME:-ssgqa-v2}"
PY_VERSION="${PY_VERSION:-3.10}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/JONAHMASTERS/MASTERS-WORK}"
MIMIC_CXR_ROOT="${MIMIC_CXR_ROOT:-$HOME/JONAHMASTERS/jpgdataset/physionet.org/files/mimic-cxr-jpg/2.1.0}"
MIMIC_QA_ROOT="${MIMIC_QA_ROOT:-$PROJECT_ROOT/data/mimic-ext-cxr-qba}"
SG_ROOT="${SG_ROOT:-$PROJECT_ROOT/data/chest-imagenome}"
SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"

# ---------------- ANSI colours ---------------------------------------------
G='\033[0;32m'; R='\033[0;31m'; Y='\033[1;33m'; B='\033[0;34m'; N='\033[0m'
log()  { printf "${B}[setup]${N} %s\n" "$*"; }
ok()   { printf "${G}[ ok ]${N} %s\n" "$*"; }
warn() { printf "${Y}[warn]${N} %s\n" "$*"; }
err()  { printf "${R}[fail]${N} %s\n" "$*" 1>&2; }

# ---------------- torch CUDA verification + auto-recovery -----------------
# Returns 0 if torch.cuda.is_available() is True after this call. Otherwise
# uninstalls torch and reinstalls from the supplied wheel index, then
# re-verifies. Hard-fails if reinstall doesn't fix it.
#
# Args:
#   $1  pip wheel index URL (e.g. https://download.pytorch.org/whl/cu121)
#         pass empty string to skip CUDA enforcement (CPU-only host).
ensure_torch_cuda() {
    local idx="${1:-}"
    if [[ -z "$idx" || "$idx" == *"/cpu" ]]; then
        log "  (skipping torch CUDA check — CPU index in use)"
        return 0
    fi

    local verify_cmd='import torch
print(f"torch={torch.__version__}  cuda_build={torch.version.cuda}  is_available={torch.cuda.is_available()}")'

    local out
    out="$(python -c "$verify_cmd" 2>&1 || true)"
    log "  $out"
    if [[ "$out" == *"is_available=True"* ]]; then
        ok "torch CUDA verified"
        return 0
    fi

    warn "torch is CPU-only or broken — wiping and reinstalling from $idx"
    python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
    python -m pip install --no-cache-dir --index-url "$idx" \
        "torch>=2.3,<2.6" "torchvision" "torchaudio"

    out="$(python -c "$verify_cmd" 2>&1 || true)"
    log "  after reinstall: $out"
    if [[ "$out" == *"is_available=True"* ]]; then
        ok "torch CUDA verified after reinstall"
        return 0
    fi

    err "torch still has no CUDA after reinstall."
    err "  Driver-reported CUDA: ${DRV_CUDA:-unknown}"
    err "  Wheel index used:    $idx"
    err "  Manual recovery:"
    err "    conda activate $ENV_NAME"
    err "    pip uninstall -y torch torchvision torchaudio"
    err "    pip install --no-cache-dir --index-url $idx 'torch>=2.3,<2.6' torchvision torchaudio"
    err "    python -c 'import torch; print(torch.cuda.is_available())'"
    return 1
}

# Pick a torch wheel index from the driver-reported CUDA. Echoes the URL so
# callers can capture it: TORCH_INDEX="$(pick_torch_index "$DRV_CUDA")"
pick_torch_index() {
    local drv="${1:-}"
    case "$drv" in
        13.*|12.[1-9]*|12.[1-9]) echo "https://download.pytorch.org/whl/cu121" ;;
        12.0*|11.8*|11.[8-9])    echo "https://download.pytorch.org/whl/cu118" ;;
        "")                      echo "https://download.pytorch.org/whl/cpu"   ;;
        *)                       echo "https://download.pytorch.org/whl/cu121" ;;
    esac
}

# Detect driver CUDA via nvidia-smi (safe under set -euo pipefail).
detect_driver_cuda() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo ""
        return
    fi
    nvidia-smi --query-gpu=driver_version,cuda_version \
               --format=csv,noheader 2>/dev/null \
        | head -n1 \
        | awk -F',' '{gsub(/ /,"",$2); print $2}' \
        || echo ""
}

# ---------------- conda discovery / auto-install --------------------------
# Look for an existing conda first; if none and AUTO_INSTALL_CONDA != 0,
# silently install Miniconda to $HOME/miniconda3. Returns 0 if conda is
# usable on PATH after this function returns.
ensure_conda() {
    if command -v conda >/dev/null 2>&1; then
        ok "conda found: $(command -v conda)"
        return 0
    fi
    for d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
             "/opt/conda" "/opt/miniconda3" "/opt/anaconda3"; do
        if [[ -x "$d/bin/conda" ]]; then
            export PATH="$d/bin:$PATH"
            ok "found conda at $d"
            return 0
        fi
    done
    if [[ "${AUTO_INSTALL_CONDA:-1}" != "1" ]]; then
        err "conda not found and AUTO_INSTALL_CONDA=0; install it manually."
        return 1
    fi

    local arch url installer
    arch="$(uname -m)"
    case "$arch" in
        x86_64|amd64)  url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" ;;
        aarch64|arm64) url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh" ;;
        *) err "no Miniconda installer for arch '$arch'; install conda manually."; return 1 ;;
    esac
    log "conda not found — auto-installing Miniconda to \$HOME/miniconda3 (~400 MB)"
    log "  (skip with: AUTO_INSTALL_CONDA=0 bash scripts/setup_marconi.sh)"
    installer="/tmp/miniconda_installer_$$.sh"
    if command -v curl >/dev/null 2>&1; then
        curl -fSL "$url" -o "$installer"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --show-progress "$url" -O "$installer"
    else
        err "neither curl nor wget is available; cannot download Miniconda."
        return 1
    fi
    bash "$installer" -b -u -p "$HOME/miniconda3"
    rm -f "$installer"
    export PATH="$HOME/miniconda3/bin:$PATH"
    # Initialize conda for THIS shell (no permanent .bashrc edit)
    # so the activate calls below work without sourcing .bashrc.
    "$HOME/miniconda3/bin/conda" init bash >/dev/null 2>&1 || true
    ok "Miniconda installed at \$HOME/miniconda3"
    return 0
}

# ---------------- ONLY_SMOKE shortcut --------------------------------------
if [[ "${ONLY_SMOKE:-0}" = "1" ]]; then
    SKIP_ENV=1; SKIP_DEPS=1
fi

# ---------------- 0. Sanity ------------------------------------------------
log "Project root:      $PROJECT_ROOT"
log "MIMIC-CXR-JPG:     $MIMIC_CXR_ROOT"
log "MIMIC-Ext-CXR-QBA: $MIMIC_QA_ROOT"
log "Chest ImaGenome:   $SG_ROOT"
log "Conda env:         $ENV_NAME (Python $PY_VERSION)"
log "Smoke-test model:  $SMOKE_MODEL"

if [[ ! -d "$PROJECT_ROOT" ]]; then
    err "Project root not found: $PROJECT_ROOT"
    exit 1
fi
cd "$PROJECT_ROOT"

if [[ ! -f models/ssg_vqa_net_v2.py ]]; then
    err "models/ssg_vqa_net_v2.py is missing. Sync the repo first."
    exit 1
fi

# ---------------- 1. Conda env --------------------------------------------
if [[ "${SKIP_ENV:-0}" != "1" ]]; then
    log "[1/5] Conda env"
    if ! ensure_conda; then
        err "conda setup failed; aborting"
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"

    # Accept Anaconda channel ToS (required since 2024 for repo.anaconda.com).
    # The `|| true` keeps this safe on older conda versions that lack `tos`.
    conda tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
    conda tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true

    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        ok "env '$ENV_NAME' already exists"
    else
        log "creating env '$ENV_NAME' (Python $PY_VERSION) from conda-forge"
        # conda-forge is community-maintained, has no ToS gate, and ships
        # newer Python builds. Override defaults so we don't hit Anaconda's
        # ToS even if the auto-accept above silently failed.
        conda create -n "$ENV_NAME" \
            -c conda-forge --override-channels \
            "python=${PY_VERSION}" -y
        ok "env created"
    fi
    conda activate "$ENV_NAME"
    ok "activated $ENV_NAME (python: $(python --version))"
else
    warn "[1/5] Skipping conda env creation (SKIP_ENV=1)"
    # Best-effort activate so downstream pip targets the right env
    if ensure_conda 2>/dev/null; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh" || true
        conda activate "$ENV_NAME" 2>/dev/null || warn "could not activate $ENV_NAME"
    fi
fi

# ---------------- 2. Dependencies -----------------------------------------
DRV_CUDA="$(detect_driver_cuda)"
TORCH_INDEX="$(pick_torch_index "$DRV_CUDA")"

if [[ "${SKIP_DEPS:-0}" != "1" ]]; then
    log "[2/5] Installing dependencies"
    log "driver-reported CUDA: ${DRV_CUDA:-unknown}"
    log "torch wheel index:    $TORCH_INDEX"

    log "  → upgrading pip / wheel / setuptools"
    python -m pip install --upgrade pip wheel setuptools

    # ---- 2a. Install everything EXCEPT torch first ----------------------
    # Order matters. If torch is installed first and then transformers /
    # peft / bitsandbytes / deepspeed are pip-installed without a wheel
    # index, those packages can pull in a CPU `torch` from PyPI as a
    # transitive dependency, silently overwriting the GPU build. So we
    # install torch LAST and force-pin it to the cu* index.
    log "  → installing transformers + peft + bitsandbytes + accelerate + deepspeed"
    python -m pip install \
        "transformers>=4.45,<5" \
        "peft>=0.11" \
        "accelerate>=0.30" \
        "bitsandbytes>=0.43" \
        "safetensors>=0.4" \
        "deepspeed>=0.14" \
        "sentencepiece>=0.1.99" \
        "tokenizers>=0.20"

    if [[ -f requirements.txt ]]; then
        log "  → installing requirements.txt (non-fatal if any package fails)"
        python -m pip install -r requirements.txt || warn "requirements.txt had failures (non-fatal)"
    else
        log "  → installing minimum project libs (pillow / numpy / pandas / nltk / ...)"
        python -m pip install \
            "pillow>=10" "numpy>=1.24,<2" "pandas>=2" \
            "scikit-learn>=1.3" "scipy>=1.10" \
            "nltk>=3.8" "tqdm>=4.65" "pyyaml>=6"
    fi

    # ---- 2b. Install PyTorch LAST, authoritative ------------------------
    # Wipe any CPU torch that the steps above may have dragged in, then
    # install fresh from the chosen wheel index. ensure_torch_cuda will
    # verify and re-attempt once if the first install still lands CPU.
    log "  → wiping any pre-existing torch and installing from $TORCH_INDEX"
    python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
    python -m pip install --no-cache-dir --index-url "$TORCH_INDEX" \
        "torch>=2.3,<2.6" "torchvision" "torchaudio"

    log "  → torch install summary"
    python -m pip show torch 2>/dev/null \
        | grep -E "^(Name|Version|Location):" || true

    log "  → verifying torch CUDA build"
    if ! ensure_torch_cuda "$TORCH_INDEX"; then
        exit 1
    fi

    log "  → downloading NLTK data"
    python - <<'PY' || warn "NLTK download had issues"
import nltk
for pkg in ("punkt", "punkt_tab", "wordnet"):
    try: nltk.download(pkg, quiet=True)
    except Exception as e: print(f"  nltk {pkg}: {e}")
PY
    ok "deps installed"
else
    warn "[2/5] Skipping pip install (SKIP_DEPS=1)"
fi

# ---------------- 3. Dataset path config ----------------------------------
log "[3/5] Writing configs/paths.yaml"
mkdir -p configs data
cat > configs/paths.yaml <<EOF
# Auto-generated by scripts/setup_marconi.sh — paths for marconi server.
# Override any of these via the matching env var before re-running this script.
data:
  mimic_cxr_jpg_path: "${MIMIC_CXR_ROOT}"
  mimic_ext_cxr_qba_path: "${MIMIC_QA_ROOT}"
  chest_imagenome_path: "${SG_ROOT}"
  metadata_csv: "${MIMIC_CXR_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz"
  chexpert_csv: "${MIMIC_CXR_ROOT}/mimic-cxr-2.0.0-chexpert.csv.gz"
  split_csv: "${MIMIC_CXR_ROOT}/mimic-cxr-2.0.0-split.csv.gz"
EOF
ok "paths.yaml written"

# Verify what's actually on disk and warn loudly about what's missing
[[ -d "${MIMIC_CXR_ROOT}/files" ]] && ok "MIMIC-CXR-JPG images present" \
    || err "MIMIC-CXR-JPG images NOT found at ${MIMIC_CXR_ROOT}/files"

[[ -f "${MIMIC_CXR_ROOT}/mimic-cxr-2.0.0-metadata.csv.gz" ]] && ok "metadata CSV present" \
    || warn "mimic-cxr-2.0.0-metadata.csv.gz missing"

if [[ ! -d "${MIMIC_QA_ROOT}" ]]; then
    warn "MIMIC-Ext-CXR-QBA NOT found at ${MIMIC_QA_ROOT}"
    warn "  → required for Stages 2-4. Download from the paper's release."
    if [[ -f "${PROJECT_ROOT}/scene_graph_ssgqa.zip" ]]; then
        warn "  → found scene_graph_ssgqa.zip in project root; unzip it to ${MIMIC_QA_ROOT}"
    fi
fi

if [[ ! -d "${SG_ROOT}" ]]; then
    warn "Chest ImaGenome NOT found at ${SG_ROOT}"
    warn "  → required for Stage 1 SG-generator pretraining (PhysioNet credentialed)."
fi

# ---------------- 4. Hardware report --------------------------------------
log "[4/5] Hardware report"
python - <<'PY'
import torch, os, sys
print(f"  python:      {sys.version.split()[0]}")
print(f"  torch:       {torch.__version__}  (CUDA build: {torch.version.cuda})")
print(f"  CUDA runtime available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("  ⚠ no CUDA — smoke test will run on CPU and be very slow")
else:
    n = torch.cuda.device_count()
    print(f"  GPUs: {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"    [{i}] {p.name}  cc={p.major}.{p.minor}  "
              f"vram={p.total_memory/1024**3:.1f} GiB  free={free/1024**3:.1f} GiB")
PY

# Run the project's own hardware util (richer summary + recommended config)
if [[ -f utils/hardware_utils.py ]]; then
    log "running utils/hardware_utils.py"
    PYTHONPATH="$PROJECT_ROOT" python utils/hardware_utils.py || warn "hardware_utils failed"
fi

# ---------------- 5. Smoke test on every GPU ------------------------------
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
    log "[5/5] Smoke test on every GPU (model: $SMOKE_MODEL)"
    if [[ ! -f scripts/smoke_test_v2.py ]]; then
        err "scripts/smoke_test_v2.py is missing — pull the latest project files."
        exit 1
    fi

    # Path A baked in: regardless of how this script was entered (SKIP_DEPS,
    # ONLY_SMOKE, second run, etc.), make sure torch in this env actually
    # sees CUDA before we try to load Qwen onto a GPU. If it doesn't, the
    # helper uninstalls and reinstalls from the right wheel index.
    log "  → pre-smoke torch CUDA sanity check"
    if ! ensure_torch_cuda "$TORCH_INDEX"; then
        err "Cannot start the smoke test — torch is not CUDA-enabled."
        exit 1
    fi

    PYTHONPATH="$PROJECT_ROOT" python scripts/smoke_test_v2.py \
        --model_id "$SMOKE_MODEL" \
        --batch_size 1 \
        --max_new_tokens 16
    ok "smoke test finished"
else
    warn "[5/5] Skipping smoke test (SKIP_SMOKE=1)"
fi

printf "\n${G}========================================${N}\n"
printf "${G} setup_marconi.sh complete.${N}\n"
printf "${G}========================================${N}\n"
echo
echo "Activate the env in future shells with:"
echo "  conda activate $ENV_NAME"
echo
echo "Next steps:"
echo "  1. python analyze_data.py                     # validate the dataset"
echo "  2. python scripts/verify_pretrain_pipeline.py # end-to-end pipeline check"
echo "  3. python train_mimic_cxr.py --config configs/default_config.yaml"
