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
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        ok "env '$ENV_NAME' already exists"
    else
        log "creating env '$ENV_NAME' (Python $PY_VERSION)"
        conda create -n "$ENV_NAME" "python=${PY_VERSION}" -y
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
if [[ "${SKIP_DEPS:-0}" != "1" ]]; then
    log "[2/5] Installing dependencies"

    # Detect CUDA version reported by the driver (not the toolkit). This is
    # what bitsandbytes and torch wheels need to match.
    if command -v nvidia-smi >/dev/null 2>&1; then
        DRV_CUDA="$(nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader 2>/dev/null \
                    | head -n1 | awk -F',' '{gsub(/ /,"",$2); print $2}')"
        log "driver-reported CUDA: ${DRV_CUDA:-unknown}"
    else
        warn "nvidia-smi not found — installing CPU PyTorch (you can override later)"
        DRV_CUDA=""
    fi

    # Pick a torch wheel index. bnb-0.43+ supports cu118 and cu121 wheels.
    case "${DRV_CUDA}" in
        12.[1-9]*|12.[1-9]) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
        12.0*|11.8*|11.[8-9])     TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
        "")                       TORCH_INDEX="https://download.pytorch.org/whl/cpu"   ;;
        *)                        TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    esac
    log "torch wheel index: $TORCH_INDEX"

    python -m pip install --upgrade pip wheel setuptools

    # PyTorch first (so transformers picks the right ABI)
    python -m pip install --index-url "$TORCH_INDEX" \
        "torch>=2.3,<2.6" "torchvision" "torchaudio"

    # v2 stack
    python -m pip install \
        "transformers>=4.45,<5" \
        "peft>=0.11" \
        "accelerate>=0.30" \
        "bitsandbytes>=0.43" \
        "safetensors>=0.4" \
        "deepspeed>=0.14" \
        "sentencepiece>=0.1.99" \
        "tokenizers>=0.20"

    # Project-side libs (whatever requirements.txt has). Prefer it if present.
    if [[ -f requirements.txt ]]; then
        log "installing requirements.txt"
        python -m pip install -r requirements.txt || warn "requirements.txt had failures (non-fatal)"
    else
        # Minimum set needed by the dataset / metrics modules
        python -m pip install \
            "pillow>=10" "numpy>=1.24,<2" "pandas>=2" \
            "scikit-learn>=1.3" "scipy>=1.10" \
            "nltk>=3.8" "tqdm>=4.65" "pyyaml>=6"
    fi

    # NLTK data
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
