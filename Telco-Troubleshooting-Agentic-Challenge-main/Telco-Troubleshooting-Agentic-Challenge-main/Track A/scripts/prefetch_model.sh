#!/usr/bin/env bash
# scripts/prefetch_model.sh
#
# Fully download the base model into the HF cache, with proper progress bar
# and resume on interruption. Runs once. After it succeeds, every subsequent
# llm_server / pipeline run is a true cache hit (no network).
#
#   bash scripts/prefetch_model.sh
#
# Override:
#   MODEL_NAME=Qwen/Qwen3-30B-A3B bash scripts/prefetch_model.sh
#   HF_HOME=/mnt/big_disk/.hf bash scripts/prefetch_model.sh
#   HF_HUB_ENABLE_HF_TRANSFER=1 bash scripts/prefetch_model.sh   # ~3x faster downloads

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HF_HOME/hub"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"

[ -d ".venv" ] || { echo "Run scripts/quickrun.sh first to create .venv"; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate

# Make sure huggingface_hub CLI is present.
pip install -q "huggingface_hub[cli]>=0.25" hf_transfer

echo "===== Prefetch ====="
echo "MODEL_NAME    = $MODEL_NAME"
echo "HF_HOME       = $HF_HOME"
echo "HF_TRANSFER   = ${HF_HUB_ENABLE_HF_TRANSFER:-0}"
echo

echo "Pre-flight: cache state for $MODEL_NAME"
SAFE="models--${MODEL_NAME//\//--}"
CACHE_DIR="$HF_HOME/hub/$SAFE"
if [ -d "$CACHE_DIR" ]; then
    echo "  cache dir exists: $CACHE_DIR"
    du -sh "$CACHE_DIR" || true
    echo "  shards / blobs in cache:"
    ls -lh "$CACHE_DIR/blobs" 2>/dev/null | tail -n +2 | wc -l || true
else
    echo "  cache dir missing: $CACHE_DIR (first download)"
fi
echo

# Use huggingface-cli with retry. It resumes any half-downloaded files.
echo "Downloading (will resume any partial files; press Ctrl-C to pause and resume later)..."
ATTEMPT=0
MAX_ATTEMPTS=5
until huggingface-cli download "$MODEL_NAME" \
        --cache-dir "$HF_HOME/hub" \
        --resume-download
do
    ATTEMPT=$((ATTEMPT + 1))
    if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
        echo "Download failed after $MAX_ATTEMPTS attempts." >&2
        exit 1
    fi
    echo "Download attempt $ATTEMPT/$MAX_ATTEMPTS failed; retrying in 10s..."
    sleep 10
done

echo
echo "===== Done ====="
du -sh "$CACHE_DIR"
echo "Total shards in blobs/:"
ls "$CACHE_DIR/blobs" | wc -l
echo
echo "If size is <50 GB the download is still incomplete; rerun this script."
echo "Otherwise you can now run:  HF_HUB_OFFLINE=1 bash scripts/full_pipeline.sh"
echo "(HF_HUB_OFFLINE=1 forbids any further network calls — guarantees no redownload.)"
