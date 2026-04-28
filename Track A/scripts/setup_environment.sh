#!/usr/bin/env bash
# scripts/setup_environment.sh
# Idempotent setup for the Telco Troubleshooting Agentic Challenge (Track A).
# Run from anywhere: REPO_ROOT defaults to ~/JONAHMASTERS/MASTERS-WORK/Telco-Troubleshooting-Agentic-Challenge
#
# Usage:
#   bash scripts/setup_environment.sh
#   REPO_ROOT=/custom/path bash scripts/setup_environment.sh

set -euo pipefail

CURRENT_STEP="0 (init)"
trap 'echo "SETUP FAILED at step ${CURRENT_STEP}" >&2' ERR

step() {
    CURRENT_STEP="$1"
    echo ""
    echo "================================================================"
    echo "STEP $1: $2"
    echo "================================================================"
}

# -------- Locate the project --------
REPO_ROOT="${REPO_ROOT:-$HOME/JONAHMASTERS/MASTERS-WORK/Telco-Troubleshooting-Agentic-Challenge}"
if [ -f "$REPO_ROOT/Track A/server.py" ]; then
    PROJECT_DIR="$REPO_ROOT/Track A"
elif [ -f "$REPO_ROOT/server.py" ]; then
    PROJECT_DIR="$REPO_ROOT"
else
    CURRENT_STEP="0 (locate project)"
    echo "Cannot find server.py at '$REPO_ROOT/Track A' or '$REPO_ROOT'." >&2
    echo "Set REPO_ROOT to the directory that contains server.py (or its 'Track A' parent)." >&2
    exit 1
fi
echo "REPO_ROOT   = $REPO_ROOT"
echo "PROJECT_DIR = $PROJECT_DIR"

cd "$PROJECT_DIR"

# -------- STEP 1: conda env --------
step 1 "Create/activate conda env 'telco' (Python 3.10)"
if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH" >&2
    exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "telco"; then
    echo "Env 'telco' already exists; reusing."
else
    conda create -y -n telco python=3.10
fi
conda activate telco
echo "Active env: ${CONDA_DEFAULT_ENV} | python = $(python --version 2>&1) | which = $(command -v python)"

# -------- STEP 2: git-lfs + pull train.json --------
step 2 "Install git-lfs and pull train.json (with manual-fallback message)"
if ! command -v git-lfs >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -y
        sudo apt-get install -y git-lfs
    else
        conda install -y -c conda-forge git-lfs
    fi
fi
git lfs install --skip-repo

LFS_OK=0
if ( cd "$REPO_ROOT" && git rev-parse --is-inside-work-tree >/dev/null 2>&1 ); then
    if ( cd "$REPO_ROOT" && git lfs pull ); then
        LFS_OK=1
    fi
fi
if [ "$LFS_OK" -ne 1 ]; then
    echo ""
    echo "WARNING: 'git lfs pull' did not run cleanly."
    echo "If train.json is still a 133-byte LFS pointer, download it MANUALLY:"
    echo "  1. Open the Zindi challenge data tab (Track A)."
    echo "  2. Download train.json directly from Zindi."
    echo "  3. Place it at: $PROJECT_DIR/data/Phase_1/train.json"
    echo "Then re-run this script. Continuing so step 3 can verify."
fi

# -------- STEP 3: verify train.json --------
step 3 "Verify train.json: ~24 MB, 2000 scenarios, all with ground-truth answers"
TRAIN_JSON="$PROJECT_DIR/data/Phase_1/train.json"
if [ ! -f "$TRAIN_JSON" ]; then
    echo "train.json not found at $TRAIN_JSON" >&2
    exit 1
fi
SIZE_BYTES=$(stat -c%s "$TRAIN_JSON" 2>/dev/null || stat -f%z "$TRAIN_JSON")
echo "train.json size: $SIZE_BYTES bytes"
if [ "$SIZE_BYTES" -lt 1000000 ]; then
    echo "train.json is only $SIZE_BYTES bytes (likely an LFS pointer). Aborting." >&2
    exit 1
fi
TRAIN_PATH="$TRAIN_JSON" python - <<'PYEOF'
import json, os, sys
path = os.environ["TRAIN_PATH"]
with open(path) as f:
    data = json.load(f)
if len(data) != 2000:
    print(f"FAIL: expected 2000 scenarios, got {len(data)}", file=sys.stderr)
    sys.exit(1)
bad = [s for s in data if not s.get("answer") or s["answer"] == "To be determined"]
if bad:
    print(f"FAIL: {len(bad)} scenarios missing or placeholder answers", file=sys.stderr)
    sys.exit(1)
print(f"OK: {len(data)} scenarios, all with ground-truth answers")
PYEOF

# -------- STEP 4: install requirements --------
step 4 "Install pip requirements (locked + agent; unsloth last)"
python -m pip install --upgrade pip setuptools wheel
pip install -r "$PROJECT_DIR/requirements.txt"

AGENT_REQ="$PROJECT_DIR/requirements-agent.txt"
if [ ! -f "$AGENT_REQ" ]; then
    echo "requirements-agent.txt missing at $AGENT_REQ" >&2
    exit 1
fi

TMP_NO_UNSLOTH="$(mktemp)"
TMP_UNSLOTH="$(mktemp)"
trap 'rm -f "$TMP_NO_UNSLOTH" "$TMP_UNSLOTH"' EXIT

# Strip blank lines and comments; split into "everything else" and "unsloth lines"
grep -vE '^\s*(#|$)' "$AGENT_REQ" | grep -viE '^\s*unsloth' > "$TMP_NO_UNSLOTH" || true
grep -vE '^\s*(#|$)' "$AGENT_REQ" | grep  -iE '^\s*unsloth' > "$TMP_UNSLOTH"     || true

if [ -s "$TMP_NO_UNSLOTH" ]; then
    pip install -r "$TMP_NO_UNSLOTH"
fi
if [ -s "$TMP_UNSLOTH" ]; then
    pip install -r "$TMP_UNSLOTH"
fi

# -------- STEP 5: chmod 444 on locked files --------
step 5 "Apply chmod 444 to locked files"
LOCKED_FILES=(server.py _types.py utils.py requirements.txt)
for f in "${LOCKED_FILES[@]}"; do
    if [ ! -f "$PROJECT_DIR/$f" ]; then
        echo "Missing locked file: $f" >&2
        exit 1
    fi
    chmod 444 "$PROJECT_DIR/$f"
    echo "  locked: $f"
done
shopt -s nullglob
for f in "$PROJECT_DIR"/data/Phase_1/*.json; do
    chmod 444 "$f"
    echo "  locked: ${f#$PROJECT_DIR/}"
done
shopt -u nullglob

# -------- STEP 6: hashes + verify_locks.sh --------
step 6 "Generate .locked_hashes and scripts/verify_locks.sh"
mkdir -p "$PROJECT_DIR/scripts"
(
    cd "$PROJECT_DIR"
    sha256sum server.py _types.py utils.py requirements.txt data/Phase_1/*.json > .locked_hashes
)
echo "Wrote $PROJECT_DIR/.locked_hashes"

cat > "$PROJECT_DIR/scripts/verify_locks.sh" <<'VERIFY_EOF'
#!/usr/bin/env bash
# Verifies that locked files have not been modified since setup.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -f .locked_hashes ]; then
    echo "ERROR: .locked_hashes not found. Run scripts/setup_environment.sh first." >&2
    exit 1
fi
fail=0
while read -r expected_hash filename; do
    if [ ! -f "$filename" ]; then
        echo "ERROR: missing locked file: $filename" >&2
        fail=1
        continue
    fi
    actual=$(sha256sum "$filename" | awk '{print $1}')
    if [ "$actual" != "$expected_hash" ]; then
        echo "ERROR: $filename has been modified" >&2
        echo "  expected $expected_hash" >&2
        echo "  actual   $actual" >&2
        fail=1
    fi
done < .locked_hashes
if [ "$fail" -eq 0 ]; then
    echo "All locked files intact."
else
    exit 1
fi
VERIFY_EOF
chmod +x "$PROJECT_DIR/scripts/verify_locks.sh"
"$PROJECT_DIR/scripts/verify_locks.sh"

# -------- STEP 7: env vars in ~/.bashrc + current session --------
step 7 "Persist env vars to ~/.bashrc and current session"
BASHRC="$HOME/.bashrc"
touch "$BASHRC"
add_to_bashrc() {
    local line="$1"
    if ! grep -qxF "$line" "$BASHRC"; then
        echo "$line" >> "$BASHRC"
        echo "  added to ~/.bashrc: $line"
    else
        echo "  already in ~/.bashrc: $line"
    fi
}
add_to_bashrc 'export VLLM_ATTENTION_BACKEND=XFORMERS'
add_to_bashrc 'export TOKENIZERS_PARALLELISM=false'
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
echo "Current session: VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND, TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"

# -------- DONE --------
trap - ERR
echo ""
echo "================================================================"
echo "SETUP COMPLETE"
echo "================================================================"
echo "Project dir : $PROJECT_DIR"
echo "Conda env   : telco (python $(python --version 2>&1 | awk '{print $2}'))"
echo "Locked files verified via scripts/verify_locks.sh"
echo ""
echo "NOTE: env-var exports apply to this shell only. New shells will pick"
echo "      them up via ~/.bashrc. To apply to your CURRENT shell, run:"
echo "         source ~/.bashrc"
echo ""
echo "NEXT: python scripts/quant_smoke_test.py"
