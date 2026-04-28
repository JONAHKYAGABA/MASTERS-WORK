#!/usr/bin/env bash
# scripts/quickrun.sh
#
# Self-healing fast runner: verifies project files, fetches missing ones,
# creates a venv, installs deps, runs the agent end-to-end against an
# OpenAI-compatible LLM endpoint, and scores against ground truth.
#
# Single command from the project root (the dir that should contain server.py):
#     bash scripts/quickrun.sh
#
# Common overrides:
#     N_SCENARIOS=50  bash scripts/quickrun.sh
#     MODEL_URL=https://my-endpoint/v1   MODEL_NAME=Qwen/...   bash ...
#     PYTHON_BIN=python3.10              bash scripts/quickrun.sh
#     REPO_URL=https://raw.githubusercontent.com/<user>/<repo>/main/Track%20A
#         bash scripts/quickrun.sh        # auto-curl missing project files
#     TRAIN_URL=https://example.com/train.json   bash scripts/quickrun.sh
#     SKIP_INSTALL=1   SKIP_SERVER=1     bash scripts/quickrun.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

step()  { printf "\n=== %s ===\n" "$*"; }
fail()  { printf "ERROR: %s\n" "$*" >&2; exit 1; }

step "Project: $PROJECT_DIR"

# ---------- 0. verify / fetch project files ----------
step "[0/7] Verify project files"

REQUIRED_FILES=(server.py _types.py utils.py logger.py main.py requirements.txt)
REQUIRED_DATA=(data/Phase_1/train.json data/Phase_1/test.json)

REQUIREMENTS_FALLBACK="fastapi
openai
pydantic
pandas
datasets"

# Helper: fetch <relpath> from $REPO_URL via curl. Returns 0 on success.
fetch_one() {
    local rel="$1"
    [ -n "${REPO_URL:-}" ] || return 1
    local url="${REPO_URL%/}/$rel"
    mkdir -p "$(dirname "$rel")"
    if curl -fL --retry 2 -sS -o "$rel" "$url"; then
        echo "  fetched $rel from $url"
        return 0
    else
        echo "  failed to fetch $rel from $url" >&2
        return 1
    fi
}

missing_files=()
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        missing_files+=("$f")
    fi
done

if [ "${#missing_files[@]}" -gt 0 ]; then
    echo "Missing project files: ${missing_files[*]}"
    if [ -n "${REPO_URL:-}" ]; then
        echo "Trying to fetch from REPO_URL=$REPO_URL"
        for f in "${missing_files[@]}"; do
            fetch_one "$f" || true
        done
    fi
fi

# Special-case: if requirements.txt is STILL missing, write the known content.
if [ ! -f requirements.txt ]; then
    echo "Writing requirements.txt fallback (5 packages)"
    printf '%s\n' "$REQUIREMENTS_FALLBACK" > requirements.txt
fi

# Re-check; non-data files are non-negotiable.
still_missing=()
for f in "${REQUIRED_FILES[@]}"; do
    [ -f "$f" ] || still_missing+=("$f")
done

if [ "${#still_missing[@]}" -gt 0 ]; then
    cat >&2 <<EOF

The following required files are still missing:
EOF
    for f in "${still_missing[@]}"; do echo "  - $f" >&2; done
    cat >&2 <<EOF

Pick ONE of these and re-run:

  1. Set REPO_URL to a base URL that contains these files. For a public
     GitHub repo it looks like:
        REPO_URL=https://raw.githubusercontent.com/<user>/<repo>/main/Track%20A \\
            bash scripts/quickrun.sh

  2. Re-clone the repo into this directory:
        git clone <repo-url> /tmp/repo
        cp -r /tmp/repo/Track\\ A/* $PROJECT_DIR/

  3. scp the missing files from the machine where you have them.

EOF
    exit 1
fi
echo "All required code files present."

# ---------- 1. venv ----------
step "[1/7] Python venv"
# Prefer 3.10 or 3.11 if available; some wheels (e.g. datasets, pandas) may
# lag behind on Python 3.13.
if [ -z "${PYTHON_BIN:-}" ]; then
    for cand in python3.11 python3.10 python3.12 python3; do
        if command -v "$cand" >/dev/null 2>&1; then
            PYTHON_BIN="$cand"
            break
        fi
    done
fi
[ -n "${PYTHON_BIN:-}" ] || fail "no python3 on PATH"
echo "PYTHON_BIN=$PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"

# If an existing .venv was made with a python that no longer matches, scrap it.
if [ -d .venv ]; then
    if [ -x .venv/bin/python ]; then
        venv_ver=$(.venv/bin/python --version 2>&1 || true)
        wanted_ver=$($PYTHON_BIN --version 2>&1 || true)
        if [ "$venv_ver" != "$wanted_ver" ] && [ "${REUSE_VENV:-0}" != "1" ]; then
            echo "Existing .venv ($venv_ver) != desired ($wanted_ver). Recreating."
            rm -rf .venv
        fi
    fi
fi

if [ ! -d .venv ]; then
    "$PYTHON_BIN" -m venv .venv
    echo "Created .venv with $($PYTHON_BIN --version 2>&1)"
else
    echo "Reusing existing .venv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "active: $(python --version 2>&1) at $(command -v python)"

# ---------- 2. deps ----------
step "[2/7] Install dependencies"
if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    echo "SKIP_INSTALL=1 → not touching pip"
else
    python -m pip install -q --upgrade pip wheel setuptools
    if ! pip install -q -r requirements.txt; then
        echo "Locked requirements.txt failed to install on $(python --version)."
        echo "Falling back to a minimal subset (skipping 'datasets')."
        pip install -q fastapi openai "pydantic>=2" pandas
    fi
    pip install -q "openai>=1.50.0" httpx requests python-dateutil tqdm \
                  "uvicorn[standard]" python-multipart
    echo "deps installed"
fi

# ---------- 3. data ----------
step "[3/7] Verify training data"
TRAIN_JSON="data/Phase_1/train.json"
TEST_JSON="data/Phase_1/test.json"

ok_size() {
    local f="$1" min="${2:-1000000}"
    [ -f "$f" ] || return 1
    local s
    s=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
    [ "$s" -ge "$min" ]
}

if ! ok_size "$TRAIN_JSON"; then
    echo "$TRAIN_JSON missing or too small. Trying recovery..."
    # 3a. git-lfs
    if command -v git-lfs >/dev/null 2>&1 \
       && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git lfs install --skip-repo 2>/dev/null || true
        git lfs pull 2>/dev/null || true
    fi
    # 3b. TRAIN_URL fallback
    if ! ok_size "$TRAIN_JSON" && [ -n "${TRAIN_URL:-}" ]; then
        mkdir -p "$(dirname "$TRAIN_JSON")"
        echo "Downloading from TRAIN_URL..."
        curl -fL --retry 3 -o "$TRAIN_JSON" "$TRAIN_URL"
    fi
    # 3c. REPO_URL fallback
    if ! ok_size "$TRAIN_JSON" && [ -n "${REPO_URL:-}" ]; then
        echo "Trying REPO_URL for train.json..."
        fetch_one "$TRAIN_JSON" || true
    fi
fi

if ! ok_size "$TRAIN_JSON"; then
    cat <<EOF >&2

train.json missing or too small (likely a 133-byte LFS pointer).

Recovery options:

  1. Set TRAIN_URL to a direct download (raw GitHub or signed S3):
        TRAIN_URL=https://... bash scripts/quickrun.sh

  2. Re-clone with git-lfs:
        git lfs install
        git clone <repo-url>

  3. Download manually from the Zindi data tab to:
        $PROJECT_DIR/$TRAIN_JSON
EOF
    exit 1
fi
echo "$TRAIN_JSON OK ($(wc -c < "$TRAIN_JSON" | tr -d ' ') bytes)"
ok_size "$TEST_JSON" \
    && echo "$TEST_JSON OK" \
    || echo "$TEST_JSON missing (only needed for Phase 1 submission)"

# ---------- 4. LLM endpoint ----------
step "[4/7] LLM endpoint check"
MODEL_URL="${MODEL_URL:-http://localhost:8001/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
AGENT_API_KEY_VAL="${AGENT_API_KEY:-dummy}"
echo "MODEL_URL=$MODEL_URL"
echo "MODEL_NAME=$MODEL_NAME"

# Real validation: GET /v1/models must return JSON with a "data" field listing
# the served model. Anything else (HTML, 4xx/5xx, JupyterHub redirects) is
# rejected — those will silently 403 every chat.completions call.
endpoint_check() {
    python - <<PY
import json, sys, urllib.request
url = "$MODEL_URL/models"
key = "$AGENT_API_KEY_VAL"
req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
try:
    with urllib.request.urlopen(req, timeout=5) as r:
        body = r.read(4096)
        ctype = r.headers.get("Content-Type", "")
except Exception as e:
    print(f"FAIL: cannot reach {url}: {e}", file=sys.stderr); sys.exit(2)
if "json" not in ctype.lower():
    print(f"FAIL: {url} returned Content-Type={ctype!r}, not JSON.", file=sys.stderr)
    print("       This is likely NOT an OpenAI-compatible LLM endpoint.", file=sys.stderr)
    print(f"       First 200 bytes: {body[:200]!r}", file=sys.stderr)
    sys.exit(3)
try:
    obj = json.loads(body)
except Exception as e:
    print(f"FAIL: response not JSON: {e}", file=sys.stderr); sys.exit(4)
data = obj.get("data") if isinstance(obj, dict) else None
if not isinstance(data, list) or not data:
    print(f"FAIL: response missing 'data' list: {obj!r}", file=sys.stderr); sys.exit(5)
ids = [m.get("id") for m in data if isinstance(m, dict)]
print(f"OK: served models = {ids}")
PY
}

if ! endpoint_check; then
    cat <<EOF >&2

$MODEL_URL is not an OpenAI-compatible LLM endpoint that serves the
expected model. Common causes:

  - Port already used by another service (JupyterHub, Jupyter, etc.).
    JupyterHub returns 200/HTML/403 to /v1/models which fools naive checks.
    Pick a different port:
        MODEL_URL=http://localhost:8001/v1 bash scripts/quickrun.sh

  - vLLM is not running. Start it in another terminal:
        pip install vllm                              # one-time, ~10 GB
        python -m vllm.entrypoints.openai.api_server \\
          --model $MODEL_NAME \\
          --port 8001 --dtype float16 \\
          --tensor-parallel-size 2 --enforce-eager \\
          --quantization bitsandbytes --load-format bitsandbytes \\
          --max-model-len 16384 \\
          --enable-auto-tool-choice --tool-call-parser hermes \\
          --trust-remote-code

  - Endpoint requires a different API key. Set:
        export AGENT_API_KEY="<your-real-key>"
EOF
    exit 1
fi

# ---------- 5. server.py ----------
step "[5/7] Start locked server.py"
DATA_SPLIT="${DATA_SPLIT:-train}"
SERVER_LOG="$PROJECT_DIR/server.log"
SERVER_PID=""

cleanup_server() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup_server EXIT

if [ "${SKIP_SERVER:-0}" = "1" ] || curl -sf http://localhost:7860/health >/dev/null 2>&1; then
    echo "server.py already running (or SKIP_SERVER=1)"
else
    DATA_SPLIT="$DATA_SPLIT" nohup python server.py > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    echo "server.py pid=$SERVER_PID  log=$SERVER_LOG"
    for i in $(seq 1 30); do
        sleep 2
        if curl -sf http://localhost:7860/health >/dev/null 2>&1; then
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "server.py exited. Last 60 lines of log:" >&2
            tail -60 "$SERVER_LOG" >&2 || true
            exit 1
        fi
    done
    if ! curl -sf http://localhost:7860/health >/dev/null 2>&1; then
        echo "server.py never reached /health. Last 60 lines:" >&2
        tail -60 "$SERVER_LOG" >&2 || true
        exit 1
    fi
fi

# ---------- 6. agent ----------
step "[6/7] Run agent"
N_SCENARIOS="${N_SCENARIOS:-5}"
EXP_NAME="${EXP_NAME:-quickrun_${N_SCENARIOS}}"
SAVE_DIR="eval/results/$EXP_NAME"
mkdir -p "$SAVE_DIR"

START_T=$(date +%s)
python main.py \
    --server_url http://localhost:7860 \
    --model_url "$MODEL_URL" \
    --model_name "$MODEL_NAME" \
    --max_samples "$N_SCENARIOS" \
    --num_attempts "${NUM_ATTEMPTS:-1}" \
    --max_iterations "${MAX_ITER:-2}" \
    --save_dir "$SAVE_DIR" \
    --log_file "$SAVE_DIR/agent.log" \
    --quiet
END_T=$(date +%s)
echo "agent wall-clock: $((END_T - START_T))s for $N_SCENARIOS scenarios"

# ---------- 7. score ----------
step "[7/7] Score"
python scripts/score_results.py \
    --results_dir "$SAVE_DIR" \
    --train_file "$TRAIN_JSON" \
    --target_acc "${TARGET_ACC:-0.35}" \
    --target_latency_s "${TARGET_LATENCY_S:-90}"

echo
echo "========================================"
echo "DONE."
echo "  result.csv         : $SAVE_DIR/result.csv"
echo "  completions.jsonl  : $SAVE_DIR/completions.jsonl"
echo "  metrics.json       : $SAVE_DIR/metrics.json"
echo "  SUMMARY.md         : $SAVE_DIR/SUMMARY.md"
echo "========================================"
