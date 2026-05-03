"""
scripts/submit_now.py

ONE script. Reads data/Phase_1/test.json, calls the local llm_server for each
scenario with a TERSE prompt (no tools, no multi-turn, max_tokens=256,
temperature=0), extracts \\boxed{...}. If the LLM emits nothing parseable
(empty, timeout, parse error), falls back to the validated heuristic from
build_baseline_submission.py (26.8% on train).

Outputs: eval/results/submit_now/result.csv (Zindi format).

Resumable: if interrupted, restart skips scenarios already in completions.jsonl.

Assumes llm_server is already running at http://localhost:8001 (do not start
it from here — the user already has it loaded). server.py is NOT required;
all scenario data is inlined in test.json.

Usage:
    python scripts/submit_now.py
    python scripts/submit_now.py --max_samples 5     # smoke test
    python scripts/submit_now.py --no_llm            # heuristic-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
sys.path.insert(0, str(PROJECT_DIR))

# Heuristic fallback (validated 26.8% on train).
from scripts.build_baseline_submission import pick_answer as heuristic_pick  # noqa: E402
from scripts.build_baseline_submission import is_multi as task_is_multi  # noqa: E402


# ---------------------------------------------------------------- LLM call

def _scenario_block(scenario: Dict[str, Any]) -> str:
    d = scenario.get("data") or {}
    parts: List[str] = []
    cfg = d.get("network_configuration_data")
    if cfg:
        parts.append("## Network Configuration\n```\n" + cfg.strip() + "\n```")
    up = d.get("user_plane_data")
    if up:
        parts.append("## User-Plane Time Series\n```\n" + up.strip() + "\n```")
    sig = d.get("signaling_plane_data")
    if sig:
        parts.append("## Signaling Plane Events\n```\n" + sig.strip() + "\n```")
    traffic = d.get("traffic_data")
    if traffic:
        parts.append("## Cell-Level Traffic KPIs\n```\n" + traffic.strip() + "\n```")
    mr = d.get("mr_data")
    if mr:
        parts.append("## Measurement Reports Sample\n```\n" + mr.strip() + "\n```")
    return "\n\n".join(parts)


SYSTEM_PROMPT_SINGLE = (
    "You are a 5G RAN troubleshooting expert. Read the scenario and pick the "
    "single best optimization action from the option list. "
    "Reply with ONLY one line in this exact format: \\boxed{Cx} "
    "where x is the option number. No explanation, no preamble, no thinking. "
    "Just the boxed answer."
)

SYSTEM_PROMPT_MULTI = (
    "You are a 5G RAN troubleshooting expert. Read the scenario and pick 2 to "
    "4 optimization actions from the option list. "
    "Reply with ONLY one line in this exact format: \\boxed{Cx|Cy|Cz} "
    "with the option numbers in ascending order, separated by | with no spaces. "
    "No explanation, no preamble, no thinking. Just the boxed answer."
)


_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_CX_RE = re.compile(r"\bC\d+\b")


def _extract_boxed(text: str, valid_options: List[str]) -> str:
    """Pull a \\boxed{...} answer out, validate option ids, normalize."""
    if not text:
        return ""
    matches = _BOXED_RE.findall(text)
    if not matches:
        # Last-ditch: look for any Cx mention.
        cx = _CX_RE.findall(text)
        if not cx:
            return ""
        valid = set(valid_options)
        cx = [c for c in cx if c in valid]
        if not cx:
            return ""
        # Treat as single answer.
        return cx[0]
    inner = matches[-1]
    # Strip braces, whitespace, leading colons.
    inner = re.sub(r"[{}\s]", "", inner).lstrip(":").rstrip("./")
    if not inner:
        return ""
    parts = [p.strip() for p in inner.split("|") if p.strip()]
    valid = set(valid_options)
    parts = [p for p in parts if p in valid]
    if not parts:
        return ""

    def _key(s: str) -> int:
        m = re.search(r"\d+", s)
        return int(m.group()) if m else 0

    parts = sorted(set(parts), key=_key)
    return "|".join(parts)


def _ask_llm(
    scenario: Dict[str, Any],
    base_url: str,
    model_name: str,
    timeout_s: float,
    max_tokens: int,
) -> str:
    options = (scenario.get("task") or {}).get("options", []) or []
    options_block = "\n".join(f"  {o['id']}: {o['label']}" for o in options if "id" in o)
    is_multi = task_is_multi(scenario)
    system = SYSTEM_PROMPT_MULTI if is_multi else SYSTEM_PROMPT_SINGLE

    user_prompt = (
        _scenario_block(scenario)
        + "\n\n## Task\n"
        + ((scenario.get("task") or {}).get("description") or "")
        + "\n\n## Options\n"
        + options_block
        + "\n\nAnswer:"
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {os.environ.get('AGENT_API_KEY', 'sk-dummy')}"},
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content") or "")
    except Exception as exc:
        print(f"  [llm-error] {exc}", file=sys.stderr)
        return ""


# ---------------------------------------------------------------- main loop

def _load_completions(path: Path) -> Dict[str, str]:
    done: Dict[str, str] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sid = rec.get("scenario_id")
                ans = rec.get("answer", "")
                if sid:
                    done[sid] = ans
            except json.JSONDecodeError:
                continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", default="data/Phase_1/test.json")
    ap.add_argument("--out_dir", default="eval/results/submit_now")
    ap.add_argument("--llm_url", default=os.environ.get("LLM_URL", "http://localhost:8001"))
    ap.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B"))
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--llm_timeout_s", type=float, default=60.0)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--no_llm", action="store_true",
                    help="Skip LLM entirely; use heuristic for all scenarios.")
    args = ap.parse_args()

    test_path = (PROJECT_DIR / args.test_file).resolve()
    out_dir = (PROJECT_DIR / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    completions_path = out_dir / "completions.jsonl"
    csv_path = out_dir / "result.csv"

    if not test_path.exists():
        print(f"FATAL: {test_path} not found", file=sys.stderr)
        return 1

    # Health check (skip if --no_llm).
    if not args.no_llm:
        try:
            h = requests.get(f"{args.llm_url}/health", timeout=5).json()
            print(f"[llm] {args.llm_url} -> {h}")
            if h.get("status") != "ok":
                print(f"[llm] not ready, falling back to --no_llm mode", file=sys.stderr)
                args.no_llm = True
        except Exception as exc:
            print(f"[llm] not reachable at {args.llm_url}: {exc}", file=sys.stderr)
            print(f"[llm] falling back to --no_llm mode", file=sys.stderr)
            args.no_llm = True

    with test_path.open("r", encoding="utf-8") as f:
        scenarios = json.load(f)
    if args.max_samples is not None:
        scenarios = scenarios[: args.max_samples]

    done = _load_completions(completions_path)
    print(f"[run] {len(scenarios)} total, {len(done)} already done, "
          f"{len(scenarios) - len(done)} remaining")

    n_llm = 0
    n_fallback = 0
    n_done = 0
    rows: List[Dict[str, str]] = []
    for sid, ans in done.items():
        rows.append({"scenario_id": sid, "answers": ans})

    f_jsonl = completions_path.open("a", encoding="utf-8")
    t_start = time.time()
    try:
        for i, scenario in enumerate(scenarios):
            sid = scenario.get("scenario_id", "")
            if sid in done:
                continue

            options = (scenario.get("task") or {}).get("options", []) or []
            valid_ids = [o["id"] for o in options if "id" in o]

            llm_text = ""
            answer = ""
            t0 = time.time()
            if not args.no_llm:
                llm_text = _ask_llm(
                    scenario,
                    args.llm_url,
                    args.model_name,
                    args.llm_timeout_s,
                    args.max_tokens,
                )
                answer = _extract_boxed(llm_text, valid_ids)

            source = "llm"
            if not answer:
                answer = heuristic_pick(scenario)
                source = "heuristic"
                n_fallback += 1
            else:
                n_llm += 1

            elapsed = time.time() - t0
            rec = {
                "scenario_id": sid,
                "answer": answer,
                "source": source,
                "elapsed_s": round(elapsed, 2),
                "llm_text_head": (llm_text or "")[:200],
            }
            f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_jsonl.flush()
            rows.append({"scenario_id": sid, "answers": answer})
            n_done += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(scenarios):
                _write_csv(rows, csv_path)
            running_s = time.time() - t_start
            avg = running_s / max(n_done, 1)
            eta = avg * (len(scenarios) - len(done) - n_done)
            print(
                f"[{i+1:4d}/{len(scenarios)}] {sid[:8]} ans={answer:<18s} "
                f"src={source:9s} {elapsed:5.1f}s  "
                f"(llm={n_llm} fb={n_fallback}, eta={eta/60:.1f}min)"
            )
    finally:
        f_jsonl.close()
        _write_csv(rows, csv_path)

    # Build the 3 Zindi submission copies.
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "result_v1_raw.csv", index=False)
    df.to_csv(out_dir / "result_v2_multi_recall.csv", index=False)
    df.to_csv(out_dir / "result_v3_insurance.csv", index=False)

    print()
    print(f"=== DONE ===")
    print(f"  total: {len(rows)}  (llm={n_llm}  heuristic_fallback={n_fallback})")
    print(f"  csv  : {csv_path}")
    print(f"  zindi submissions:")
    print(f"    {out_dir / 'result_v1_raw.csv'}")
    print(f"    {out_dir / 'result_v2_multi_recall.csv'}")
    print(f"    {out_dir / 'result_v3_insurance.csv'}")
    return 0


def _write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    sys.exit(main())
