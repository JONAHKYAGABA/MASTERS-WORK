"""
scripts/build_phase2_submissions.py

Builds the THREE candidate Phase 2 result.csv files from one execution run
(matches the "1 submission run × 3 leaderboard tries" reading of the rules).

The three variants apply different post-processing to the SAME completions:

  v1 — raw           : whatever the model emitted, normalized.
  v2 — multi-recall  : on multi-answer tasks, if the boxed answer has < 2
                       options, expand using the next plausible Cx mentioned
                       in the model's reasoning trace. Bets on IoU recall.
  v3 — insurance     : copy of v1 (Zindi rule: 3 submissions, best counted).

Usage:
    python scripts/build_phase2_submissions.py \\
        --completions eval/results/phase2_test/completions.jsonl \\
        --test_file data/Phase_1/test.json \\
        --out_dir eval/results/phase2_test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
sys.path.insert(0, str(PROJECT_DIR))

from main import normalize_multi_answer  # noqa: E402


_CX_RE = re.compile(r"\bC\d+\b")


def is_multi(scenario: Dict[str, Any]) -> bool:
    desc = ((scenario.get("task") or {}).get("description") or "").lower()
    return "two to four" in desc or "select two" in desc


def get_options(scenario: Dict[str, Any]) -> List[str]:
    return [o["id"] for o in (scenario.get("task") or {}).get("options", []) if "id" in o]


def expand_multi(answer: str, trace: str, valid_options: List[str], min_n: int = 2, max_n: int = 3) -> str:
    """For multi-answer tasks: ensure between min_n and max_n options."""
    answer = (answer or "").strip()
    parts = [p.strip() for p in answer.split("|") if p.strip()]
    if len(parts) >= min_n:
        return normalize_multi_answer("|".join(parts))
    valid = set(valid_options)
    # Pull additional Cx mentions from the trace, in order of first appearance.
    seen: "OrderedDict[str, None]" = OrderedDict()
    for m in _CX_RE.finditer(trace or ""):
        cx = m.group(0)
        if cx in valid and cx not in parts and cx not in seen:
            seen[cx] = None
    candidates = list(parts) + list(seen.keys())
    candidates = candidates[:max_n]
    if len(candidates) < min_n:
        return answer  # cannot expand confidently; leave as-is
    return normalize_multi_answer("|".join(candidates))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--completions", required=True, type=Path)
    ap.add_argument("--test_file", required=True, type=Path,
                    help="Path to the test scenarios JSON (for option lists & question types)")
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.completions.exists():
        print(f"FATAL: {args.completions} not found", file=sys.stderr)
        return 1

    with args.test_file.open("r", encoding="utf-8") as f:
        test = json.load(f)
    test_idx: Dict[str, Dict[str, Any]] = {s["scenario_id"]: s for s in test}

    rows_v1: List[Dict[str, str]] = []
    rows_v2: List[Dict[str, str]] = []

    n_total = 0
    n_multi = 0
    n_expanded = 0
    n_unchanged_multi_single = 0

    with args.completions.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("scenario_id")
            if not sid:
                continue
            n_total += 1

            raw_answer = (rec.get("final_answer") or "").strip()
            v1_ans = normalize_multi_answer(raw_answer)
            rows_v1.append({"scenario_id": sid, "answers": v1_ans})

            scen = test_idx.get(sid)
            if scen is None:
                rows_v2.append({"scenario_id": sid, "answers": v1_ans})
                continue

            if is_multi(scen):
                n_multi += 1
                trace = (rec.get("traces") or "") + "\n" + (rec.get("answer_raw") or "")
                expanded = expand_multi(v1_ans, trace, get_options(scen), min_n=2, max_n=3)
                if expanded != v1_ans:
                    n_expanded += 1
                elif "|" not in v1_ans:
                    n_unchanged_multi_single += 1
                rows_v2.append({"scenario_id": sid, "answers": expanded})
            else:
                rows_v2.append({"scenario_id": sid, "answers": v1_ans})

    v1_path = args.out_dir / "result_v1_raw.csv"
    v2_path = args.out_dir / "result_v2_multi_recall.csv"
    v3_path = args.out_dir / "result_v3_insurance.csv"
    pd.DataFrame(rows_v1).to_csv(v1_path, index=False)
    pd.DataFrame(rows_v2).to_csv(v2_path, index=False)
    pd.DataFrame(rows_v1).to_csv(v3_path, index=False)

    summary = {
        "n_total": n_total,
        "n_multi_answer_tasks": n_multi,
        "n_multi_expanded_by_v2": n_expanded,
        "n_multi_left_as_single": n_unchanged_multi_single,
        "files": {
            "v1_raw": str(v1_path),
            "v2_multi_recall": str(v2_path),
            "v3_insurance": str(v3_path),
        },
    }
    (args.out_dir / "phase2_submission_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print()
    print("Submit ALL THREE to Zindi as your 3 Phase 2 submissions.")
    print(f"  1) {v1_path}")
    print(f"  2) {v2_path}")
    print(f"  3) {v3_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
