"""
scripts/rescue_answers.py

Rebuild result.csv from completions.jsonl when the agent finished but failed
to extract \\boxed{...} answers. Tries a cascade of increasingly lenient
extractors and falls back to "Insufficient data" only as a last resort.

Usage:
    python scripts/rescue_answers.py \\
        --results_dir eval/results/phase1_test \\
        --test_file data/Phase_1/test.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
sys.path.insert(0, str(PROJECT_DIR))

from main import normalize_multi_answer  # noqa: E402


# Cascade of patterns, most-specific first.
_PATTERNS = [
    re.compile(r"\\boxed\{\s*([Cc]\d+(?:\s*\|\s*[Cc]\d+)*)\s*\}"),
    re.compile(r"\\boxed\{\s*([^}]+?)\s*\}"),
    re.compile(r"\bAnswer\s*[:=]\s*([Cc]\d+(?:\s*\|\s*[Cc]\d+)*)\b", re.IGNORECASE),
    re.compile(r"\bFinal answer\s*[:=]\s*([Cc]\d+(?:\s*\|\s*[Cc]\d+)*)\b", re.IGNORECASE),
    re.compile(r"\bChoose\s+([Cc]\d+(?:\s*\|\s*[Cc]\d+)*)\b", re.IGNORECASE),
    re.compile(r"\bI (?:choose|select|pick)\s+([Cc]\d+(?:\s*\|\s*[Cc]\d+)*)\b", re.IGNORECASE),
]
_CX_RE = re.compile(r"\b[Cc]\d+\b")


def find_insufficient(options: List[Dict[str, str]]) -> Optional[str]:
    for o in options or []:
        if "insufficient" in (o.get("label") or "").lower():
            return o.get("id")
    return None


def is_multi(scenario: Dict[str, Any]) -> bool:
    desc = ((scenario.get("task") or {}).get("description") or "").lower()
    return "two to four" in desc or "select two" in desc


def extract(text: str, valid_options: List[str], multi: bool) -> str:
    if not text:
        return ""
    valid = {o.upper() for o in valid_options}
    # Pattern cascade
    for pat in _PATTERNS:
        m = list(pat.finditer(text))
        if m:
            cand = m[-1].group(1).strip()
            cand = cand.replace(" ", "")
            parts = [p.upper() for p in cand.split("|") if p.strip()]
            parts = [p for p in parts if p in valid]
            if parts:
                if multi:
                    parts = sorted(set(parts), key=lambda s: int(re.search(r"\d+", s).group()))[:4]
                    return "|".join(parts)
                return parts[0]
    # Last-ditch: Cx mentions in trace, take most frequent in valid options.
    counts = Counter(m.group(0).upper() for m in _CX_RE.finditer(text) if m.group(0).upper() in valid)
    if counts:
        if multi:
            top = [c for c, _ in counts.most_common(3)]
            top = sorted(top, key=lambda s: int(re.search(r"\d+", s).group()))
            return "|".join(top)
        return counts.most_common(1)[0][0]
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, type=Path)
    ap.add_argument("--test_file", required=True, type=Path)
    ap.add_argument("--out_csv", default="result_rescued.csv")
    args = ap.parse_args()

    completions_path = args.results_dir / "completions.jsonl"
    if not completions_path.exists():
        print(f"FATAL: {completions_path} not found", file=sys.stderr)
        return 1

    with args.test_file.open("r", encoding="utf-8") as f:
        scenarios = {s["scenario_id"]: s for s in json.load(f)}

    rows = []
    n_already = 0
    n_rescued = 0
    n_fallback = 0

    with completions_path.open("r", encoding="utf-8") as f:
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
            scen = scenarios.get(sid)
            options = (scen or {}).get("task", {}).get("options", []) or []
            valid_ids = [o["id"] for o in options if "id" in o]

            existing = (rec.get("final_answer") or "").strip()
            if existing:
                rows.append({"scenario_id": sid, "answers": normalize_multi_answer(existing)})
                n_already += 1
                continue

            multi = bool(scen and is_multi(scen))
            pool = (rec.get("answer_raw") or "") + "\n" + (rec.get("traces") or "")
            ans = extract(pool, valid_ids, multi)
            if ans:
                rows.append({"scenario_id": sid, "answers": normalize_multi_answer(ans)})
                n_rescued += 1
            else:
                fallback = find_insufficient(options) or (valid_ids[0] if valid_ids else "C1")
                rows.append({"scenario_id": sid, "answers": fallback})
                n_fallback += 1

    out_path = args.results_dir / args.out_csv
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"  already had answer       : {n_already}")
    print(f"  rescued via lenient parse: {n_rescued}")
    print(f"  fell back to default     : {n_fallback}")
    print(f"  total rows               : {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
