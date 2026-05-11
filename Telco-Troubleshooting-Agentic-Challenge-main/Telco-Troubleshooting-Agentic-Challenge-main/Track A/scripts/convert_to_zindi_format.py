"""
Convert our result.csv (scenario_id, answers) to the Zindi submission format
(scenario_id, Track A, Track B) using the sample template's row order and IDs.

Usage:
    python scripts/convert_to_zindi_format.py \\
        --our_csv      eval/results/submit_now/result_v1_raw.csv \\
        --sample_csv   ../submission/Phase_1/result.csv \\
        --out          eval/results/submit_now/result_v1_zindi.csv \\
        --track A
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--our_csv", required=True, type=Path)
    ap.add_argument("--sample_csv", required=True, type=Path,
                    help="Path to Zindi's sample submission template")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--track", choices=["A", "B"], default="A")
    args = ap.parse_args()

    if not args.our_csv.exists():
        print(f"FATAL: {args.our_csv} not found", file=sys.stderr)
        return 1
    if not args.sample_csv.exists():
        print(f"FATAL: {args.sample_csv} not found", file=sys.stderr)
        return 1

    template = pd.read_csv(args.sample_csv, dtype=str).fillna("")
    ours = pd.read_csv(args.our_csv, dtype=str).fillna("")

    # Validate template columns
    expected = {"scenario_id", "Track A", "Track B"}
    if set(template.columns) != expected:
        print(f"WARN: template columns are {list(template.columns)}, expected {sorted(expected)}",
              file=sys.stderr)

    # Build a lookup from our results
    if "scenario_id" not in ours.columns or "answers" not in ours.columns:
        print(f"FATAL: our_csv must have columns 'scenario_id,answers', got {list(ours.columns)}",
              file=sys.stderr)
        return 1
    answer_by_id = dict(zip(ours["scenario_id"], ours["answers"]))

    # Fill the chosen track column from our answers; the other track stays untouched.
    target_col = f"Track {args.track}"
    n_filled = 0
    n_missing = 0
    for i, sid in enumerate(template["scenario_id"]):
        if sid in answer_by_id:
            template.at[i, target_col] = answer_by_id[sid]
            n_filled += 1
        else:
            n_missing += 1

    # Zindi expects the ID column to be literally named "ID", not "scenario_id".
    template = template.rename(columns={"scenario_id": "ID"})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(args.out, index=False)

    print(f"Wrote {len(template)} rows to {args.out}")
    print(f"  filled '{target_col}' for {n_filled} scenarios")
    print(f"  no answer for {n_missing} scenarios (left blank — likely the other track)")
    print(f"  columns: {list(template.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
