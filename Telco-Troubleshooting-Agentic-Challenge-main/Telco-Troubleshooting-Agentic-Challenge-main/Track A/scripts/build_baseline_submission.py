"""
scripts/build_baseline_submission.py

Pure-Python heuristic baseline for the Telco Track A challenge. No LLM,
no server, no GPU. Reads test.json directly, applies a hand-tuned decision
tree derived from training-set label statistics, and writes a Zindi-format
result.csv in seconds.

Validated on the 2000 train scenarios at ~26.8% accuracy (vs ~6% for trivial
baselines and 0% for the timed-out LLM pipeline). Use this to get a real
submission on the leaderboard while the LLM path is being debugged.

Usage:
    python scripts/build_baseline_submission.py
    # writes eval/results/baseline/result.csv
    # plus result_v1_raw.csv / result_v2_multi_recall.csv / result_v3_insurance.csv
    # (3 identical copies — rule-based is deterministic, so the variants only
    #  exist to slot into Zindi's 3-submission allowance.)

Optional:
    --test_file <path>       default data/Phase_1/test.json
    --out_dir   <path>       default eval/results/baseline
    --score_train            also score on train.json and print accuracy
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------- table parsing

def parse_table(text: Optional[str]) -> pd.DataFrame:
    if not text or not text.strip():
        return pd.DataFrame()
    try:
        return pd.read_csv(StringIO(text), sep="|")
    except Exception:
        return pd.DataFrame()


def _col(df: pd.DataFrame, *needles: str) -> Optional[str]:
    """First column whose name contains ALL needle substrings (case-insensitive)."""
    for c in df.columns:
        cl = c.lower()
        if all(n.lower() in cl for n in needles):
            return c
    return None


# --------------------------------------------------------- failure-mode classify

def find_drop(df_up: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Locate the row with the largest sustained throughput drop and return
    the RSRP/SINR delta around it plus the serving PCI at that row."""
    if df_up.empty:
        return None
    tput_c = _col(df_up, "DL Throughput")
    rsrp_c = _col(df_up, "Serving SS-RSRP")
    sinr_c = _col(df_up, "Serving SS-SINR")
    pci_c = next(
        (c for c in df_up.columns if "Serving PCI" in c and "Neighbor" not in c),
        None,
    )
    if not all([tput_c, rsrp_c, sinr_c, pci_c]):
        return None

    tput = pd.to_numeric(df_up[tput_c], errors="coerce")
    rsrp = pd.to_numeric(df_up[rsrp_c], errors="coerce")
    sinr = pd.to_numeric(df_up[sinr_c], errors="coerce")
    if len(tput.dropna()) < 4:
        return None

    baseline = tput.shift(1).rolling(3, min_periods=1).mean()
    drop_pct = (baseline - tput) / baseline.replace(0, float("nan"))
    if drop_pct.dropna().empty:
        return None
    idx = int(drop_pct.idxmax())
    if pd.isna(drop_pct.loc[idx]) or drop_pct.loc[idx] < 0.30:
        return None

    pre_idx = max(0, idx - 3)
    rsrp_pre = rsrp.iloc[pre_idx:idx].mean()
    sinr_pre = sinr.iloc[pre_idx:idx].mean()
    rsrp_at = rsrp.iloc[idx]
    sinr_at = sinr.iloc[idx]

    pci = df_up[pci_c].iloc[idx]
    try:
        pci = int(pci)
    except (TypeError, ValueError):
        pass

    return {
        "rsrp_delta": float(rsrp_pre - rsrp_at) if pd.notna(rsrp_pre) and pd.notna(rsrp_at) else 0.0,
        "sinr_delta": float(sinr_pre - sinr_at) if pd.notna(sinr_pre) and pd.notna(sinr_at) else 0.0,
        "serving_pci": pci,
    }


def classify_mode(diag: Optional[Dict[str, Any]]) -> str:
    if diag is None:
        return "NO_DROP"
    rd, sd = diag["rsrp_delta"], diag["sinr_delta"]
    if rd > 6 and sd > 5:
        return "COVERAGE"
    if sd > 5 and rd < 3:
        return "INTERFERENCE"
    if rd < 3 and sd < 3:
        return "SCHEDULER"
    if sd > 5:
        return "INTERFERENCE_QUALITY"
    return "AMBIG"


# Keyword priors per failure mode — derived empirically from train.json
# (label-keyword distribution conditioned on classified mode).
KEYWORD_PRIOR: Dict[str, List[str]] = {
    "COVERAGE":             ["add_neighbor", "a3_decrease"],
    "INTERFERENCE":         ["covinterfreq_decrease", "a3_decrease"],
    "INTERFERENCE_QUALITY": ["a3_decrease", "add_neighbor"],
    "SCHEDULER":            ["pdcch_2sym", "tilt_lift", "check_test_server", "insufficient"],
    "AMBIG":                ["tilt_lift", "covinterfreq_decrease", "check_test_server"],
    "NO_DROP":              ["insufficient", "check_test_server"],
}


def label_keyword(label: str) -> str:
    L = (label or "").lower()
    if "insufficient" in L:                  return "insufficient"
    if "check test server" in L:             return "check_test_server"
    if "covinterfreq" in L:                  return "covinterfreq_decrease"
    if "pdcchoccupied" in L:                 return "pdcch_2sym"
    if "a3 offset" in L:
        return "a3_decrease" if "decrease" in L else "a3_increase"
    if "tilt" in L:
        return "tilt_lift" if "lift" in L else "tilt_down"
    if "azimuth" in L:                       return "azimuth"
    if "increase transmission power" in L:   return "tx_power_up"
    if "decrease transmission power" in L:   return "tx_power_down"
    if "add neighbor" in L:                  return "add_neighbor"
    return "other"


# ------------------------------------------------------------- PCI → cell mapping

def pci_to_cell(cfg_text: str, pci: Any) -> Optional[str]:
    """Map a PCI from user_plane_data to '{gNodeB ID}_{Cell ID}' as it appears
    in option labels."""
    df = parse_table(cfg_text)
    if df.empty or pci is None:
        return None
    pci_col = next((c for c in df.columns if c.strip().lower() == "pci"), None)
    g_col = next((c for c in df.columns if "gnodeb" in c.lower() and "id" in c.lower()), None)
    c_col = next(
        (c for c in df.columns if "cell" in c.lower() and "id" in c.lower() and "gnodeb" not in c.lower()),
        None,
    )
    if not all([pci_col, g_col, c_col]):
        return None
    try:
        pci_int: Any = int(pci)
    except (TypeError, ValueError):
        pci_int = pci
    matches = df[pd.to_numeric(df[pci_col], errors="coerce") == pci_int]
    if matches.empty:
        return None
    row = matches.iloc[0]
    try:
        return f"{int(row[g_col])}_{int(row[c_col])}"
    except (TypeError, ValueError):
        return f"{row[g_col]}_{row[c_col]}"


# ------------------------------------------------------------------- option pick

def is_multi(scenario: Dict[str, Any]) -> bool:
    desc = ((scenario.get("task") or {}).get("description") or "").lower()
    return "two to four" in desc or "select two" in desc


def score_option(label: str, mode: str, serving_cell: Optional[str]) -> float:
    pref = KEYWORD_PRIOR.get(mode, [])
    kw = label_keyword(label)
    sc = 0.0
    if kw in pref:
        sc = 5.0 - pref.index(kw) * 0.5
    if serving_cell and serving_cell in (label or ""):
        sc += 2.0
    return sc


def _sort_cx(ids: List[str]) -> List[str]:
    def key(s: str) -> int:
        m = re.search(r"\d+", s)
        return int(m.group()) if m else 0
    return sorted(set(ids), key=key)


def pick_answer(scenario: Dict[str, Any]) -> str:
    options = (scenario.get("task") or {}).get("options", []) or []
    diag = find_drop(parse_table((scenario.get("data") or {}).get("user_plane_data")))
    mode = classify_mode(diag)
    serving_cell = pci_to_cell(
        (scenario.get("data") or {}).get("network_configuration_data", ""),
        diag["serving_pci"] if diag else None,
    )

    scored: List[Tuple[float, str]] = [
        (score_option(o.get("label", ""), mode, serving_cell), o.get("id", ""))
        for o in options
    ]
    scored.sort(reverse=True, key=lambda x: x[0])

    if is_multi(scenario):
        top = [s for s in scored if s[0] > 0][:3]
        if len(top) < 2:
            top = scored[:3]
        return "|".join(_sort_cx([s[1] for s in top]))

    if scored and scored[0][0] > 0:
        return scored[0][1]
    # Fallback: pick the explicit "Insufficient data" option if present.
    for o in options:
        if "insufficient" in (o.get("label") or "").lower():
            return o["id"]
    return options[0]["id"] if options else ""


# ------------------------------------------------------------------------- main

def _score_match(pred: str, gt: str) -> float:
    if not pred or not gt:
        return 0.0
    if "|" in gt:
        gts = set(gt.split("|"))
        if "|" in pred:
            preds = set(pred.split("|"))
            inter = preds & gts
            union = preds | gts
            return len(inter) / len(union) if union else 0.0
        return 1.0 if pred in gts else 0.0
    if "|" in pred:
        return 1.0 if gt in set(pred.split("|")) else 0.0
    return 1.0 if pred == gt else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", default="data/Phase_1/test.json")
    ap.add_argument("--out_dir", default="eval/results/baseline")
    ap.add_argument("--score_train", action="store_true",
                    help="Also run on train.json and print accuracy.")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent.parent  # Track A/
    test_path = (here / args.test_file).resolve()
    out_dir = (here / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.score_train:
        train_path = (here / "data" / "Phase_1" / "train.json").resolve()
        with train_path.open("r", encoding="utf-8") as f:
            train = json.load(f)
        total = len(train)
        score = sum(_score_match(pick_answer(s), s.get("answer", "")) for s in train)
        print(f"[train] {score:.1f}/{total}  ->  {score/total*100:.2f}% accuracy")

    if not test_path.exists():
        print(f"FATAL: {test_path} not found", file=sys.stderr)
        return 1

    with test_path.open("r", encoding="utf-8") as f:
        test = json.load(f)

    rows: List[Dict[str, str]] = []
    for s in test:
        rows.append({
            "scenario_id": s.get("scenario_id", ""),
            "answers": pick_answer(s),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "result.csv", index=False)
    df.to_csv(out_dir / "result_v1_raw.csv", index=False)
    df.to_csv(out_dir / "result_v2_multi_recall.csv", index=False)
    df.to_csv(out_dir / "result_v3_insurance.csv", index=False)

    n_multi = sum(1 for s in test if is_multi(s))
    print(f"Wrote {len(rows)} rows to {out_dir}/result.csv  "
          f"(single={len(rows)-n_multi}, multi={n_multi})")
    print(f"Submission files (3 identical copies for the 3 Zindi tries):")
    print(f"  {out_dir}/result_v1_raw.csv")
    print(f"  {out_dir}/result_v2_multi_recall.csv")
    print(f"  {out_dir}/result_v3_insurance.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
