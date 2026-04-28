"""
scripts/score_results.py

Reads completions.jsonl produced by main.py and writes SUMMARY.md.
Joins answers against train.json ground truth (DATA_SPLIT=train at run time)
to give a conclusive accuracy number, plus per-template, per-environment,
per-question-type breakdowns and the wall-clock distribution.

Usage:
    python scripts/score_results.py \\
        --results_dir eval/results/stage_b_50 \\
        --train_file data/Phase_1/train.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Reuse locked compute_score so the score matches the leaderboard rule.
HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
sys.path.insert(0, str(PROJECT_DIR))
from utils import compute_score  # type: ignore  # noqa: E402


# Action template extraction: strip the trailing cell ID so options like
# "Increase tilt of cell 1234567_1 by 2 degrees" collapse to a template.
_CELL_ID_PAT = re.compile(r"\b\d{6,}(?:_\d+)?\b")
_NUM_PAT = re.compile(r"\b\d+(?:\.\d+)?\b")


def derive_template(label: str) -> str:
    if not label:
        return "(unknown)"
    s = _CELL_ID_PAT.sub("<CELL>", label)
    s = _NUM_PAT.sub("<N>", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_train_index(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {s["scenario_id"]: s for s in data}


def is_multi_answer(scenario: Dict[str, Any]) -> bool:
    desc = (scenario.get("task", {}) or {}).get("description", "") or ""
    return "two to four" in desc.lower() or "select two" in desc.lower()


def env_size(scenario: Dict[str, Any]) -> str:
    info = (scenario.get("context", {}) or {}).get("wireless_network_information", {}) or {}
    n = info.get("num_base_stations") or "?"
    return f"{n}-BS"


def find_template_for_answer(scenario: Dict[str, Any], gt: str) -> str:
    options = (scenario.get("task", {}) or {}).get("options", []) or []
    if "|" in gt:
        first = gt.split("|")[0].strip()
    else:
        first = gt.strip()
    for opt in options:
        if opt.get("id") == first:
            return derive_template(opt.get("label", ""))
    return "(no-match)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, type=Path)
    ap.add_argument("--train_file", required=True, type=Path)
    ap.add_argument("--summary_name", default="SUMMARY.md")
    ap.add_argument("--target_acc", type=float, default=0.35,
                    help="Pass threshold for prompted-only ceiling (default 0.35)")
    ap.add_argument("--target_latency_s", type=float, default=90.0,
                    help="Pass threshold for mean wall-clock per scenario (default 90s)")
    args = ap.parse_args()

    results_dir: Path = args.results_dir
    completions_path = results_dir / "completions.jsonl"
    if not completions_path.exists():
        print(f"FAIL: {completions_path} not found", file=sys.stderr)
        return 1

    completions = load_jsonl(completions_path)
    if not completions:
        print(f"FAIL: {completions_path} is empty", file=sys.stderr)
        return 1

    train_idx = load_train_index(args.train_file)

    n = len(completions)
    matched = 0
    correct_sum = 0.0
    latencies: List[float] = []
    tool_calls: List[int] = []
    timeouts = 0
    unresolved = 0

    per_template_acc: Dict[str, List[float]] = defaultdict(list)
    per_env_acc: Dict[str, List[float]] = defaultdict(list)
    per_qtype_acc: Dict[str, List[float]] = defaultdict(list)

    answer_distribution: Counter[str] = Counter()
    failure_examples: List[Dict[str, Any]] = []

    rescored: List[Dict[str, Any]] = []

    for c in completions:
        sid = c.get("scenario_id")
        if not sid:
            continue
        scen = train_idx.get(sid)
        gt = (scen or {}).get("answer") if scen else c.get("ground_truth")
        ans = c.get("final_answer", "") or ""

        if scen is None or not gt:
            score = float(c.get("accuracy", 0.0) or 0.0)
        else:
            try:
                score = float(bool(compute_score(ans, gt)))
            except Exception:
                score = 0.0
            matched += 1

        correct_sum += score
        latencies.append(float(c.get("latency", 0.0) or 0.0))
        tool_calls.append(int(c.get("num_tool_calls", 0) or 0))
        if c.get("status") == "timeout":
            timeouts += 1
        if c.get("status") == "unresolved":
            unresolved += 1

        if ans:
            answer_distribution[ans] += 1

        if scen is not None:
            tmpl = find_template_for_answer(scen, gt or "")
            per_template_acc[tmpl].append(score)
            per_env_acc[env_size(scen)].append(score)
            per_qtype_acc["multi" if is_multi_answer(scen) else "single"].append(score)

            if score < 1.0 and len(failure_examples) < 5:
                failure_examples.append({
                    "scenario_id": sid,
                    "ground_truth": gt,
                    "predicted": ans,
                    "tool_calls": c.get("num_tool_calls", 0),
                    "latency_s": c.get("latency", 0.0),
                    "status": c.get("status"),
                    "reason": c.get("reason"),
                })

        rescored.append({**c, "rescored": score})

    mean_acc = correct_sum / n if n else 0.0
    mean_latency = statistics.fmean(latencies) if latencies else 0.0
    p50_latency = statistics.median(latencies) if latencies else 0.0
    p95_latency = (sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0.0)
    max_latency = max(latencies) if latencies else 0.0
    mean_tools = statistics.fmean(tool_calls) if tool_calls else 0.0

    pass_acc = mean_acc >= args.target_acc
    pass_latency = mean_latency <= args.target_latency_s

    if pass_acc and pass_latency:
        verdict = "GO — proceed to Stage C (self-distillation)"
    elif pass_latency and not pass_acc:
        verdict = "PARTIAL — prompt/system tweaks before distillation"
    elif pass_acc and not pass_latency:
        verdict = "PARTIAL — accuracy is fine but wall-clock too high; trim further"
    else:
        verdict = "NO-GO — investigate before committing GPU-hours"

    summary_path = results_dir / args.summary_name
    lines: List[str] = []
    lines.append(f"# Stage B conclusive experiment — {results_dir.name}")
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append(f"- Scenarios scored: **{n}** (matched to train.json: {matched})")
    lines.append(f"- Mean accuracy (IoU/exact-match per locked compute_score): **{mean_acc:.3f}**")
    lines.append(f"  - Pass threshold {args.target_acc:.2f} → {'YES' if pass_acc else 'NO'}")
    lines.append(f"- Mean wall-clock per scenario: **{mean_latency:.1f} s**")
    lines.append(f"  - Median {p50_latency:.1f} s | p95 {p95_latency:.1f} s | max {max_latency:.1f} s")
    lines.append(f"  - Pass threshold {args.target_latency_s:.0f} s → {'YES' if pass_latency else 'NO'}")
    lines.append(f"- Mean tool calls per scenario: {mean_tools:.2f}")
    lines.append(f"- Timeouts: {timeouts} | unresolved: {unresolved}")
    lines.append("")

    lines.append("## Per-question-type accuracy")
    lines.append("")
    lines.append("| Type | N | Acc |")
    lines.append("|---|---:|---:|")
    for k, vs in sorted(per_qtype_acc.items()):
        lines.append(f"| {k} | {len(vs)} | {statistics.fmean(vs):.3f} |")
    lines.append("")

    lines.append("## Per-environment accuracy")
    lines.append("")
    lines.append("| Env | N | Acc |")
    lines.append("|---|---:|---:|")
    for k in sorted(per_env_acc):
        vs = per_env_acc[k]
        lines.append(f"| {k} | {len(vs)} | {statistics.fmean(vs):.3f} |")
    lines.append("")

    lines.append("## Per-template accuracy (top 10 by sample count)")
    lines.append("")
    lines.append("| Template | N | Acc |")
    lines.append("|---|---:|---:|")
    top_templates = sorted(per_template_acc.items(), key=lambda kv: -len(kv[1]))[:10]
    for tmpl, vs in top_templates:
        short = (tmpl[:90] + "…") if len(tmpl) > 90 else tmpl
        lines.append(f"| {short} | {len(vs)} | {statistics.fmean(vs):.3f} |")
    lines.append("")

    lines.append("## Answer distribution (top 10)")
    lines.append("")
    lines.append("| Answer | Count |")
    lines.append("|---|---:|")
    for a, cnt in answer_distribution.most_common(10):
        lines.append(f"| `{a}` | {cnt} |")
    lines.append("")

    if failure_examples:
        lines.append("## First 5 failures")
        lines.append("")
        for f in failure_examples:
            lines.append(
                f"- **{f['scenario_id']}** — gt=`{f['ground_truth']}`, "
                f"pred=`{f['predicted']}`, tools={f['tool_calls']}, "
                f"{f['latency_s']:.1f}s, status={f['status']}"
            )
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    metrics = {
        "n_scenarios": n,
        "matched_to_train": matched,
        "mean_accuracy": mean_acc,
        "mean_latency_s": mean_latency,
        "p50_latency_s": p50_latency,
        "p95_latency_s": p95_latency,
        "max_latency_s": max_latency,
        "mean_tool_calls": mean_tools,
        "timeouts": timeouts,
        "unresolved": unresolved,
        "pass_accuracy": pass_acc,
        "pass_latency": pass_latency,
        "verdict": verdict,
        "per_template_acc": {k: statistics.fmean(v) for k, v in per_template_acc.items()},
        "per_env_acc": {k: statistics.fmean(v) for k, v in per_env_acc.items()},
        "per_qtype_acc": {k: statistics.fmean(v) for k, v in per_qtype_acc.items()},
    }
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Wrote {summary_path}")
    print(f"VERDICT: {verdict}")
    print(f"acc={mean_acc:.3f} latency_mean={mean_latency:.1f}s n={n}")
    return 0 if pass_acc and pass_latency else 2


if __name__ == "__main__":
    sys.exit(main())
