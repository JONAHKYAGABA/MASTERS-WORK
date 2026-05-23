#!/usr/bin/env python
"""
Read-only analysis of a per-stage training log.

Parses the tqdm progress + val + checkpoint + warning lines and prints:
  - Loss progression (every Nth step or per-epoch summary)
  - Val metrics per epoch (acc, IoU, AUROC, BLEU, etc.)
  - Best model promotion events
  - Anomalies (NaN warnings, OOM, NCCL timeouts, CUDA assertions)
  - Final summary (trained N steps, best val on metric M, time taken)

Usage:
    PYTHONPATH=$PWD python scripts/analyze_stage_log.py logs/curriculum_budget_stage1_sg_only_20260520_155237.log
    # Show fewer loss points (50 instead of default 100):
    PYTHONPATH=$PWD python scripts/analyze_stage_log.py <log> --loss_points 50
    # Save a CSV of all parsed losses:
    PYTHONPATH=$PWD python scripts/analyze_stage_log.py <log> --csv losses.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# -------------------------------------------------------------------------
# Patterns to pull from log lines
# -------------------------------------------------------------------------

# Format from the trainer's tqdm postfix dict:
#   "Epoch 1: ... loss=4.1234, vqa=0.66, chex=0.62, gen=2.64, sg=9.83, grd=1.29, step=46"
LOSS_RE = re.compile(
    r"Epoch\s+(\d+):.*?loss=([\d.]+),\s*vqa=([\d.]+),\s*chex=([\d.]+),"
    r"\s*gen=([\d.]+),\s*sg=([\d.]+),\s*grd=([\d.]+),\s*step=(\d+)"
)

# Val metric lines
VAL_RE = re.compile(r"Val\s+([A-Za-z][\w\s]*?):\s+([-\d.]+)")
EPOCH_END_RE = re.compile(r"Train Loss:\s+([\d.]+)")
NEW_BEST_RE = re.compile(r"New best (\w+):\s+([\d.]+)")
CKPT_RE = re.compile(r"Saved checkpoint to (\S+)")
BEST_CKPT_RE = re.compile(r"Saved best model to (\S+)")
FINAL_CKPT_RE = re.compile(r"Saved FINAL MODEL to (\S+)")
TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")

# Anomaly patterns
ANOMALY_RES = [
    ("NaN/Inf detected", re.compile(r"NaN.*detected|Inf.*detected", re.I)),
    ("OOM",              re.compile(r"OutOfMemoryError|out of memory", re.I)),
    ("CUDA assertion",   re.compile(r"Assertion `.*` failed|device-side assert", re.I)),
    ("NCCL timeout",     re.compile(r"NCCL.*timeout|ncclCommWatchdog", re.I)),
    ("HF push failed",   re.compile(r"Failed to push to hub", re.I)),
    ("HF DNS failure",   re.compile(r"NameResolutionError.*huggingface", re.I)),
    ("Network blip",     re.compile(r"Failed to resolve.*api\.", re.I)),
    ("Loss spike",       None),  # filled programmatically
]


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def parse_timestamp(line: str) -> Optional[datetime]:
    m = TIMESTAMP_RE.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def human_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def thin(samples: list, target: int) -> list:
    """Reduce a list to ~target items, evenly spaced."""
    if len(samples) <= target:
        return samples
    stride = max(1, len(samples) // target)
    return samples[::stride]


# -------------------------------------------------------------------------
# Main parse
# -------------------------------------------------------------------------

def parse_log(path: Path) -> Dict:
    losses: List[Tuple[int, int, float, float, float, float, float, float, datetime]] = []
    # (epoch, step, loss, vqa, chex, gen, sg, grd, timestamp)
    val_metrics_per_epoch: Dict[int, Dict[str, float]] = {}
    train_loss_per_epoch: Dict[int, float] = {}
    best_promotions: List[Tuple[str, float, Optional[datetime]]] = []
    checkpoints: List[Tuple[str, Optional[datetime]]] = []
    best_checkpoints: List[Tuple[str, Optional[datetime]]] = []
    final_checkpoint: Optional[Tuple[str, Optional[datetime]]] = None
    anomalies: List[Tuple[str, int, str]] = []  # (kind, line_num, snippet)
    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    cur_epoch = 0

    with path.open("r", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            ts = parse_timestamp(line)
            if ts:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

            # Loss tqdm line
            m = LOSS_RE.search(line)
            if m:
                ep = int(m.group(1))
                losses.append((
                    ep,
                    int(m.group(8)),  # step
                    float(m.group(2)),  # loss
                    float(m.group(3)),  # vqa
                    float(m.group(4)),  # chex
                    float(m.group(5)),  # gen
                    float(m.group(6)),  # sg
                    float(m.group(7)),  # grd
                    last_ts or first_ts,
                ))
                cur_epoch = ep

            # Train loss at end of epoch
            m = EPOCH_END_RE.search(line)
            if m:
                train_loss_per_epoch[cur_epoch] = float(m.group(1))

            # Val metric lines
            m = VAL_RE.search(line)
            if m:
                key = m.group(1).strip()
                try:
                    val = float(m.group(2))
                    val_metrics_per_epoch.setdefault(cur_epoch, {})[key] = val
                except ValueError:
                    pass

            m = NEW_BEST_RE.search(line)
            if m:
                best_promotions.append((m.group(1), float(m.group(2)), last_ts))

            m = CKPT_RE.search(line)
            if m:
                checkpoints.append((m.group(1), last_ts))

            m = BEST_CKPT_RE.search(line)
            if m:
                best_checkpoints.append((m.group(1), last_ts))

            m = FINAL_CKPT_RE.search(line)
            if m:
                final_checkpoint = (m.group(1), last_ts)

            for kind, pat in ANOMALY_RES:
                if pat is None:
                    continue
                if pat.search(line):
                    anomalies.append((kind, lineno, line.strip()[:200]))
                    break

    # Detect loss spikes: any step where loss > 2x median of surrounding 50 steps
    if len(losses) > 100:
        median_loss = sorted(l[2] for l in losses)[len(losses) // 2]
        spike_threshold = max(median_loss * 2, 10.0)
        spikes = [(ep, st, val) for ep, st, val, *_ in losses if val > spike_threshold]
        for ep, st, val in spikes[:10]:  # cap to first 10
            anomalies.append(("Loss spike", 0, f"step={st} loss={val:.2f} (median={median_loss:.2f})"))

    return {
        "path": path,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "losses": losses,
        "train_loss_per_epoch": train_loss_per_epoch,
        "val_metrics_per_epoch": val_metrics_per_epoch,
        "best_promotions": best_promotions,
        "checkpoints": checkpoints,
        "best_checkpoints": best_checkpoints,
        "final_checkpoint": final_checkpoint,
        "anomalies": anomalies,
    }


# -------------------------------------------------------------------------
# Report
# -------------------------------------------------------------------------

def report(data: Dict, loss_points: int = 100, csv_out: Optional[Path] = None) -> None:
    losses = data["losses"]

    print("=" * 80)
    print(f"  STAGE LOG ANALYSIS: {data['path']}")
    print("=" * 80)

    if data["first_ts"] and data["last_ts"]:
        elapsed = (data["last_ts"] - data["first_ts"]).total_seconds()
        print(f"\nWall time: {data['first_ts']} → {data['last_ts']}  ({human_duration(elapsed)})")

    if not losses:
        print("\n❌ NO TRAINING DATA FOUND IN LOG")
        print("   The log doesn't contain any 'Epoch N: ... loss=X.XX, ... step=Y' lines.")
        print("   Either training never started or this isn't a per-stage log.")
        return

    n_steps = losses[-1][1]
    n_epochs_seen = max(l[0] for l in losses)
    print(f"Training reached: epoch {n_epochs_seen}, dataloader iteration {n_steps:,}")

    # --- Loss progression ---
    print("\n" + "-" * 80)
    print(f"  LOSS PROGRESSION  ({len(losses)} measurements, showing {min(loss_points, len(losses))})")
    print("-" * 80)
    print(f"\n  {'epoch':>5} {'step':>8} {'total':>8} {'vqa':>7} {'chex':>7} {'gen':>7} {'sg':>7} {'grd':>7}")
    sampled = thin(losses, loss_points)
    for ep, st, loss, vqa, chex, gen, sg, grd, _ts in sampled:
        print(f"  {ep:>5} {st:>8,} {loss:>8.3f} {vqa:>7.3f} {chex:>7.3f} {gen:>7.3f} {sg:>7.3f} {grd:>7.3f}")

    first = losses[0]
    last = losses[-1]
    print(f"\n  First  : loss={first[2]:.3f}  vqa={first[3]:.3f}  chex={first[4]:.3f}  gen={first[5]:.3f}  sg={first[6]:.3f}  grd={first[7]:.3f}")
    print(f"  Last   : loss={last[2]:.3f}  vqa={last[3]:.3f}  chex={last[4]:.3f}  gen={last[5]:.3f}  sg={last[6]:.3f}  grd={last[7]:.3f}")
    delta = lambda i: last[i] - first[i]
    print(f"  Δ      : loss={delta(2):+.3f}  vqa={delta(3):+.3f}  chex={delta(4):+.3f}  gen={delta(5):+.3f}  sg={delta(6):+.3f}  grd={delta(7):+.3f}")

    # --- Per-epoch summary ---
    print("\n" + "-" * 80)
    print("  PER-EPOCH SUMMARY")
    print("-" * 80)
    print(f"\n  {'epoch':>5} {'train_loss':>12} {'val_loss':>10} {'cls_acc':>10} {'sg_ent_acc':>12} {'grd_iou':>10} {'chex_auroc':>12}")
    for ep in sorted(set([0] + list(data["train_loss_per_epoch"].keys()) + list(data["val_metrics_per_epoch"].keys()))):
        if ep == 0:
            continue
        train_loss = data["train_loss_per_epoch"].get(ep, float("nan"))
        vm = data["val_metrics_per_epoch"].get(ep, {})
        print(f"  {ep:>5} "
              f"{train_loss:>12.4f} "
              f"{vm.get('Loss', float('nan')):>10.4f} "
              f"{vm.get('Accuracy', float('nan')):>10.4f} "
              f"{vm.get('SG Entity Acc', float('nan')):>12.4f} "
              f"{vm.get('Grounding IoU', float('nan')):>10.4f} "
              f"{vm.get('CheXpert AUROC', float('nan')):>12.4f}")

    # --- Best model promotions ---
    print("\n" + "-" * 80)
    print("  BEST MODEL PROMOTIONS")
    print("-" * 80)
    if data["best_promotions"]:
        for metric, val, ts in data["best_promotions"]:
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "?"
            print(f"  [{ts_str}]  new best {metric}: {val:.4f}")
    else:
        print("  (none — best metric never improved past zero/init)")

    # --- Checkpoints ---
    print("\n" + "-" * 80)
    print(f"  CHECKPOINTS WRITTEN  ({len(data['checkpoints'])} step-checkpoints, "
          f"{len(data['best_checkpoints'])} best, "
          f"{'1 final' if data['final_checkpoint'] else '0 final'})")
    print("-" * 80)
    for path, ts in data["checkpoints"][-5:]:
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "?"
        print(f"  [{ts_str}]  {path}")
    if data["best_checkpoints"]:
        print()
        for path, ts in data["best_checkpoints"][-3:]:
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "?"
            print(f"  [{ts_str}]  BEST → {path}")
    if data["final_checkpoint"]:
        path, ts = data["final_checkpoint"]
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "?"
        print(f"  [{ts_str}]  FINAL → {path}")

    # --- Anomalies ---
    print("\n" + "-" * 80)
    print(f"  ANOMALIES / WARNINGS  ({len(data['anomalies'])})")
    print("-" * 80)
    if not data["anomalies"]:
        print("  ✓ none detected")
    else:
        counts: Dict[str, int] = {}
        for kind, _, _ in data["anomalies"]:
            counts[kind] = counts.get(kind, 0) + 1
        for kind, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {n:>4} × {kind}")
        print("\n  First 5 examples:")
        for kind, lineno, snippet in data["anomalies"][:5]:
            print(f"    [{kind}@line{lineno}]  {snippet[:140]}")

    # --- CSV dump ---
    if csv_out:
        with csv_out.open("w") as f:
            f.write("epoch,step,loss,vqa,chex,gen,sg,grd,timestamp\n")
            for ep, st, loss, vqa, chex, gen, sg, grd, ts in losses:
                ts_str = ts.isoformat() if ts else ""
                f.write(f"{ep},{st},{loss},{vqa},{chex},{gen},{sg},{grd},{ts_str}\n")
        print(f"\n[csv] {len(losses):,} loss points → {csv_out}")

    # --- Final verdict ---
    print("\n" + "=" * 80)
    print("  VERDICT")
    print("=" * 80)
    converged = first[2] - last[2] > 0.5
    has_best = bool(data["best_promotions"])
    has_final = data["final_checkpoint"] is not None
    if converged and has_best and has_final:
        print("  ✅ Stage completed cleanly. Loss decreased, best promotions happened,")
        print("     final checkpoint exists. Safe to chain to next stage.")
    elif has_final and not converged:
        print("  ⚠️  Stage finished but loss didn't decrease meaningfully.")
        print("     Did you train enough steps? Check the loss curve above.")
    elif not has_final:
        print("  ❌ Stage did NOT complete — no FINAL MODEL save line.")
        print("     Likely crashed mid-training. See ANOMALIES above for clues.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, help="Per-stage training log file")
    ap.add_argument("--loss_points", type=int, default=100,
                    help="How many loss samples to print (default 100)")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Optional: write all loss points to this CSV")
    args = ap.parse_args()

    if not args.log.exists():
        print(f"ERROR: log not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    data = parse_log(args.log)
    report(data, loss_points=args.loss_points, csv_out=args.csv)


if __name__ == "__main__":
    main()
