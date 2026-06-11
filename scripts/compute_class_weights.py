#!/usr/bin/env python3
"""
scripts/compute_class_weights.py — derive inverse-frequency class weights
from MIMIC-Ext-CXR-QBA's `study_observations.csv` + `study_regions.csv`.

Why this matters
----------------
The dataset is overwhelmingly negative ("no pneumothorax", "lungs clear",
"bony structures intact"). On the 894 entities, the most common have a
pos/neg ratio of 1:1000 or worse. Untuned cross-entropy reduces by
predicting neg + the most common positive entity for everything — that's
exactly the "prosthesis @ hemiazygos vein" mode collapse we saw at
inference.

Solution: per-class loss weights inversely proportional to frequency, so
rare positive findings carry the same effective gradient as common ones.

This script:
  - reads `data/mimic-ext-cxr-qba/stats/study_observations.csv` (entity freq)
  - reads `data/mimic-ext-cxr-qba/stats/study_regions.csv` (region freq)
  - maps entity_name → entity_id via dataset_info.json
  - computes:
        weight[i] = (sqrt(total) / sqrt(count[i]))   ← muted inverse-freq
        # sqrt is gentler than 1/count; pure inverse-freq amplifies noise
        # in long-tail classes that have only a handful of samples.
  - writes:
        configs/class_weights/entity_weights.json   (232 weights)
        configs/class_weights/region_weights.json   (311 weights)
        configs/class_weights/polarity_weights.json (just 2 weights)

These files are consumed by training/loss.py — see the new
`--class_weights_dir` CLI flag in train_mimic_cxr.py.

Usage:
    .venv/bin/python scripts/compute_class_weights.py
    .venv/bin/python scripts/compute_class_weights.py --power 1.0    # full inv-freq
    .venv/bin/python scripts/compute_class_weights.py --positive_only # ignore neg
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("class_weights")

QA_ROOT = _ROOT / "data" / "mimic-ext-cxr-qba"
STATS_DIR = QA_ROOT / "stats"
VOCAB = QA_ROOT / "metadata" / "dataset_info.json"
OUT_DIR = _ROOT / "configs" / "class_weights"


def load_vocab():
    di = json.loads(VOCAB.read_text())
    regions = di.get("regions") or di.get("region_names") or []
    entities = di.get("finding_entities") or di.get("entity_names") or []
    region_to_idx = {r.lower(): i for i, r in enumerate(regions)}
    entity_to_idx = {e.lower(): i for i, e in enumerate(entities)}
    return regions, entities, region_to_idx, entity_to_idx


def freq_to_weights(
    counts: Dict[int, int],
    vocab_size: int,
    power: float = 0.5,
    min_count: int = 1,
    clip_min: float = 0.1,
    clip_max: float = 10.0,
) -> List[float]:
    """Inverse-frequency weights with a sqrt damping and clipping.

    - `power=0.5` → sqrt(N/n_i)         (default — gentle re-balancing)
    - `power=1.0` → N/n_i              (full inverse-freq — aggressive)
    - `power=0.0` → uniform             (no re-balancing)

    Classes with count < min_count default to clip_max (treat as very rare).
    Final weights are normalised to mean = 1.0 so loss scale is preserved.
    """
    import numpy as np
    n_total = max(1, sum(counts.values()))
    weights = []
    for i in range(vocab_size):
        c = max(min_count, counts.get(i, 0))
        # Inverse-freq^power, normalised against mean count
        w = (n_total / vocab_size / c) ** power
        w = max(clip_min, min(clip_max, w))
        weights.append(w)
    arr = np.array(weights)
    arr = arr / arr.mean()  # preserve loss scale
    return arr.tolist()


def compute_entity_weights(entity_to_idx, args) -> List[float]:
    """Read study_observations.csv and produce entity weights."""
    import pandas as pd
    df = pd.read_csv(STATS_DIR / "study_observations.csv")
    log.info(f"entities CSV: shape={df.shape}, columns={list(df.columns)}")

    n_classes = max(entity_to_idx.values()) + 1
    counts: Dict[int, int] = {}
    not_in_vocab = 0
    for _, row in df.iterrows():
        name = str(row["entity_name"]).lower().strip()
        eid = entity_to_idx.get(name)
        if eid is None:
            not_in_vocab += 1
            continue
        c = int(row["pos_mention"]) if args.positive_only else int(row["total_mention"])
        counts[eid] = counts.get(eid, 0) + c

    log.info(f"  matched {len(counts):,}/{len(df):,} entity rows to vocab "
             f"({not_in_vocab} not in vocab — fine, vocab subset of stats)")
    log.info(f"  per-class count: min={min(counts.values()) if counts else 0} "
             f"max={max(counts.values()) if counts else 0} "
             f"mean={sum(counts.values())/max(1,len(counts)):.0f}")

    return freq_to_weights(counts, n_classes, power=args.power,
                            min_count=args.min_count,
                            clip_min=args.clip_min, clip_max=args.clip_max)


def compute_region_weights(region_to_idx, args) -> List[float]:
    import pandas as pd
    df = pd.read_csv(STATS_DIR / "study_regions.csv")
    log.info(f"regions CSV: shape={df.shape}, columns={list(df.columns)}")
    n_classes = max(region_to_idx.values()) + 1
    counts: Dict[int, int] = {}
    not_in_vocab = 0
    for _, row in df.iterrows():
        name = str(row["region_name"]).lower().strip()
        rid = region_to_idx.get(name)
        if rid is None:
            not_in_vocab += 1
            continue
        c = int(row["pos_mention"]) if args.positive_only else int(row["total_mention"])
        counts[rid] = counts.get(rid, 0) + c
    log.info(f"  matched {len(counts):,}/{len(df):,} region rows to vocab "
             f"({not_in_vocab} not in vocab — fine)")
    return freq_to_weights(counts, n_classes, power=args.power,
                            min_count=args.min_count,
                            clip_min=args.clip_min, clip_max=args.clip_max)


def compute_polarity_weights() -> List[float]:
    """Polarity weights — derived directly from the entity CSV's pos/neg sums.
    Always 2 classes (0=neg, 1=pos)."""
    import pandas as pd
    df = pd.read_csv(STATS_DIR / "study_observations.csv")
    n_pos = int(df["pos_mention"].sum())
    n_neg = int(df["neg_mention"].sum())
    total = n_pos + n_neg
    log.info(f"  polarity totals: pos={n_pos:,}  neg={n_neg:,}  ratio=1:{n_neg/max(1,n_pos):.1f}")
    # Inverse-freq, normalised so mean=1
    raw = [total / (2 * n_neg) if n_neg else 1.0,
           total / (2 * n_pos) if n_pos else 1.0]
    import numpy as np
    arr = np.array(raw)
    arr = arr / arr.mean()
    return arr.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--power",     type=float, default=0.5,
                    help="0.5=sqrt damped (default); 1.0=pure inverse-freq.")
    ap.add_argument("--min_count", type=int,   default=10,
                    help="Floor on counts to avoid extreme weights for rare classes.")
    ap.add_argument("--clip_min",  type=float, default=0.1)
    ap.add_argument("--clip_max",  type=float, default=10.0)
    ap.add_argument("--positive_only", action="store_true",
                    help="Use only pos_mention (not total). Best if most loss "
                         "comes from positive-finding samples.")
    ap.add_argument("--output_dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"output dir: {args.output_dir}")
    log.info(f"hyperparams: power={args.power} min_count={args.min_count} "
             f"clip=[{args.clip_min},{args.clip_max}] positive_only={args.positive_only}")

    regions, entities, region_to_idx, entity_to_idx = load_vocab()
    log.info(f"vocab: {len(regions)} regions, {len(entities)} entities")

    log.info("=== entity weights ===")
    entity_weights = compute_entity_weights(entity_to_idx, args)
    log.info(f"  entity weights: min={min(entity_weights):.3f} "
             f"max={max(entity_weights):.3f}")

    log.info("=== region weights ===")
    region_weights = compute_region_weights(region_to_idx, args)
    log.info(f"  region weights: min={min(region_weights):.3f} "
             f"max={max(region_weights):.3f}")

    log.info("=== polarity weights ===")
    polarity_weights = compute_polarity_weights()
    log.info(f"  polarity weights: neg={polarity_weights[0]:.3f}  "
             f"pos={polarity_weights[1]:.3f}  (ratio={polarity_weights[1]/polarity_weights[0]:.2f}x)")

    # Save each set as a small JSON payload that includes the metadata
    # so the trainer can sanity-check it matches the expected vocab sizes.
    common = {
        "power": args.power,
        "min_count": args.min_count,
        "clip_min": args.clip_min,
        "clip_max": args.clip_max,
        "positive_only": args.positive_only,
    }
    payloads = {
        "entity_weights.json": {
            **common, "n_classes": len(entity_weights),
            "vocab_source": "dataset_info.json:finding_entities|entity_names",
            "weights": entity_weights,
        },
        "region_weights.json": {
            **common, "n_classes": len(region_weights),
            "vocab_source": "dataset_info.json:regions|region_names",
            "weights": region_weights,
        },
        "polarity_weights.json": {
            **common, "n_classes": 2,
            "vocab_source": "study_observations.csv (pos vs neg sums)",
            "weights": polarity_weights,
        },
    }
    for name, payload in payloads.items():
        out = args.output_dir / name
        out.write_text(json.dumps(payload, indent=2))
        log.info(f"wrote {out}")

    # Top-10 most-amplified and least-amplified entity classes (sanity)
    pairs = sorted(zip(entities, entity_weights), key=lambda kv: -kv[1])[:10]
    log.info("Top-10 UPWEIGHTED entities (rare → high weight):")
    for name, w in pairs:
        log.info(f"  {w:.3f}  {name}")
    pairs = sorted(zip(entities, entity_weights), key=lambda kv: kv[1])[:10]
    log.info("Top-10 DOWNWEIGHTED entities (very common → low weight):")
    for name, w in pairs:
        log.info(f"  {w:.3f}  {name}")


if __name__ == "__main__":
    main()
