#!/usr/bin/env python3
"""
scripts/partition_curriculum.py — deterministic disjoint splits for the curriculum.

Why this exists
---------------
The original curriculum had all 4 stages pulling from the same MIMIC train
split, distinguished only by ``quality_grade`` filter. That's three bugs at
once:

  1) Data leak: Stage 4 finetune's val_loss is inflated because the model
     already saw most A-grade studies during Stage 3 pretrain (B = "B and
     above" includes A).
  2) Wrong supervision for the SG generator: Stage 1 trains on
     quality_grade="all", so the detector learns from low-quality scene
     graphs and inherits their noise (this is part of why we saw "prosthesis
     @ hemiazygos vein" mode collapse — that pair is over-represented in
     low-quality annotations).
  3) No held-out final eval slice: the only "validation" is built from the
     dataset's validate split, which is fine for cross-validation but doesn't
     give us a true test set for the report.

What this script does
---------------------
Partitions the train split's study_ids deterministically (seeded hash by
study_id) into four disjoint pools, then writes the pools to JSON files the
trainer can load directly. The partition rule is:

    A-grade studies → split 90/10 into:
        - sg_train_A     (90%, for Stage 1 = SG generator training)
        - finetune_A     (10%, for Stage 4, NEVER seen during Stages 1–3)

    B-grade studies → split into:
        - alignment_B    (40%, for Stage 2)
        - pretrain_B     (60%, for Stage 3)

    A-grade studies in sg_train_A are also added to alignment_B and
    pretrain_B (training the alignment + full pipeline on clean A-grade
    samples in addition to noisier B-grade is good practice and doesn't
    leak because Stage 4's finetune_A slice is held out).

    Studies with no quality grade (or grade C) → discarded (untrusted).

The trainer reads the partition file via a new --partition_file flag
and filters its dataset accordingly. See the bottom of this docstring for
the changes needed in train_mimic_cxr.py.

Usage
-----
    # Run ONCE before launching the curriculum; writes 4 JSON files
    python scripts/partition_curriculum.py \\
        --mimic_qa_path data/mimic-ext-cxr-qba \\
        --output_dir data/partitions \\
        --seed 42

    # Then in the curriculum:
    python train_mimic_cxr.py --phase sg_only    --partition_file data/partitions/sg_train_A.json    ...
    python train_mimic_cxr.py --phase alignment  --partition_file data/partitions/alignment_B.json   ...
    python train_mimic_cxr.py --phase pretrain   --partition_file data/partitions/pretrain_B.json    ...
    python train_mimic_cxr.py --phase finetune   --partition_file data/partitions/finetune_A.json    ...

Verify with audit_dataset.py afterwards — the Stage4 ↔ Stage1/2/3 Jaccard
overlap should drop to ~0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("partition")


def stable_hash_fraction(key: str, seed: int) -> float:
    """Deterministic [0, 1) hash so the same study_id ends up in the same
    split across runs and across machines."""
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    # Take first 8 bytes as uint64, divide by 2^64
    return int.from_bytes(h[:8], "big") / 2**64


def collect_studies_by_grade(mimic_qa_path: Path, split: str = "train") -> Dict[str, Set[str]]:
    """Walk the MIMIC-Ext-CXR-QBA QA files and bucket study_ids by their
    extraction_quality grade. Returns {'A': {sid, ...}, 'B': {...}, ...}."""
    qa_root = mimic_qa_path / split
    if not qa_root.exists():
        raise FileNotFoundError(f"QA root not found: {qa_root}")

    buckets: Dict[str, Set[str]] = defaultdict(set)
    study_to_grades: Dict[str, Counter] = defaultdict(Counter)

    qa_files = list(qa_root.glob("**/qa.json"))
    log.info(f"scanning {len(qa_files)} qa.json files under {qa_root}")
    for i, p in enumerate(qa_files):
        try:
            qa = json.loads(p.read_text())
        except Exception as e:
            log.debug(f"could not parse {p}: {e}")
            continue
        # Try both common schemas
        sid = qa.get("study_id") or p.parent.name.lstrip("s")
        questions = qa.get("questions", []) or qa.get("qa_pairs", [])
        for q in questions:
            grade = (q.get("extraction_quality") or "").upper().strip()
            if grade in {"A", "B", "C"}:
                study_to_grades[str(sid)][grade] += 1
        if (i + 1) % 5000 == 0:
            log.info(f"  scanned {i+1}/{len(qa_files)}")

    # A study is "grade X" if at least one of its questions has grade X.
    # If a study has both A and B questions, prefer the higher grade (A).
    # Studies with no grade or only grade C → discarded as untrusted.
    for sid, counts in study_to_grades.items():
        if counts.get("A", 0) > 0:
            buckets["A"].add(sid)
        elif counts.get("B", 0) > 0:
            buckets["B"].add(sid)
        # C-only studies fall through (excluded)

    log.info(f"grade A studies: {len(buckets['A']):,}")
    log.info(f"grade B studies: {len(buckets['B']):,}")
    log.info(f"discarded (C-only or ungraded): "
             f"{len(study_to_grades) - len(buckets['A']) - len(buckets['B']):,}")
    return dict(buckets)


def partition(
    studies_by_grade: Dict[str, Set[str]],
    seed: int,
    finetune_holdout_frac: float = 0.10,
    alignment_frac_of_B: float = 0.40,
) -> Dict[str, List[str]]:
    """Apply the partition rule. Returns {pool_name: sorted study_ids}."""
    a = sorted(studies_by_grade.get("A", set()))
    b = sorted(studies_by_grade.get("B", set()))

    # A-grade split: 90% sg_train_A (Stage 1), 10% finetune_A (Stage 4 hold-out)
    sg_train_A: List[str] = []
    finetune_A: List[str] = []
    for sid in a:
        bucket = stable_hash_fraction(sid, seed)
        if bucket < finetune_holdout_frac:
            finetune_A.append(sid)
        else:
            sg_train_A.append(sid)

    # B-grade split: 40% alignment, 60% pretrain
    alignment_B: List[str] = []
    pretrain_B: List[str] = []
    for sid in b:
        bucket = stable_hash_fraction(sid, seed + 1)
        if bucket < alignment_frac_of_B:
            alignment_B.append(sid)
        else:
            pretrain_B.append(sid)

    # Stages 2 and 3 can ALSO see the sg_train_A studies — that's not leak,
    # because finetune_A is held out from both. Adding the clean A-grade
    # samples to the pretrain pool gives stronger supervision for the
    # alignment + full-pipeline phases.
    alignment_pool = sorted(set(alignment_B) | set(sg_train_A))
    pretrain_pool  = sorted(set(pretrain_B)  | set(sg_train_A))

    return {
        "sg_train_A":   sg_train_A,    # Stage 1 only
        "alignment":    alignment_pool, # Stage 2: B(40%) + A(90%)
        "pretrain":     pretrain_pool,  # Stage 3: B(60%) + A(90%)
        "finetune_A":   finetune_A,     # Stage 4 only — held out from 1/2/3
    }


def write_partitions(partitions: Dict[str, List[str]], output_dir: Path,
                     seed: int, finetune_holdout_frac: float,
                     alignment_frac_of_B: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, sids in partitions.items():
        path = output_dir / f"{name}.json"
        payload = {
            "stage": name,
            "seed": seed,
            "finetune_holdout_frac": finetune_holdout_frac,
            "alignment_frac_of_B": alignment_frac_of_B,
            "n_studies": len(sids),
            "study_ids": sids,
        }
        path.write_text(json.dumps(payload, indent=2))
        log.info(f"wrote {path}  ({len(sids):,} studies)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mimic_qa_path", type=Path, default=Path("data/mimic-ext-cxr-qba"))
    ap.add_argument("--output_dir", type=Path, default=Path("data/partitions"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--finetune_holdout_frac", type=float, default=0.10,
                    help="Fraction of A-grade studies reserved for Stage 4 (default 10%%)")
    ap.add_argument("--alignment_frac_of_B", type=float, default=0.40,
                    help="Fraction of B-only studies for Stage 2 (rest goes to Stage 3)")
    ap.add_argument("--split", default="train",
                    help="Which MIMIC split to partition (default: train)")
    args = ap.parse_args()

    studies_by_grade = collect_studies_by_grade(args.mimic_qa_path, args.split)
    partitions = partition(
        studies_by_grade,
        seed=args.seed,
        finetune_holdout_frac=args.finetune_holdout_frac,
        alignment_frac_of_B=args.alignment_frac_of_B,
    )
    write_partitions(
        partitions, args.output_dir,
        seed=args.seed,
        finetune_holdout_frac=args.finetune_holdout_frac,
        alignment_frac_of_B=args.alignment_frac_of_B,
    )

    # Sanity-print: verify the disjointness invariant
    sg, ft = set(partitions["sg_train_A"]), set(partitions["finetune_A"])
    aln, pre = set(partitions["alignment"]), set(partitions["pretrain"])
    print()
    print("Disjointness check (should ALL be 0):")
    print(f"  sg_train_A ∩ finetune_A : {len(sg & ft)}")
    print(f"  alignment  ∩ finetune_A : {len(aln & ft)}")
    print(f"  pretrain   ∩ finetune_A : {len(pre & ft)}")
    print()
    print("Inclusion sanity (sg_train_A studies should appear in alignment + pretrain):")
    print(f"  sg_train_A ⊆ alignment  : {sg.issubset(aln)}")
    print(f"  sg_train_A ⊆ pretrain   : {sg.issubset(pre)}")
    print()
    print(f"Next: run scripts/audit_dataset.py and confirm Stage4 ↔ Stage1/2/3 "
          f"Jaccard overlap drops to ~0.")


if __name__ == "__main__":
    main()
