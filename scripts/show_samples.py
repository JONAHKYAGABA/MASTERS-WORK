#!/usr/bin/env python3
"""
scripts/show_samples.py — minimal Q + scene-graph viewer.

For each of the first N samples in the train split, prints:
  - the question text
  - the reference answer (one line truncated)
  - the GT scene graph as a small table (entity, region, polarity, bbox)

That's it. No statistics, no aggregation, just raw data so you can eyeball it.

Usage:
    .venv/bin/python scripts/show_samples.py
    .venv/bin/python scripts/show_samples.py --n 20 --quality_grade A
    .venv/bin/python scripts/show_samples.py --min_objects 5   # skip empties
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_vocab(qa_path: Path):
    """Return (region_names, entity_names) so we can resolve IDs to text."""
    for p in (qa_path / "metadata" / "dataset_info.json",
              qa_path / "dataset_info.json"):
        if p.exists():
            try:
                info = json.loads(p.read_text())
                regs = info.get("regions") or info.get("region_names") or []
                ents = info.get("finding_entities") or info.get("entity_names") or []
                return regs, ents
            except Exception:
                continue
    return [], []


def name_or_id(names: List[str], idx: int) -> str:
    return names[idx] if 0 <= idx < len(names) else f"id_{idx}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_cxr_path", default="data/mimic-cxr-jpg")
    ap.add_argument("--mimic_qa_path",  default="data/mimic-ext-cxr-qba")
    ap.add_argument("--quality_grade",  default="all", choices=["A", "B", "all"])
    ap.add_argument("--split",          default="train",
                    choices=["train", "validate", "test"])
    ap.add_argument("--n",              type=int, default=10,
                    help="How many samples to print")
    ap.add_argument("--max_load",       type=int, default=2000,
                    help="Upper bound on how many to load while filtering "
                         "(useful with --min_objects)")
    ap.add_argument("--min_objects",    type=int, default=0,
                    help="Skip samples whose scene graph has fewer than this "
                         "many objects (e.g. 5 to filter out the empties)")
    ap.add_argument("--max_objects_show", type=int, default=15,
                    help="Truncate very long scene graphs in the printout")
    args = ap.parse_args()

    from data.mimic_cxr_dataset import MIMICCXRVQADataset

    region_names, entity_names = load_vocab(Path(args.mimic_qa_path))

    ds = MIMICCXRVQADataset(
        mimic_cxr_path=args.mimic_cxr_path,
        mimic_qa_path=args.mimic_qa_path,
        split=args.split,
        quality_grade=args.quality_grade,
        max_samples=args.max_load,
        use_cache=True,
    )

    shown = 0
    for i in range(len(ds)):
        try:
            item = ds[i]
        except Exception as e:
            print(f"[skip] item {i}: {e}")
            continue

        ents = item.get("gt_sg_entities")
        regs = item.get("gt_sg_regions")
        bbs  = item.get("gt_sg_bboxes")
        pos  = item.get("gt_sg_positiveness")
        n_objs = len(ents) if ents is not None else 0
        if n_objs < args.min_objects:
            continue

        sid_short = f"s{item.get('study_id')}"
        qt = item.get("question_types") or "?"
        question = (item.get("question_text") or "").strip()
        ref_ans  = (item.get("reference_answer") or "").strip()

        print()
        print(f"━━━ Sample {shown+1}  [{sid_short}]  question_type={qt}  "
              f"sg_objects={n_objs} ━━━")
        print(f"Q: {question}")
        print(f"A: {ref_ans[:300]}{'…' if len(ref_ans) > 300 else ''}")

        if n_objs == 0:
            print("(scene graph is empty for this sample)")
        else:
            print(f"Scene graph ({n_objs} object{'s' if n_objs != 1 else ''}):")
            print(f"  {'#':>2} | {'entity':<26} | {'region':<26} | pol  | bbox (x1,y1,x2,y2)")
            print(f"  {'-'*2:>2}-+-{'-'*26:<26}-+-{'-'*26:<26}-+-{'-'*4}-+-{'-'*30}")
            show_n = min(n_objs, args.max_objects_show)
            for j in range(show_n):
                e_name = name_or_id(entity_names, int(ents[j]))[:26]
                r_name = (name_or_id(region_names, int(regs[j]))[:26]
                          if regs is not None and j < len(regs) else "—")
                p = "+pos" if (pos is not None and j < len(pos) and int(pos[j]) == 1) else " neg"
                if bbs is not None and j < len(bbs):
                    b = bbs[j]
                    bbox_str = f"[{float(b[0]):.2f},{float(b[1]):.2f},{float(b[2]):.2f},{float(b[3]):.2f}]"
                else:
                    bbox_str = "—"
                print(f"  {j+1:>2} | {e_name:<26} | {r_name:<26} | {p} | {bbox_str}")
            if n_objs > show_n:
                print(f"  ...({n_objs - show_n} more truncated; --max_objects_show to see)")

        shown += 1
        if shown >= args.n:
            break

    print()
    print(f"Printed {shown} sample(s) "
          f"(quality_grade={args.quality_grade}, split={args.split}, "
          f"min_objects={args.min_objects}).")


if __name__ == "__main__":
    main()
