#!/usr/bin/env python3
"""
scripts/audit_dataset.py — diagnose the training data BEFORE scaling to 500K.

What this script answers (the questions you actually need to know):

1. How many samples does each stage filter resolve to?
2. What fraction of samples have usable GT scene graphs / GT bboxes /
   grounding bboxes / chexpert labels?
3. What does the GT scene-graph distribution look like — are entities and
   regions balanced, or are a few labels dominating? (If "prosthesis" is
   60% of all entities in the training set, no wonder the model collapses
   to predicting "prosthesis" for everything.)
4. What's the SPATIAL distribution of GT bboxes — are they all near image
   center, or do they spread? (If the data is centre-biased, the
   centerness-top-K detection head will be too.)
5. How does the per-stage filter (quality_grade A vs B vs all) change
   those distributions? (i.e. does fine-tune data look very different from
   pre-train data?)
6. CRITICAL: how much do the stages OVERLAP at the study level? If Stage 4
   finetune's val_loss is computed on studies the model already trained on
   in Stage 3 pretrain, the reported numbers are inflated by memorisation.
   This script reports per-stage study_id sets + pairwise Jaccard overlaps
   so you can see leakage directly.

What it does NOT do:
  - Load the model. Pure dataset audit, no GPU needed.
  - Mutate anything on disk. Read-only.
  - Stream the full train split — by default samples ``--per_stage``
    items per stage (5000 is a stable default; bump if you want tighter
    statistics for the full 500K analysis).

Usage:
    python scripts/audit_dataset.py
    python scripts/audit_dataset.py --per_stage 20000 --top_k 40
    python scripts/audit_dataset.py --output_md audit_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("audit")

# Stage filter mapping — mirrors what train_mimic_cxr.py + the curriculum
# config files use. Update if you change the curriculum.
STAGE_FILTERS: Dict[str, Dict[str, Any]] = {
    "stage1_sg_only":   {"quality_grade": "all", "config_yaml": "pretrain_config.yaml"},
    "stage2_alignment": {"quality_grade": "all", "config_yaml": "pretrain_config.yaml"},
    "stage3_pretrain":  {"quality_grade": "B",   "config_yaml": "pretrain_config.yaml"},
    "stage4_finetune":  {"quality_grade": "A",   "config_yaml": "finetune_config.yaml"},
}


# ---------------------------------------------------------------------------
# Stats accumulators
# ---------------------------------------------------------------------------

class StageStats:
    """Roll up per-sample observations into a single stage summary."""

    def __init__(self, name: str):
        self.name = name
        self.n_total = 0
        self.n_with_sg_objects = 0
        self.n_with_sg_bboxes  = 0
        self.n_with_grounding_bbox = 0
        self.n_with_chexpert = 0
        self.objects_per_sample: List[int] = []
        self.entity_counts:    Counter = Counter()
        self.region_counts:    Counter = Counter()
        self.pair_counts:      Counter = Counter()
        self.positiveness_counts: Counter = Counter()
        self.bbox_areas:       List[float] = []  # normalised in [0,1]
        self.bbox_centers:     List[Tuple[float, float]] = []
        self.grounding_areas:  List[float] = []
        self.grounding_centers: List[Tuple[float, float]] = []
        self.question_types:   Counter = Counter()
        # Study-level identifier set — used for cross-stage overlap (data-leak) detection
        self.study_ids: set = set()
        # Examples to render in the report — we keep a stratified mix so the
        # user can see a variety of question types, scene-graph sizes, and
        # finding polarities (not just the most common boring sample).
        self.examples: List[Dict[str, Any]] = []
        # Examples per "bucket" we want to fill (small/medium/large SG,
        # positive/negative). Keep 1-2 per bucket = 6-12 total.
        self._examples_per_bucket = 2
        self._bucket_counts: Counter = Counter()

    def _bucket_for(self, item: Dict[str, Any]) -> str:
        """Stratify examples so we get variety, not 10 of the same thing."""
        n_objs = len(item.get("gt_sg_entities") or [])
        pos = item.get("gt_sg_positiveness")
        any_pos = bool(pos is not None and any(int(p) == 1 for p in pos))
        size = "empty" if n_objs == 0 else "small" if n_objs <= 3 else "medium" if n_objs <= 10 else "large"
        polarity = "pos" if any_pos else "neg"
        return f"{size}_{polarity}"

    def _maybe_keep_example(self, item: Dict[str, Any]):
        """Stash a copy of the sample if its bucket isn't full yet."""
        bucket = self._bucket_for(item)
        if self._bucket_counts[bucket] >= self._examples_per_bucket:
            return
        # Make a serialisable snapshot (no tensors, no numpy)
        def _safe(v):
            if v is None:
                return None
            if hasattr(v, "tolist"):
                return v.tolist()
            return v
        snap = {
            "bucket": bucket,
            "subject_id": item.get("subject_id"),
            "study_id":   item.get("study_id"),
            "dicom_id":   item.get("dicom_id"),
            "question":      item.get("question_text"),
            "question_type": item.get("question_types"),
            "reference_answer": item.get("reference_answer"),
            "structured_answer_text": (
                str(item.get("structured_answer_text") or "")[:600]
            ),
            "n_sg_objects": len(item.get("gt_sg_entities") or []),
            "gt_sg_entities":     _safe(item.get("gt_sg_entities")),
            "gt_sg_regions":      _safe(item.get("gt_sg_regions")),
            "gt_sg_bboxes":       _safe(item.get("gt_sg_bboxes")),
            "gt_sg_positiveness": _safe(item.get("gt_sg_positiveness")),
            "grounding_bbox":  _safe(item.get("grounding_bbox")),
            "grounding_valid": _safe(item.get("grounding_valid")),
            "answer_entities": item.get("answer_entities"),
            "answer_regions":  item.get("answer_regions"),
            "answer_positiveness": item.get("answer_positiveness"),
            "image_path_hint": f"p{str(item.get('subject_id'))[:2]}/p{item.get('subject_id')}/s{item.get('study_id')}",
        }
        self.examples.append(snap)
        self._bucket_counts[bucket] += 1

    def add(self, item: Dict[str, Any]):
        self.n_total += 1
        sid = item.get("study_id")
        if sid is not None:
            self.study_ids.add(str(sid))
        self._maybe_keep_example(item)

        # Scene-graph GT counts
        gt_entities = item.get("gt_sg_entities")
        gt_regions  = item.get("gt_sg_regions")
        gt_bboxes   = item.get("gt_sg_bboxes")
        gt_pos      = item.get("gt_sg_positiveness")

        n_objs = 0
        if gt_entities is not None and len(gt_entities) > 0:
            n_objs = len(gt_entities)
            self.n_with_sg_objects += 1
        self.objects_per_sample.append(n_objs)

        if gt_bboxes is not None and len(gt_bboxes) > 0:
            self.n_with_sg_bboxes += 1
            for i in range(min(n_objs, len(gt_bboxes))):
                b = gt_bboxes[i]
                if len(b) == 4:
                    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    w, h = max(0, x2 - x1), max(0, y2 - y1)
                    self.bbox_areas.append(w * h)
                    self.bbox_centers.append(((x1 + x2) / 2, (y1 + y2) / 2))

        if gt_entities is not None and gt_regions is not None:
            n = min(len(gt_entities), len(gt_regions))
            for i in range(n):
                e_id = int(gt_entities[i])
                r_id = int(gt_regions[i])
                self.entity_counts[e_id] += 1
                self.region_counts[r_id] += 1
                self.pair_counts[(e_id, r_id)] += 1
                if gt_pos is not None and i < len(gt_pos):
                    self.positiveness_counts[int(gt_pos[i])] += 1

        # Grounding bbox (the head's primary supervision)
        gb = item.get("grounding_bbox")
        gv = item.get("grounding_valid")
        if gb is not None and (gv is None or float(gv) > 0.5):
            try:
                x1, y1, x2, y2 = (float(v) for v in gb)
                w, h = max(0, x2 - x1), max(0, y2 - y1)
                if w * h > 0:
                    self.n_with_grounding_bbox += 1
                    self.grounding_areas.append(w * h)
                    self.grounding_centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
            except Exception:
                pass

        # CheXpert mask
        cm = item.get("chexpert_mask")
        if cm is not None and float(np.array(cm).sum()) > 0:
            self.n_with_chexpert += 1

        # Question type
        qt = item.get("question_types")
        if qt is not None:
            self.question_types[str(qt)] += 1

    def to_dict(self, entity_names: List[str], region_names: List[str],
                top_k: int = 30) -> Dict[str, Any]:
        def name_or_id(names: List[str], idx: int) -> str:
            return names[idx] if 0 <= idx < len(names) else f"id_{idx}"

        npa = np.array(self.objects_per_sample, dtype=np.int32)
        coverage = {
            "samples_total": self.n_total,
            "with_sg_objects":    f"{self.n_with_sg_objects} ({100*self.n_with_sg_objects/max(1,self.n_total):.1f}%)",
            "with_sg_bboxes":     f"{self.n_with_sg_bboxes} ({100*self.n_with_sg_bboxes/max(1,self.n_total):.1f}%)",
            "with_grounding_bbox": f"{self.n_with_grounding_bbox} ({100*self.n_with_grounding_bbox/max(1,self.n_total):.1f}%)",
            "with_chexpert_labels": f"{self.n_with_chexpert} ({100*self.n_with_chexpert/max(1,self.n_total):.1f}%)",
        }

        objects_dist = {}
        if npa.size:
            objects_dist = {
                "mean":   float(npa.mean()),
                "median": float(np.median(npa)),
                "p10":    float(np.percentile(npa, 10)),
                "p90":    float(np.percentile(npa, 90)),
                "min":    int(npa.min()),
                "max":    int(npa.max()),
            }

        # Areas + centres
        bbox_dist = {}
        if self.bbox_areas:
            ba = np.array(self.bbox_areas)
            bbox_dist = {
                "n_bboxes":     int(ba.size),
                "area_mean":    float(ba.mean()),
                "area_median":  float(np.median(ba)),
                "area_p10":     float(np.percentile(ba, 10)),
                "area_p90":     float(np.percentile(ba, 90)),
                "tiny_pct (<0.005)":  f"{100*(ba<0.005).mean():.1f}%",
                "small_pct (<0.05)":  f"{100*(ba<0.05).mean():.1f}%",
                "large_pct (>0.3)":   f"{100*(ba>0.3).mean():.1f}%",
            }
        center_grid = _grid_distribution(self.bbox_centers) if self.bbox_centers else None
        grounding_center_grid = _grid_distribution(self.grounding_centers) if self.grounding_centers else None

        # Top-k entities, regions, pairs
        total_objs = sum(self.entity_counts.values())
        top_entities = [
            {"name": name_or_id(entity_names, eid), "count": c,
             "pct": f"{100*c/max(1,total_objs):.2f}%"}
            for eid, c in self.entity_counts.most_common(top_k)
        ]
        total_regs = sum(self.region_counts.values())
        top_regions = [
            {"name": name_or_id(region_names, rid), "count": c,
             "pct": f"{100*c/max(1,total_regs):.2f}%"}
            for rid, c in self.region_counts.most_common(top_k)
        ]
        total_pairs = sum(self.pair_counts.values())
        top_pairs = [
            {"entity": name_or_id(entity_names, eid),
             "region": name_or_id(region_names, rid),
             "count": c,
             "pct": f"{100*c/max(1,total_pairs):.2f}%"}
            for (eid, rid), c in self.pair_counts.most_common(top_k)
        ]

        # Concentration metrics (top-1 / top-5 / top-10 share)
        def share(counter: Counter, k: int) -> float:
            total = sum(counter.values()) or 1
            return sum(c for _, c in counter.most_common(k)) / total

        concentration = {
            "entity_top1_share":   f"{100*share(self.entity_counts, 1):.1f}%",
            "entity_top5_share":   f"{100*share(self.entity_counts, 5):.1f}%",
            "entity_top10_share":  f"{100*share(self.entity_counts, 10):.1f}%",
            "region_top1_share":   f"{100*share(self.region_counts, 1):.1f}%",
            "region_top5_share":   f"{100*share(self.region_counts, 5):.1f}%",
            "region_top10_share":  f"{100*share(self.region_counts, 10):.1f}%",
            "unique_entities_seen": len(self.entity_counts),
            "unique_regions_seen":  len(self.region_counts),
            "unique_pairs_seen":    len(self.pair_counts),
        }

        # Resolve entity/region IDs to names in the stashed examples so the
        # markdown report shows human-readable rows.
        def _enrich(ex: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(ex)
            ents = ex.get("gt_sg_entities") or []
            regs = ex.get("gt_sg_regions")  or []
            boxes = ex.get("gt_sg_bboxes")  or []
            pos   = ex.get("gt_sg_positiveness") or []
            rows = []
            for i in range(len(ents)):
                rows.append({
                    "entity_id": int(ents[i]),
                    "entity":    name_or_id(entity_names, int(ents[i])),
                    "region_id": int(regs[i]) if i < len(regs) else None,
                    "region":    name_or_id(region_names, int(regs[i])) if i < len(regs) else None,
                    "bbox":      [round(float(v), 3) for v in boxes[i]] if i < len(boxes) and boxes[i] is not None else None,
                    "polarity":  int(pos[i]) if i < len(pos) else None,
                })
            out["sg_resolved"] = rows
            return out

        return {
            "stage": self.name,
            "coverage": coverage,
            "objects_per_sample": objects_dist,
            "bbox_geometry": bbox_dist,
            "bbox_center_grid_3x3": center_grid,
            "grounding_center_grid_3x3": grounding_center_grid,
            "concentration": concentration,
            "polarity_distribution": dict(self.positiveness_counts),
            "top_entities":     top_entities,
            "top_regions":      top_regions,
            "top_entity_region_pairs": top_pairs,
            "question_type_distribution": dict(self.question_types.most_common(10)),
            "examples": [_enrich(ex) for ex in self.examples],
        }


def _grid_distribution(centers: List[Tuple[float, float]]) -> Dict[str, str]:
    """Bin (x, y) centres into a 3x3 grid for at-a-glance spatial bias."""
    grid = np.zeros((3, 3), dtype=np.int64)
    for cx, cy in centers:
        col = min(2, max(0, int(cx * 3)))
        row = min(2, max(0, int(cy * 3)))
        grid[row, col] += 1
    total = grid.sum() or 1
    rows = []
    for r in range(3):
        row_pct = [f"{100*grid[r,c]/total:5.1f}%" for c in range(3)]
        rows.append(" | ".join(row_pct))
    return {
        "row_top":    rows[0],
        "row_middle": rows[1],
        "row_bottom": rows[2],
        "note": "L | M | R columns. Healthy data spreads across all 9 cells; "
                "an undertrained detector / centre-biased data shows >50% in middle cell.",
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def render_stage_md(d: Dict[str, Any], top_k_show: int = 20) -> str:
    parts = []
    parts.append(f"## {d['stage']}\n")

    parts.append("### Coverage\n")
    for k, v in d["coverage"].items():
        parts.append(f"- **{k}**: {v}")
    parts.append("")

    parts.append("### Objects per sample\n")
    if d["objects_per_sample"]:
        ops = d["objects_per_sample"]
        parts.append(f"- mean={ops['mean']:.1f}, median={ops['median']:.0f}, "
                     f"p10={ops['p10']:.0f}, p90={ops['p90']:.0f}, "
                     f"min={ops['min']}, max={ops['max']}")
    parts.append("")

    parts.append("### Bbox geometry\n")
    for k, v in d.get("bbox_geometry", {}).items():
        if isinstance(v, float):
            parts.append(f"- {k}: {v:.4f}")
        else:
            parts.append(f"- {k}: {v}")
    parts.append("")

    parts.append("### GT bbox center 3x3 grid (% of bboxes)\n")
    g = d.get("bbox_center_grid_3x3")
    if g:
        parts.append("```")
        parts.append(g["row_top"])
        parts.append(g["row_middle"])
        parts.append(g["row_bottom"])
        parts.append("```")
        parts.append(f"_{g['note']}_")
    parts.append("")

    parts.append("### Class-concentration warning signs\n")
    parts.append("> **Red flag**: top-1 share > 30% means one label dominates → trained model will collapse to it.\n")
    for k, v in d["concentration"].items():
        parts.append(f"- {k}: {v}")
    parts.append("")

    parts.append("### Polarity distribution\n")
    for k, v in d["polarity_distribution"].items():
        parts.append(f"- polarity={k}: {v}")
    parts.append("")

    parts.append(f"### Top {top_k_show} entities\n")
    parts.append("| # | entity | count | % of objects |")
    parts.append("|---|---|---|---|")
    for i, row in enumerate(d["top_entities"][:top_k_show], 1):
        parts.append(f"| {i} | {row['name']} | {row['count']} | {row['pct']} |")
    parts.append("")

    parts.append(f"### Top {top_k_show} regions\n")
    parts.append("| # | region | count | % of objects |")
    parts.append("|---|---|---|---|")
    for i, row in enumerate(d["top_regions"][:top_k_show], 1):
        parts.append(f"| {i} | {row['name']} | {row['count']} | {row['pct']} |")
    parts.append("")

    parts.append(f"### Top {top_k_show} (entity, region) pairs\n")
    parts.append("| # | entity | region | count | % of pairs |")
    parts.append("|---|---|---|---|---|")
    for i, row in enumerate(d["top_entity_region_pairs"][:top_k_show], 1):
        parts.append(f"| {i} | {row['entity']} | {row['region']} | {row['count']} | {row['pct']} |")
    parts.append("")

    # =====================================================
    # Example samples — what the raw data actually looks like
    # =====================================================
    examples = d.get("examples", [])
    parts.append(f"### Example samples ({len(examples)} stratified across "
                 "scene-graph size + polarity)\n")
    if not examples:
        parts.append("_(no examples captured)_\n")
    for i, ex in enumerate(examples, 1):
        parts.append(f"#### Example {i} — bucket: `{ex.get('bucket', '?')}` "
                     f"(study `{ex.get('image_path_hint', '?')}`)")
        parts.append("")
        parts.append(f"- **Question** ({ex.get('question_type', '?')}): "
                     f"_{(ex.get('question') or '').strip()[:300]}_")
        ref_ans = (ex.get('reference_answer') or '').strip()
        parts.append(f"- **Reference answer**: _{ref_ans[:400] or '(empty)'}_")
        ans_ents = ex.get('answer_entities') or []
        ans_regs = ex.get('answer_regions') or []
        ans_pos  = ex.get('answer_positiveness') or ''
        if ans_ents or ans_regs:
            parts.append(f"- **Answer entities**: `{ans_ents}`")
            parts.append(f"- **Answer regions**:  `{ans_regs}`")
            parts.append(f"- **Answer polarity**: `{ans_pos}`")
        gb = ex.get('grounding_bbox')
        gv = ex.get('grounding_valid')
        if gb is not None:
            valid_str = "valid" if (gv is None or float(gv) > 0.5) else "INVALID"
            parts.append(f"- **Grounding bbox**: `{[round(float(v), 3) for v in gb]}` ({valid_str})")
        n_sg = ex.get('n_sg_objects', 0)
        parts.append(f"- **Scene graph**: {n_sg} object{'s' if n_sg != 1 else ''}")
        rows = ex.get('sg_resolved', [])
        if rows:
            parts.append("")
            parts.append("  | # | entity | region | polarity | bbox |")
            parts.append("  |---|---|---|---|---|")
            for j, r in enumerate(rows[:15], 1):
                pol = '+pos' if r.get('polarity') == 1 else 'neg'
                parts.append(f"  | {j} | `{r.get('entity')}` | `{r.get('region')}` "
                             f"| {pol} | `{r.get('bbox')}` |")
            if len(rows) > 15:
                parts.append(f"  | ... | _({len(rows) - 15} more truncated)_ | | | |")
        sat = ex.get('structured_answer_text', '')
        if sat:
            parts.append("")
            parts.append("  <details><summary>Structured answer template (what Qwen "
                         "LM sees as supervision)</summary>")
            parts.append("")
            parts.append("  ```")
            parts.append("  " + sat[:600].replace("\n", " "))
            parts.append("  ```")
            parts.append("  </details>")
        parts.append("")
    parts.append("")

    return "\n".join(parts)


def render_comparison_md(all_stages: Dict[str, Dict[str, Any]]) -> str:
    """One-table summary across all stages — useful before scaling to 500K."""
    parts = []
    parts.append("# Dataset audit — cross-stage comparison\n")
    parts.append("| metric | " + " | ".join(all_stages.keys()) + " |")
    parts.append("|---" + "|---" * len(all_stages) + "|")

    def row(label: str, getter):
        cells = [getter(d) for d in all_stages.values()]
        parts.append(f"| {label} | " + " | ".join(str(c) for c in cells) + " |")

    row("samples_total",          lambda d: d["coverage"]["samples_total"])
    row("with_sg_objects",        lambda d: d["coverage"]["with_sg_objects"])
    row("with_sg_bboxes",         lambda d: d["coverage"]["with_sg_bboxes"])
    row("with_grounding_bbox",    lambda d: d["coverage"]["with_grounding_bbox"])
    row("with_chexpert_labels",   lambda d: d["coverage"]["with_chexpert_labels"])
    row("avg objects/sample",     lambda d: f"{d['objects_per_sample'].get('mean', 0):.1f}")
    row("median objects/sample",  lambda d: f"{d['objects_per_sample'].get('median', 0):.0f}")
    row("unique entities seen",   lambda d: d["concentration"]["unique_entities_seen"])
    row("unique regions seen",    lambda d: d["concentration"]["unique_regions_seen"])
    row("entity top-1 share",     lambda d: d["concentration"]["entity_top1_share"])
    row("entity top-5 share",     lambda d: d["concentration"]["entity_top5_share"])
    row("region top-1 share",     lambda d: d["concentration"]["region_top1_share"])
    row("region top-5 share",     lambda d: d["concentration"]["region_top5_share"])
    row("tiny bbox % (<0.005)",   lambda d: d["bbox_geometry"].get("tiny_pct (<0.005)", "—"))
    row("small bbox % (<0.05)",   lambda d: d["bbox_geometry"].get("small_pct (<0.05)", "—"))
    row("large bbox % (>0.3)",    lambda d: d["bbox_geometry"].get("large_pct (>0.3)", "—"))
    parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def audit_one_stage(name: str, filters: Dict[str, Any], per_stage: int,
                    mimic_cxr_path: str, mimic_qa_path: str,
                    top_k: int, examples_per_bucket: int = 2) -> Tuple[Dict[str, Any], set]:
    """Returns (stats_dict, study_id_set) — the set is used for cross-stage overlap."""
    from data.mimic_cxr_dataset import MIMICCXRVQADataset

    log.info(f"=== {name}  (quality_grade={filters['quality_grade']}, max_samples={per_stage}) ===")
    ds = MIMICCXRVQADataset(
        mimic_cxr_path=mimic_cxr_path,
        mimic_qa_path=mimic_qa_path,
        split="train",
        quality_grade=filters["quality_grade"],
        max_samples=per_stage,
        use_cache=True,
    )
    log.info(f"  loaded {len(ds)} samples")

    # Vocab for human-readable names
    region_names: List[str] = []
    entity_names: List[str] = []
    for p in (
        Path(mimic_qa_path) / "metadata" / "dataset_info.json",
        Path(mimic_qa_path) / "dataset_info.json",
    ):
        if p.exists():
            try:
                info = json.loads(p.read_text())
                region_names = info.get("regions") or info.get("region_names") or []
                entity_names = info.get("finding_entities") or info.get("entity_names") or []
                break
            except Exception:
                pass

    stats = StageStats(name)
    stats._examples_per_bucket = examples_per_bucket
    for i in range(len(ds)):
        try:
            item = ds[i]
        except Exception as e:
            log.debug(f"  skipped sample {i}: {e}")
            continue
        stats.add(item)
        if (i + 1) % 500 == 0:
            log.info(f"  processed {i+1}/{len(ds)}")
    log.info(f"  done — n_total={stats.n_total}  unique_studies={len(stats.study_ids)}  "
             f"examples_kept={len(stats.examples)}")

    return stats.to_dict(entity_names, region_names, top_k=top_k), stats.study_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mimic_cxr_path", default="data/mimic-cxr-jpg")
    ap.add_argument("--mimic_qa_path",  default="data/mimic-ext-cxr-qba")
    ap.add_argument("--per_stage",  type=int, default=5000,
                    help="How many samples to inspect PER stage. Default 5000 "
                         "(fast). Use 20000+ for tighter stats before 500K runs.")
    ap.add_argument("--top_k", type=int, default=30,
                    help="Number of top entities/regions/pairs to keep per stage.")
    ap.add_argument("--top_k_show", type=int, default=20,
                    help="Number to render in the markdown tables.")
    ap.add_argument("--examples_per_bucket", type=int, default=2,
                    help="Per stage we stratify examples into buckets "
                         "(small/medium/large × pos/neg). This controls how many "
                         "items per bucket to keep (default 2 → up to ~12 per stage).")
    ap.add_argument("--output_md",  default="docs/dataset_audit.md",
                    help="Markdown report path.")
    ap.add_argument("--output_json", default="docs/dataset_audit.json",
                    help="Full raw JSON (for downstream tooling).")
    ap.add_argument("--stages", nargs="+", default=list(STAGE_FILTERS.keys()),
                    help="Subset of stages to audit (default: all 4).")
    args = ap.parse_args()

    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    all_stages: Dict[str, Dict[str, Any]] = {}
    stage_study_ids: Dict[str, set] = {}
    for name in args.stages:
        if name not in STAGE_FILTERS:
            log.warning(f"unknown stage {name}, skipping")
            continue
        d, sids = audit_one_stage(
            name, STAGE_FILTERS[name], args.per_stage,
            args.mimic_cxr_path, args.mimic_qa_path, args.top_k,
            examples_per_bucket=args.examples_per_bucket,
        )
        all_stages[name] = d
        stage_study_ids[name] = sids

    # ===== Cross-stage study-id overlap (the data-leak diagnostic) =====
    overlap_md = ["## Cross-stage study_id overlap (DATA-LEAK DIAGNOSTIC)\n"]
    overlap_md.append(
        "> Pairwise Jaccard: |A ∩ B| / |A ∪ B|. "
        "Values near 1.0 mean the two stages are training on the SAME studies "
        "(memorisation, not generalisation). Values near 0 mean disjoint partitions."
    )
    overlap_md.append("> For a clean curriculum: stages 1–3 share data freely "
                      "(curriculum stages on a common pretrain pool), but **Stage 4 "
                      "should be disjoint from Stages 1–3** (held-out finetune set).\n")
    stage_names = list(stage_study_ids.keys())
    overlap_md.append("| | " + " | ".join(stage_names) + " |")
    overlap_md.append("|---" + "|---" * len(stage_names) + "|")
    overlap_summary: Dict[str, Dict[str, float]] = {}
    for a in stage_names:
        row_cells = [f"**{a}**"]
        overlap_summary[a] = {}
        for b in stage_names:
            sa, sb = stage_study_ids[a], stage_study_ids[b]
            if not sa or not sb:
                row_cells.append("—")
                continue
            inter = len(sa & sb)
            union = len(sa | sb)
            jacc = inter / union if union else 0.0
            overlap_summary[a][b] = jacc
            cell = f"{jacc:.2f}" if a != b else "1.00"
            if a != b and jacc > 0.5:
                cell += " ⚠"   # leak warning
            row_cells.append(cell)
        overlap_md.append("| " + " | ".join(row_cells) + " |")
    overlap_md.append("")
    overlap_md.append("**⚠ in any off-diagonal cell = significant overlap = your "
                      "model is being evaluated on data it trained on.**")
    overlap_text = "\n".join(overlap_md)

    # Attach overlap info to the JSON payload too
    payload = {
        "per_stage": all_stages,
        "study_overlap_jaccard": overlap_summary,
    }

    # Persist raw JSON
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info(f"wrote raw JSON: {args.output_json}")

    # Markdown report — order: cross-stage summary, OVERLAP table, then per-stage detail
    md_parts = [render_comparison_md(all_stages), overlap_text]
    for name in all_stages:
        md_parts.append(render_stage_md(all_stages[name], top_k_show=args.top_k_show))
    md = "\n\n".join(md_parts)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md)
    log.info(f"wrote markdown report: {args.output_md}")

    # Also print summary table to stdout for quick read
    print()
    print(render_comparison_md(all_stages))
    print()
    print(overlap_text)
    print()
    print("Red flags to scan for:")
    print("  * top-1 entity share > 30%      → mode collapse risk")
    print("  * tiny bbox % > 50%             → centerness top-K will overfit to nothing")
    print("  * with_sg_objects coverage <70% → sg loss will be zero most of the time")
    print("  * Stage4 ↔ Stage1/2/3 Jaccard > 0.5  → DATA LEAK; finetune metrics inflated")
    print(f"\nFull report: {args.output_md}")
    print(f"Raw JSON:    {args.output_json}")


if __name__ == "__main__":
    main()
