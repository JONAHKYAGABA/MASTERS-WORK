#!/usr/bin/env python3
"""
scripts/dataset_deep_analysis.py — comprehensive MIMIC-CXR + MIMIC-Ext-CXR-QBA
structural analysis.

Designed to answer the *structural* questions before scaling to 500K:

  Section 1  Filesystem inventory     (~30s — counts files, NO content reads)
  Section 2  Scene-graph structure    (samples N scene_graph.json files,
                                       reverse-engineers the data model)
  Section 3  Hypothesis tests         (test: are bboxes really anatomical
                                       defaults? — the central question from
                                       the last conversation)
  Section 4  Q/A patterns             (question types, quality grades,
                                       grounding bbox availability,
                                       answer length distribution)
  Section 5  Cross-reference          (do scene-graph entities match
                                       answer entities? polarity alignment?)
  Section 6  Visual samples           (saves N annotated images so you can
                                       LOOK at what the data really is)
  Section 7  Per-stage filtering      (what each curriculum stage sees,
                                       with study-id Jaccard between stages)

Everything is logged to stdout with timestamps so `tail -f` works.

USAGE (recommended — overnight run):
    cd /root/code/MASTERS-WORK
    mkdir -p logs docs/data_analysis_samples
    setsid nohup .venv/bin/python scripts/dataset_deep_analysis.py \\
        --max_studies 5000 \\
        --max_visual_samples 30 \\
        > logs/data_analysis_$(date +%Y%m%d_%H%M).log 2>&1 < /dev/null &
    tail -f logs/data_analysis_*.log

FAST PREVIEW (~2 min, no images):
    .venv/bin/python scripts/dataset_deep_analysis.py \\
        --max_studies 500 --max_visual_samples 0

FULL CORPUS (~all 227K studies, ~3-6 hours):
    setsid nohup .venv/bin/python scripts/dataset_deep_analysis.py \\
        --max_studies 0 --max_visual_samples 50 \\
        > logs/data_analysis_full_$(date +%Y%m%d_%H%M).log 2>&1 < /dev/null &

Outputs (under docs/):
    data_analysis_report.md      — human-readable
    data_analysis_report.json    — raw data, downstream tooling
    data_analysis_samples/*.png  — annotated images with scene-graph bboxes
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Logging setup — flushes every line so `tail -f` shows progress immediately
# ---------------------------------------------------------------------------

class _UnbufferedHandler(logging.StreamHandler):
    """logging.StreamHandler that flushes after every record (essential under
    nohup since stdout is line-buffered by default — without this you'd see
    the log update only every ~4 KB which makes monitoring useless)."""
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[_UnbufferedHandler(sys.stdout)],
)
log = logging.getLogger("data_analysis")


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Section 1 — filesystem inventory (no content reads, just file counts)
# ---------------------------------------------------------------------------

def section1_filesystem_inventory(mimic_cxr_path: Path, mimic_qa_path: Path) -> Dict[str, Any]:
    log.info("=" * 70)
    log.info("SECTION 1 — filesystem inventory")
    log.info("=" * 70)

    def count_glob(root: Path, pattern: str, label: str) -> int:
        log.info(f"  counting {label} ({root}/{pattern}) ...")
        t0 = time.time()
        n = sum(1 for _ in root.glob(pattern))
        log.info(f"    → {n:,} ({time.time()-t0:.1f}s)")
        return n

    inv: Dict[str, Any] = {}
    if mimic_cxr_path.exists():
        inv["mimic_cxr_jpg"] = {
            "root_exists": True,
            "n_images_jpg": count_glob(mimic_cxr_path, "files/p*/p*/s*/*.jpg",
                                        "MIMIC-CXR-JPG images"),
            "n_studies":   count_glob(mimic_cxr_path, "files/p*/p*/s*",
                                        "study directories"),
            "metadata_csv": (mimic_cxr_path / "mimic-cxr-2.0.0-metadata.csv.gz").exists(),
            "chexpert_csv": (mimic_cxr_path / "mimic-cxr-2.0.0-chexpert.csv.gz").exists(),
        }
    else:
        log.warning(f"  MIMIC-CXR-JPG path does not exist: {mimic_cxr_path}")
        inv["mimic_cxr_jpg"] = {"root_exists": False}

    if mimic_qa_path.exists():
        # Actual layout (discovered 2026-06-10):
        #   data/mimic-ext-cxr-qba/qa/p{XX}/p{patient}/s{study}.qa.json
        #   data/mimic-ext-cxr-qba/scene_data/p{XX}/p{patient}/s{study}.scene_graph.json
        #   data/mimic-ext-cxr-qba/scene_data/p{XX}/p{patient}/s{study}.metadata.json
        # NO train/validate/test directories — files are by patient ID only.
        # Splits must be determined from a manifest (quality_mappings.csv or
        # metadata/dataset_info.json) outside this filesystem walk.
        inv["mimic_ext_cxr_qba"] = {
            "root_exists": True,
            "n_qa":           count_glob(mimic_qa_path, "qa/p*/p*/s*.qa.json",                "*.qa.json"),
            "n_scene_graph":  count_glob(mimic_qa_path, "scene_data/p*/p*/s*.scene_graph.json", "*.scene_graph.json"),
            "n_metadata":     count_glob(mimic_qa_path, "scene_data/p*/p*/s*.metadata.json",  "*.metadata.json"),
            "dataset_info":   (mimic_qa_path / "metadata" / "dataset_info.json").exists(),
            "quality_mappings_csv": (mimic_qa_path / "quality_mappings.csv").exists(),
            "exports_dir":    (mimic_qa_path / "exports").exists(),
        }
    else:
        log.warning(f"  MIMIC-Ext path does not exist: {mimic_qa_path}")
        inv["mimic_ext_cxr_qba"] = {"root_exists": False}

    log.info("filesystem inventory done")
    return inv


# ---------------------------------------------------------------------------
# Section 2 — scene-graph structural analysis
# ---------------------------------------------------------------------------

def section2_scene_graph_structure(mimic_qa_path: Path, max_studies: int,
                                    split: str = "train") -> Tuple[Dict[str, Any], List[Path]]:
    log.info("=" * 70)
    log.info(f"SECTION 2 — scene-graph structure "
             f"(max_studies={max_studies or 'ALL'})")
    log.info("=" * 70)
    # Note: `split` arg is kept for backward compat but ignored — there are
    # no train/validate/test subdirs in the actual layout. Splits are
    # determined elsewhere (manifest / dataset_info.json).
    sg_paths = sorted((mimic_qa_path / "scene_data").glob("p*/p*/s*.scene_graph.json"))
    log.info(f"found {len(sg_paths):,} scene_graph.json files")

    if max_studies > 0 and len(sg_paths) > max_studies:
        random.seed(42)
        sg_paths = random.sample(sg_paths, max_studies)
        log.info(f"  sampling {len(sg_paths):,} for analysis")

    # Reverse-engineer the schema
    top_level_keys: Counter = Counter()
    objects_field_keys: Counter = Counter()    # keys present in each object
    relations_field_keys: Counter = Counter()
    attributes_field_keys: Counter = Counter()
    object_count_dist: List[int] = []
    relation_count_dist: List[int] = []

    schema_examples: List[Dict[str, Any]] = []
    parse_failures = 0

    for i, p in enumerate(sg_paths):
        try:
            sg = json.loads(p.read_text())
        except Exception as e:
            parse_failures += 1
            log.debug(f"  parse failed for {p}: {e}")
            continue

        if isinstance(sg, dict):
            for k in sg.keys():
                top_level_keys[k] += 1
            # Look for 'objects' or 'nodes' or 'entities' — multiple possible
            # schemas across SG datasets.
            for objects_field in ("objects", "nodes", "entities", "attributes"):
                if objects_field in sg and isinstance(sg[objects_field], list):
                    object_count_dist.append(len(sg[objects_field]))
                    for o in sg[objects_field][:3]:
                        if isinstance(o, dict):
                            for k in o.keys():
                                objects_field_keys[k] += 1
                    break
            if "relations" in sg and isinstance(sg["relations"], list):
                relation_count_dist.append(len(sg["relations"]))
                for r in sg["relations"][:3]:
                    if isinstance(r, dict):
                        for k in r.keys():
                            relations_field_keys[k] += 1
            if "attributes" in sg and isinstance(sg["attributes"], list):
                for a in sg["attributes"][:3]:
                    if isinstance(a, dict):
                        for k in a.keys():
                            attributes_field_keys[k] += 1

        if len(schema_examples) < 3:
            schema_examples.append({
                "path": str(p),
                "top_level_keys": list(sg.keys()) if isinstance(sg, dict) else "(not a dict)",
                "preview": json.dumps(sg, indent=2)[:3000] if isinstance(sg, dict) else str(sg)[:3000],
            })

        if (i + 1) % 500 == 0:
            log.info(f"  scanned {i+1:,}/{len(sg_paths):,} scene_graph.json files")

    summary: Dict[str, Any] = {
        "n_scanned":            len(sg_paths) - parse_failures,
        "n_parse_failures":     parse_failures,
        "top_level_keys":       dict(top_level_keys.most_common()),
        "objects_field_keys":   dict(objects_field_keys.most_common()),
        "relations_field_keys": dict(relations_field_keys.most_common()),
        "attributes_field_keys": dict(attributes_field_keys.most_common()),
        "objects_per_sg": _dist(object_count_dist),
        "relations_per_sg": _dist(relation_count_dist),
        "schema_examples": schema_examples,
    }
    log.info(f"  top-level keys (with frequency):")
    for k, c in top_level_keys.most_common(10):
        log.info(f"    {k}: {c}/{len(sg_paths)} ({100*c/max(1,len(sg_paths)):.1f}%)")
    log.info(f"  object-list field keys: {dict(objects_field_keys.most_common(15))}")
    log.info(f"  objects/sg distribution: {summary['objects_per_sg']}")
    log.info(f"  relations/sg distribution: {summary['relations_per_sg']}")
    return summary, sg_paths


# ---------------------------------------------------------------------------
# Section 3 — hypothesis tests
# ---------------------------------------------------------------------------

def section3_bbox_hypothesis(sg_paths: List[Path]) -> Dict[str, Any]:
    """
    Test the hypothesis from our last conversation:
        "All bboxes are anatomical-region defaults, not per-finding."

    Concretely: group bboxes by (region_id, study), check if WITHIN a single
    study there's a 1:1 region→bbox mapping (i.e., all entities in 'lungs'
    share one bbox). If yes, the hypothesis is confirmed → per-finding
    localization isn't learnable from this GT.
    """
    log.info("=" * 70)
    log.info("SECTION 3 — bbox hypothesis tests")
    log.info("=" * 70)

    # H1: region → unique bbox mapping (per study)
    # For each study, group entities by region_id, count distinct bboxes per region.
    # If almost ALL are 1 distinct bbox per region, the hypothesis is confirmed.
    studies_examined = 0
    h1_region_unique_bbox_counts: List[int] = []
    h1_per_region_distinct_bbox: Counter = Counter()  # distinct-bbox-per-region histogram

    # H2: across studies, does the SAME region have the SAME bbox?
    # If yes, the bbox is a global template; if no, it's at least study-adaptive.
    region_global_bboxes: Dict[str, Set[Tuple[float, float, float, float]]] = defaultdict(set)

    # H3: do duplicate (entity, region) pairs share a bbox?
    # If consolidation@lungs always has bbox X regardless of context, that's a template.
    pair_bboxes: Dict[Tuple[str, str], Set[Tuple[float, float, float, float]]] = defaultdict(set)

    # H4: are entity bboxes derived from REGION bbox? Look at the data raw.
    # Sample bboxes-by-region for inspection.
    region_bbox_samples: Dict[str, List[Tuple[str, Tuple[float, float, float, float]]]] = defaultdict(list)

    for i, p in enumerate(sg_paths):
        try:
            sg = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(sg, dict):
            continue

        objects_list = (sg.get("objects") or sg.get("nodes")
                        or sg.get("entities") or sg.get("attributes") or [])
        if not isinstance(objects_list, list) or len(objects_list) == 0:
            continue

        # group bboxes by region within this study
        per_region: Dict[str, Set[Tuple[float, float, float, float]]] = defaultdict(set)
        for o in objects_list:
            if not isinstance(o, dict):
                continue
            # Field names vary across schemas; pick the most common ones
            region = str(o.get("region") or o.get("bbox_name")
                         or o.get("anatomy") or o.get("location") or "")
            bbox = o.get("bbox") or o.get("bbox_2d") or o.get("box")
            if not region or not bbox or len(bbox) != 4:
                continue
            try:
                tup = (round(float(bbox[0]), 4), round(float(bbox[1]), 4),
                       round(float(bbox[2]), 4), round(float(bbox[3]), 4))
            except Exception:
                continue
            per_region[region].add(tup)
            region_global_bboxes[region].add(tup)
            entity = str(o.get("entity") or o.get("name") or o.get("attribute") or "?")
            pair_bboxes[(entity, region)].add(tup)
            if len(region_bbox_samples[region]) < 10:
                region_bbox_samples[region].append((entity, tup))

        if per_region:
            studies_examined += 1
            for region, bboxes in per_region.items():
                h1_region_unique_bbox_counts.append(len(bboxes))
                h1_per_region_distinct_bbox[len(bboxes)] += 1

        if (i + 1) % 500 == 0:
            log.info(f"  scanned {i+1:,}/{len(sg_paths):,}  "
                     f"studies_with_bboxes={studies_examined:,}")

    # H1 verdict
    h1_dist = h1_per_region_distinct_bbox.most_common()
    h1_n_singleton = h1_per_region_distinct_bbox.get(1, 0)
    h1_total = sum(h1_per_region_distinct_bbox.values()) or 1
    h1_verdict = h1_n_singleton / h1_total
    log.info(f"  H1: distinct-bboxes-per-region histogram: {h1_dist}")
    log.info(f"  H1: P(region has exactly 1 bbox per study) = "
             f"{100*h1_verdict:.1f}%  "
             f"({'CONFIRMED — bboxes are region-templated' if h1_verdict > 0.8 else 'rejected — bboxes vary within region'})")

    # H2 verdict (global)
    region_global_count = Counter()
    for r, b_set in region_global_bboxes.items():
        region_global_count[r] = len(b_set)
    h2_dist = Counter(region_global_count.values())
    log.info(f"  H2: distinct-bboxes-per-region (globally across all studies): "
             f"top counts {dict(h2_dist.most_common(5))}")
    h2_constant_regions = sum(1 for r, c in region_global_count.items() if c <= 2)
    h2_total = max(1, len(region_global_count))
    log.info(f"  H2: P(region has <= 2 distinct bboxes globally) = "
             f"{100*h2_constant_regions/h2_total:.1f}%")

    # H3 verdict (entity-region pair → bbox)
    pair_distinct_bbox = Counter()
    for (e, r), bset in pair_bboxes.items():
        pair_distinct_bbox[len(bset)] += 1
    h3_pair_dist = pair_distinct_bbox.most_common()
    log.info(f"  H3: distinct-bboxes-per-(entity,region) pair: {h3_pair_dist[:10]}")

    summary: Dict[str, Any] = {
        "studies_examined": studies_examined,
        "H1_distinct_bbox_per_region_per_study_histogram": dict(h1_dist),
        "H1_p_singleton": h1_verdict,
        "H1_verdict": ("CONFIRMED: bboxes are region templates"
                       if h1_verdict > 0.8
                       else "REJECTED: bboxes vary within region per study"),
        "H2_global_distinct_bbox_per_region_top10": dict(region_global_count.most_common(10)),
        "H2_global_singleton_histogram": dict(h2_dist),
        "H3_pair_distinct_bbox_histogram": dict(h3_pair_dist),
        "n_unique_regions_seen": len(region_global_bboxes),
        "n_unique_entity_region_pairs": len(pair_bboxes),
        "region_bbox_samples_first_20_regions": {
            r: samples for r, samples in list(region_bbox_samples.items())[:20]
        },
    }
    return summary


# ---------------------------------------------------------------------------
# Section 4 — Q/A patterns
# ---------------------------------------------------------------------------

def section4_qa_patterns(mimic_qa_path: Path, max_studies: int,
                          split: str = "train") -> Dict[str, Any]:
    log.info("=" * 70)
    log.info("SECTION 4 — Q/A patterns")
    log.info("=" * 70)

    qa_paths = sorted((mimic_qa_path / "qa").glob("p*/p*/s*.qa.json"))
    log.info(f"found {len(qa_paths):,} qa files")
    if max_studies > 0 and len(qa_paths) > max_studies:
        random.seed(42)
        qa_paths = random.sample(qa_paths, max_studies)
        log.info(f"  sampling {len(qa_paths):,} for analysis")

    questions_per_study: List[int] = []
    question_type_counts: Counter = Counter()
    quality_grade_counts: Counter = Counter()
    localization_quality_counts: Counter = Counter()
    answer_lengths: List[int] = []
    question_lengths: List[int] = []
    has_grounding_bbox: int = 0
    has_obs_entities: int = 0
    qa_field_keys: Counter = Counter()
    question_field_keys: Counter = Counter()
    parse_failures = 0

    for i, p in enumerate(qa_paths):
        try:
            qa = json.loads(p.read_text())
        except Exception:
            parse_failures += 1
            continue
        if isinstance(qa, dict):
            for k in qa.keys():
                qa_field_keys[k] += 1
            questions = qa.get("questions") or qa.get("qa_pairs") or []
        elif isinstance(qa, list):
            questions = qa
        else:
            continue

        questions_per_study.append(len(questions))
        for q in questions:
            if not isinstance(q, dict):
                continue
            for k in q.keys():
                question_field_keys[k] += 1
            qt = q.get("question_type", "?")
            question_type_counts[str(qt)] += 1
            gr = (q.get("extraction_quality") or "").strip().upper()
            quality_grade_counts[gr or "(none)"] += 1
            lq = (q.get("question_img_localization_quality") or "").strip()
            localization_quality_counts[lq or "(none)"] += 1
            q_text = q.get("question") or ""
            question_lengths.append(len(q_text))

            answers = q.get("answers") or []
            for a in answers if isinstance(answers, list) else []:
                if isinstance(a, dict):
                    a_text = a.get("text") or ""
                    answer_lengths.append(len(a_text))
                    if a.get("obs_entities"):
                        has_obs_entities += 1
                    if a.get("localization") or a.get("grounding_bbox") or a.get("bbox"):
                        has_grounding_bbox += 1

        if (i + 1) % 500 == 0:
            log.info(f"  scanned {i+1:,}/{len(qa_paths):,} qa.json files  "
                     f"questions_so_far={sum(question_type_counts.values()):,}")

    summary = {
        "n_qa_files":           len(qa_paths) - parse_failures,
        "parse_failures":       parse_failures,
        "questions_per_study":  _dist(questions_per_study),
        "total_questions":      sum(question_type_counts.values()),
        "question_field_keys":  dict(question_field_keys.most_common(20)),
        "question_type_distribution":  dict(question_type_counts.most_common()),
        "quality_grade_distribution":  dict(quality_grade_counts.most_common()),
        "localization_quality_distribution": dict(localization_quality_counts.most_common()),
        "question_length_chars":  _dist(question_lengths),
        "answer_length_chars":    _dist(answer_lengths),
        "n_answers_with_obs_entities":   has_obs_entities,
        "n_answers_with_grounding_bbox": has_grounding_bbox,
    }
    log.info(f"  total questions: {summary['total_questions']:,}  "
             f"with_grounding_bbox: {has_grounding_bbox:,}  "
             f"with_obs_entities: {has_obs_entities:,}")
    log.info(f"  question_type distribution (top 10): "
             f"{dict(question_type_counts.most_common(10))}")
    log.info(f"  quality_grade distribution: {summary['quality_grade_distribution']}")
    return summary


# ---------------------------------------------------------------------------
# Section 5 — cross-reference (does the scene graph agree with the answers?)
# ---------------------------------------------------------------------------

def section5_cross_reference(mimic_qa_path: Path, max_studies: int,
                              split: str = "train") -> Dict[str, Any]:
    log.info("=" * 70)
    log.info("SECTION 5 — cross-reference (sg vs answer)")
    log.info("=" * 70)

    qa_paths = sorted((mimic_qa_path / "qa").glob("p*/p*/s*.qa.json"))
    if max_studies > 0 and len(qa_paths) > max_studies:
        random.seed(43)
        qa_paths = random.sample(qa_paths, max_studies)

    studies_checked = 0
    sg_entities_seen_in_answer = 0
    sg_entities_total = 0
    answer_polarity_matches_sg = 0
    answer_polarity_total = 0

    for i, qa_p in enumerate(qa_paths):
        # Map qa path → scene_graph path:
        #   qa/p10/p10000032/s50414267.qa.json
        # → scene_data/p10/p10000032/s50414267.scene_graph.json
        sid_part = qa_p.name.replace(".qa.json", "")
        rel_dir = qa_p.parent.relative_to(mimic_qa_path / "qa")
        sg_p = mimic_qa_path / "scene_data" / rel_dir / f"{sid_part}.scene_graph.json"
        if not sg_p.exists():
            continue
        try:
            qa = json.loads(qa_p.read_text())
            sg = json.loads(sg_p.read_text())
        except Exception:
            continue

        # Collect SG entity names
        sg_obj_list = (sg.get("objects") or sg.get("nodes")
                        or sg.get("entities") or [])
        sg_entity_polarity: Dict[str, int] = {}  # entity → polarity (last wins; usually one each)
        for o in sg_obj_list:
            if not isinstance(o, dict):
                continue
            e = str(o.get("entity") or o.get("name") or "").lower().strip()
            if not e:
                continue
            pol = o.get("positiveness") or o.get("polarity")
            if pol is not None:
                try:
                    sg_entity_polarity[e] = int(pol)
                except Exception:
                    pass

        if not sg_entity_polarity:
            continue
        studies_checked += 1

        questions = qa.get("questions") or qa.get("qa_pairs") or []
        for q in questions:
            if not isinstance(q, dict):
                continue
            answers = q.get("answers") or []
            if not isinstance(answers, list):
                continue
            for a in answers:
                if not isinstance(a, dict):
                    continue
                obs_ents = a.get("obs_entities") or []
                pos = (a.get("positiveness") or "").lower().strip()
                # Translate positiveness label to int (variable across datasets)
                pos_int = (1 if pos in {"pos", "positive", "yes", "1"}
                           else 0 if pos in {"neg", "negative", "no", "0"} else None)
                for e in obs_ents if isinstance(obs_ents, list) else []:
                    e_norm = str(e).lower().strip()
                    sg_entities_total += 1
                    if e_norm in sg_entity_polarity:
                        sg_entities_seen_in_answer += 1
                        if pos_int is not None:
                            answer_polarity_total += 1
                            if pos_int == sg_entity_polarity[e_norm]:
                                answer_polarity_matches_sg += 1

        if (i + 1) % 500 == 0:
            log.info(f"  scanned {i+1:,}/{len(qa_paths):,}  "
                     f"studies_checked={studies_checked:,}")

    summary = {
        "studies_checked":   studies_checked,
        "answer_entity_in_sg_rate": (sg_entities_seen_in_answer / max(1, sg_entities_total)),
        "answer_polarity_matches_sg_rate": (answer_polarity_matches_sg / max(1, answer_polarity_total)),
        "n_total_answer_entities_checked": sg_entities_total,
        "n_total_polarity_comparisons":   answer_polarity_total,
    }
    log.info(f"  answer-entity-in-SG rate: {100*summary['answer_entity_in_sg_rate']:.1f}%")
    log.info(f"  answer-polarity-matches-SG rate: {100*summary['answer_polarity_matches_sg_rate']:.1f}%")
    return summary


# ---------------------------------------------------------------------------
# Section 6 — visual samples (LOOK at the data, with bboxes drawn)
# ---------------------------------------------------------------------------

def section6_visual_samples(mimic_cxr_path: Path, mimic_qa_path: Path,
                             output_dir: Path, n_samples: int,
                             split: str = "train") -> List[Dict[str, Any]]:
    log.info("=" * 70)
    log.info(f"SECTION 6 — visual samples (saving up to {n_samples} annotated images)")
    log.info("=" * 70)
    output_dir.mkdir(parents=True, exist_ok=True)

    if n_samples <= 0:
        log.info("  skipped (n_samples=0)")
        return []

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        log.error("PIL not available — `pip install pillow`")
        return []

    sg_paths = sorted((mimic_qa_path / "scene_data").glob("p*/p*/s*.scene_graph.json"))
    random.seed(44)
    random.shuffle(sg_paths)

    # Color palette by region (so same region in different samples uses same color)
    PALETTE = ["#FF3B30", "#FF9500", "#FFCC00", "#34C759", "#00C7BE",
               "#30B0C7", "#007AFF", "#5856D6", "#AF52DE", "#FF2D92"]
    color_for: Dict[str, str] = {}
    def col(region: str) -> str:
        if region not in color_for:
            color_for[region] = PALETTE[len(color_for) % len(PALETTE)]
        return color_for[region]

    out_summary: List[Dict[str, Any]] = []
    n_saved = 0
    for p in sg_paths:
        if n_saved >= n_samples:
            break
        # SG path: scene_data/p10/p10000032/s50414267.scene_graph.json
        # Image path: mimic_cxr/files/p10/p10000032/s50414267/*.jpg
        sid = p.name.replace(".scene_graph.json", "")  # "s50414267"
        patient_dir = p.parent          # p10000032
        p_short_dir = p.parent.parent   # p10
        img_dir = (mimic_cxr_path / "files" / p_short_dir.name /
                   patient_dir.name / sid)
        jpgs = list(img_dir.glob("*.jpg"))
        if not jpgs:
            continue
        try:
            sg = json.loads(p.read_text())
            objects_list = (sg.get("objects") or sg.get("nodes")
                             or sg.get("entities") or [])
            if not isinstance(objects_list, list) or len(objects_list) == 0:
                continue

            pil = Image.open(jpgs[0]).convert("RGB")
            W, H = pil.size
            draw = ImageDraw.Draw(pil, "RGBA")
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    max(14, min(W, H) // 60),
                )
            except Exception:
                font = None

            lw = max(3, min(W, H) // 200)
            drawn = 0
            for o in objects_list[:25]:
                if not isinstance(o, dict):
                    continue
                bbox = o.get("bbox") or o.get("bbox_2d") or o.get("box")
                if not bbox or len(bbox) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = (float(v) for v in bbox)
                except Exception:
                    continue
                if 0 <= x1 <= 1 and 0 <= y1 <= 1:
                    x1, y1, x2, y2 = x1 * W, y1 * H, x2 * W, y2 * H
                if x2 <= x1 + 4 or y2 <= y1 + 4:
                    continue
                entity = str(o.get("entity") or o.get("name") or "?")
                region = str(o.get("region") or o.get("bbox_name") or "?")
                polarity = o.get("positiveness") or o.get("polarity")
                color = col(region)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
                label = f"{entity[:18]}@{region[:18]}"
                if polarity is not None:
                    label += f" ({'+' if int(polarity) == 1 else '-'})"
                if font:
                    tb = draw.textbbox((0, 0), label, font=font)
                    tw, th = tb[2] - tb[0], tb[3] - tb[1]
                else:
                    tw, th = 200, 16
                draw.rectangle([x1, max(0, y1 - th - 4),
                                 min(W, x1 + tw + 8), y1], fill=color)
                kw = {"font": font} if font else {}
                draw.text((x1 + 4, max(0, y1 - th - 2)), label, fill="white", **kw)
                drawn += 1

            out_path = output_dir / f"{p_short_dir.name}_{patient_dir.name}_{sid}.png"
            pil.save(out_path)
            out_summary.append({
                "image": str(out_path.relative_to(_ROOT)),
                "source_xray": str(jpgs[0]),
                "scene_graph": str(p),
                "n_objects_drawn": drawn,
                "n_objects_total": len(objects_list),
            })
            n_saved += 1
            if n_saved % 5 == 0:
                log.info(f"  saved {n_saved}/{n_samples}")
        except Exception as e:
            log.debug(f"  failed on {p}: {e}")
            continue

    log.info(f"  saved {n_saved} visual samples under {output_dir}")
    return out_summary


# ---------------------------------------------------------------------------
# Section 7 — per-stage filtering preview
# ---------------------------------------------------------------------------

def section7_per_stage_preview(mimic_qa_path: Path, max_studies: int,
                                split: str = "train") -> Dict[str, Any]:
    log.info("=" * 70)
    log.info("SECTION 7 — per-stage filter preview")
    log.info("=" * 70)
    # Note: the dataset's quality grade lives in
    # question["answers"][i]["answer_quality"]["rating"] with values like
    # "0_", "1_", "3_B", "4_A", "5_A+", "6_A++". We bucket by the first
    # letter of the rating suffix. The trainer's current filter uses
    # extraction_quality which is a sub-dict of numerical scores — that's
    # a different (coarser) signal; we report both for comparison.

    qa_paths = sorted((mimic_qa_path / "qa").glob("p*/p*/s*.qa.json"))
    if max_studies > 0 and len(qa_paths) > max_studies:
        random.seed(45)
        qa_paths = random.sample(qa_paths, max_studies)

    studies_by_grade: Dict[str, Set[str]] = defaultdict(set)
    questions_by_grade: Counter = Counter()

    for p in qa_paths:
        try:
            qa = json.loads(p.read_text())
        except Exception:
            continue
        sid = str(qa.get("study_id", p.name.replace(".qa.json", "").lstrip("s")))
        questions = qa.get("questions") or qa.get("qa_pairs") or []
        seen_grades_for_study: Set[str] = set()
        for q in questions if isinstance(questions, list) else []:
            if not isinstance(q, dict):
                continue
            # New (correct) grade source: answer_quality.rating, e.g. "4_A".
            # Extract the trailing letter (A, B, C, etc.) — first letter of
            # the part after the underscore. Skip "0_" / unrated entries.
            answers = q.get("answers") or []
            gr = None
            for a in answers if isinstance(answers, list) else []:
                if not isinstance(a, dict):
                    continue
                aq = a.get("answer_quality") or {}
                rating = (aq.get("rating") or "") if isinstance(aq, dict) else ""
                if isinstance(rating, str) and "_" in rating and len(rating) > 2:
                    suffix = rating.split("_", 1)[1].strip()
                    if suffix and suffix[0].isalpha():
                        gr = suffix[0].upper()
                        break
            if gr in {"A", "B", "C"}:
                questions_by_grade[gr] += 1
                seen_grades_for_study.add(gr)
        for gr in seen_grades_for_study:
            studies_by_grade[gr].add(sid)

    a_studies = studies_by_grade.get("A", set())
    b_studies = studies_by_grade.get("B", set())

    # Curriculum effective filters
    pools = {
        "stage1_sg_only_current_all":  studies_by_grade["A"] | studies_by_grade["B"] | studies_by_grade["C"],
        "stage1_sg_only_PROPOSED_A":   a_studies,
        "stage3_pretrain_B_or_above":  a_studies | b_studies,  # mimics quality_grade="B"
        "stage4_finetune_A_only":      a_studies,
    }

    # Pairwise Jaccard
    overlap: Dict[str, Dict[str, float]] = {}
    for a, sa in pools.items():
        overlap[a] = {}
        for b, sb in pools.items():
            union = sa | sb
            overlap[a][b] = (len(sa & sb) / len(union)) if union else 0.0

    summary = {
        "studies_total":         len(set().union(*studies_by_grade.values())),
        "studies_A":             len(a_studies),
        "studies_B":             len(b_studies),
        "studies_C":             len(studies_by_grade.get("C", set())),
        "questions_by_grade":    dict(questions_by_grade),
        "pool_sizes":            {k: len(v) for k, v in pools.items()},
        "jaccard_overlap":       overlap,
    }
    log.info(f"  studies (by grade): A={len(a_studies):,}  B={len(b_studies):,}")
    log.info(f"  pool sizes: {summary['pool_sizes']}")
    log.info(f"  Stage4_A vs Stage3_BorAbove Jaccard: "
             f"{overlap['stage4_finetune_A_only']['stage3_pretrain_B_or_above']:.2f}  "
             f"(should be low for clean curriculum)")
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dist(values: List[float]) -> Dict[str, float]:
    import numpy as np
    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    return {
        "n":      int(arr.size),
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "p10":    float(np.percentile(arr, 10)),
        "p90":    float(np.percentile(arr, 90)),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
    }


def render_markdown(report: Dict[str, Any]) -> str:
    md = ["# MIMIC-CXR + MIMIC-Ext-CXR-QBA — deep dataset analysis\n"]
    md.append(f"_Generated at section completion — see logs for timing._\n")

    md.append("## Section 1 — filesystem inventory\n")
    md.append("```json")
    md.append(json.dumps(report.get("section1", {}), indent=2))
    md.append("```\n")

    s2 = report.get("section2", {})
    md.append("## Section 2 — scene-graph structure\n")
    md.append(f"- files scanned: **{s2.get('n_scanned', 0):,}**")
    md.append(f"- parse failures: {s2.get('n_parse_failures', 0)}")
    md.append(f"- top-level keys present: `{list(s2.get('top_level_keys', {}).keys())}`")
    md.append(f"- object-list field keys: `{s2.get('objects_field_keys', {})}`")
    md.append(f"- relation-list field keys: `{s2.get('relations_field_keys', {})}`")
    md.append(f"- objects per scene_graph: `{s2.get('objects_per_sg', {})}`")
    md.append(f"- relations per scene_graph: `{s2.get('relations_per_sg', {})}`\n")
    md.append("### Example schema preview\n")
    for ex in s2.get("schema_examples", [])[:2]:
        md.append(f"**{ex.get('path', '?')}**")
        md.append("```json")
        md.append(ex.get("preview", "")[:1500])
        md.append("```\n")

    s3 = report.get("section3", {})
    md.append("## Section 3 — bbox hypothesis tests\n")
    md.append(f"- **H1 verdict** (region-templated bboxes): **{s3.get('H1_verdict', '?')}**")
    md.append(f"  - P(region has exactly 1 distinct bbox within a study) = "
              f"**{100*s3.get('H1_p_singleton', 0):.1f}%**")
    md.append(f"  - histogram of distinct-bboxes-per-region-per-study: "
              f"`{s3.get('H1_distinct_bbox_per_region_per_study_histogram', {})}`")
    md.append(f"- H2 (globally): top regions by distinct-bbox count: "
              f"`{s3.get('H2_global_distinct_bbox_per_region_top10', {})}`")
    md.append(f"- H3 (entity-region pair has multiple bboxes?): "
              f"`{s3.get('H3_pair_distinct_bbox_histogram', {})}`")
    md.append(f"- unique regions seen: {s3.get('n_unique_regions_seen', 0)}")
    md.append(f"- unique (entity, region) pairs seen: {s3.get('n_unique_entity_region_pairs', 0)}\n")
    md.append("### Region → bbox samples (first 20 regions)\n")
    for region, samples in (s3.get("region_bbox_samples_first_20_regions") or {}).items():
        md.append(f"**{region}**")
        for ent, b in samples[:5]:
            md.append(f"  - {ent}: `{b}`")
        md.append("")

    s4 = report.get("section4", {})
    md.append("## Section 4 — Q/A patterns\n")
    md.append(f"- qa.json files scanned: **{s4.get('n_qa_files', 0):,}**")
    md.append(f"- total questions: **{s4.get('total_questions', 0):,}**")
    md.append(f"- questions per study: `{s4.get('questions_per_study', {})}`")
    md.append(f"- question_type distribution: `{s4.get('question_type_distribution', {})}`")
    md.append(f"- quality_grade distribution: `{s4.get('quality_grade_distribution', {})}`")
    md.append(f"- localization_quality distribution: `{s4.get('localization_quality_distribution', {})}`")
    md.append(f"- question length (chars): `{s4.get('question_length_chars', {})}`")
    md.append(f"- answer length (chars): `{s4.get('answer_length_chars', {})}`")
    md.append(f"- answers with obs_entities: **{s4.get('n_answers_with_obs_entities', 0):,}**")
    md.append(f"- answers with grounding bbox: **{s4.get('n_answers_with_grounding_bbox', 0):,}**\n")

    s5 = report.get("section5", {})
    md.append("## Section 5 — cross-reference (SG ⟷ answers)\n")
    md.append(f"- studies checked: {s5.get('studies_checked', 0):,}")
    md.append(f"- **answer entity present in scene graph**: "
              f"**{100*s5.get('answer_entity_in_sg_rate', 0):.1f}%**")
    md.append(f"- **answer polarity matches SG polarity**: "
              f"**{100*s5.get('answer_polarity_matches_sg_rate', 0):.1f}%**")
    md.append(f"- n entities compared: {s5.get('n_total_answer_entities_checked', 0):,}")
    md.append(f"- n polarity comparisons: {s5.get('n_total_polarity_comparisons', 0):,}\n")

    s6 = report.get("section6", [])
    md.append("## Section 6 — visual samples\n")
    md.append(f"Saved **{len(s6)}** annotated images under `docs/data_analysis_samples/`.\n")
    for item in s6[:10]:
        md.append(f"- `{item.get('image')}`  ({item.get('n_objects_drawn')}/"
                  f"{item.get('n_objects_total')} bboxes drawn)")
    if len(s6) > 10:
        md.append(f"- _(+{len(s6) - 10} more in the samples directory)_")
    md.append("")

    s7 = report.get("section7", {})
    md.append("## Section 7 — per-stage filter preview\n")
    md.append(f"- studies total: {s7.get('studies_total', 0):,}")
    md.append(f"- studies with A-grade questions: {s7.get('studies_A', 0):,}")
    md.append(f"- studies with B-grade questions: {s7.get('studies_B', 0):,}")
    md.append(f"- studies with C-grade questions: {s7.get('studies_C', 0):,}")
    md.append(f"- questions by grade: {s7.get('questions_by_grade', {})}")
    md.append(f"- proposed pool sizes: `{s7.get('pool_sizes', {})}`\n")
    md.append("### Pairwise Jaccard overlap (leak diagnostic)\n")
    pools = list(s7.get("pool_sizes", {}).keys())
    md.append("| | " + " | ".join(pools) + " |")
    md.append("|---" + "|---" * len(pools) + "|")
    overlap = s7.get("jaccard_overlap", {})
    for a in pools:
        cells = [f"{overlap.get(a, {}).get(b, 0):.2f}" for b in pools]
        md.append(f"| **{a}** | " + " | ".join(cells) + " |")
    md.append("")

    return "\n".join(md)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mimic_cxr_path", type=Path, default=Path("data/mimic-cxr-jpg"))
    ap.add_argument("--mimic_qa_path",  type=Path, default=Path("data/mimic-ext-cxr-qba"))
    ap.add_argument("--split",          default="train", choices=["train", "validate", "test"])
    ap.add_argument("--max_studies",    type=int, default=5000,
                    help="Cap on studies scanned per section. 0 = ALL.")
    ap.add_argument("--max_visual_samples", type=int, default=30,
                    help="How many annotated images to save under "
                         "docs/data_analysis_samples/. 0 to skip.")
    ap.add_argument("--output_md",      type=Path, default=Path("docs/data_analysis_report.md"))
    ap.add_argument("--output_json",    type=Path, default=Path("docs/data_analysis_report.json"))
    ap.add_argument("--samples_dir",    type=Path, default=Path("docs/data_analysis_samples"))
    ap.add_argument("--skip_sections",  type=int, nargs="+", default=[],
                    help="Section numbers to skip (e.g. --skip_sections 6 if "
                         "you don't want image dumps).")
    args = ap.parse_args()

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    log.info("=" * 70)
    log.info(f"START data deep analysis")
    log.info(f"  mimic_cxr_path={args.mimic_cxr_path}")
    log.info(f"  mimic_qa_path={args.mimic_qa_path}")
    log.info(f"  split={args.split}  max_studies={args.max_studies}")
    log.info(f"  max_visual_samples={args.max_visual_samples}")
    log.info(f"  skip_sections={args.skip_sections}")
    log.info("=" * 70)

    report: Dict[str, Any] = {}

    if 1 not in args.skip_sections:
        report["section1"] = section1_filesystem_inventory(args.mimic_cxr_path, args.mimic_qa_path)
    if 2 not in args.skip_sections:
        s2, sg_paths = section2_scene_graph_structure(args.mimic_qa_path, args.max_studies, args.split)
        report["section2"] = s2
    else:
        sg_paths = sorted((args.mimic_qa_path / args.split).glob("p*/p*/s*/scene_graph.json"))
        if args.max_studies > 0 and len(sg_paths) > args.max_studies:
            random.seed(42)
            sg_paths = random.sample(sg_paths, args.max_studies)

    if 3 not in args.skip_sections:
        report["section3"] = section3_bbox_hypothesis(sg_paths)
    if 4 not in args.skip_sections:
        report["section4"] = section4_qa_patterns(args.mimic_qa_path, args.max_studies, args.split)
    if 5 not in args.skip_sections:
        report["section5"] = section5_cross_reference(args.mimic_qa_path, args.max_studies, args.split)
    if 6 not in args.skip_sections:
        report["section6"] = section6_visual_samples(
            args.mimic_cxr_path, args.mimic_qa_path,
            args.samples_dir, args.max_visual_samples, args.split,
        )
    if 7 not in args.skip_sections:
        report["section7"] = section7_per_stage_preview(args.mimic_qa_path, args.max_studies, args.split)

    # Persist
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"wrote raw JSON: {args.output_json}")

    md = render_markdown(report)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md)
    log.info(f"wrote markdown report: {args.output_md}")

    elapsed = time.time() - t_start
    log.info("=" * 70)
    log.info(f"DONE in {elapsed:.0f}s")
    log.info(f"  report:  {args.output_md}")
    log.info(f"  raw:     {args.output_json}")
    if 6 not in args.skip_sections:
        log.info(f"  images:  {args.samples_dir}/")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
