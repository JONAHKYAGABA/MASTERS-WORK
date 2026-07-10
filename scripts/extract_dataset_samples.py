"""Extract N (default 10) MIMIC-Ext-CXR-QBA samples and write THREE files
per sample:

  * `sample_XX_image.png`        — the original X-ray, untouched.
  * `sample_XX_scenegraph.png`   — the X-ray with ground-truth scene-graph
                                   bboxes overlaid as thin outlines + tiny
                                   numeric tags, plus a right sidebar
                                   listing entity/region/positiveness per
                                   numbered box (no text on the X-ray
                                   itself).
  * `sample_XX.txt`              — plain-text bundle: question, answer,
                                   question type, quality grade,
                                   subject/study/dicom IDs, and the full
                                   list of observations (entity, region,
                                   bbox, positiveness).

Plus a single `index.json` summarizing all samples.

Usage (on marconi):
    cd /root/code/MASTERS-WORK
    source .venv/bin/activate
    python scripts/extract_dataset_samples.py \
        --qba_root data/mimic-ext-cxr-qba \
        --jpg_root data/mimic-cxr-jpg \
        --n 10 \
        --quality A \
        --out dataset_samples
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


_COLORS = [
    "#FF3B30", "#34C759", "#007AFF", "#FF9500", "#AF52DE",
    "#FFCC00", "#5856D6", "#FF2D55", "#00C7BE", "#A2845E",
]

_GRADE_ORDER = {"A++": 5, "A+": 4, "A": 3, "B+": 2, "B": 1, "C": 0, "D": -1}

# Per-criterion int -> letter grade mapping. Copied verbatim from
# data/mimic_cxr_dataset.py (QBA_CRITERION_INT_GRADES) so this script is
# self-contained. QBA stores quality as positional ints into each
# criterion's ordered enum, and the scale differs PER criterion -- a flat
# {0:D, 1:C, ...} mapping is wrong and rejects all A-grade samples.
_QBA_INT_GRADES = {
    'region_extract_quality': ['B', 'B', 'A', 'A', 'A++'],
    'entity_extract_quality': ['B', 'A', 'A++'],
    'sentence_name_quality':  ['B', 'A', 'A++'],
    'change_quality':         ['B', 'A', 'A', 'A++'],
    'general_issue_level':    ['D', 'C', 'B', 'A', 'A+', 'A++'],
    'localization_quality':   ['B', 'B', 'A', 'A++', 'A++'],
    # legacy aliases for older dumps
    'region_quality':         ['B', 'B', 'A', 'A', 'A++'],
    'entity_quality':         ['B', 'A', 'A++'],
    'issue_level':            ['D', 'C', 'B', 'A', 'A+', 'A++'],
}


def _try_font(size: int) -> Optional[ImageFont.FreeTypeFont]:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return None


def _qba_dict_to_grade(d: Any) -> Optional[str]:
    """Map a QBA quality dict (per-criterion ints OR strings) to the worst
    letter grade across criteria. Returns None if nothing parseable."""
    if isinstance(d, str):
        return d if d in _GRADE_ORDER else None
    if not isinstance(d, dict):
        return None
    if isinstance(d.get('overall'), str):
        return d['overall']
    if isinstance(d.get('grade'), str):
        return d['grade']
    grades: List[str] = []
    for k, v in d.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            enum = _QBA_INT_GRADES.get(k)
            if enum and 0 <= v < len(enum):
                grades.append(enum[v])
        elif isinstance(v, str) and v in _GRADE_ORDER:
            grades.append(v)
        elif isinstance(v, dict):
            sub = _qba_dict_to_grade(v)
            if sub:
                grades.append(sub)
    if not grades:
        return None
    return min(grades, key=lambda g: _GRADE_ORDER.get(g, 0))


def _find_image(jpg_root: Path, subject_id: int, study_id: int) -> Optional[Path]:
    p_group = f"p{str(subject_id)[:2]}"
    study_dir = jpg_root / "files" / p_group / f"p{subject_id}" / f"s{study_id}"
    if not study_dir.exists():
        return None
    imgs = list(study_dir.glob("*.jpg"))
    return imgs[0] if imgs else None


def _find_scene_graph(qba_root: Path, subject_id: int, study_id: int) -> Optional[Path]:
    p_group = f"p{str(subject_id)[:2]}"
    for sub in ("scene_data", "scene_graphs"):
        for suffix in (".scene_graph.json", ".json"):
            p = qba_root / sub / p_group / f"p{subject_id}" / f"s{study_id}{suffix}"
            if p.exists():
                return p
    return None


def _extract_gt_observations(
    scene_graph: Dict[str, Any],
    image_w: int,
    image_h: int,
    dicom_id: Optional[str],
    min_localization_quality: int = 0,
) -> List[Dict[str, Any]]:
    """Pull (entity, region, bbox_normalised, raw_bbox, positiveness,
    localization_quality) per observation.

    Skips observations without a real localisation bbox.

    When ``min_localization_quality > 0``, ALSO skips observations whose
    per-image localization_quality is below the threshold. This is the
    fix for the "labels distorted in positions" issue: QBA's level 0-1
    entries are fallback / whole-region defaults that don't match the
    finding's actual pixel position. Only level 3+ (BBOX_LOCALIZATION)
    are pixel-accurate. Recommended value for paper figures: 3.
    """
    observations = scene_graph.get("observations", {})
    out = []
    for obs_id, obs in observations.items():
        loc = obs.get("localization") or {}
        bbox = None
        raw_bbox = None
        loc_q: Optional[int] = None
        if isinstance(loc, dict) and loc:
            img_loc = loc.get(dicom_id) if dicom_id and dicom_id in loc else next(iter(loc.values()), {})
            if isinstance(img_loc, dict):
                bboxes = img_loc.get("bboxes") or []
                # QBA stores per-image localization_quality inside the
                # per-dicom_id block. Values 0-4 map to
                # NO_LOCALIZATION / FALLBACK / INCOMPLETE / BBOX / BBOX+MASK.
                raw_lq = img_loc.get("localization_quality")
                if isinstance(raw_lq, int):
                    loc_q = raw_lq
                if bboxes:
                    raw = bboxes[0]
                    if len(raw) == 4:
                        raw_bbox = [float(v) for v in raw]
                        x1, y1, x2, y2 = raw_bbox
                        x1, x2 = sorted((x1, x2))
                        y1, y2 = sorted((y1, y2))
                        if image_w > 0 and image_h > 0:
                            bbox = [
                                max(0.0, min(1.0, x1 / image_w)),
                                max(0.0, min(1.0, y1 / image_h)),
                                max(0.0, min(1.0, x2 / image_w)),
                                max(0.0, min(1.0, y2 / image_h)),
                            ]
                            if bbox[2] - bbox[0] < 0.01 or bbox[3] - bbox[1] < 0.01:
                                bbox = None
        if bbox is None:
            continue
        if min_localization_quality > 0:
            # Reject anything below the threshold. None (missing quality
            # tag) is treated as "unknown" and dropped when we're being
            # strict, since we can't defend that placement in the paper.
            if loc_q is None or loc_q < min_localization_quality:
                continue
        entities = obs.get("obs_entities") or []
        regions = obs.get("regions") or []
        entity = entities[0] if entities else "?"
        if regions:
            region = regions[0]["region"] if isinstance(regions[0], dict) else str(regions[0])
        else:
            region = "?"
        out.append({
            "obs_id": obs_id,
            "entity": str(entity),
            "region": str(region),
            "bbox": bbox,
            "raw_bbox": raw_bbox,
            "positiveness": obs.get("positiveness", "?"),
            "localization_quality": loc_q,
        })
    return out


# ---------------------------------------------------------------------------
# CheXpert labels (separate from scene graph -- per-study CSV).
# CSV columns: subject_id, study_id, then 14 finding columns with values:
#   1.0 = positive, 0.0 = negative (mentioned as absent),
#   -1.0 = uncertain, NaN/empty = not mentioned in report.
# ---------------------------------------------------------------------------
CHEXPERT_CATEGORIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]


def _load_chexpert(csv_path: Path) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Load CheXpert CSV into {(subject_id, study_id): {category: value}}.
    Returns empty dict if file is missing or unreadable."""
    if not csv_path or not csv_path.exists():
        return {}
    import csv
    import gzip
    opener = gzip.open if str(csv_path).endswith(".gz") else open
    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    with opener(csv_path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (int(row["subject_id"]), int(row["study_id"]))
            except (KeyError, ValueError):
                continue
            vals: Dict[str, float] = {}
            for cat in CHEXPERT_CATEGORIES:
                v = row.get(cat, "").strip()
                if v == "" or v.lower() == "nan":
                    vals[cat] = float("nan")
                else:
                    try:
                        vals[cat] = float(v)
                    except ValueError:
                        vals[cat] = float("nan")
            out[key] = vals
    return out


def _chexpert_value_label(v: float) -> Tuple[str, Tuple[int, int, int]]:
    """Render a CheXpert value as (text, RGB color)."""
    if v != v:  # NaN
        return ("not mentioned", (110, 110, 110))
    if v == 1.0:
        return ("POSITIVE", (220, 60, 60))
    if v == 0.0:
        return ("negative", (60, 160, 70))
    if v == -1.0:
        return ("uncertain", (220, 170, 40))
    return (f"{v}", (180, 180, 180))


def _scan_questions(
    qba_root: Path,
    jpg_root: Path,
    n: int,
    quality: Optional[str],
) -> List[Dict[str, Any]]:
    """Walk `qa/p**/p*/s*.qa.json` in lex order and return the first N
    studies whose image + scene graph exist and whose first matching question
    passes the quality filter. Stops as soon as N is reached."""
    q_root = qba_root / "qa"
    if not q_root.exists():
        raise SystemExit(f"qa root not found: {q_root}")
    out = []
    for p_group in sorted(q_root.iterdir()):
        if not (p_group.is_dir() and p_group.name.startswith("p")):
            continue
        for patient_dir in sorted(p_group.iterdir()):
            if not (patient_dir.is_dir() and patient_dir.name.startswith("p")):
                continue
            try:
                subject_id = int(patient_dir.name[1:])
            except ValueError:
                continue
            for qa_file in sorted(patient_dir.glob("s*.qa.json")):
                try:
                    study_id = int(qa_file.stem.split(".")[0][1:])
                except Exception:
                    continue
                img_path = _find_image(jpg_root, subject_id, study_id)
                if img_path is None:
                    continue
                sg_path = _find_scene_graph(qba_root, subject_id, study_id)
                if sg_path is None:
                    continue
                with open(qa_file) as f:
                    qa_data = json.load(f)
                for q in qa_data.get("questions", []):
                    qg = None
                    if quality:
                        ex_q = _qba_dict_to_grade(q.get("extraction_quality"))
                        legacy = _qba_dict_to_grade(
                            q.get("question_quality", q.get("quality")))
                        grades = [g for g in (ex_q, legacy) if g]
                        qg = (min(grades, key=lambda g: _GRADE_ORDER.get(g, 0))
                              if grades else "B")
                        if _GRADE_ORDER.get(qg, 0) < _GRADE_ORDER.get(quality, 0):
                            continue
                    out.append({
                        "subject_id": subject_id,
                        "study_id": study_id,
                        "image_path": img_path,
                        "scene_graph_path": sg_path,
                        "question_id": q.get("question_id", ""),
                        "question": q.get("question", ""),
                        "question_type": q.get("question_type", "?"),
                        "answers": q.get("answers", []),
                        "quality": qg,
                        "obs_ids": q.get("obs_ids", []),
                    })
                    print(f"[hit {len(out):>2d}/{n}] subj={subject_id} "
                          f"study={study_id} qtype={q.get('question_type', '?')} "
                          f"grade={qg or '-'}", flush=True)
                    break
                if len(out) >= n:
                    return out
    return out


def _main_answer_text(answers: List[Dict[str, Any]]) -> str:
    """Pull the human-readable main answer text from a QBA answers block."""
    if not answers:
        return ""
    mains = [a.get("text", "") for a in answers
             if a.get("answer_type", "main_answer") == "main_answer"]
    out = " ".join(t for t in mains if t).strip()
    return out or " ".join(a.get("text", "") for a in answers).strip()


def _render_image_only(sample: Dict[str, Any], out_path: Path,
                       target_w: int = 720) -> Tuple[int, int]:
    """Save the original X-ray with no annotations. Returns the rendered
    (width, height) so the scene-graph render can use the same canvas."""
    img = Image.open(sample["image_path"]).convert("RGB")
    if img.width > target_w:
        scale = target_w / img.width
        img = img.resize((target_w, int(img.height * scale)))
    img.save(out_path, format="PNG")
    return img.size


def _render_scene_graph(sample: Dict[str, Any], observations: List[Dict[str, Any]],
                        out_path: Path, render_size: Tuple[int, int]) -> None:
    """Render the X-ray with thin GT bbox outlines + tiny numeric tags,
    plus a sidebar listing each observation. No text sits on the X-ray."""
    W, H = render_size
    base = Image.open(sample["image_path"]).convert("RGB")
    if base.size != (W, H):
        base = base.resize((W, H))

    sidebar_w = 340
    canvas = Image.new("RGB", (W + sidebar_w, H), (24, 24, 24))
    canvas.paste(base, (0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.line([(W, 0), (W, H)], fill=(80, 80, 80), width=1)

    fs_legend = 13
    fs_title = 13
    fs_tag = max(9, min(13, min(W, H) // 80))
    font_legend = _try_font(fs_legend)
    font_title = _try_font(fs_title)
    font_tag = _try_font(fs_tag)

    lw = max(1, min(W, H) // 400)
    for i, obs in enumerate(observations):
        color = _COLORS[i % len(_COLORS)]
        x1, y1, x2, y2 = obs["bbox"]
        px1, py1 = int(x1 * W), int(y1 * H)
        px2, py2 = int(x2 * W), int(y2 * H)
        draw.rectangle([px1, py1, px2, py2], outline=color, width=lw)
        tag = str(i + 1)
        if font_tag:
            tb = draw.textbbox((0, 0), tag, font=font_tag)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        else:
            tw, th = 8, 10
        pad = 2
        cx1, cy1 = px1, py1
        cx2, cy2 = cx1 + tw + 2 * pad, cy1 + th + 2 * pad
        draw.rectangle([cx1, cy1, cx2, cy2], fill=color)
        kw = {"font": font_tag} if font_tag else {}
        draw.text((cx1 + pad, cy1 + pad - 1), tag, fill="white", **kw)

    x0 = W + 10
    y = 12
    if font_title:
        draw.text((x0, y), "Ground-truth scene graph",
                  fill=(220, 220, 220), font=font_title)
        y += fs_title + 8
    if not observations:
        draw.text((x0, y), "(no localised observations)",
                  fill=(180, 180, 180), font=font_legend)
    line_h = fs_legend + 4
    for i, obs in enumerate(observations):
        color = _COLORS[i % len(_COLORS)]
        chip = 16
        draw.rectangle([x0, y, x0 + chip, y + chip], fill=color)
        if font_tag:
            tb = draw.textbbox((0, 0), str(i + 1), font=font_tag)
            tx = x0 + (chip - (tb[2] - tb[0])) // 2
            ty = y + (chip - (tb[3] - tb[1])) // 2 - 1
            draw.text((tx, ty), str(i + 1), fill="white", font=font_tag)
        tx0 = x0 + chip + 6
        pos = obs["positiveness"]
        pos_color = (90, 200, 90) if pos == "pos" else (
            (220, 90, 90) if pos == "neg" else (200, 200, 200))
        ent = obs["entity"]
        if len(ent) > 30:
            ent = ent[:29] + "."
        draw.text((tx0, y), ent, fill=(240, 240, 240), font=font_legend)
        # Quality tag next to the entity so readers can see at a glance
        # which sidebar entries have trustworthy pixel-level bboxes.
        loc_q = obs.get("localization_quality")
        if font_legend:
            _quality_names = {
                0: "no-loc", 1: "fallback", 2: "partial",
                3: "bbox", 4: "bbox+mask",
            }
            q_txt = _quality_names.get(loc_q, "?") if loc_q is not None else "?"
            q_color = ((90, 200, 90) if isinstance(loc_q, int) and loc_q >= 3
                       else (220, 170, 60) if loc_q == 2
                       else (220, 90, 90))
            q_str = f"[Q={loc_q if loc_q is not None else '?'}:{q_txt}]"
            draw.text((tx0 + 200, y), q_str, fill=q_color, font=font_legend)
        reg_line = f"@ {obs['region']}"
        if len(reg_line) > 32:
            reg_line = reg_line[:31] + "."
        draw.text((tx0, y + line_h), reg_line,
                  fill=(170, 170, 170), font=font_legend)
        draw.text((tx0 + 220, y + line_h), f"[{pos}]",
                  fill=pos_color, font=font_legend)
        y += 2 * line_h + 6
        if y > H - 20:
            break

    canvas.save(out_path, format="PNG")


def _write_text_bundle(sample: Dict[str, Any],
                       observations: List[Dict[str, Any]],
                       chexpert_row: Optional[Dict[str, float]],
                       out_path: Path) -> None:
    """Write question, answer, metadata, observations, CheXpert labels.
    Plain text, readable in any editor."""
    ans_text = _main_answer_text(sample["answers"])
    lines = [
        f"sample id           : {sample.get('_idx', '?')}",
        f"subject_id          : {sample['subject_id']}",
        f"study_id            : {sample['study_id']}",
        f"dicom_id            : {sample['image_path'].stem}",
        f"question_id         : {sample['question_id']}",
        f"question_type       : {sample['question_type']}",
        f"quality_grade       : {sample.get('quality') or '(unfiltered)'}",
        f"image_path          : {sample['image_path']}",
        f"scene_graph_path    : {sample['scene_graph_path']}",
        "",
        "=" * 70,
        "QUESTION",
        "=" * 70,
        sample["question"] or "(empty)",
        "",
        "=" * 70,
        "ANSWER (main)",
        "=" * 70,
        ans_text or "(empty)",
    ]
    if sample["answers"]:
        lines += ["", "-" * 70, "FULL ANSWERS BLOCK", "-" * 70]
        for j, a in enumerate(sample["answers"], 1):
            lines.append(
                f"  [{j}] type={a.get('answer_type', '?')}  "
                f"positiveness={a.get('positiveness', '?')}"
            )
            if a.get("text"):
                lines.append(f"      text: {a['text']}")
            if a.get("regions"):
                lines.append(f"      regions: {a['regions']}")
            if a.get("obs_entities"):
                lines.append(f"      obs_entities: {a['obs_entities']}")
            if a.get("modifiers"):
                lines.append(f"      modifiers: {a['modifiers']}")

    # Clinical context (INDICATION / HISTORY / EXAM_TECHNIQUE) pulled from
    # the scene_graph.json's `sentences` block. This is what the trainer's
    # --use_reports flag prepends to the question as "Clinical context:".
    clinical = sample.get("_clinical") or {}
    if clinical:
        lines += [
            "",
            "=" * 70,
            "CLINICAL CONTEXT (from scene_graph.json report sentences)",
            "=" * 70,
            f"Indication      : {clinical.get('indication_text', '') or '(none)'}",
            f"History         : {clinical.get('history_text', '') or '(none)'}",
            f"Exam technique  : {clinical.get('technique_text', '') or '(none)'}",
            f"Findings        : {clinical.get('findings_text', '') or '(none)'}",
            f"Impression      : {clinical.get('impression_text', '') or '(none)'}",
        ]
        struct = clinical.get("indication_struct") or {}
        if isinstance(struct, dict) and struct:
            lines += ["", "-" * 70, "STRUCTURED INDICATION BLOCK", "-" * 70]
            for k in ("indication_summary", "patient_info", "clinical_indication",
                      "expected_evaluation", "answer_for_indication"):
                v = struct.get(k)
                if v:
                    if isinstance(v, (list, dict)):
                        v = json.dumps(v, ensure_ascii=False)
                    lines.append(f"  {k:<22s}: {v}")

    lines += [
        "",
        "=" * 70,
        f"OBSERVATIONS (ground-truth scene graph) — {len(observations)} localised",
        "=" * 70,
        "raw_bbox is what's literally in scene_graph.json (pixel coords).",
        "normalised is raw_bbox / image_size, used to draw on the viz.",
        "Imprecise placements come from QBA's localization_quality<3 entries",
        "(NO_LOCALIZATION / FALLBACK / INCOMPLETE) — not a viz bug.",
        "",
    ]
    if not observations:
        lines.append("(no observations with localisation bboxes)")
    _quality_names = {
        0: "NO_LOCALIZATION", 1: "FALLBACK_LOCALIZATION",
        2: "INCOMPLETE_LOCALIZATION",
        3: "BBOX_LOCALIZATION", 4: "BBOX_AND_MASK_LOCALIZATION",
    }
    for i, obs in enumerate(observations, 1):
        x1, y1, x2, y2 = obs["bbox"]
        loc_q = obs.get("localization_quality")
        q_str = (f"{loc_q} ({_quality_names.get(loc_q, '?')})"
                 if loc_q is not None else "?")
        lines.append(
            f"  [{i}] entity={obs['entity']!r}  region={obs['region']!r}  "
            f"positiveness={obs['positiveness']!r}  "
            f"localization_quality={q_str}"
        )
        if obs.get("raw_bbox"):
            r = obs["raw_bbox"]
            lines.append(
                f"       raw_bbox(px xyxy)       = "
                f"[{r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f}, {r[3]:.1f}]"
            )
        lines.append(
            f"       bbox(normalised xyxy)   = "
            f"[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]"
        )

    # CheXpert section
    lines += [
        "",
        "=" * 70,
        "CHEXPERT LABELS (per-study, from mimic-cxr-2.0.0-chexpert.csv)",
        "=" * 70,
        "Values: 1.0 = positive, 0.0 = negative (mentioned absent),",
        "       -1.0 = uncertain, NaN/empty = not mentioned in report.",
        "",
    ]
    if chexpert_row is None:
        lines.append("(no CheXpert CSV provided OR study not found in CSV)")
    else:
        for cat in CHEXPERT_CATEGORIES:
            v = chexpert_row.get(cat, float("nan"))
            label, _ = _chexpert_value_label(v)
            lines.append(f"  {cat:<30s} {v!s:>6s}   {label}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _render_chexpert(sample: Dict[str, Any],
                     chexpert_row: Optional[Dict[str, float]],
                     out_path: Path) -> None:
    """Render the 14 CheXpert labels for this study as a compact dashboard
    PNG: one row per category with a colored value chip."""
    cell_h = 28
    title_h = 36
    pad = 12
    W = 520
    H = title_h + cell_h * len(CHEXPERT_CATEGORIES) + 2 * pad
    canvas = Image.new("RGB", (W, H), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    font_title = _try_font(15)
    font = _try_font(13)
    font_b = _try_font(13)

    sid = sample["study_id"]
    title = f"CheXpert labels — study {sid}"
    if font_title:
        draw.text((pad, pad), title, fill=(230, 230, 230), font=font_title)
        sub = f"subject {sample['subject_id']}"
        draw.text((pad, pad + 18), sub, fill=(140, 140, 140), font=font)

    y = pad + title_h
    draw.line([(pad, y - 4), (W - pad, y - 4)], fill=(60, 60, 60), width=1)

    if chexpert_row is None:
        draw.text((pad, y + 4),
                  "(CheXpert CSV not provided or study not in CSV)",
                  fill=(200, 120, 120), font=font)
        canvas.save(out_path, format="PNG")
        return

    for cat in CHEXPERT_CATEGORIES:
        v = chexpert_row.get(cat, float("nan"))
        label, color = _chexpert_value_label(v)
        # category name
        draw.text((pad, y + 6), cat, fill=(220, 220, 220), font=font)
        # value chip on the right
        chip_w, chip_h = 130, cell_h - 8
        cx1 = W - pad - chip_w
        cy1 = y + 4
        draw.rectangle([cx1, cy1, cx1 + chip_w, cy1 + chip_h], fill=color)
        if font_b:
            tb = draw.textbbox((0, 0), label, font=font_b)
            tx = cx1 + (chip_w - (tb[2] - tb[0])) // 2
            ty = cy1 + (chip_h - (tb[3] - tb[1])) // 2 - 1
            draw.text((tx, ty), label, fill="white", font=font_b)
        # divider
        draw.line([(pad, y + cell_h), (W - pad, y + cell_h)],
                  fill=(50, 50, 50), width=1)
        y += cell_h

    canvas.save(out_path, format="PNG")


def _extract_clinical_context(sg: Dict[str, Any]) -> Dict[str, Any]:
    """Pull INDICATION + HISTORY + EXAM_TECHNIQUE from a QBA scene_graph.json.

    QBA embeds the original report text under two shapes we care about:

    1. ``sentences``: a dict keyed by section_type where each entry is
       either a list of strings or a list of dicts with a ``text`` field.
       Section types include INDICATION, HISTORY, EXAM_TECHNIQUE,
       FINDINGS, IMPRESSION, COMPARISON, WET_READ.
    2. ``indication``: a structured dict with keys such as
       ``indication_summary``, ``patient_info``, ``clinical_indication``,
       ``expected_evaluation``, ``answer_for_indication``. This is what
       the QBA question-generation pipeline uses to construct
       "Indication" questions and is exactly what the trainer's
       ``--use_reports`` flag prepends to the question.

    Returns a dict:
        {
          "indication_text":  concatenated INDICATION sentences (str),
          "history_text":     concatenated HISTORY sentences (str),
          "technique_text":   concatenated EXAM_TECHNIQUE sentences (str),
          "findings_text":    concatenated FINDINGS sentences (str),
          "impression_text":  concatenated IMPRESSION sentences (str),
          "indication_struct": raw dict from sg["indication"] (or {}),
        }

    Any missing section is returned as "".
    """
    def _flatten(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, list):
            parts = []
            for item in v:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    for k in ("text", "sentence", "value"):
                        if k in item and isinstance(item[k], str):
                            parts.append(item[k])
                            break
            return " ".join(p.strip() for p in parts if p and p.strip())
        if isinstance(v, dict):
            for k in ("text", "sentence", "value"):
                if k in v and isinstance(v[k], str):
                    return v[k].strip()
        return ""

    sentences = sg.get("sentences") or {}
    if not isinstance(sentences, dict):
        sentences = {}

    def _sec(name: str) -> str:
        # try exact + lowercased + a couple of aliases
        for key in (name, name.lower(), name.upper(),
                    name.title().replace("_", " ")):
            if key in sentences:
                return _flatten(sentences[key])
        return ""

    result = {
        "indication_text":  _sec("INDICATION"),
        "history_text":     _sec("HISTORY"),
        "technique_text":   _sec("EXAM_TECHNIQUE"),
        "findings_text":    _sec("FINDINGS"),
        "impression_text":  _sec("IMPRESSION"),
        "indication_struct": sg.get("indication") if isinstance(sg.get("indication"), dict) else {},
    }
    return result


def _render_paper_figure(sample: Dict[str, Any],
                         observations: List[Dict[str, Any]],
                         clinical: Dict[str, Any],
                         out_path: Path,
                         render_size: Tuple[int, int]) -> None:
    """Render a single publication-ready PNG combining, top-to-bottom:

        [ Clinical context text box    ]
        [ X-ray  |  Scene-graph legend ]
        [ Question + Answer row        ]

    Designed for direct inclusion in the paper as a "sample walkthrough"
    figure. Nothing is drawn on the X-ray itself apart from thin colored
    bbox outlines + tiny digit tags; all readable text lives in the panels
    above and beside the image so the anatomy is never occluded.
    """
    W, H = render_size
    xray = Image.open(sample["image_path"]).convert("RGB")
    if xray.size != (W, H):
        xray = xray.resize((W, H))

    # Layout constants
    sidebar_w = 360
    context_h = 180
    footer_h = 130
    canvas_w = W + sidebar_w
    canvas_h = context_h + H + footer_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 247))
    draw = ImageDraw.Draw(canvas, "RGBA")

    # Fonts
    f_title = _try_font(15)
    f_hdr   = _try_font(13)
    f_body  = _try_font(12)
    f_small = _try_font(11)
    f_tag   = _try_font(max(9, min(13, min(W, H) // 80)))

    # ---- 1. Clinical-context header ---------------------------------
    ctx_x0, ctx_y0 = 20, 12
    draw.rectangle([10, 6, canvas_w - 10, context_h - 6],
                   fill=(255, 255, 255), outline=(200, 200, 210), width=1)
    if f_title:
        draw.text((ctx_x0, ctx_y0), "Clinical context (from QBA scene graph)",
                  fill=(30, 30, 40), font=f_title)
        ctx_y0 += 22

    def _wrap_lines(text: str, max_w: int, font) -> List[str]:
        if not text or not font:
            return [text] if text else []
        words = text.split()
        lines, cur = [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            tb = draw.textbbox((0, 0), trial, font=font)
            if tb[2] - tb[0] <= max_w:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    avail_w = canvas_w - 40
    lh = 15
    for label, val in (
        ("Indication:", clinical.get("indication_text", "").strip() or "(none)"),
        ("History:",    clinical.get("history_text", "").strip() or "(none)"),
        ("Technique:",  clinical.get("technique_text", "").strip() or "(none)"),
    ):
        if f_hdr:
            draw.text((ctx_x0, ctx_y0), label, fill=(80, 80, 100), font=f_hdr)
            lbl_w = draw.textbbox((0, 0), label, font=f_hdr)[2] + 6
        else:
            lbl_w = 90
        wrapped = _wrap_lines(val, avail_w - lbl_w, f_body)[:2]
        for i, ln in enumerate(wrapped):
            draw.text((ctx_x0 + lbl_w, ctx_y0 + i * lh), ln,
                      fill=(30, 30, 40), font=f_body)
        ctx_y0 += max(lh, lh * len(wrapped)) + 6
        if ctx_y0 > context_h - 20:
            break

    # ---- 2. X-ray + scene-graph annotations -------------------------
    canvas.paste(xray, (0, context_h))
    draw.line([(W, context_h), (W, context_h + H)],
              fill=(200, 200, 210), width=1)

    # thin bboxes + digit tags on the X-ray
    lw = max(1, min(W, H) // 400)
    for i, obs in enumerate(observations):
        color = _COLORS[i % len(_COLORS)]
        x1, y1, x2, y2 = obs["bbox"]
        px1, py1 = int(x1 * W), context_h + int(y1 * H)
        px2, py2 = int(x2 * W), context_h + int(y2 * H)
        draw.rectangle([px1, py1, px2, py2], outline=color, width=lw)
        tag = str(i + 1)
        if f_tag:
            tb = draw.textbbox((0, 0), tag, font=f_tag)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        else:
            tw, th = 8, 10
        pad = 2
        cx2 = px1 + tw + 2 * pad
        cy2 = py1 + th + 2 * pad
        draw.rectangle([px1, py1, cx2, cy2], fill=color)
        kw = {"font": f_tag} if f_tag else {}
        draw.text((px1 + pad, py1 + pad - 1), tag, fill="white", **kw)

    # ---- 3. Scene-graph legend sidebar ------------------------------
    lx = W + 12
    ly = context_h + 10
    if f_title:
        draw.text((lx, ly), "Ground-truth scene graph",
                  fill=(30, 30, 40), font=f_title)
        ly += f_title.size + 8
    if not observations:
        if f_body:
            draw.text((lx, ly), "(no localised observations)",
                      fill=(120, 120, 130), font=f_body)
    line_h = 14
    for i, obs in enumerate(observations):
        color = _COLORS[i % len(_COLORS)]
        chip = 16
        draw.rectangle([lx, ly, lx + chip, ly + chip], fill=color)
        if f_tag:
            tb = draw.textbbox((0, 0), str(i + 1), font=f_tag)
            tx = lx + (chip - (tb[2] - tb[0])) // 2
            ty = ly + (chip - (tb[3] - tb[1])) // 2 - 1
            draw.text((tx, ty), str(i + 1), fill="white", font=f_tag)
        tx0 = lx + chip + 6
        pos = obs["positiveness"]
        pos_color = ((60, 160, 70) if pos == "pos"
                     else (220, 60, 60) if pos == "neg"
                     else (140, 140, 140))
        ent = obs["entity"][:32]
        if f_body:
            draw.text((tx0, ly), ent, fill=(30, 30, 40), font=f_body)
        # Quality tag next to the entity: [Q=3] means BBOX_LOCALIZATION;
        # [Q=fallback] etc. warns the reader that a bbox is a QBA default
        # rather than a semantically-accurate position.
        loc_q = obs.get("localization_quality")
        if f_small:
            _quality_names = {
                0: "no-loc", 1: "fallback", 2: "partial",
                3: "bbox", 4: "bbox+mask",
            }
            q_txt = _quality_names.get(loc_q, "?") if loc_q is not None else "?"
            q_color = ((60, 160, 70) if isinstance(loc_q, int) and loc_q >= 3
                       else (200, 140, 40) if loc_q == 2
                       else (200, 80, 60))
            q_str = f"[Q={loc_q if loc_q is not None else '?'}:{q_txt}]"
            draw.text((tx0 + 200, ly), q_str, fill=q_color, font=f_small)
        reg = "@ " + obs["region"][:28]
        if f_small:
            draw.text((tx0, ly + line_h), reg,
                      fill=(100, 100, 120), font=f_small)
            draw.text((tx0 + 220, ly + line_h), f"[{pos}]",
                      fill=pos_color, font=f_small)
        ly += 2 * line_h + 4
        if ly > context_h + H - 20:
            break

    # ---- 4. Question/Answer footer ----------------------------------
    fy = context_h + H + 12
    draw.rectangle([10, fy - 6, canvas_w - 10, canvas_h - 6],
                   fill=(255, 255, 255), outline=(200, 200, 210), width=1)
    ans_text = _main_answer_text(sample.get("answers", [])) or "(no answer)"
    for label, val, color in (
        ("Question:", sample.get("question", "") or "(empty)", (30, 30, 40)),
        ("Answer:",   ans_text,                                 (10, 80, 40)),
    ):
        if f_hdr:
            draw.text((20, fy), label, fill=(80, 80, 100), font=f_hdr)
            lbl_w = draw.textbbox((0, 0), label, font=f_hdr)[2] + 6
        else:
            lbl_w = 80
        lines = _wrap_lines(val, canvas_w - 40 - lbl_w, f_body)[:2]
        for i, ln in enumerate(lines):
            draw.text((20 + lbl_w, fy + i * lh), ln, fill=color, font=f_body)
        fy += max(lh, lh * len(lines)) + 6

    canvas.save(out_path, format="PNG")


def _render_sample(sample: Dict[str, Any], out_dir: Path, sample_id: int,
                   chexpert_row: Optional[Dict[str, float]] = None,
                   paper_figure: bool = False,
                   min_localization_quality: int = 0) -> None:
    """Emit four artifacts for one sample: clean X-ray, scene-graph viz,
    CheXpert dashboard, text bundle. When ``paper_figure=True``, ALSO
    emit a fifth artifact: sample_XX_paper.png -- a single combined
    figure with clinical context header + X-ray + scene-graph legend +
    Q/A footer, suitable for direct inclusion in the thesis.

    ``min_localization_quality`` is forwarded to _extract_gt_observations
    so paper figures can drop QBA fallback / whole-region bboxes that
    would otherwise be drawn at semantically-wrong positions.
    """
    sample["_idx"] = sample_id
    base = f"sample_{sample_id:02d}"

    # 1. clean X-ray
    img_path = out_dir / f"{base}_image.png"
    render_size = _render_image_only(sample, img_path)

    # 2. scene graph viz (uses original dims for bbox normalisation)
    orig_w, orig_h = Image.open(sample["image_path"]).size
    with open(sample["scene_graph_path"]) as f:
        sg = json.load(f)
    observations = _extract_gt_observations(
        sg, orig_w, orig_h, sample["image_path"].stem,
        min_localization_quality=min_localization_quality,
    )[:10]
    sg_path = out_dir / f"{base}_scenegraph.png"
    _render_scene_graph(sample, observations, sg_path, render_size)

    # Extract clinical context once; used by text bundle AND paper figure.
    clinical = _extract_clinical_context(sg)
    # Attach to the sample dict so the text bundle can persist it too.
    sample["_clinical"] = clinical

    # 3. CheXpert dashboard
    chx_path = out_dir / f"{base}_chexpert.png"
    _render_chexpert(sample, chexpert_row, chx_path)

    # 4. text bundle (includes scene-graph + CheXpert + clinical context)
    txt_path = out_dir / f"{base}.txt"
    _write_text_bundle(sample, observations, chexpert_row, txt_path)

    # 5. combined publication figure (image + indication + scene graph)
    if paper_figure:
        paper_path = out_dir / f"{base}_paper.png"
        _render_paper_figure(sample, observations, clinical,
                             paper_path, render_size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qba_root", required=True, type=Path,
                   help="data/mimic-ext-cxr-qba")
    p.add_argument("--jpg_root", required=True, type=Path,
                   help="data/mimic-cxr-jpg")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--quality", default="none",
                   help="Minimum grade (A++, A+, A, B+, B, C, D, or 'none'). "
                        "Default 'none' = take first N studies in lex order "
                        "regardless of grade — fastest, guaranteed to find N.")
    p.add_argument("--out", type=Path, default=Path("dataset_samples"))
    p.add_argument("--chexpert_csv", type=Path, default=None,
                   help="Override path to mimic-cxr-2.0.0-chexpert.csv(.gz). "
                        "If omitted, looks for <jpg_root>/mimic-cxr-2.0.0-chexpert.csv.gz "
                        "(same default as the training loader).")
    p.add_argument("--paper_figures", action="store_true",
                   help="Additionally emit sample_XX_paper.png -- a single "
                        "publication-ready figure per sample with clinical "
                        "context header + X-ray + scene-graph legend + Q/A "
                        "footer, suitable for direct inclusion in the thesis.")
    p.add_argument("--min_localization_quality", type=int, default=0,
                   help="When >0, drop scene-graph observations whose per-image "
                        "localization_quality is below this level. Level 3 = "
                        "BBOX_LOCALIZATION (recommended for paper figures -- "
                        "fixes the 'labels distorted in positions' effect where "
                        "QBA fallback / whole-region bboxes get drawn at "
                        "semantically-wrong positions). Levels 0-2 are "
                        "fallback / incomplete / whole-region defaults.")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    quality = None if args.quality.lower() in ("none", "", "all") else args.quality

    # Resolve CheXpert CSV using the same fallback as data/mimic_cxr_dataset.py
    chexpert_path = args.chexpert_csv
    if chexpert_path is None:
        for candidate in (args.jpg_root / "mimic-cxr-2.0.0-chexpert.csv.gz",
                          args.jpg_root / "mimic-cxr-2.0.0-chexpert.csv"):
            if candidate.exists():
                chexpert_path = candidate
                break
    chexpert = _load_chexpert(chexpert_path) if chexpert_path else {}
    if chexpert:
        print(f"[chexpert] loaded {len(chexpert)} studies from {chexpert_path}",
              flush=True)
    else:
        print("[chexpert] no CSV found, dashboards will say '(not provided)'",
              flush=True)

    samples = _scan_questions(args.qba_root, args.jpg_root, args.n, quality)
    if not samples:
        raise SystemExit("no samples matched filters")

    index = []
    for i, s in enumerate(samples, 1):
        try:
            chx_row = chexpert.get((s["subject_id"], s["study_id"]))
            _render_sample(s, args.out, i, chexpert_row=chx_row,
                            paper_figure=args.paper_figures,
                            min_localization_quality=args.min_localization_quality)
            base = f"sample_{i:02d}"
            print(f"[render {i:>2d}/{len(samples)}] wrote {base}_image.png, "
                  f"{base}_scenegraph.png, {base}_chexpert.png, {base}.txt",
                  flush=True)
            index.append({
                "id": i,
                "image_png": f"{base}_image.png",
                "scene_graph_png": f"{base}_scenegraph.png",
                "chexpert_png": f"{base}_chexpert.png",
                "text_bundle": f"{base}.txt",
                "subject_id": s["subject_id"],
                "study_id": s["study_id"],
                "question": s["question"],
                "question_type": s["question_type"],
                "quality": s.get("quality"),
                "answers": [a.get("text", "") for a in s["answers"]],
                "chexpert_available": chx_row is not None,
                "source_image_path": str(s["image_path"]),
                "source_scene_graph_path": str(s["scene_graph_path"]),
            })
        except Exception as e:
            print(f"FAILED sample {i}: {e}", file=sys.stderr)

    (args.out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nwrote {len(index)} samples (4 files each) + index.json to {args.out}")


if __name__ == "__main__":
    main()
