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
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


_COLORS = [
    "#FF3B30", "#34C759", "#007AFF", "#FF9500", "#AF52DE",
    "#FFCC00", "#5856D6", "#FF2D55", "#00C7BE", "#A2845E",
]

_GRADE_ORDER = {"A++": 5, "A+": 4, "A": 3, "B+": 2, "B": 1, "C": 0, "D": -1}


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


def _qba_int_to_grade(v: Any) -> Optional[str]:
    """QBA stores per-criterion quality as int 0-4.  Pick the worst across
    all criteria and map to a letter grade.  Returns None if v is empty or
    can't be interpreted."""
    if v is None:
        return None
    if isinstance(v, str):
        return v if v in _GRADE_ORDER else None
    if isinstance(v, int):
        return ["D", "C", "B", "A", "A++"][max(0, min(4, v))]
    if isinstance(v, dict):
        grades = []
        for cv in v.values():
            g = _qba_int_to_grade(cv)
            if g:
                grades.append(g)
        if not grades:
            return None
        return min(grades, key=lambda g: _GRADE_ORDER.get(g, 0))
    return None


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
) -> List[Dict[str, Any]]:
    """Pull (entity, region, bbox_normalised, positiveness) per observation.
    Skips observations without a real localisation bbox so the visualisation
    only shows boxes that were actually grounded."""
    observations = scene_graph.get("observations", {})
    out = []
    for obs_id, obs in observations.items():
        loc = obs.get("localization") or {}
        bbox = None
        if isinstance(loc, dict) and loc:
            img_loc = loc.get(dicom_id) if dicom_id and dicom_id in loc else next(iter(loc.values()), {})
            if isinstance(img_loc, dict):
                bboxes = img_loc.get("bboxes") or []
                if bboxes:
                    raw = bboxes[0]
                    if len(raw) == 4:
                        x1, y1, x2, y2 = (float(v) for v in raw)
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
            "positiveness": obs.get("positiveness", "?"),
        })
    return out


def _scan_questions(
    qba_root: Path,
    jpg_root: Path,
    n: int,
    quality: Optional[str],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Walk `qa/p**/p*/s*.qa.json` and collect N samples that satisfy the
    filters AND have both an image and a scene graph on disk.

    MIMIC-Ext-CXR-QBA actual layout (matches the training loader):
      qa/p10/p10000032/s50414267.qa.json
      scene_data/p10/p10000032/s50414267.scene_graph.json
    """
    q_root = qba_root / "qa"
    if not q_root.exists():
        raise SystemExit(f"qa root not found: {q_root}")
    rng = random.Random(seed)
    # Use iterator instead of materialising the full list (it's ~377K files)
    # — shuffle a randomised view by sampling p-groups + patients in random
    # order, but stop as soon as we have enough hits.
    p_groups = sorted([p for p in q_root.iterdir()
                       if p.is_dir() and p.name.startswith("p")])
    rng.shuffle(p_groups)
    out = []
    scanned = 0
    for p_group in p_groups:
        patients = [p for p in p_group.iterdir()
                    if p.is_dir() and p.name.startswith("p")]
        rng.shuffle(patients)
        for patient_dir in patients:
            qa_files = list(patient_dir.glob("s*.qa.json"))
            rng.shuffle(qa_files)
            for qa_file in qa_files:
                scanned += 1
                try:
                    subject_id = int(patient_dir.name[1:])
                    # stem is "s50414267.qa" -> first split is "s50414267"
                    study_id = int(qa_file.stem.split(".")[0][1:])
                except Exception:
                    continue
                try:
                    with open(qa_file) as f:
                        qa_data = json.load(f)
                except Exception:
                    continue
                img_path = _find_image(jpg_root, subject_id, study_id)
                if img_path is None:
                    continue
                sg_path = _find_scene_graph(qba_root, subject_id, study_id)
                if sg_path is None:
                    continue
                for q in qa_data.get("questions", []):
                    qg = None
                    if quality:
                        ex_q = _qba_int_to_grade(q.get("extraction_quality"))
                        legacy = _qba_int_to_grade(
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
                    break  # one question per study
                if len(out) >= n:
                    print(f"scanned {scanned} qa files, collected {len(out)}",
                          file=sys.stderr)
                    return out
    print(f"scanned {scanned} qa files, collected {len(out)}", file=sys.stderr)
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
                       out_path: Path) -> None:
    """Write question, answer, metadata, and the full observation list to a
    plain-text file readable in any editor."""
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

    lines += [
        "",
        "=" * 70,
        f"OBSERVATIONS (ground-truth scene graph) — {len(observations)} localised",
        "=" * 70,
    ]
    if not observations:
        lines.append("(no observations with localisation bboxes)")
    for i, obs in enumerate(observations, 1):
        x1, y1, x2, y2 = obs["bbox"]
        lines.append(
            f"  [{i}] entity={obs['entity']!r}  region={obs['region']!r}  "
            f"positiveness={obs['positiveness']!r}"
        )
        lines.append(
            f"       bbox(normalised xyxy) = "
            f"[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _render_sample(sample: Dict[str, Any], out_dir: Path, sample_id: int) -> None:
    """Emit three artifacts for one sample: clean X-ray, scene-graph viz,
    text bundle."""
    sample["_idx"] = sample_id
    base = f"sample_{sample_id:02d}"

    # 1. clean X-ray
    img_path = out_dir / f"{base}_image.png"
    render_size = _render_image_only(sample, img_path)

    # 2. scene graph viz (uses original dims for bbox normalisation)
    orig_w, orig_h = Image.open(sample["image_path"]).size
    with open(sample["scene_graph_path"]) as f:
        sg = json.load(f)
    observations = _extract_gt_observations(sg, orig_w, orig_h,
                                            sample["image_path"].stem)[:10]
    sg_path = out_dir / f"{base}_scenegraph.png"
    _render_scene_graph(sample, observations, sg_path, render_size)

    # 3. text bundle
    txt_path = out_dir / f"{base}.txt"
    _write_text_bundle(sample, observations, txt_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qba_root", required=True, type=Path,
                   help="data/mimic-ext-cxr-qba")
    p.add_argument("--jpg_root", required=True, type=Path,
                   help="data/mimic-cxr-jpg")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--quality", default="A",
                   help="Minimum grade (A++, A+, A, B+, B, C, D, or 'none')")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("dataset_samples"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    quality = None if args.quality.lower() in ("none", "", "all") else args.quality

    samples = _scan_questions(args.qba_root, args.jpg_root, args.n, quality, args.seed)
    if not samples:
        raise SystemExit("no samples matched filters")

    index = []
    for i, s in enumerate(samples, 1):
        try:
            _render_sample(s, args.out, i)
            base = f"sample_{i:02d}"
            print(f"wrote {base}_image.png, {base}_scenegraph.png, {base}.txt")
            index.append({
                "id": i,
                "image_png": f"{base}_image.png",
                "scene_graph_png": f"{base}_scenegraph.png",
                "text_bundle": f"{base}.txt",
                "subject_id": s["subject_id"],
                "study_id": s["study_id"],
                "question": s["question"],
                "question_type": s["question_type"],
                "quality": s.get("quality"),
                "answers": [a.get("text", "") for a in s["answers"]],
                "source_image_path": str(s["image_path"]),
                "source_scene_graph_path": str(s["scene_graph_path"]),
            })
        except Exception as e:
            print(f"FAILED sample {i}: {e}", file=sys.stderr)

    (args.out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nwrote {len(index)} samples (3 files each) + index.json to {args.out}")


if __name__ == "__main__":
    main()
