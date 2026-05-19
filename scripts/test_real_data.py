#!/usr/bin/env python3
"""
scripts/test_real_data.py — test SSGVQANetV2 on N real MIMIC+QBA samples.

Walks qa/ directly using the same logic as scripts/prebuild_cache.py
(_map_qa_file at lines 681-702) but skips the split.csv filter — that filter
is for train/val/test separation during training, irrelevant for smoke tests.

Required on disk (project's standard layout):
    data/mimic-cxr-jpg/files/p10..p19/p<subject>/s<study>/<dicom>.jpg
    data/mimic-ext-cxr-qba/qa/p10..p19/p<subject>/s<study>.qa.json   (qa.zip extracted)
    data/mimic-ext-cxr-qba/scene_data/p10..p19/p<subject>/s<study>.scene_graph.json  (extracted)

Usage:
    python scripts/test_real_data.py                  # 3 samples, GPU 0
    python scripts/test_real_data.py --n 5 --gpu 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def load_paths() -> Dict[str, str]:
    import yaml
    with open(_ROOT / "configs" / "paths.yaml") as f:
        return yaml.safe_load(f)["data"]


# ---------------------------------------------------------------------------
# Sample loader — mirrors prebuild_cache.py:_map_qa_file
# ---------------------------------------------------------------------------


def _walk_qa_files(qa_dir: Path, max_visit: int = 200) -> List[Path]:
    """Yield up to max_visit qa.json paths from qa/p*/p*/s*.qa.json."""
    out: List[Path] = []
    if not qa_dir.exists():
        return out
    for p_group in sorted(qa_dir.iterdir()):
        if not (p_group.is_dir() and p_group.name.startswith("p")):
            continue
        for patient_dir in sorted(p_group.iterdir()):
            if not (patient_dir.is_dir() and patient_dir.name.startswith("p")):
                continue
            for qa_file in sorted(patient_dir.glob("s*.qa.json")):
                out.append(qa_file)
                if len(out) >= max_visit:
                    return out
    return out


def _load_sample(
    qa_file: Path, mimic_cxr_path: Path, sg_dir: Path
) -> Optional[Dict[str, Any]]:
    """
    Build a sample dict from one qa.json. Returns None if the image or
    scene graph aren't on disk. Mirrors prebuild_cache.py:_map_qa_file
    but uses the first image (no view-position selection — fine for smoke).
    """
    try:
        subject_id = int(qa_file.parent.name[1:])            # p10000032 → 10000032
        study_id = int(qa_file.stem.split(".")[0][1:])       # s50414267.qa → 50414267
    except (ValueError, IndexError):
        return None

    p_prefix = f"p{str(subject_id)[:2]}"
    study_dir = mimic_cxr_path / "files" / p_prefix / f"p{subject_id}" / f"s{study_id}"
    if not study_dir.exists():
        return None

    jpg_files = sorted(study_dir.glob("*.jpg"))
    if not jpg_files:
        return None
    img_path = jpg_files[0]
    dicom_id = img_path.stem

    sg_path = sg_dir / p_prefix / f"p{subject_id}" / f"s{study_id}.scene_graph.json"
    if not sg_path.exists():
        return None

    try:
        with open(qa_file) as f:
            qa_data = json.load(f)
    except Exception:
        return None

    questions = qa_data.get("questions", [])
    if not questions:
        return None

    q = questions[0]   # first question per study is enough for a smoke
    return {
        "subject_id": subject_id,
        "study_id": study_id,
        "dicom_id": dicom_id,
        "image_path": str(img_path),
        "scene_graph_path": str(sg_path),
        "question_id": q.get("question_id", ""),
        "question_type": q.get("question_type", "unknown"),
        "question": q.get("question", ""),
        "answers": q.get("answers", []),
    }


def collect_samples(
    mimic_cxr_path: Path, qa_root: Path, n: int, max_visit: int = 200, verbose: bool = True
) -> List[Dict[str, Any]]:
    """Walk qa/ until we have n samples whose image+scene_graph both exist."""
    qa_dir = qa_root / "qa"
    sg_dir = qa_root / "scene_data"

    qa_files = _walk_qa_files(qa_dir, max_visit=max_visit)
    if verbose:
        print(f"[walk] visited {len(qa_files)} candidate qa.json files")

    samples: List[Dict[str, Any]] = []
    skipped = {"no_image_dir": 0, "no_jpg": 0, "no_sg": 0, "no_questions": 0, "parse": 0}
    for qa_file in qa_files:
        s = _load_sample(qa_file, mimic_cxr_path, sg_dir)
        if s is None:
            # Coarse skip-reason inference (cheap)
            try:
                subject_id = int(qa_file.parent.name[1:])
                study_id = int(qa_file.stem.split(".")[0][1:])
                p_prefix = f"p{str(subject_id)[:2]}"
                study_dir = mimic_cxr_path / "files" / p_prefix / f"p{subject_id}" / f"s{study_id}"
                if not study_dir.exists():
                    skipped["no_image_dir"] += 1
                elif not list(study_dir.glob("*.jpg")):
                    skipped["no_jpg"] += 1
                else:
                    sg_path = sg_dir / p_prefix / f"p{subject_id}" / f"s{study_id}.scene_graph.json"
                    if not sg_path.exists():
                        skipped["no_sg"] += 1
                    else:
                        skipped["no_questions"] += 1
            except Exception:
                skipped["parse"] += 1
            continue

        samples.append(s)
        if verbose:
            print(f"  ✓ sample {len(samples)}: study={s['study_id']} dicom={s['dicom_id'][:12]}...  "
                  f"Q: {s['question'][:80]}")
        if len(samples) >= n:
            break

    if verbose and len(samples) < n:
        print(f"\n[walk] found {len(samples)}/{n}; skip reasons: {skipped}")

    return samples


# ---------------------------------------------------------------------------
# Scene graph + answer formatting
# ---------------------------------------------------------------------------


def sg_to_model_dict(
    sg_json: Dict[str, Any],
    dicom_id: str,
    image_w: int,
    image_h: int,
    num_regions: int = 310,
    num_entities: int = 237,
) -> Dict[str, Any]:
    """QBA scene graph → SceneGraphEncoderV2 dict. Stable name→id hash."""
    import numpy as np

    if not hasattr(sg_to_model_dict, "_rmap"):
        sg_to_model_dict._rmap, sg_to_model_dict._emap = {}, {}
    rmap, emap = sg_to_model_dict._rmap, sg_to_model_dict._emap

    def _id(name: str, table: Dict[str, int], mod: int) -> int:
        if name not in table:
            table[name] = (hash(name) & 0x7FFFFFFF) % mod
        return table[name]

    bboxes, ents, regs, pos = [], [], [], []
    for obs in (sg_json.get("observations") or {}).values():
        loc = obs.get("localization", {}).get(dicom_id, {})
        bx = loc.get("bboxes", []) if isinstance(loc, dict) else []
        if not bx:
            continue
        x1, y1, x2, y2 = bx[0]
        bb = [
            max(0.0, min(1.0, float(x1) / max(image_w, 1))),
            max(0.0, min(1.0, float(y1) / max(image_h, 1))),
            max(0.0, min(1.0, float(x2) / max(image_w, 1))),
            max(0.0, min(1.0, float(y2) / max(image_h, 1))),
        ]
        if bb[2] <= bb[0] or bb[3] <= bb[1]:
            continue
        rlist = obs.get("regions", [])
        rname = (rlist[0]["region"] if rlist and isinstance(rlist[0], dict)
                 else (str(rlist[0]) if rlist else "unknown"))
        elist = obs.get("obs_entities", [])
        ename = str(elist[0]) if elist else "unknown"
        bboxes.append(bb)
        regs.append(_id(rname, rmap, num_regions))
        ents.append(_id(ename, emap, num_entities))
        pos.append({"pos": 1, "neg": 0}.get(obs.get("positiveness"), 2))

    return {
        "bboxes": np.asarray(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32),
        "entity_ids": np.asarray(ents, dtype=np.int64),
        "region_ids": np.asarray(regs, dtype=np.int64),
        "positiveness": np.asarray(pos, dtype=np.int64),
        "num_objects": len(bboxes),
    }


def format_answer_text(answers: List[Dict[str, Any]], dicom_id: str, w: int, h: int) -> Tuple[str, List[float]]:
    """Format QBA answer dicts into '<think>...</think><box>...</box><answer>...</answer>' + parsed bbox."""
    mains = [a for a in answers if isinstance(a, dict) and a.get("answer_type") == "main_answer"]
    a = mains[0] if mains else (answers[0] if answers and isinstance(answers[0], dict) else {})
    text = a.get("text", "Unknown.")

    bx_list = [0.0, 0.0, 1.0, 1.0]
    loc = a.get("localization", {}).get(dicom_id, {})
    if isinstance(loc, dict):
        bboxes = loc.get("bboxes", [])
        if bboxes:
            x1, y1, x2, y2 = bboxes[0]
            bx_list = [
                max(0.0, min(1.0, x1 / max(w, 1))),
                max(0.0, min(1.0, y1 / max(h, 1))),
                max(0.0, min(1.0, x2 / max(w, 1))),
                max(0.0, min(1.0, y2 / max(h, 1))),
            ]

    bbox_str = ",".join(f"{v:.3f}" for v in bx_list)
    regions = ", ".join(str(r) for r in a.get("regions", []) if r)
    entities = ", ".join(str(e) for e in a.get("obs_entities", []) if e)
    cot_parts = []
    if regions:
        cot_parts.append(f"Region(s): {regions}.")
    if entities:
        cot_parts.append(f"Observed: {entities}.")
    cot = " ".join(cot_parts) if cot_parts else "Reviewing the chest radiograph."

    return f"<think>{cot}</think><box>{bbox_str}</box><answer>{text}</answer>", bx_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Test SSGVQANetV2 on real samples (cache-free)")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--max_visit", type=int, default=200,
                   help="Max qa.json files to scan before giving up")
    p.add_argument("--max_side", type=int, default=448,
                   help="Resize MIMIC images to this short-side (default 448, "
                        "Qwen-recommended for medical). Bump to 896 if VRAM "
                        "permits; lower to 336 if you OOM.")
    args = p.parse_args(argv)

    print(f"=== SSGVQANetV2 real-data test (n={args.n}) — cache-free ===\n")

    paths = load_paths()
    mimic_cxr = Path(paths["mimic_cxr_jpg_path"])
    qa_root = Path(paths["mimic_ext_cxr_qba_path"])
    print(f"MIMIC-CXR-JPG : {mimic_cxr}")
    print(f"QBA           : {qa_root}\n")

    samples = collect_samples(mimic_cxr, qa_root, args.n, max_visit=args.max_visit)
    if not samples:
        print("\n✗ Found no usable samples. Check:")
        print(f"   data/mimic-cxr-jpg/files/p*    (must contain p10..p19)")
        print(f"   data/mimic-ext-cxr-qba/qa/p*   (qa.zip extracted)")
        print(f"   data/mimic-ext-cxr-qba/scene_data/p*  (scene_data.zip extracted)")
        return 1

    # ---- Build v2 inputs ---------------------------------------------------
    # IMPORTANT: MIMIC-CXR-JPG is ~2048x2500. Qwen's dynamic-resolution ViT
    # would emit ~8K patch tokens per image, blowing past 40+ GB VRAM on the
    # cross_entropy(logits, labels) step (logits = seq × 152K vocab in fp16).
    # Resize to 448 short-side — Qwen's recommended size for medical imaging
    # and what every reasonable training config uses. Bbox normalization
    # already happened against the original dims so the gt stays correct.
    from PIL import Image
    MAX_SIDE = int(args.max_side)
    pil_images, questions, answer_texts, scene_graphs, gt_bboxes = [], [], [], [], []
    for s in samples:
        pil = Image.open(s["image_path"]).convert("RGB")
        iw, ih = pil.size
        # Build SG/answer using ORIGINAL dims (bboxes are normalized so resize
        # downstream doesn't break them)
        with open(s["scene_graph_path"]) as f:
            sg_json = json.load(f)
        sg_dict = sg_to_model_dict(sg_json, s["dicom_id"], iw, ih)
        ans_text, gt_bbox = format_answer_text(s["answers"], s["dicom_id"], iw, ih)

        # Now actually resize the PIL we hand to Qwen
        pil_small = pil.copy()
        pil_small.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

        pil_images.append(pil_small)
        questions.append(s["question"])
        answer_texts.append(ans_text)
        scene_graphs.append(sg_dict)
        gt_bboxes.append(gt_bbox)
        print(f"  → orig={iw}x{ih} resized={pil_small.size[0]}x{pil_small.size[1]}  "
              f"sg_objects={sg_dict['num_objects']}  bbox_gt={[round(b,3) for b in gt_bbox]}")

    # ---- Model setup -------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        cc = torch.cuda.get_device_capability(args.gpu)
        dtype = torch.bfloat16 if cc >= (8, 0) else torch.float16
        device = torch.device(f"cuda:{args.gpu}")
        force_qlora = cc < (8, 0)
    else:
        dtype, device, force_qlora = torch.float32, torch.device("cpu"), False
    print(f"\n[model] {args.model_id}  device={device}  dtype={dtype}  qlora={force_qlora}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    from models import SSGVQANetV2
    t0 = time.time()
    model = SSGVQANetV2(
        qwen_model_id=args.model_id,
        use_quantization=force_qlora,
        num_sg_tokens=4,
        training_mode="pretrain",
        torch_dtype=dtype,
        max_answer_length=32,
    )
    for name in ("sg_encoder", "sg_projector", "grounding_head", "aux_heads"):
        getattr(model, name).to(device=device, dtype=dtype)
    model.sg_generator.to(device=device)
    print(f"[model] built in {time.time()-t0:.1f}s")

    gt_bbox_t = torch.tensor(gt_bboxes, dtype=torch.float, device=device)

    # ---- Training step -----------------------------------------------------
    print(f"\n[train] forward + backward + step")
    model.train()
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    try:
        t1 = time.time()
        out = model(
            images=None, pil_images=pil_images, questions=questions,
            answer_texts=answer_texts, scene_graphs=scene_graphs,
            gt_grounding_bboxes=gt_bbox_t,
        )
        loss = out["lm_loss"]
        if loss is None or not torch.isfinite(loss):
            print(f"  ✗ non-finite lm_loss: {loss}")
            return 1
        print(f"  ✓ forward: lm_loss={loss.item():.4f}  ({time.time()-t1:.1f}s)")
        t2 = time.time()
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        print(f"  ✓ backward + step ({time.time()-t2:.1f}s)")
    except Exception as e:
        print(f"  ✗ training failed: {e}")
        traceback.print_exc()
        return 1

    bb = out["grounding_outputs"]["bbox_pred"]
    print(f"\n[grounding] pred vs gt:")
    for i, s in enumerate(samples):
        pr = bb[i].tolist()
        gt = gt_bboxes[i]
        print(f"  {i+1} pred=[{pr[0]:.3f},{pr[1]:.3f},{pr[2]:.3f},{pr[3]:.3f}]"
              f"  gt=[{gt[0]:.3f},{gt[1]:.3f},{gt[2]:.3f},{gt[3]:.3f}]")

    # ---- Generation --------------------------------------------------------
    print(f"\n[infer] generation")
    model.eval()
    try:
        t3 = time.time()
        with torch.no_grad():
            gen = model(
                images=None, pil_images=pil_images, questions=questions,
                scene_graphs=scene_graphs,
            )
        print(f"  ✓ generated in {time.time()-t3:.1f}s")
        for i, t in enumerate(gen.get("generated_answer_text") or []):
            ref = answer_texts[i].split("<answer>")[-1].split("</answer>")[0]
            print(f"\n  sample {i+1}")
            print(f"    Q  : {questions[i][:100]}")
            print(f"    ref: {ref[:100]}")
            print(f"    gen: {(t or '')[:160]}")
    except Exception as e:
        print(f"  ✗ inference failed: {e}")
        traceback.print_exc()
        return 1

    print(f"\n✓ Real-data test PASSED on {len(samples)} samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
