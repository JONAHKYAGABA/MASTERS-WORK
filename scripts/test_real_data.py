#!/usr/bin/env python3
"""
scripts/test_real_data.py — test SSGVQANetV2 on real cached samples.

This script does NO data discovery. It expects a sample cache built by
scripts/prebuild_cache.py — the project's canonical, validated builder that
walks qa/*/*/*.qa.json, picks the best frontal image per study, and writes a
pickled list of sample dicts.

Workflow:
    # one-time: build a 50-sample cache using the project's own builder
    python scripts/prebuild_cache.py \\
        --mimic_cxr_path data/mimic-cxr-jpg \\
        --mimic_qa_path  data/mimic-ext-cxr-qba \\
        --max_samples 50 --num_workers 4 --split train

    # then: run the v2 model on N of those samples
    python scripts/test_real_data.py --n 3 --gpu 0

If the cache doesn't exist, this script prints the exact prebuild command
to run and exits.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Cache discovery
# ---------------------------------------------------------------------------


def find_latest_cache(cache_dir: Path, split: str = "train") -> Path | None:
    """Return the newest samples_<split>*.pkl in cache_dir, or None."""
    if not cache_dir.exists():
        return None
    candidates = sorted(
        cache_dir.glob(f"samples_{split}*.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_cache(cache_path: Path) -> List[Dict[str, Any]]:
    with open(cache_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Sample → v2 inputs
# ---------------------------------------------------------------------------


def sg_dict_from_scene_graph(
    sg_json: Dict[str, Any],
    dicom_id: str,
    image_w: int,
    image_h: int,
    num_regions: int = 310,
    num_entities: int = 237,
) -> Dict[str, Any]:
    """
    Convert a QBA scene graph JSON into the dict format SceneGraphEncoderV2
    expects: bboxes (normalized), entity_ids, region_ids, positiveness,
    num_objects. Mirrors the project's SceneGraphProcessor logic.
    """
    import numpy as np

    # Stable per-process name → int id table (cached on the function)
    if not hasattr(sg_dict_from_scene_graph, "_region_map"):
        sg_dict_from_scene_graph._region_map = {}
        sg_dict_from_scene_graph._entity_map = {}
    region_map = sg_dict_from_scene_graph._region_map
    entity_map = sg_dict_from_scene_graph._entity_map

    def _id(name: str, table: Dict[str, int], mod: int) -> int:
        if name not in table:
            table[name] = (hash(name) & 0x7FFFFFFF) % mod
        return table[name]

    bboxes: List[List[float]] = []
    entity_ids: List[int] = []
    region_ids: List[int] = []
    positiveness: List[int] = []

    for obs in (sg_json.get("observations") or {}).values():
        loc = obs.get("localization", {}).get(dicom_id, {})
        obs_bboxes = loc.get("bboxes", []) if isinstance(loc, dict) else []
        if not obs_bboxes:
            continue
        x1, y1, x2, y2 = obs_bboxes[0]
        bb = [
            max(0.0, min(1.0, float(x1) / max(image_w, 1))),
            max(0.0, min(1.0, float(y1) / max(image_h, 1))),
            max(0.0, min(1.0, float(x2) / max(image_w, 1))),
            max(0.0, min(1.0, float(y2) / max(image_h, 1))),
        ]
        if bb[2] <= bb[0] or bb[3] <= bb[1]:
            continue
        regions = obs.get("regions", [])
        region_name = (
            regions[0]["region"] if regions and isinstance(regions[0], dict)
            else (str(regions[0]) if regions else "unknown")
        )
        entities = obs.get("obs_entities", [])
        entity_name = str(entities[0]) if entities else "unknown"
        pos = obs.get("positiveness", "unknown")
        pos_id = {"pos": 1, "neg": 0}.get(pos, 2)

        bboxes.append(bb)
        region_ids.append(_id(region_name, region_map, num_regions))
        entity_ids.append(_id(entity_name, entity_map, num_entities))
        positiveness.append(pos_id)

    return {
        "bboxes": np.asarray(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32),
        "entity_ids": np.asarray(entity_ids, dtype=np.int64),
        "region_ids": np.asarray(region_ids, dtype=np.int64),
        "positiveness": np.asarray(positiveness, dtype=np.int64),
        "num_objects": len(bboxes),
    }


def format_answer_text(sample: Dict[str, Any], image_w: int, image_h: int) -> str:
    """Format a sample's first main-answer as '<think>...</think><box>...</box><answer>...</answer>'."""
    answers = sample.get("answers") or []
    mains = [a for a in answers if a.get("answer_type") == "main_answer"]
    a = mains[0] if mains else (answers[0] if answers else {})
    text = a.get("text", "Unknown.") if isinstance(a, dict) else "Unknown."

    # Best bbox for the chosen image
    bbox_str = "0.000,0.000,1.000,1.000"
    loc = a.get("localization", {}).get(sample["dicom_id"], {}) if isinstance(a, dict) else {}
    if isinstance(loc, dict):
        bboxes = loc.get("bboxes", [])
        if bboxes:
            x1, y1, x2, y2 = bboxes[0]
            bbox_str = (
                f"{x1/max(image_w,1):.3f},{y1/max(image_h,1):.3f},"
                f"{x2/max(image_w,1):.3f},{y2/max(image_h,1):.3f}"
            )

    regions = ", ".join(str(r) for r in a.get("regions", []) if r) if isinstance(a, dict) else ""
    entities = ", ".join(str(e) for e in a.get("obs_entities", []) if e) if isinstance(a, dict) else ""
    cot_parts = []
    if regions:
        cot_parts.append(f"Region(s) of interest: {regions}.")
    if entities:
        cot_parts.append(f"Observed: {entities}.")
    cot = " ".join(cot_parts) if cot_parts else "Reviewing the chest radiograph."

    return f"<think>{cot}</think><box>{bbox_str}</box><answer>{text}</answer>"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Test SSGVQANetV2 on cached real samples")
    p.add_argument("--cache_dir", default=".cache/dataset_samples",
                   help="Where prebuild_cache.py wrote samples_<split>*.pkl")
    p.add_argument("--cache_path", default=None,
                   help="Specific cache .pkl (overrides --cache_dir auto-pick)")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    args = p.parse_args(argv)

    print(f"=== SSGVQANetV2 cached-sample test (n={args.n}) ===\n")

    # ---- Find the cache ---------------------------------------------------
    if args.cache_path:
        cache_path = Path(args.cache_path)
    else:
        cache_path = find_latest_cache(Path(args.cache_dir), args.split)

    if cache_path is None or not cache_path.exists():
        print(f"✗ No cache found in {args.cache_dir} for split '{args.split}'.")
        print("\nBuild a small one with the project's own canonical builder:")
        print()
        print("  python scripts/prebuild_cache.py \\")
        print(f"      --mimic_cxr_path data/mimic-cxr-jpg \\")
        print(f"      --mimic_qa_path  data/mimic-ext-cxr-qba \\")
        print(f"      --max_samples 50 --num_workers 4 --split {args.split}")
        print()
        print("Then re-run this script.")
        return 1

    print(f"[cache] {cache_path}")
    samples = load_cache(cache_path)
    print(f"[cache] {len(samples)} samples available; using first {args.n}\n")
    samples = samples[: args.n]

    # ---- Build PIL images, scene_graph dicts, structured answers ----------
    from PIL import Image
    pil_images: List[Any] = []
    questions: List[str] = []
    answer_texts: List[str] = []
    scene_graphs: List[Dict[str, Any]] = []
    gt_bboxes: List[List[float]] = []

    for i, s in enumerate(samples):
        img_path = Path(s["image_path"])
        if not img_path.exists():
            print(f"  ✗ sample {i+1}: image missing at {img_path}")
            return 1
        pil = Image.open(img_path).convert("RGB")
        iw, ih = pil.size

        sg_dict = {"bboxes": [], "entity_ids": [], "region_ids": [], "positiveness": [], "num_objects": 0}
        sg_path = s.get("scene_graph_path")
        if sg_path and Path(sg_path).exists():
            with open(sg_path) as f:
                sg_json = json.load(f)
            sg_dict = sg_dict_from_scene_graph(sg_json, s["dicom_id"], iw, ih)

        ans_text = format_answer_text(s, iw, ih)
        # Parse bbox out of the structured answer for the model's training-time init
        try:
            bbox_str = ans_text.split("<box>")[1].split("</box>")[0]
            bx = [float(x) for x in bbox_str.split(",")]
        except Exception:
            bx = [0.0, 0.0, 1.0, 1.0]

        pil_images.append(pil)
        questions.append(s.get("question", ""))
        answer_texts.append(ans_text)
        scene_graphs.append(sg_dict)
        gt_bboxes.append(bx)

        print(f"  ✓ sample {i+1}: study={s['study_id']} dicom={s['dicom_id']}  "
              f"size={iw}x{ih}  sg_objects={sg_dict['num_objects']}")
        print(f"      Q: {s.get('question', '')[:100]}")
        print(f"      A: {ans_text.split('<answer>')[-1].split('</answer>')[0][:100]}")

    # ---- Model + forward + backward + generate ----------------------------
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        cc = torch.cuda.get_device_capability(args.gpu)
        dtype = torch.bfloat16 if cc >= (8, 0) else torch.float16
        device = torch.device(f"cuda:{args.gpu}")
        force_qlora = cc < (8, 0)
    else:
        dtype = torch.float32
        device = torch.device("cpu")
        force_qlora = False

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

    print(f"\n[train] forward + loss + backward + step")
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
        print(f"  ✗ training step failed: {e}")
        traceback.print_exc()
        return 1

    bb = out["grounding_outputs"]["bbox_pred"]
    print(f"\n[grounding] pred vs gt (normalized):")
    for i in range(len(samples)):
        pr = bb[i].tolist()
        gt = gt_bboxes[i]
        print(f"  {i+1}  pred=[{pr[0]:.3f},{pr[1]:.3f},{pr[2]:.3f},{pr[3]:.3f}]"
              f"  gt=[{gt[0]:.3f},{gt[1]:.3f},{gt[2]:.3f},{gt[3]:.3f}]")

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
