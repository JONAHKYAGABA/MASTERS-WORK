#!/usr/bin/env python3
"""
scripts/predict_and_visualize.py — clinician-facing inference + bbox overlay.

Workflow:
  1. Doctor uploads an X-ray (any size, any view)
  2. Script resizes to Qwen's input range
  3. Model generates: scene graph (predicted from the image) +
     <think>reasoning</think><box>x1,y1,x2,y2</box><answer>...</answer>
  4. Bbox is drawn on the original-resolution image
  5. Outputs: annotated PNG + console-printed answer + JSON sidecar

Usage:
    # Untrained baseline (sanity check)
    python scripts/predict_and_visualize.py --image /path/to/xray.jpg

    # With a fine-tuned LoRA checkpoint
    python scripts/predict_and_visualize.py \\
        --image /path/to/xray.jpg \\
        --checkpoint training/checkpoints/run_v1/best_lora \\
        --question "Are there any signs of pneumonia? Highlight the location."

    # Multiple images, batched
    python scripts/predict_and_visualize.py \\
        --image_dir /path/to/xrays/ \\
        --output_dir /path/to/predictions/

Output (per image):
    <stem>_annotated.png   — original image with bbox + label overlay
    <stem>_prediction.json — full structured output: bbox, answer, reasoning,
                              chexpert probabilities, etc.

Honest expectations:
  - Untrained model: bbox will be near the centre anchor [0.25,0.25,0.75,0.75]
    and the answer will be Qwen's default medical-knowledge response. Not
    clinically useful, but proves the plumbing works.
  - After Stage 3 LoRA fine-tuning: bboxes track real findings, answers
    follow the radiology-report style from QBA training data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------


def build_model(
    model_id: str,
    gpu: int,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[Any, torch.device, torch.dtype]:
    """Build customvqamodel, optionally load fine-tuned LoRA adapter."""
    from models import customvqamodel

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        cc = torch.cuda.get_device_capability(gpu)
        dtype = torch.bfloat16 if cc >= (8, 0) else torch.float16
        device = torch.device(f"cuda:{gpu}")
        force_qlora = cc < (8, 0)
    else:
        dtype, device, force_qlora = torch.float32, torch.device("cpu"), False

    print(f"[model] {model_id}  device={device}  dtype={dtype}  qlora={force_qlora}")
    t0 = time.time()

    if checkpoint_path and checkpoint_path.exists():
        print(f"[model] loading from checkpoint: {checkpoint_path}")
        model = customvqamodel.from_pretrained(
            str(checkpoint_path),
            torch_dtype=dtype,
            use_quantization=force_qlora,
        )
    else:
        if checkpoint_path:
            warnings.warn(f"Checkpoint not found at {checkpoint_path} — using untrained model")
        model = customvqamodel(
            qwen_model_id=model_id,
            use_quantization=force_qlora,
            num_sg_tokens=4,
            training_mode="finetune",
            torch_dtype=dtype,
            max_answer_length=128,
        )

    for name in ("sg_encoder", "sg_projector", "grounding_head", "aux_heads", "view_proj"):
        getattr(model, name).to(device=device, dtype=dtype)
    model.sg_generator.to(device=device)
    model.eval()
    print(f"[model] ready in {time.time()-t0:.1f}s")
    return model, device, dtype


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def load_and_resize(image_path: Path, max_side: int = 448) -> Tuple[Image.Image, Image.Image, Tuple[int, int]]:
    """Returns (original_pil, resized_pil_for_model, original_size_wh)."""
    original = Image.open(image_path).convert("RGB")
    orig_size = original.size
    resized = original.copy()
    resized.thumbnail((max_side, max_side), Image.LANCZOS)
    return original, resized, orig_size


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


_BOX_RE = re.compile(
    r"<box>\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*</box>"
)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_all_bboxes(text: str) -> List[List[float]]:
    """Extract every <box>...</box> in the generated text (model may emit multiple)."""
    out = []
    for m in _BOX_RE.finditer(text or ""):
        try:
            coords = [max(0.0, min(1.0, float(m.group(i)))) for i in (1, 2, 3, 4)]
            if coords[2] > coords[0] and coords[3] > coords[1]:
                out.append(coords)
        except (ValueError, IndexError):
            pass
    return out


def run_inference(
    model: Any,
    resized_pil: Image.Image,
    question: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Run the model on one (image, question) pair and return parsed predictions."""
    t0 = time.time()
    with torch.no_grad():
        out = model(
            images=None,
            pil_images=[resized_pil],
            questions=[question],
            scene_graphs=None,    # let the SG generator predict
        )
    elapsed = time.time() - t0

    raw_text = (out.get("generated_answer_text") or [""])[0]
    answer_match = _ANSWER_RE.search(raw_text)
    think_match = _THINK_RE.search(raw_text)
    text_bboxes = parse_all_bboxes(raw_text)

    refined_bbox = out["grounding_outputs"]["bbox_pred"][0].tolist()  # (4,)

    # CheXpert per-class probabilities (14 classes)
    chexpert_names = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
        "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
        "Pneumothorax", "Support Devices",
    ]
    chexpert_logits = out.get("chexpert_logits")
    chex_probs = {}
    if chexpert_logits is not None:
        probs = torch.sigmoid(chexpert_logits[0].float()).tolist()
        chex_probs = {n: round(p, 3) for n, p in zip(chexpert_names, probs)}

    # Scene graph predictions (from the SG generator if it ran)
    sg = out.get("generated_scene_graphs")
    sg_summary = None
    if sg and len(sg) > 0:
        sg_summary = {"num_objects": sg[0].get("num_objects", 0)}

    return {
        "question": question,
        "answer": (answer_match.group(1).strip() if answer_match else raw_text.strip()),
        "reasoning": (think_match.group(1).strip() if think_match else None),
        "bbox_refined": [round(v, 3) for v in refined_bbox],          # from grounding head
        "bboxes_from_text": [[round(v, 3) for v in b] for b in text_bboxes],  # from <box> tokens
        "chexpert_probs": chex_probs,
        "scene_graph": sg_summary,
        "raw_generation": raw_text,
        "inference_seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


# Color palette for multiple findings
_BBOX_COLORS = [
    "#FF3B30", "#FF9500", "#FFCC00", "#34C759",
    "#00C7BE", "#30B0C7", "#007AFF", "#AF52DE",
]


def _font(size: int = 18) -> Optional[ImageFont.FreeTypeFont]:
    """Best-effort font load — falls back to PIL default if no truetype available."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return None


def draw_annotation(
    original: Image.Image,
    prediction: Dict[str, Any],
    out_path: Path,
) -> Path:
    """Draw bbox(es) + answer text on the original image and save."""
    img = original.copy().convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    # Pick which bboxes to draw: prefer multi-bbox from text if available,
    # otherwise fall back to the single refined bbox from the grounding head.
    bboxes = prediction["bboxes_from_text"] or [prediction["bbox_refined"]]

    line_width = max(3, min(W, H) // 200)
    font_label = _font(max(14, min(W, H) // 60))

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        px1, py1, px2, py2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
        color = _BBOX_COLORS[i % len(_BBOX_COLORS)]

        # Rectangle
        draw.rectangle([px1, py1, px2, py2], outline=color, width=line_width)

        # Label background
        label = f"#{i+1}"
        if font_label:
            text_bbox = draw.textbbox((0, 0), label, font=font_label)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]
        else:
            tw, th = 24, 18
        draw.rectangle([px1, py1 - th - 6, px1 + tw + 10, py1], fill=color)
        if font_label:
            draw.text((px1 + 5, py1 - th - 4), label, fill="white", font=font_label)
        else:
            draw.text((px1 + 5, py1 - th - 4), label, fill="white")

    # Answer box at the bottom — semi-transparent background + wrapped text
    answer = prediction["answer"][:400]
    if answer:
        font_ans = _font(max(14, min(W, H) // 70))
        # Word-wrap manually
        max_chars = max(40, W // 12)
        words, lines, cur = answer.split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 <= max_chars:
                cur = (cur + " " + w).strip()
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)

        line_h = (font_ans.size + 6) if font_ans else 20
        box_h = line_h * len(lines) + 16
        draw.rectangle([0, H - box_h, W, H], fill=(0, 0, 0, 180))
        for j, line in enumerate(lines):
            y = H - box_h + 8 + j * line_h
            if font_ans:
                draw.text((10, y), line, fill="white", font=font_ans)
            else:
                draw.text((10, y), line, fill="white")

    img.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process_one(
    model: Any,
    device: torch.device,
    image_path: Path,
    out_dir: Path,
    question: str,
    max_side: int,
) -> Dict[str, Any]:
    """Run the full pipeline on one image."""
    print(f"\n[image] {image_path}")
    original, resized, orig_size = load_and_resize(image_path, max_side=max_side)
    print(f"  size: {orig_size[0]}x{orig_size[1]} → resized {resized.size[0]}x{resized.size[1]}")

    pred = run_inference(model, resized, question, device)

    print(f"  inference: {pred['inference_seconds']}s")
    print(f"  answer   : {pred['answer'][:200]}")
    if pred["reasoning"]:
        print(f"  reasoning: {pred['reasoning'][:200]}")
    print(f"  bbox(refined)    : {pred['bbox_refined']}")
    if pred["bboxes_from_text"]:
        print(f"  bbox(es from text): {pred['bboxes_from_text']}")

    # Top 3 chexpert findings by probability
    if pred["chexpert_probs"]:
        top3 = sorted(pred["chexpert_probs"].items(), key=lambda kv: -kv[1])[:3]
        print(f"  top chexpert: " + ", ".join(f"{k}({v:.2f})" for k, v in top3))

    # Save annotated image + JSON
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    annotated_path = out_dir / f"{stem}_annotated.png"
    draw_annotation(original, pred, annotated_path)
    print(f"  ✓ wrote {annotated_path}")

    json_path = out_dir / f"{stem}_prediction.json"
    with open(json_path, "w") as f:
        json.dump(pred, f, indent=2)
    print(f"  ✓ wrote {json_path}")

    return pred


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Predict + visualize on a clinician-uploaded X-ray")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="Single image file")
    src.add_argument("--image_dir", type=Path, help="Directory of images (processes *.jpg / *.png)")

    p.add_argument("--output_dir", type=Path, default=Path("predictions"),
                   help="Where to write annotated images + JSON sidecars")
    p.add_argument("--question", type=str,
                   default="Identify any abnormalities visible in this chest X-ray "
                           "and highlight their location with a bounding box.",
                   help="Question/prompt to ask about each image")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Fine-tuned customvqamodel checkpoint dir (saved via save_pretrained)")
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                   help="Base model. Use Qwen3-VL-4B-Instruct for faster smoke runs.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max_side", type=int, default=448,
                   help="Resize so the longest image side ≤ this value before model input")
    args = p.parse_args(argv)

    if args.image and not args.image.exists():
        print(f"✗ image not found: {args.image}")
        return 1
    if args.image_dir and not args.image_dir.exists():
        print(f"✗ image_dir not found: {args.image_dir}")
        return 1

    images: List[Path]
    if args.image:
        images = [args.image]
    else:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
        images = []
        for e in exts:
            images.extend(sorted(args.image_dir.glob(e)))
        if not images:
            print(f"✗ no images found in {args.image_dir}")
            return 1
        print(f"[batch] found {len(images)} images")

    model, device, _ = build_model(args.model_id, args.gpu, args.checkpoint)

    summary = []
    for img_path in images:
        try:
            pred = process_one(model, device, img_path, args.output_dir,
                               args.question, args.max_side)
            summary.append({"image": str(img_path), "answer": pred["answer"]})
        except Exception as e:
            print(f"  ✗ failed: {e}")
            import traceback; traceback.print_exc()
            summary.append({"image": str(img_path), "error": str(e)})

    # Batch summary
    print(f"\n=== Done. Processed {len(summary)} images. ===")
    print(f"Annotated images + JSON sidecars in: {args.output_dir.resolve()}")
    return 0 if all("error" not in s for s in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
