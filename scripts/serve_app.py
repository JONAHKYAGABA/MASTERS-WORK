#!/usr/bin/env python3
"""
scripts/serve_app.py — FastAPI inference server + optional ngrok tunnel.

A minimal web UI that lets a user upload a chest X-ray and ask a question.
The server:
  - Loads the trained SSGVQANetV2 once at startup (the Stage 4 best_model by
    default; override with --checkpoint).
  - Serves a single HTML page at GET / with an upload form.
  - Accepts multipart POST /predict (image + question), runs inference, and
    returns JSON containing:
      * generated answer (parsed <answer>...</answer>)
      * reasoning trace (parsed <think>...</think>)
      * refined grounding bbox (from the grounding head)
      * any <box> coordinates the LM emitted (often multiple)
      * scene graph: per-object bboxes + entity names + region names
      * CheXpert top-5 findings with probabilities
      * the original image with bboxes drawn on it (base64-encoded PNG)
      * the raw LM generation for debugging

Why this file exists separately from predict_and_visualize.py:
  predict_and_visualize.py is a CLI batch tool that writes annotated PNGs and
  JSON sidecars to disk. The serving path needs the full scene-graph dict
  (not just a num_objects summary) and has to return bytes, so we inline the
  inference + drawing here rather than try to retrofit the CLI script.

Usage:
    pip install fastapi uvicorn python-multipart pyngrok pillow

    # Local-only (LAN access only)
    python scripts/serve_app.py

    # With public ngrok tunnel
    NGROK_AUTHTOKEN=your_token_here python scripts/serve_app.py --tunnel

    # Override defaults
    python scripts/serve_app.py \\
        --checkpoint ./checkpoints/stage4_finetune_qwen3vl8b_5k/best_model \\
        --port 8000 \\
        --gpu 0 \\
        --tunnel

Get a free ngrok auth token at https://dashboard.ngrok.com/get-started/your-authtoken
Then either:
  - export NGROK_AUTHTOKEN=...   (read by this script when --tunnel is set)
  - or run `ngrok config add-authtoken <token>` once to save it globally
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

# FastAPI imports MUST be at module top — `from __future__ import annotations`
# (line 41) turns every annotation into a string ForwardRef. When FastAPI calls
# pydantic's TypeAdapter on `image: UploadFile = File(...)`, pydantic evals the
# string "UploadFile" against the MODULE-GLOBAL namespace, NOT the function's
# local scope. Importing UploadFile inside build_app() therefore raises
# NameError: name 'UploadFile' is not defined at endpoint-registration time.
# Hoisting these imports to module top fixes that.
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

# Make `from models import ...` work whether this is run from repo root or scripts/.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("serve_app")


# ---------------------------------------------------------------------------
# Vocab loading — turns entity_id 42 into "consolidation" in the response
# ---------------------------------------------------------------------------

def load_vocab(dataset_info_path: Optional[Path]) -> Tuple[List[str], List[str]]:
    """Return (region_names, entity_names). Empty lists if file missing."""
    if not dataset_info_path or not dataset_info_path.exists():
        log.warning(
            f"Vocab not found at {dataset_info_path}; scene-graph response "
            "will show numeric IDs instead of names."
        )
        return [], []
    with open(dataset_info_path) as f:
        info = json.load(f)
    regions = info.get("regions") or info.get("region_names") or []
    entities = info.get("finding_entities") or info.get("entity_names") or []
    log.info(f"Loaded vocab: {len(regions)} regions, {len(entities)} entities")
    return regions, entities


# ---------------------------------------------------------------------------
# Model loading — mirrors scripts/predict_and_visualize.py:build_model
# ---------------------------------------------------------------------------

def build_model(
    model_id: str,
    gpu: int,
    checkpoint_path: Optional[Path],
    trust_checkpoint: bool = False,
):
    """Build a fresh SSGVQANetV2 and load weights from pytorch_model.bin.

    We deliberately AVOID ``SSGVQANetV2.from_pretrained`` because the
    config.json saved next to the model is the full training config
    (nested: ``{model: {...}, training: {...}, data: {...}}``), not the
    flat init kwargs the model constructor accepts. Calling
    from_pretrained therefore raises:

        TypeError: SSGVQANetV2.__init__() got an unexpected keyword
        argument 'model'

    Instead we mirror exactly what train_mimic_cxr.py does on resume:
    construct the model with explicit kwargs, then load
    ``pytorch_model.bin`` with strict=False (bitsandbytes quant metadata
    intentionally produces "unexpected keys" — harmless).
    """
    from models import SSGVQANetV2

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        cc = torch.cuda.get_device_capability(gpu)
        dtype = torch.bfloat16 if cc >= (8, 0) else torch.float16
        device = torch.device(f"cuda:{gpu}")
        force_qlora = cc < (8, 0)
    else:
        dtype, device, force_qlora = torch.float32, torch.device("cpu"), False

    log.info(f"device={device}  dtype={dtype}  qlora={force_qlora}  model={model_id}")
    t0 = time.time()

    # Always construct fresh — single code path, no from_pretrained surprises.
    # num_sg_tokens=8 MUST match what training used (visible in the training
    # log: `num_sg_tokens=8`). It controls two shapes that affect state-dict
    # compatibility:
    #   - tokenizer vocab grows by num_sg_tokens (adds <|sg_token_N|> tokens),
    #     so qwen.base_model.model.model.language_model.embed_tokens.weight
    #     and qwen.base_model.model.lm_head.weight have shape
    #     [151669 + num_sg_tokens, 4096]
    #   - sg_projector.queries has shape [num_sg_tokens, 128]
    # Setting this lower than training would raise size_mismatch on those
    # exact tensors (which is what happened with num_sg_tokens=4).
    #
    # max_answer_length=384: the model emits the structured template
    #   <think>...</think><box>...</box><answer>...</answer>
    # which is ~50-80 tokens of wrapper around an answer that itself can run
    # 100-200 tokens for verbose findings. 128 tokens (training default)
    # truncates mid-<answer> — the parser then can't find </answer>, so the
    # entire raw generation (including <think> and <box> tags) leaks into
    # the "Answer" field. 384 gives ~300 tokens for the actual answer text
    # while keeping inference under ~12s on Qwen3-VL-8B QLoRA.
    log.info("constructing SSGVQANetV2 with explicit kwargs (training_mode=finetune)")
    model = SSGVQANetV2(
        qwen_model_id=model_id,
        use_quantization=force_qlora,
        num_sg_tokens=8,
        training_mode="finetune",
        torch_dtype=dtype,
        max_answer_length=384,
    )

    # Cast trainable / non-quantised modules to device+dtype
    for name in ("sg_encoder", "sg_projector", "grounding_head", "aux_heads", "view_proj"):
        sub = getattr(model, name, None)
        if sub is not None:
            sub.to(device=device, dtype=dtype)
    if getattr(model, "sg_generator", None) is not None:
        model.sg_generator.to(device=device)
    # Recast mHC to fp32 — same Option-C stability fix used in training.
    if hasattr(model, "grounding_head") and getattr(model.grounding_head, "mhc", None) is not None:
        model.grounding_head.mhc.to(dtype=torch.float32)

    # ---- Load fine-tuned weights from pytorch_model.bin ----
    if checkpoint_path:
        bin_path = checkpoint_path / "pytorch_model.bin"
        if bin_path.exists():
            log.info(f"loading weights from {bin_path}")
            # weights_only=True is the safe default in torch >= 2.6, but the
            # trainer pickles a fan-out of numpy types alongside the tensors
            # (numpy scalars, parameterised dtypes like numpy.dtype[float64],
            # etc. — whatever sklearn/numpy stashed into metric values that
            # got swept into the state_dict). Whitelisting each one is
            # whack-a-mole; the principled answer is:
            #
            #   - When --trust_checkpoint is set, load with weights_only=False.
            #     This permits arbitrary unpickling — safe ONLY for files the
            #     operator authored or fetched from a known source.
            #
            #   - Otherwise, try the strict path with as many numpy globals
            #     whitelisted as we know about, and raise a clear error
            #     (with the --trust_checkpoint hint) on failure.
            if trust_checkpoint:
                log.warning(
                    "--trust_checkpoint set: loading with weights_only=False. "
                    "This permits arbitrary code execution from the file. "
                    "Make sure you trust the origin of this checkpoint."
                )
                state = torch.load(str(bin_path), map_location="cpu", weights_only=False)
            else:
                try:
                    import numpy as _np
                    _allow = []
                    for _attr in ("scalar", "_reconstruct", "ndarray"):
                        obj = getattr(_np.core.multiarray, _attr, None)
                        if obj is not None:
                            _allow.append(obj)
                    for _attr in ("dtype", "ndarray"):
                        obj = getattr(_np, _attr, None)
                        if obj is not None:
                            _allow.append(obj)
                    # Parameterised dtypes (numpy >= 1.25). These are concrete
                    # classes like Float64DType / Int64DType / etc.
                    dtypes_mod = getattr(_np, "dtypes", None)
                    if dtypes_mod is not None:
                        for _attr in dir(dtypes_mod):
                            if _attr.endswith("DType"):
                                obj = getattr(dtypes_mod, _attr, None)
                                if obj is not None:
                                    _allow.append(obj)
                    if _allow:
                        torch.serialization.add_safe_globals(_allow)
                except Exception as _e:
                    log.warning(f"could not whitelist numpy globals: {_e}")
                try:
                    state = torch.load(str(bin_path), map_location="cpu", weights_only=True)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load {bin_path} with weights_only=True. "
                        f"Underlying error: {e}\n\n"
                        "If you trust this checkpoint (e.g. you trained it yourself), "
                        "re-run with --trust_checkpoint to enable weights_only=False."
                    ) from e
            # ---- Unwrap if the checkpoint is a save_checkpoint() bundle ----
            # save_checkpoint() in train_mimic_cxr.py:385 saves
            # {'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
            #  'epoch', 'global_step', 'rng_state', 'scaler_state_dict', 'metrics'}
            # — roughly 8 top-level keys. We only want model_state_dict; loading
            # the wrapper directly matches zero weight tensors and silently leaves
            # the model with random LoRA + heads (produces garbage at inference).
            if isinstance(state, dict) and "model_state_dict" in state:
                log.info(
                    f"unwrapped checkpoint bundle (top-level keys: "
                    f"{list(state.keys())}); extracting model_state_dict"
                )
                state = state["model_state_dict"]

            # ---- Strip DDP "module." prefix from saved keys ----
            # The trainer saves model.state_dict() AFTER wrapping with
            # DistributedDataParallel, so every parameter key starts with
            # "module.". A non-DDP model can't match those keys and
            # load_state_dict raises (some failures bypass strict=False).
            if isinstance(state, dict) and state:
                sample_key = next(iter(state.keys()))
                if sample_key.startswith("module."):
                    log.info("stripping DDP 'module.' prefix from checkpoint keys")
                    state = {k[len("module."):]: v for k, v in state.items()}
            # The training save writes the trainable / custom-component
            # state_dict; the bnb-quantised base weights are NOT in this
            # file (they live in the HF cache and get loaded by the
            # constructor above). Strict=False is REQUIRED.
            missing, unexpected = model.load_state_dict(state, strict=False)
            # missing = params in the model but not in the file (mostly bnb-
            # quantised base weights, which load from the HF cache via the
            # constructor — expected).
            # unexpected = keys in the file but not in the model (mostly bnb
            # quant_state metadata — also expected).
            log.info(
                f"loaded {len(state)} weight keys from checkpoint "
                f"(model has {len(missing)} keys not in file, file has "
                f"{len(unexpected)} keys not in model — both mostly bnb "
                f"quant_state, expected)"
            )
        else:
            warnings.warn(
                f"Checkpoint path {checkpoint_path} has no pytorch_model.bin; "
                "serving with UNTRAINED LoRA + heads. Outputs will be garbage."
            )

    model.eval()
    log.info(f"model ready in {time.time()-t0:.1f}s")
    return model, device, dtype


# ---------------------------------------------------------------------------
# Inference — replicates run_inference but returns the full scene graph
# ---------------------------------------------------------------------------

_BOX_RE = re.compile(
    r"<box>\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*</box>"
)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# Truncation-tolerant fallback: when the LM hits max_answer_length mid-output
# the closing </answer> tag is missing. Match <answer> through end of string
# so we still extract the partial answer text instead of dumping the entire
# raw generation (including <think> + <box>) into the "Answer" field.
_ANSWER_OPEN_RE = re.compile(r"<answer>(.*)", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# Truncation-tolerant for <think> too — same reason.
_THINK_OPEN_RE = re.compile(r"<think>(.*?)(?=<box>|<answer>|$)", re.DOTALL)

CHEXPERT_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]


def parse_text_bboxes(text: str) -> List[List[float]]:
    out = []
    for m in _BOX_RE.finditer(text or ""):
        try:
            coords = [max(0.0, min(1.0, float(m.group(i)))) for i in (1, 2, 3, 4)]
            if coords[2] > coords[0] and coords[3] > coords[1]:
                out.append(coords)
        except (ValueError, IndexError):
            pass
    return out


def _to_list(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def run_inference(
    model,
    pil_image: Image.Image,
    question: str,
    region_names: List[str],
    entity_names: List[str],
) -> Dict[str, Any]:
    """Run the model and return a serialisable dict including the FULL scene graph."""
    t0 = time.time()
    with torch.no_grad():
        out = model(
            images=None,
            pil_images=[pil_image],
            questions=[question],
            scene_graphs=None,  # let the SG generator predict from the image
        )
    elapsed = time.time() - t0

    raw_text = (out.get("generated_answer_text") or [""])[0]

    # Answer extraction with truncation-tolerant fallback.
    # Order: closed <answer>...</answer> → open <answer>... (truncated at end)
    #        → empty (rather than dumping <think> + <box> into the answer field
    #          which is what the old "show raw_text as fallback" did).
    m_closed = _ANSWER_RE.search(raw_text)
    if m_closed:
        answer = m_closed.group(1).strip()
        answer_truncated = False
    else:
        m_open = _ANSWER_OPEN_RE.search(raw_text)
        if m_open:
            answer = m_open.group(1).strip()
            answer_truncated = True  # ← mark so the UI can show a hint
        else:
            answer = ""
            answer_truncated = False

    # Reasoning extraction with the same fallback approach.
    m_think_closed = _THINK_RE.search(raw_text)
    if m_think_closed:
        reasoning = m_think_closed.group(1).strip()
    else:
        m_think_open = _THINK_OPEN_RE.search(raw_text)
        reasoning = m_think_open.group(1).strip() if m_think_open else None

    text_bboxes = parse_text_bboxes(raw_text)

    # Grounding head's refined bbox (the one the model is most confident about)
    refined = out.get("grounding_outputs", {}).get("bbox_pred")
    bbox_refined = _to_list(refined[0]) if refined is not None else [0.0, 0.0, 1.0, 1.0]
    bbox_refined = [float(max(0.0, min(1.0, v))) for v in bbox_refined]

    # CheXpert probabilities → top-5
    chex_logits = out.get("chexpert_logits")
    chex_top = {}
    if chex_logits is not None:
        probs = torch.sigmoid(chex_logits[0].float()).tolist()
        chex_pairs = sorted(
            zip(CHEXPERT_NAMES, probs), key=lambda kv: -kv[1]
        )[:5]
        chex_top = {n: round(p, 3) for n, p in chex_pairs}

    # Scene graph: pull the dict produced by _sg_outputs_to_dicts
    # (bboxes, entity_ids, region_ids, positiveness, relations, num_objects)
    sg_objects: List[Dict[str, Any]] = []
    sg_list = out.get("generated_scene_graphs") or []
    if sg_list:
        sg = sg_list[0]
        n = int(sg.get("num_objects", 0))
        bboxes = sg.get("bboxes", [])
        ent_ids = sg.get("entity_ids", [])
        reg_ids = sg.get("region_ids", [])
        pos = sg.get("positiveness", [])
        for i in range(n):
            ent_idx = int(ent_ids[i]) if i < len(ent_ids) else -1
            reg_idx = int(reg_ids[i]) if i < len(reg_ids) else -1
            sg_objects.append({
                "bbox": [float(max(0.0, min(1.0, v))) for v in _to_list(bboxes[i])],
                "entity_id": ent_idx,
                "entity": (entity_names[ent_idx] if 0 <= ent_idx < len(entity_names)
                           else f"entity_{ent_idx}"),
                "region_id": reg_idx,
                "region": (region_names[reg_idx] if 0 <= reg_idx < len(region_names)
                           else f"region_{reg_idx}"),
                "positiveness": int(pos[i]) if i < len(pos) else 0,
            })

    return {
        "question": question,
        "answer": answer,
        "answer_truncated": answer_truncated,
        "reasoning": reasoning,
        "bbox_refined": [round(v, 3) for v in bbox_refined],
        "bboxes_from_text": [[round(v, 3) for v in b] for b in text_bboxes],
        "scene_graph": {
            "num_objects": len(sg_objects),
            "objects": sg_objects,
        },
        "chexpert_top": chex_top,
        "raw_generation": raw_text,
        "inference_seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Drawing — overlay refined bbox + scene-graph objects + answer
# ---------------------------------------------------------------------------

_BBOX_COLORS = [
    "#FF3B30", "#FF9500", "#FFCC00", "#34C759",
    "#00C7BE", "#30B0C7", "#007AFF", "#AF52DE",
]


def _try_font(size: int) -> Optional[ImageFont.FreeTypeFont]:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return None


def _encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _iou_xyxy(a, b):
    """IoU on (x1,y1,x2,y2) tuples in any unit (we normalise to 0-1)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _dedupe_sg_objects(objects, iou_thresh=0.7):
    """Merge near-identical scene-graph proposals.

    The undertrained / mode-collapsed SG generator emits the same anatomy
    20+ times with bboxes within ~0.02 of each other. Drawing all 20 makes
    the image unreadable. Group by (entity, region) and merge when bbox IoU
    is high — keep the median box, attach a ``count`` so the legend can
    show "lungs x20".
    """
    groups = []  # each: {entity, region, bbox, count, bboxes}
    for o in objects:
        bb = o.get("bbox")
        if not bb or len(bb) != 4:
            continue
        ent = (o.get("entity") or "?").strip()
        reg = (o.get("region") or "?").strip()
        matched = None
        for g in groups:
            if g["entity"] == ent and g["region"] == reg:
                if _iou_xyxy(bb, g["bbox"]) > iou_thresh:
                    matched = g
                    break
        if matched is None:
            groups.append({"entity": ent, "region": reg, "bbox": list(bb),
                           "count": 1, "bboxes": [list(bb)]})
        else:
            matched["count"] += 1
            matched["bboxes"].append(list(bb))
            # update to the median box (more stable than first-occurrence)
            n = len(matched["bboxes"])
            if n >= 3 and n % 2 == 1:  # only re-median on odd counts to avoid jitter
                xs1 = sorted(b[0] for b in matched["bboxes"])
                ys1 = sorted(b[1] for b in matched["bboxes"])
                xs2 = sorted(b[2] for b in matched["bboxes"])
                ys2 = sorted(b[3] for b in matched["bboxes"])
                matched["bbox"] = [xs1[n // 2], ys1[n // 2],
                                   xs2[n // 2], ys2[n // 2]]
    return groups


def annotate_image(pil: Image.Image, pred: Dict[str, Any]) -> bytes:
    """Draw refined bbox + deduped scene-graph with labels OUTSIDE the X-ray.

    DESIGN:
      * Canvas is extended with a right sidebar (legend) and a bottom strip
        (answer). The X-ray itself only receives thin bbox outlines + tiny
        numeric tags (1, 2, 3, R for refined) at each box's top-left corner.
        Nothing larger than a single digit ever sits on top of the image.
      * Boxes are deduped (IoU > 0.7) so 20 identical proposals collapse
        to one entry showing "x20" in the legend.
      * Capped at 5 distinct scene-graph boxes.
    Always returns valid PNG bytes.
    """
    try:
        xray = pil.copy().convert("RGB")
    except Exception as e:
        log.warning(f"could not copy/convert PIL image: {e}")
        return _encode_png(Image.new("RGB", (64, 64), (60, 60, 60)))

    W, H = xray.size
    short_side = min(W, H)

    # Sidebar / bottom-strip sizes — proportional to the image but capped
    sidebar_w = max(180, min(320, W // 2))
    bottom_h = max(70, min(160, H // 4))

    # Font sizes (all text lives in sidebar/bottom, never on the X-ray)
    fs_legend = 13
    fs_legend_title = 12
    fs_ans = 14
    fs_tag = max(9, min(13, short_side // 80))  # tiny digit tags on boxes
    font_legend = _try_font(fs_legend)
    font_legend_title = _try_font(fs_legend_title)
    font_ans = _try_font(fs_ans)
    font_tag = _try_font(fs_tag)

    # ------------------------------------------------------------------
    # Pass 1: collect deduped boxes + assign 1-based numeric tags
    # ------------------------------------------------------------------
    sg_objects = pred.get("scene_graph", {}).get("objects", []) or []
    deduped = _dedupe_sg_objects(sg_objects, iou_thresh=0.7)[:5]

    items = []  # each: {tag, color, bbox, label, count}
    for i, g in enumerate(deduped):
        color = _BBOX_COLORS[(i + 1) % len(_BBOX_COLORS)]
        items.append({
            "tag": str(i + 1),
            "color": color,
            "bbox": g["bbox"],
            "entity": g["entity"],
            "region": g["region"],
            "count": g["count"],
            "kind": "sg",
        })
    for i, bb in enumerate(pred.get("bboxes_from_text", [])[:3]):
        items.append({
            "tag": f"L{i+1}",
            "color": "#AF52DE",
            "bbox": bb,
            "entity": "LM bbox",
            "region": "from text",
            "count": 1,
            "kind": "lm",
        })
    if pred.get("bbox_refined"):
        items.append({
            "tag": "R",
            "color": "#FF3B30",
            "bbox": pred["bbox_refined"],
            "entity": "refined grounding",
            "region": "answer bbox",
            "count": 1,
            "kind": "refined",
        })

    # ------------------------------------------------------------------
    # Pass 2: draw thin outlines + tiny corner digit tags on the X-ray
    # ------------------------------------------------------------------
    draw_x = ImageDraw.Draw(xray, "RGBA")
    lw = max(1, short_side // 400)
    MIN_PX = 4
    skipped = 0

    def _normalise(bbox):
        if not bbox or len(bbox) != 4:
            raise ValueError("need 4 floats")
        x1, y1, x2, y2 = (float(v) for v in bbox)
        x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
        px1 = max(0, min(W - 1, int(x1 * W)))
        py1 = max(0, min(H - 1, int(y1 * H)))
        px2 = max(0, min(W,     int(x2 * W)))
        py2 = max(0, min(H,     int(y2 * H)))
        if (px2 - px1) < MIN_PX or (py2 - py1) < MIN_PX:
            raise ValueError("degenerate")
        return px1, py1, px2, py2

    for it in items:
        try:
            px1, py1, px2, py2 = _normalise(it["bbox"])
        except ValueError:
            skipped += 1
            it["_drawn"] = False
            continue
        # outline only — no fill, no label-on-image
        draw_x.rectangle([px1, py1, px2, py2], outline=it["color"], width=lw)
        # tiny tag at top-left, ~16 px square, sits at box corner (or just
        # outside if the box is at the very top edge of the image)
        try:
            tag = it["tag"]
            if font_tag:
                tb = draw_x.textbbox((0, 0), tag, font=font_tag)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
            else:
                tw, th = 8, 10
            pad = 2
            chip_w = tw + 2 * pad
            chip_h = th + 2 * pad
            # default: just inside box top-left
            cx1, cy1 = px1, py1
            cx2, cy2 = cx1 + chip_w, cy1 + chip_h
            # if box hugs the top of the image, push tag down inside the box
            if py1 < chip_h + 2:
                cy1 = py1
                cy2 = cy1 + chip_h
            draw_x.rectangle([cx1, cy1, cx2, cy2], fill=it["color"])
            kw = {"font": font_tag} if font_tag else {}
            draw_x.text((cx1 + pad, cy1 + pad - 1), tag, fill="white", **kw)
        except Exception as e:
            log.debug(f"tag draw failed: {e}")
        it["_drawn"] = True

    if skipped:
        log.info(f"annotate_image: skipped {skipped} degenerate bboxes")

    # ------------------------------------------------------------------
    # Pass 3: build extended canvas (X-ray + sidebar + bottom strip)
    # ------------------------------------------------------------------
    canvas_w = W + sidebar_w
    canvas_h = H + bottom_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (24, 24, 24))
    canvas.paste(xray, (0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")

    # Subtle divider lines so the panels feel intentional
    draw.line([(W, 0), (W, canvas_h)], fill=(80, 80, 80), width=1)
    draw.line([(0, H), (canvas_w, H)], fill=(80, 80, 80), width=1)

    # ---- LEGEND in the right sidebar ----
    try:
        if items and font_legend:
            x0 = W + 10
            y = 12
            if font_legend_title:
                draw.text((x0, y), "Detections", fill=(220, 220, 220),
                          font=font_legend_title)
                y += fs_legend_title + 8
            line_h = fs_legend + 6
            max_text_w = sidebar_w - 30  # text region width
            for it in items:
                if not it.get("_drawn"):
                    continue
                # colored chip with the tag
                chip = 16
                draw.rectangle([x0, y, x0 + chip, y + chip], fill=it["color"])
                if font_tag:
                    tb = draw.textbbox((0, 0), it["tag"], font=font_tag)
                    tx = x0 + (chip - (tb[2] - tb[0])) // 2
                    ty = y + (chip - (tb[3] - tb[1])) // 2 - 1
                    draw.text((tx, ty), it["tag"], fill="white", font=font_tag)
                # entity + region text wrapped to fit the sidebar
                tx0 = x0 + chip + 6
                ent = it["entity"]
                reg = it["region"]
                count_suffix = f"  x{it['count']}" if it["count"] > 1 else ""
                line1 = ent + count_suffix
                line2 = f"@ {reg}" if reg and reg != "?" else ""
                # truncate each line to fit
                def _fit(s, max_w, f):
                    if not s or not f:
                        return s
                    tb = draw.textbbox((0, 0), s, font=f)
                    if tb[2] - tb[0] <= max_w:
                        return s
                    while s and (draw.textbbox((0, 0), s + ".", font=f)[2]
                                 - draw.textbbox((0, 0), s + ".", font=f)[0]
                                 > max_w):
                        s = s[:-1]
                    return s + "."
                line1 = _fit(line1, max_text_w - chip - 6, font_legend)
                line2 = _fit(line2, max_text_w - chip - 6, font_legend)
                draw.text((tx0, y), line1, fill=(240, 240, 240),
                          font=font_legend)
                if line2:
                    draw.text((tx0, y + line_h - 2), line2,
                              fill=(170, 170, 170), font=font_legend)
                    y += 2 * line_h + 4
                else:
                    y += line_h + 6
                if y > canvas_h - 20:
                    break
    except Exception as e:
        log.debug(f"legend draw failed: {e}")

    # ---- ANSWER in the bottom strip ----
    try:
        ans = (pred.get("answer") or "").strip()
        if ans and font_ans:
            avail_w = canvas_w - 20
            # naive word-wrap by pixel width
            words = ans.split()
            lines, cur = [], ""
            for w in words:
                trial = (cur + " " + w).strip()
                tb = draw.textbbox((0, 0), trial, font=font_ans)
                if tb[2] - tb[0] <= avail_w:
                    cur = trial
                else:
                    if cur:
                        lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
            line_h = fs_ans + 4
            max_lines = max(1, (bottom_h - 16) // line_h)
            lines = lines[:max_lines]
            for j, line in enumerate(lines):
                draw.text((10, H + 8 + j * line_h), line,
                          fill=(240, 240, 240), font=font_ans)
    except Exception as e:
        log.debug(f"answer overlay failed: {e}")

    try:
        return _encode_png(canvas)
    except Exception as e:
        log.warning(f"PNG encode failed: {e}; falling back to original")
        return _encode_png(pil.convert("RGB"))


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

HTML_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SSG-VQA — Chest X-Ray QA</title>
<!-- Scene-graph visualization is rendered with inline SVG (vanilla JS) —
     no CDN, no Subresource Integrity to manage, no dependency to vendor.
     The bipartite entity↔region layout is actually a better fit for our
     data than a generic force-directed graph from vis-network. -->
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         max-width: 920px; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; }
  h1 { margin-bottom: .25rem; }
  .sub { color: #666; margin-top: 0; }
  form { display: flex; flex-direction: column; gap: .75rem;
         padding: 1rem; border: 1px solid #ccc; border-radius: 8px; }
  label { font-weight: 600; }
  textarea, input[type=file] { font-size: 1rem; padding: .5rem; width: 100%;
         border: 1px solid #bbb; border-radius: 4px; font-family: inherit; }
  textarea { min-height: 4rem; resize: vertical; }
  button { padding: .75rem 1rem; font-size: 1.05rem; background: #007AFF;
           color: white; border: 0; border-radius: 6px; cursor: pointer; }
  button:disabled { opacity: .5; cursor: progress; }
  #result { margin-top: 2rem; }
  #result img { max-width: 100%; border: 1px solid #ccc; border-radius: 6px; }
  pre { background: #f4f4f4; padding: .75rem; overflow-x: auto;
        border-radius: 4px; font-size: .85rem; }
  table { border-collapse: collapse; width: 100%; margin-top: .5rem; }
  th, td { text-align: left; padding: .35rem .5rem; border-bottom: 1px solid #eee; }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 999px;
          background: #eee; font-size: .85rem; margin-right: 4px; }
  details { margin-top: 1rem; }
  summary { cursor: pointer; font-weight: 600; }
  #sg-graph { width: 100%; min-height: 360px;
              border: 1px solid #ccc; border-radius: 6px;
              background: #fafafa; overflow-x: auto; padding: 8px 0; }
  #sg-graph svg { display: block; margin: 0 auto; }
  #sg-graph .node-entity { fill: #007AFF; stroke: #003e80; }
  #sg-graph .node-region { fill: #34C759; stroke: #1a6e2e; }
  #sg-graph .node-label { fill: white; font-size: 12px; font-weight: 600;
                          text-anchor: middle; dominant-baseline: middle;
                          pointer-events: none; font-family: inherit; }
  #sg-graph .edge { stroke: #888; fill: none; opacity: .55; }
  #sg-graph .edge.pos { stroke: #FF3B30; }
  #sg-graph .edge.neg { stroke: #888; }
  #sg-graph .col-label { fill: #555; font-size: 11px; font-weight: 600;
                         text-anchor: middle; text-transform: uppercase;
                         letter-spacing: .05em; }
  .legend { display: flex; gap: 1rem; font-size: .85rem; flex-wrap: wrap;
            color: #666; margin: .5rem 0 .25rem; }
  .legend .swatch { display: inline-block; width: 12px; height: 12px;
                    border-radius: 3px; margin-right: 4px; vertical-align: middle; }
  @media (prefers-color-scheme: dark) {
    body { background: #1a1a1a; color: #eee; }
    form, pre { background: #2a2a2a; border-color: #444; }
    th, td { border-bottom-color: #333; }
    .pill { background: #333; }
    #sg-graph { background: #2a2a2a; border-color: #444; }
    #sg-graph .col-label { fill: #aaa; }
    #sg-graph .edge { opacity: .7; }
  }
</style></head><body>
<h1>SSG-VQA Demo</h1>
<p class="sub">Upload a chest X-ray + ask a question. Server: Qwen3-VL-8B + SG generator + grounding head.</p>

<form id="f">
  <label>X-ray image <input type="file" name="image" accept="image/*" required></label>
  <label>Question
    <textarea name="question" required>Are there any abnormalities visible? If so, point them out.</textarea>
  </label>
  <button type="submit" id="go">Run inference</button>
</form>

<div id="result"></div>

<script>
const f = document.getElementById('f');
const out = document.getElementById('result');
const btn = document.getElementById('go');

function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

// Bipartite SVG renderer for the scene graph.
// Aggregates duplicate (entity, region, polarity) tuples into edge weights
// instead of drawing them as separate nodes — much more readable when the
// detector is mode-collapsed (20 of "body habitus/clothing @ thoracolumbar
// spine" → 1 fat edge instead of 20 overlapping nodes).
function renderSceneGraph(objects) {
  if (!objects || objects.length === 0) {
    return '<p><i>No scene-graph objects detected for this image.</i></p>';
  }

  // 1. Aggregate edges by (entity, region, polarity)
  const edgeMap = new Map();
  for (const o of objects) {
    const key = `${o.entity}||${o.region}||${o.positiveness}`;
    const e = edgeMap.get(key) || {
      entity: o.entity, region: o.region,
      positiveness: o.positiveness, count: 0,
    };
    e.count += 1;
    edgeMap.set(key, e);
  }
  const edges = [...edgeMap.values()];

  // 2. Build unique node lists (preserve appearance order)
  const entityList = [], entitySeen = new Set();
  const regionList = [], regionSeen = new Set();
  for (const e of edges) {
    if (!entitySeen.has(e.entity)) { entitySeen.add(e.entity); entityList.push(e.entity); }
    if (!regionSeen.has(e.region)) { regionSeen.add(e.region); regionList.push(e.region); }
  }

  // 3. Layout: 2 columns, ~80px node spacing, fixed-width nodes
  const W = 760, NODE_W = 220, NODE_H = 32, ROW_GAP = 12, COL_HEADER_H = 28;
  const leftX = 20, rightX = W - 20 - NODE_W;
  const nRows = Math.max(entityList.length, regionList.length);
  const H = COL_HEADER_H + nRows * (NODE_H + ROW_GAP) + 20;

  const entityIdx = new Map(entityList.map((e, i) => [e, i]));
  const regionIdx = new Map(regionList.map((r, i) => [r, i]));
  const ey = i => COL_HEADER_H + i * (NODE_H + ROW_GAP) + NODE_H / 2;
  const ry = i => COL_HEADER_H + i * (NODE_H + ROW_GAP) + NODE_H / 2;

  // 4. Build SVG
  const parts = [];
  parts.push(`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">`);

  // Column headers
  parts.push(`<text class="col-label" x="${leftX + NODE_W/2}" y="16">ENTITY (${entityList.length})</text>`);
  parts.push(`<text class="col-label" x="${rightX + NODE_W/2}" y="16">REGION (${regionList.length})</text>`);

  // Edges first (so nodes render on top)
  const maxCount = Math.max(...edges.map(e => e.count));
  for (const e of edges) {
    const i = entityIdx.get(e.entity), j = regionIdx.get(e.region);
    const x1 = leftX + NODE_W, y1 = ey(i);
    const x2 = rightX,         y2 = ry(j);
    // Bezier control points for a smooth curve
    const cx1 = x1 + 80, cx2 = x2 - 80;
    const strokeWidth = 1 + Math.round(4 * (e.count / maxCount));
    const cls = (e.positiveness === 1) ? 'edge pos' : 'edge neg';
    const title = `${e.entity} → ${e.region} (×${e.count}, ${e.positiveness === 1 ? 'positive' : 'negative'})`;
    parts.push(
      `<path class="${cls}" stroke-width="${strokeWidth}" ` +
      `d="M ${x1} ${y1} C ${cx1} ${y1}, ${cx2} ${y2}, ${x2} ${y2}">` +
      `<title>${esc(title)}</title></path>`
    );
    // Edge count label (small, near midpoint)
    if (e.count > 1) {
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
      parts.push(
        `<circle cx="${mx}" cy="${my}" r="10" fill="#fff" stroke="#888" stroke-width="1"/>` +
        `<text x="${mx}" y="${my}" text-anchor="middle" dominant-baseline="middle" ` +
        `font-size="10" font-weight="700" fill="#333">${e.count}</text>`
      );
    }
  }

  // Entity nodes (left column)
  for (let i = 0; i < entityList.length; i++) {
    const y = COL_HEADER_H + i * (NODE_H + ROW_GAP);
    parts.push(
      `<rect class="node-entity" x="${leftX}" y="${y}" width="${NODE_W}" height="${NODE_H}" rx="6"/>` +
      `<text class="node-label" x="${leftX + NODE_W/2}" y="${y + NODE_H/2}">` +
      `${esc(entityList[i].length > 28 ? entityList[i].slice(0, 26) + '…' : entityList[i])}</text>` +
      `<title>${esc(entityList[i])}</title>`
    );
  }

  // Region nodes (right column)
  for (let j = 0; j < regionList.length; j++) {
    const y = COL_HEADER_H + j * (NODE_H + ROW_GAP);
    parts.push(
      `<rect class="node-region" x="${rightX}" y="${y}" width="${NODE_W}" height="${NODE_H}" rx="6"/>` +
      `<text class="node-label" x="${rightX + NODE_W/2}" y="${y + NODE_H/2}">` +
      `${esc(regionList[j].length > 28 ? regionList[j].slice(0, 26) + '…' : regionList[j])}</text>` +
      `<title>${esc(regionList[j])}</title>`
    );
  }

  parts.push('</svg>');

  // Legend
  const legend = `
    <div class="legend">
      <span><span class="swatch" style="background:#007AFF"></span>entity</span>
      <span><span class="swatch" style="background:#34C759"></span>region</span>
      <span><span class="swatch" style="background:#FF3B30"></span>positive finding</span>
      <span><span class="swatch" style="background:#888"></span>negative/normal</span>
      <span style="margin-left:auto;color:#999">${objects.length} object${objects.length>1?'s':''} → ${edges.length} unique edge${edges.length>1?'s':''}</span>
    </div>`;

  return `<div id="sg-graph">${parts.join('')}</div>${legend}`;
}

f.addEventListener('submit', async (e) => {
  e.preventDefault();
  btn.disabled = true;
  out.innerHTML = '<p>Running inference… (first request may take ~15s for warm-up)</p>';
  const fd = new FormData(f);
  const t0 = performance.now();
  try {
    const r = await fetch('/predict', { method: 'POST', body: fd });
    const dt = ((performance.now()-t0)/1000).toFixed(2);
    const j = await r.json();
    if (!r.ok) {
      out.innerHTML = '<p style="color:#FF3B30">Error: ' + esc(j.detail || JSON.stringify(j)) + '</p>';
      return;
    }
    const sg = j.scene_graph;
    const chex = Object.entries(j.chexpert_top || {})
      .map(([k,v]) => `<tr><td>${esc(k)}</td><td>${v.toFixed(3)}</td></tr>`).join('');
    const objs = (sg.objects || []).slice(0, 20).map((o,i) => `
      <tr><td>${i+1}</td><td>${esc(o.entity)}</td><td>${esc(o.region)}</td>
      <td>${o.positiveness}</td>
      <td>[${o.bbox.map(v=>v.toFixed(2)).join(', ')}]</td></tr>`).join('');
    out.innerHTML = `
      <h2>Result <span class="pill">${dt}s wall</span>
                  <span class="pill">${j.inference_seconds}s model</span></h2>
      <img src="data:image/png;base64,${j.annotated_image_b64}">
      <h3>Answer ${j.answer_truncated ? '<span class="pill" style="background:#FF9500;color:white">truncated</span>' : ''}</h3>
      <p>${esc(j.answer)}${j.answer_truncated ? '<span style="color:#FF9500"> … <em>(generation hit max length; consider raising max_answer_length)</em></span>' : ''}</p>
      ${j.reasoning ? '<h3>Reasoning</h3><p>'+esc(j.reasoning)+'</p>' : ''}
      <h3>Scene Graph (${sg.num_objects} objects detected)</h3>
      ${renderSceneGraph(sg.objects || [])}
      ${sg.num_objects > 0 ? '<details><summary>Raw scene-graph table</summary><table><thead><tr><th>#</th><th>Entity</th><th>Region</th><th>Polarity</th><th>Bbox (x1,y1,x2,y2)</th></tr></thead><tbody>'+objs+'</tbody></table></details>' : ''}
      <h3>CheXpert Top-5</h3>
      ${chex ? '<table><thead><tr><th>Finding</th><th>Probability</th></tr></thead><tbody>'+chex+'</tbody></table>' : '<p><i>CheXpert head produced no output.</i></p>'}
      <details>
        <summary>Refined bbox + LM &lt;box&gt; tokens</summary>
        <pre>refined: ${JSON.stringify(j.bbox_refined)}
from_text: ${JSON.stringify(j.bboxes_from_text)}</pre>
      </details>
      <details>
        <summary>Raw LM generation</summary>
        <pre>${esc(j.raw_generation)}</pre>
      </details>`;
  } catch (err) {
    out.innerHTML = '<p style="color:#FF3B30">Network error: ' + esc(err.message) + '</p>';
  } finally {
    btn.disabled = false;
  }
});
</script>
</body></html>
"""


def build_app(model, device, region_names, entity_names):
    # NOTE on version compatibility:
    # FastAPI 0.136 had two compounding bugs with this exact endpoint shape
    # (UploadFile TypeAdapter not rebuilt, and Request injection misidentified
    # as a query param). Both vanish on FastAPI 0.115-0.119 — the standard
    # `image: UploadFile = File(...), question: str = Form(...)` signature
    # works correctly there. requirements.txt pins fastapi>=0.110,<0.120 to
    # keep this stable. If you upgrade fastapi past 0.120, retest /predict
    # before promoting the upgrade.
    # NB: FastAPI imports live at module top (see comment near line 65) —
    # they MUST be there because `from __future__ import annotations` makes
    # endpoint signatures into string ForwardRefs that pydantic resolves
    # against module globals, not function locals.
    app = FastAPI(title="SSG-VQA Inference Server")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML_PAGE

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "device": str(device)}

    @app.post("/predict")
    async def predict(image: UploadFile = File(...), question: str = Form(...)):
        if not question.strip():
            raise HTTPException(status_code=400, detail="question must be non-empty")
        try:
            raw = await image.read()
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"could not decode image: {e}")

        # Resize for the model (Qwen image processor cap is honoured downstream
        # via QWEN_MAX_PIXELS env). 448 short-side is a good radiology default.
        resized = pil.copy()
        resized.thumbnail((448, 448), Image.LANCZOS)

        try:
            pred = run_inference(model, resized, question, region_names, entity_names)
        except Exception as e:
            log.exception("inference failed")
            raise HTTPException(status_code=500, detail=f"inference error: {e}")

        # Draw on the FULL-RESOLUTION original (looks better) and return base64.
        # annotate_image() is now bulletproof — it always returns PNG bytes,
        # so this try/except is only for catastrophic failures (PIL crash, OOM).
        # If it does fail, fall back to encoding the unmodified original so the
        # frontend at least sees the input image.
        try:
            png = annotate_image(pil, pred)
            pred["annotated_image_b64"] = base64.b64encode(png).decode("ascii")
        except Exception as e:
            log.exception(f"annotation crashed: {e} — falling back to original")
            try:
                buf = io.BytesIO()
                pil.convert("RGB").save(buf, format="PNG")
                pred["annotated_image_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")
            except Exception as e2:
                log.exception(f"fallback PNG encode also failed: {e2}")
                pred["annotated_image_b64"] = ""

        return JSONResponse(pred)

    return app


# ---------------------------------------------------------------------------
# Optional ngrok tunnel — single function, easy to skip
# ---------------------------------------------------------------------------

def start_ngrok_tunnel(port: int) -> Optional[str]:
    try:
        from pyngrok import ngrok, conf
    except ImportError:
        log.error("pyngrok not installed. `pip install pyngrok` to enable --tunnel.")
        return None
    # Reading NGROK_AUTHTOKEN from the env was a footgun: an earlier
    # `export NGROK_AUTHTOKEN=YOUR_TOKEN_HERE` (literal placeholder, set
    # from a copy-pasted command) silently overrides the perfectly fine
    # token in ~/.config/ngrok/ngrok.yml because we always preferred env
    # over config. Two guards:
    #   1) skip the env var if it looks like a placeholder
    #   2) skip the env var if a real config file already has a token
    token = os.environ.get("NGROK_AUTHTOKEN", "").strip()
    placeholder_markers = ("YOUR_TOKEN", "PASTE", "REPLACE", "<token>", "xxxx")
    looks_placeholder = (
        not token
        or len(token) < 20
        or any(m.lower() in token.lower() for m in placeholder_markers)
    )
    if token and not looks_placeholder:
        log.info("using NGROK_AUTHTOKEN from environment")
        conf.get_default().auth_token = token
    elif token and looks_placeholder:
        log.warning(
            f"Ignoring NGROK_AUTHTOKEN={token!r} — looks like a placeholder. "
            "Falling back to ~/.config/ngrok/ngrok.yml. Run `unset NGROK_AUTHTOKEN` "
            "to silence this warning."
        )
    # else: no env var, pyngrok will read ~/.config/ngrok/ngrok.yml
    try:
        tunnel = ngrok.connect(port, "http")
        log.info(f"ngrok public URL: {tunnel.public_url}  →  http://localhost:{port}")
        return tunnel.public_url
    except Exception as e:
        log.error(
            f"ngrok tunnel failed: {e}. Either set NGROK_AUTHTOKEN "
            f"or run `ngrok config add-authtoken <token>` once."
        )
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--checkpoint", type=Path,
                   default=Path("./checkpoints/stage4_finetune_qwen3vl8b_5k/best_model"),
                   help="Path to a saved best_model dir. Use a Stage 1/2/3 dir to compare stages.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--vocab", type=Path,
                   default=Path("./data/mimic-ext-cxr-qba/metadata/dataset_info.json"),
                   help="dataset_info.json for entity/region names.")
    p.add_argument("--tunnel", action="store_true", help="Open a public ngrok tunnel.")
    p.add_argument("--trust_checkpoint", action="store_true",
                   help="Load the checkpoint with weights_only=False. Required when "
                        "the file pickles numpy types the strict-loader cannot accept "
                        "(e.g. parameterised dtypes). Safe ONLY for files you trust "
                        "the origin of — your own training output is the canonical "
                        "trusted case.")
    args = p.parse_args()

    region_names, entity_names = load_vocab(args.vocab)
    model, device, _ = build_model(
        args.model_id, args.gpu, args.checkpoint, trust_checkpoint=args.trust_checkpoint
    )

    public_url = None
    if args.tunnel:
        public_url = start_ngrok_tunnel(args.port)

    app = build_app(model, device, region_names, entity_names)

    import uvicorn
    log.info(f"serving on http://{args.host}:{args.port}")
    if public_url:
        log.info(f"PUBLIC URL: {public_url}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
