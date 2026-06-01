"""
Pre-compute scene graphs from the frozen Stage-1 generator once and cache
them per-study, so Stages 3 and 4 read them off disk instead of recomputing
identical graphs every epoch.

Why caching is a free win here (and not generally):
    - The SG generator is FROZEN at Stage 3+ (set_grad False, see
      models/ssg_vqa_net_v2.py:set_training_mode).
    - It produces graphs via argmax over RPN/entity/region logits — fully
      deterministic given identical inputs.
    - Therefore, per-image graph is identical across epochs and across runs
      that share the same (image, ViT, generator) config. Pay the compute
      once.

Design notes baked in by reviewer feedback:

1. Cache is keyed by a manifest (max_pixels, min_pixels, spatial_merge_size,
   qwen_model_id, dtype, num_entities, num_regions, max_objects, and a
   SHA of the SG generator state dict). Any of these change → manifest
   mismatch at load → loud failure instead of silent stale-graph serving.

2. We precompute at batch=1, NOT the same batch size training uses. Reason:
   ``_extract_qwen_vit_feature_maps`` zero-pads ViT outputs to
   ``max(H, W)`` across a batch, and the SG generator's centerness top-K
   runs on the padded grid. At inference (which is what we actually want
   to match) batch=1 = no cross-image padding. Caching at batch=1 makes
   Stage 3's SG inputs match inference semantics more closely than the
   on-the-fly batched path would. Side benefit: the resulting cached
   graphs are slightly cleaner.

3. Relations are cached at the kept-object indexing (``kept_idx``), not the
   full max_objects grid. This is what ``_sg_outputs_to_dicts`` already
   does at runtime, so the encoder's ``rel[:n, :n, :num_relations]`` crop
   becomes a no-op. Storing at the full grid would misalign rows.

4. We do NOT cache ``scene_graph_outputs`` (the raw RPN logits). At Stage 3
   the generator is frozen, so its detection loss is unoptimizable —
   recomputing the logits would be pointless. Loss-side: when
   ``scene_graphs`` is fed in (cached), the model skips ``_run_sg_generator``,
   ``outputs['scene_graph_outputs']`` is None, and MultiTaskLoss reports
   ``sg_loss = 0``. This is correct, not a regression.

Usage:
    PYTHONPATH=$PWD python scripts/precompute_sg_cache.py \
        --stage1_checkpoint checkpoints/stage1_sg_only/best_model \
        --mimic_cxr_path data/mimic-cxr-jpg \
        --mimic_qa_path data/mimic-ext-cxr-qba \
        --cache_root .cache/sg_generated \
        --split train

Run once per split (train, validate). Single process — generator is fast at
batch=1 (~0.3-0.6 s/study on RTX 8000), so all 250K studies finish in
~30-90 minutes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.ssg_vqa_net_v2 import SSGVQANetV2  # noqa: E402
from data.mimic_cxr_dataset import MIMICCXRVQADataset  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("precompute_sg_cache")


# -----------------------------------------------------------------------------
# Manifest (cache invalidation key) — Trap #1 from reviewer
# -----------------------------------------------------------------------------

def _state_sha(module: torch.nn.Module) -> str:
    """SHA-256 of a module's full state dict (parameter values, not buffers).

    Pins the cache to the exact Stage-1 checkpoint. If you re-train Stage 1
    or swap the best_model file, the SHA changes and ``assert_manifest_matches``
    fails loud at the start of Stage 3.
    """
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()[:16]


def cache_signature(model: SSGVQANetV2) -> Dict[str, Any]:
    """Everything that, if changed, invalidates the cached graphs."""
    ip = model.processor.image_processor
    base = model.qwen.get_base_model() if hasattr(model.qwen, "get_base_model") else model.qwen
    spatial_merge = int(getattr(base.config.vision_config, "spatial_merge_size", 2))

    return {
        "max_pixels":         int(ip.max_pixels),
        "min_pixels":         int(ip.min_pixels),
        "spatial_merge_size": spatial_merge,
        "qwen_model_id":      model.qwen_model_id,
        "torch_dtype":        str(model.torch_dtype),
        "num_entities":       int(model.sg_generator.entity_classifier[-1].out_features),
        "num_regions":        int(model.sg_generator.region_classifier[-1].out_features),
        "max_objects":        int(model.sg_generator.max_objects),
        "sg_generator_sha":   _state_sha(model.sg_generator),
        "schema_version":     2,
    }


def write_manifest(cache_root: Path, sig: Dict[str, Any]) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "manifest.json").write_text(json.dumps(sig, indent=2, sort_keys=True))
    logger.info(f"Wrote manifest: {cache_root / 'manifest.json'}")


def assert_manifest_matches(cache_root: Path, sig: Dict[str, Any]) -> None:
    """Called from the dataset on each run. Cheap (one JSON read + dict compare)."""
    mpath = cache_root / "manifest.json"
    if not mpath.exists():
        raise FileNotFoundError(
            f"No SG cache manifest at {mpath}. Run scripts/precompute_sg_cache.py "
            f"or pass sg_cache_root=None to fall back to on-the-fly."
        )
    saved = json.loads(mpath.read_text())
    diffs = {k: (saved.get(k), sig[k]) for k in sig if saved.get(k) != sig[k]}
    if diffs:
        raise RuntimeError(
            f"SG cache manifest mismatch — cache is stale for this run.\n"
            f"Differing keys (saved → current): {diffs}\n"
            f"Delete {cache_root} and re-run scripts/precompute_sg_cache.py."
        )


# -----------------------------------------------------------------------------
# Per-study precompute at batch=1 — Trap #2 + #3 from reviewer
# -----------------------------------------------------------------------------

@torch.no_grad()
def precompute_one(
    model: SSGVQANetV2,
    pil_image: Image.Image,
    device: torch.device,
) -> Dict[str, Any]:
    """Run the frozen SG generator on one image; return a compact dict ready
    for the encoder.

    Matches inference semantics:
      - batch=1 → no cross-image zero-padding in _extract_qwen_vit_feature_maps
      - relations stored at kept-object indexing (consistent with
        _sg_outputs_to_dicts), so the encoder's ``rel[:n, :n]`` crop is a no-op
    """
    model.eval()

    # Dummy text — we only need the pixel processing branch.
    # Using the SG placeholder block keeps the processor happy.
    proc = model.processor(
        text=[model._sg_placeholder_block()],
        images=[pil_image],
        return_tensors="pt",
        padding=True,
    ).to(device)

    sg_raw, sg_dicts = model._run_sg_generator(
        proc["pixel_values"], proc["image_grid_thw"]
    )
    d = sg_dicts[0]  # single study

    # Compact dtypes — graphs are small (n_keep <= max_objects = 20) but we
    # cache hundreds of thousands of them, so the per-file delta matters.
    return {
        "bboxes":       torch.as_tensor(d["bboxes"], dtype=torch.float16),
        "entity_ids":   torch.as_tensor(d["entity_ids"], dtype=torch.int16),
        "region_ids":   torch.as_tensor(d["region_ids"], dtype=torch.int16),
        "positiveness": torch.as_tensor(d["positiveness"], dtype=torch.int8),
        "relations":    d["relations"].to(torch.float16).cpu(),  # (n_keep, n_keep, 10)
        "num_objects":  int(d["num_objects"]),
    }


# -----------------------------------------------------------------------------
# Cache path scheme
# -----------------------------------------------------------------------------

def cache_path_for_study(cache_root: Path, subject_id: int, study_id: int) -> Path:
    """Mirrors MIMIC's p10/p10000032/s50414267 structure."""
    p_short = f"p{str(subject_id)[:2]}"
    p_full = f"p{subject_id}"
    return cache_root / p_short / p_full / f"s{study_id}.sg.pt"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1_checkpoint", type=Path, required=True,
                    help="Path to checkpoints/stage1_sg_only/best_model")
    ap.add_argument("--mimic_cxr_path", type=Path, default=Path("data/mimic-cxr-jpg"))
    ap.add_argument("--mimic_qa_path",  type=Path, default=Path("data/mimic-ext-cxr-qba"))
    ap.add_argument("--cache_root",     type=Path, default=Path(".cache/sg_generated"))
    ap.add_argument("--split",          type=str, default="train",
                    choices=["train", "validate", "test"])
    ap.add_argument("--max_samples",    type=int, default=None,
                    help="Limit (for smoke tests). Default: full split.")
    ap.add_argument("--qwen_model_id",  type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--device",         type=str, default="cuda:0")
    ap.add_argument("--skip_existing",  action="store_true", default=True,
                    help="Skip studies that already have a cache file. Default ON.")
    ap.add_argument("--force",          action="store_true",
                    help="Overwrite existing cache files (disables --skip_existing).")
    args = ap.parse_args()

    if args.force:
        args.skip_existing = False

    device = torch.device(args.device)
    cache_root = args.cache_root.resolve()

    # ---- Build the model with Stage-1 weights ----------------------------
    logger.info("Loading SSGVQANetV2 (4-bit QLoRA, frozen for inference)...")
    model = SSGVQANetV2(
        qwen_model_id=args.qwen_model_id,
        use_quantization=True,
        training_mode="pretrain",     # ensures sg_generator is FROZEN (not sg_only)
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    # Load Stage-1 weights into the SG generator (and other components)
    ckpt_bin = args.stage1_checkpoint / "pytorch_model.bin"
    if not ckpt_bin.exists():
        ckpt_bin = args.stage1_checkpoint / "ssg_components.pt"
    if not ckpt_bin.exists():
        raise FileNotFoundError(
            f"Could not find pytorch_model.bin or ssg_components.pt under "
            f"{args.stage1_checkpoint}"
        )
    logger.info(f"Loading Stage-1 weights from {ckpt_bin}")
    # weights_only=True: checkpoint is pure state_dict (tensors + plain dict keys).
    # If a future checkpoint format adds non-tensor metadata that fails this load,
    # whitelist the specific globals via torch.serialization.add_safe_globals rather
    # than reverting to weights_only=False, which permits arbitrary code execution.
    state = torch.load(ckpt_bin, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(
        f"  Loaded {len(state) - len(missing)} keys "
        f"({len(missing)} missing, {len(unexpected)} unexpected; "
        f"most unexpected are bnb quant_state — expected)"
    )

    # ---- Write manifest --------------------------------------------------
    sig = cache_signature(model)
    write_manifest(cache_root, sig)
    logger.info(f"Cache signature: sg_generator_sha={sig['sg_generator_sha']} "
                f"max_pixels={sig['max_pixels']} merge={sig['spatial_merge_size']}")

    # ---- Build dataset (we only need the metadata, not the tokenized text) -
    logger.info(f"Building dataset for split={args.split}...")
    ds = MIMICCXRVQADataset(
        mimic_cxr_path=str(args.mimic_cxr_path),
        mimic_qa_path=str(args.mimic_qa_path),
        split=args.split,
        max_samples=args.max_samples,
        quality_grade="all",  # cache for all grades — Stage 3 may filter to B
        use_cache=True,
    )
    logger.info(f"  {len(ds)} samples in split={args.split}")

    # ---- Iterate, precompute, save ---------------------------------------
    studies_seen: set = set()  # dedupe (one cache file per study, samples share studies)
    n_written = n_skipped = n_failed = 0
    t0 = time.time()

    for idx in range(len(ds)):
        try:
            item = ds.samples[idx] if hasattr(ds, "samples") else None
            if item is None:
                # Fall back to __getitem__ which loads the PIL — slower
                item = ds[idx]
            subject_id = int(item["subject_id"])
            study_id = int(item["study_id"])
        except Exception as e:
            logger.warning(f"  [{idx}] skipping — could not read metadata: {e}")
            n_failed += 1
            continue

        if (subject_id, study_id) in studies_seen:
            continue
        studies_seen.add((subject_id, study_id))

        cache_file = cache_path_for_study(cache_root, subject_id, study_id)
        if args.skip_existing and cache_file.exists():
            n_skipped += 1
            continue

        # Load the PIL image — try to read directly from path if available
        # (faster than going through __getitem__'s full collate path).
        image_path = item.get("image_path")
        if image_path is None or not os.path.exists(image_path):
            logger.warning(f"  [{idx}] no image at {image_path}; skipping")
            n_failed += 1
            continue

        try:
            pil = Image.open(image_path).convert("RGB")
            cached = precompute_one(model, pil, device)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"_sig_sha": sig["sg_generator_sha"], **cached},
                cache_file,
            )
            n_written += 1
        except Exception as e:
            logger.warning(f"  [{idx}] s={study_id} failed: {e}")
            n_failed += 1
            continue

        if n_written and n_written % 500 == 0:
            elapsed = time.time() - t0
            rate = n_written / max(elapsed, 1.0)
            remaining = max(0, len(studies_seen) - n_written)
            eta_min = (len(ds) - idx - 1) / max(rate, 0.01) / 60
            logger.info(
                f"  [{idx + 1}/{len(ds)}] written={n_written} "
                f"skipped={n_skipped} failed={n_failed} "
                f"rate={rate:.1f}/s eta~{eta_min:.0f}min"
            )

    elapsed = time.time() - t0
    logger.info(
        f"\nDone. studies_seen={len(studies_seen)} "
        f"written={n_written} skipped={n_skipped} failed={n_failed} "
        f"elapsed={elapsed/60:.1f}min"
    )
    logger.info(f"Cache root: {cache_root}")


if __name__ == "__main__":
    main()
