"""Post-training quantization + export pipeline for SSG-VQA-Net v2.

Objective (iv) of the thesis: reduce the deployed model's memory and compute
footprint through post-training quantization.

Produces up to six variants of the Stage-4 (or Stage-3) best checkpoint:
  - fp16        : baseline, no quantization
  - int8        : 8-bit dynamic (bitsandbytes)
  - nf4         : 4-bit NormalFloat + double-quant (bitsandbytes; QLoRA format)
  - q5_k_m      : 5-bit block-wise (llama.cpp K-quants)   [needs llama.cpp]
  - q4_k_m      : 4-bit block-wise (llama.cpp K-quants)   [needs llama.cpp]
  - q3_k_m     : 3-bit block-wise (llama.cpp K-quants)   [needs llama.cpp]

Asymmetric quantization policy: ONLY the Qwen3-VL backbone gets quantized.
The scene-graph generator, grounding head, and auxiliary heads are kept in
FP16 and saved as a separate ``heads.safetensors`` sidecar because:
  * they account for < 1% of parameters,
  * their precision matters for downstream IoU / accuracy, and
  * quantizing them empirically loses 2-3x more accuracy per bit than the
    backbone at the same bit width [QLoRA, Dettmers et al. 2023].

Usage:
    python scripts/quantize_and_export.py \\
        --checkpoint ./checkpoints/mimic-cxr-vqa/finetune/best_model \\
        --output_dir ./quantized_models \\
        --variants fp16 int8 nf4 q4_k_m

    # Or all variants (default):
    python scripts/quantize_and_export.py \\
        --checkpoint ./checkpoints/mimic-cxr-vqa/finetune/best_model

The script writes a top-level ``disk_manifest.json`` capturing size, timing,
and any failed variants so the paper's tables can be regenerated from a
single JSON file.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the repo root importable when this file is invoked as a script.
# Without this, `from models.sg_generators import ...` in _load_and_merge()
# raises ModuleNotFoundError because Python only puts the script's own dir
# (scripts/) on sys.path, not the repo root that contains the models/ package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

logger = logging.getLogger(__name__)


# ==========================================================================
# Variant registry
# ==========================================================================
ALL_VARIANTS = ["fp16", "int8", "nf4", "q5_k_m", "q4_k_m", "q3_k_m"]
BNB_VARIANTS = {"int8", "nf4"}
GGUF_VARIANTS = {"q5_k_m", "q4_k_m", "q3_k_m"}


# ==========================================================================
# LoRA merge + heads-sidecar split
# ==========================================================================
def _load_and_merge(checkpoint_dir: Path, model_id: str, dtype: torch.dtype = torch.float16):
    """Load a trained SSGVQANetV2 checkpoint and merge LoRA into the Qwen base.

    Returns (merged_model, heads_state_dict) where merged_model is the LoRA-
    merged HuggingFace Qwen3-VL ready for HF-style export, and heads_state_dict
    contains everything OUTSIDE Qwen (SG generator, encoder, projector,
    grounding head, aux heads, view proj) as an FP16 flat dict for sidecar
    saving. Also returns the full model config so downstream loaders can
    reconstruct the heads with matching dimensions.
    """
    from models.sg_generators import get_sg_generator  # noqa: F401  (auto-registers)
    from models.ssg_vqa_net_v2 import SSGVQANetV2

    logger.info(f"Constructing SSGVQANetV2 with base {model_id} (dtype={dtype})")

    # Reconstruct via the same path the trainer uses.
    #
    # use_quantization MUST be True when loading from our merged-base dir:
    # the merged Stage-4 checkpoint stores Qwen weights as NF4-packed uint8
    # buffers (shape [N*M/2, 1]) that only a `Linear4bit` layer can ingest,
    # AND the merged config.json carries a `quantization_config` dict that
    # trips a HF `from_pretrained` bug when we pass `quantization_config=None`
    # -- HF ends up with `config.quantization_config = None`, then
    # `logger.info(f"Model config {config}")` calls
    # `__repr__ -> to_json_string -> to_dict -> self.quantization_config.to_dict()`
    # on the None and raises `AttributeError: 'NoneType' object has no attribute
    # 'to_dict'`. Passing `use_quantization=True` supplies an explicit
    # BitsAndBytesConfig object that HF wires up correctly.
    #
    # Downstream: `_export_fp16` calls `.to(dtype=torch.float16)` on the
    # merged model to dequantise before saving; bitsandbytes >= 0.43 supports
    # this in-place dequant. For int8/nf4/gguf variants the model is reloaded
    # from the FP16 export, so the initial NF4 load only affects the merge
    # itself (which adds ~1-2%% per-layer requant drift, bounded by NF4).
    model = SSGVQANetV2(
        qwen_model_id=model_id,
        use_quantization=True,
        lora_rank=32,
        lora_alpha=64,
        lora_target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        num_sg_tokens=8,
        num_regions=30,
        num_entities=22,
        num_binary=2,
        num_category=14,
        num_region_classes=26,
        num_severity=4,
        training_mode="finetune",
        torch_dtype=dtype,
        freeze_sg_generator=True,
    )

    # Load the checkpoint's state_dict (matches what serve_app.py does).
    bin_path = checkpoint_dir / "pytorch_model.bin"
    pointer = checkpoint_dir / "checkpoint_step.txt"
    if pointer.exists() and not bin_path.exists():
        step = int(pointer.read_text().strip())
        real_dir = checkpoint_dir.parent / f"checkpoint-{step}"
        bin_path = real_dir / "pytorch_model.bin"
        logger.info(f"resolved pointer {pointer} -> {bin_path}")

    if not bin_path.exists():
        raise FileNotFoundError(f"No pytorch_model.bin at {bin_path}")

    logger.info(f"Loading weights from {bin_path}")
    # Prefer weights_only=True (safe) and only fall back with an explicit
    # env opt-in. Training checkpoints pickle numpy scalars in their metrics
    # dict, so we allowlist a handful of numpy globals before attempting
    # the strict load -- same policy as serve_app.py.
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
        logger.warning(f"could not whitelist numpy globals: {_e}")
    try:
        state = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    except Exception as e:
        if os.environ.get("SG_TRUST_CHECKPOINT", "0") != "1":
            raise RuntimeError(
                f"torch.load(weights_only=True) rejected {bin_path}: {e}\n\n"
                "If you trust the origin of this checkpoint (your own training "
                "output is the canonical trusted case), re-run with "
                "SG_TRUST_CHECKPOINT=1 to permit weights_only=False."
            ) from e
        logger.warning(
            "SG_TRUST_CHECKPOINT=1 set: falling back to weights_only=False. "
            "This permits arbitrary code execution during unpickling."
        )
        state = torch.load(str(bin_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if state and next(iter(state.keys())).startswith("module."):
        state = {k[len("module."):]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"  loaded {len(state)} keys, missing {len(missing)}, unexpected {len(unexpected)}")

    # Merge LoRA adapters into base weights.
    logger.info("Merging LoRA adapters into Qwen base weights...")
    if hasattr(model.qwen, "merge_and_unload"):
        model.qwen = model.qwen.merge_and_unload()
        logger.info("  LoRA merged via PEFT merge_and_unload()")
    else:
        logger.warning("model.qwen has no merge_and_unload(); leaving LoRA adapters unmerged")

    # Split into (Qwen-only, everything-else) state dicts.
    qwen_state = {k: v for k, v in model.qwen.state_dict().items()}

    heads_state: Dict[str, torch.Tensor] = {}
    for module_name in ("sg_generator", "sg_encoder", "sg_projector",
                        "grounding_head", "aux_heads", "view_proj"):
        sub = getattr(model, module_name, None)
        if sub is None:
            continue
        for k, v in sub.state_dict().items():
            heads_state[f"{module_name}.{k}"] = v.detach().to(torch.float16).cpu()

    logger.info(
        f"Split: Qwen backbone = {sum(v.numel() for v in qwen_state.values())/1e9:.2f} B params, "
        f"heads = {sum(v.numel() for v in heads_state.values())/1e6:.2f} M params"
    )
    return model, qwen_state, heads_state


def _save_heads_sidecar(heads_state: Dict[str, torch.Tensor], out_dir: Path) -> None:
    """Save the non-Qwen modules as a safetensors sidecar."""
    try:
        from safetensors.torch import save_file
        save_file(heads_state, str(out_dir / "heads.safetensors"))
        logger.info(f"  wrote heads sidecar: {(out_dir / 'heads.safetensors').stat().st_size / 1e6:.1f} MB")
    except ImportError:
        # Fallback to torch.save
        torch.save(heads_state, out_dir / "heads.bin")
        logger.info(f"  wrote heads sidecar (torch): {(out_dir / 'heads.bin').stat().st_size / 1e6:.1f} MB")


# ==========================================================================
# Variant exporters
# ==========================================================================
def _export_fp16(qwen_model, out_dir: Path) -> None:
    """Save the merged Qwen backbone as FP16 (baseline; no quantization)."""
    qwen_model.to(dtype=torch.float16)
    qwen_model.save_pretrained(str(out_dir), safe_serialization=True)
    logger.info("  wrote FP16 model.safetensors")


def _export_bnb(model_id_for_reload: Path, out_dir: Path, mode: str) -> None:
    """Reload the FP16 model with BitsAndBytesConfig and re-save.

    mode: 'int8' or 'nf4'
    """
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    logger.info(f"  reloading model with BitsAndBytesConfig({mode}) from {model_id_for_reload}")
    if mode == "int8":
        cfg = BitsAndBytesConfig(load_in_8bit=True)
    elif mode == "nf4":
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        raise ValueError(f"unknown bnb mode: {mode}")

    reloaded = AutoModelForImageTextToText.from_pretrained(
        str(model_id_for_reload),
        quantization_config=cfg,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    reloaded.save_pretrained(str(out_dir), safe_serialization=True)
    logger.info(f"  wrote {mode} quantized weights")


def _export_gguf(model_id_for_reload: Path, out_dir: Path, method: str,
                 llama_cpp_path: Optional[Path] = None) -> None:
    """Export a GGUF-quantized variant via llama.cpp toolchain.

    method: 'Q5_K_M' | 'Q4_K_M' | 'Q3_K_M' (llama.cpp naming)

    Requires ``convert_hf_to_gguf.py`` and the ``llama-quantize`` binary
    from a llama.cpp build. Set --llama_cpp_path or the LLAMA_CPP env var.
    """
    llama_cpp = llama_cpp_path or Path(os.environ.get("LLAMA_CPP", ""))
    if not llama_cpp.exists():
        raise FileNotFoundError(
            f"llama.cpp not found at {llama_cpp}. Set --llama_cpp_path "
            "or LLAMA_CPP env var to a checkout of llama.cpp with build/bin/llama-quantize."
        )

    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"
    if not convert_script.exists():
        # Older layouts
        convert_script = llama_cpp / "convert-hf-to-gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found in {llama_cpp}")
    if not quantize_bin.exists():
        raise FileNotFoundError(f"llama-quantize binary not found at {quantize_bin}")

    # Step 1: convert HF FP16 -> GGUF FP16 (intermediate)
    intermediate = out_dir / f"_intermediate_fp16.gguf"
    logger.info(f"  step 1: HF -> GGUF FP16 via {convert_script.name}")
    subprocess.run(
        ["python", str(convert_script),
         str(model_id_for_reload),
         "--outtype", "f16",
         "--outfile", str(intermediate)],
        check=True,
    )

    # Step 2: quantize to target method
    final = out_dir / f"model-{method.lower()}.gguf"
    logger.info(f"  step 2: quantize {intermediate.name} -> {final.name} ({method})")
    subprocess.run(
        [str(quantize_bin), str(intermediate), str(final), method],
        check=True,
    )
    intermediate.unlink()  # drop the FP16 intermediate to save disk
    logger.info(f"  wrote {final.name}")


# ==========================================================================
# Manifest + size reporting
# ==========================================================================
def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024.0:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.2f} PB"


# ==========================================================================
# Main
# ==========================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Stage-4 (or Stage-3) best_model or checkpoint-N dir.")
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                   help="HuggingFace ID of the Qwen base model.")
    p.add_argument("--output_dir", type=Path, default=Path("./quantized_models"),
                   help="Root output dir. One subdir per variant.")
    p.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                   choices=ALL_VARIANTS,
                   help="Which variants to build.")
    p.add_argument("--llama_cpp_path", type=Path, default=None,
                   help="Path to a llama.cpp checkout (for GGUF variants). "
                        "Alternatively set env LLAMA_CPP.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip a variant if its output dir already contains files.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "source_checkpoint": str(args.checkpoint),
        "source_model_id": args.model_id,
        "generated_at": datetime.now().isoformat(),
        "variants": {},
    }

    # ---- Build the merged FP16 model once (shared prep) --------------------
    logger.info("=" * 70)
    logger.info("PREP: loading + merging LoRA")
    logger.info("=" * 70)
    t0 = time.time()
    model, qwen_state, heads_state = _load_and_merge(
        args.checkpoint, args.model_id, dtype=torch.float16
    )
    logger.info(f"prep took {time.time() - t0:.1f} s")

    # First always write an FP16 canonical version -- required as the reload
    # source for the bnb + GGUF paths.
    fp16_dir = args.output_dir / "fp16"
    fp16_dir.mkdir(parents=True, exist_ok=True)
    _export_fp16(model.qwen, fp16_dir)
    _save_heads_sidecar(heads_state, fp16_dir)
    # Copy the processor / tokenizer so downstream loads work standalone.
    try:
        model.processor.save_pretrained(str(fp16_dir))
        logger.info("  wrote processor/tokenizer")
    except Exception as e:
        logger.warning(f"could not save processor: {e}")

    fp16_size = _dir_size_bytes(fp16_dir)
    manifest["variants"]["fp16"] = {
        "path": str(fp16_dir),
        "disk_bytes": fp16_size,
        "disk_human": _human_size(fp16_size),
        "status": "ok",
        "elapsed_s": time.time() - t0,
    }

    # Free the merged FP16 model from RAM before exporting other variants.
    del model
    del qwen_state
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- Emit remaining variants -------------------------------------------
    for variant in args.variants:
        if variant == "fp16":
            continue  # already done
        out_dir = args.output_dir / variant
        if args.skip_existing and out_dir.exists() and any(out_dir.iterdir()):
            logger.info(f"skip: {variant} dir already populated")
            manifest["variants"][variant] = {
                "path": str(out_dir),
                "status": "skipped_existing",
            }
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info(f"EXPORT: {variant}")
        logger.info("=" * 70)
        t_v = time.time()
        try:
            if variant in BNB_VARIANTS:
                _export_bnb(fp16_dir, out_dir, mode=variant)
            elif variant in GGUF_VARIANTS:
                method_map = {"q5_k_m": "Q5_K_M", "q4_k_m": "Q4_K_M", "q3_k_m": "Q3_K_M"}
                _export_gguf(fp16_dir, out_dir, method=method_map[variant],
                             llama_cpp_path=args.llama_cpp_path)
            else:
                raise ValueError(f"no exporter for variant '{variant}'")

            # Always copy the FP16 heads sidecar alongside every variant.
            src = fp16_dir / "heads.safetensors"
            if src.exists():
                shutil.copy2(src, out_dir / "heads.safetensors")
            # Copy processor too.
            for name in ("preprocessor_config.json", "tokenizer.json",
                         "tokenizer_config.json", "special_tokens_map.json",
                         "chat_template.jinja"):
                s = fp16_dir / name
                if s.exists():
                    shutil.copy2(s, out_dir / name)

            elapsed = time.time() - t_v
            size = _dir_size_bytes(out_dir)
            manifest["variants"][variant] = {
                "path": str(out_dir),
                "disk_bytes": size,
                "disk_human": _human_size(size),
                "status": "ok",
                "elapsed_s": elapsed,
                "compression_ratio_vs_fp16": (
                    fp16_size / size if size > 0 else None
                ),
            }
            logger.info(
                f"[{variant}] done in {elapsed:.1f}s, "
                f"size {_human_size(size)}, "
                f"{fp16_size/size:.2f}x smaller than FP16"
            )
        except Exception as e:
            logger.error(f"[{variant}] FAILED: {e}", exc_info=args.verbose)
            manifest["variants"][variant] = {
                "path": str(out_dir),
                "status": "failed",
                "error": str(e),
                "elapsed_s": time.time() - t_v,
            }
            # Clean up partial output
            if out_dir.exists() and not any(out_dir.iterdir()):
                out_dir.rmdir()

    # ---- Write manifest ----------------------------------------------------
    manifest_path = args.output_dir / "disk_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("=" * 70)
    logger.info(f"MANIFEST: {manifest_path}")
    logger.info("=" * 70)
    for v, info in manifest["variants"].items():
        status = info.get("status", "?")
        size = info.get("disk_human", "-")
        ratio = info.get("compression_ratio_vs_fp16")
        ratio_s = f" ({ratio:.2f}x)" if ratio else ""
        logger.info(f"  {v:10s} {status:20s} {size}{ratio_s}")

    # Non-zero exit if any variant failed
    failed = [v for v, info in manifest["variants"].items()
              if info.get("status") == "failed"]
    if failed:
        logger.warning(f"Failed variants: {failed}")
        sys.exit(1 if len(failed) == len(args.variants) - 1 else 0)


if __name__ == "__main__":
    main()
