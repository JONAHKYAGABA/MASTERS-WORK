#!/usr/bin/env python
"""
scripts/merge_lora_for_stage4.py

Fold Stage-3's rank-16 LoRA adapters into the Qwen backbone weights,
producing a fp16 HuggingFace directory + a heads sidecar that Stage 4 can
consume as a fresh base with a re-initialised rank-32 LoRA on top.

Why: Stage 4's YAML asks for `lora_rank: 32` with MLP targets (bigger
adaptation surface), but Stage 3's checkpoint contains rank-16 attention-only
LoRA weights. A direct `model.load_state_dict(strict=False)` crashes on
size mismatch. This script resolves the transition by *baking* Stage 3's
LoRA delta into the base, so Stage 4 starts training from the exact Stage 3
functional output but with a fresh, higher-capacity LoRA layer that can
absorb Stage 4's A-grade signal.

Output tree:
    <output>/
        qwen_merged_fp16/    # Qwen backbone with LoRA folded in (safetensors)
        heads.bin            # Non-Qwen weights: SG generator, aux heads, mHC, ...
        MERGED_BASE.json     # Sentinel + provenance metadata

Stage 4 launch flow after this runs:
    torchrun ... train_mimic_cxr.py \
        --config configs/stage4_finetune.yaml \
        --resume_from_checkpoint ./checkpoints/mimic-cxr-vqa/pretrain/final_model_merged
    # The trainer's patched loader sees MERGED_BASE.json, uses
    # qwen_merged_fp16/ as the Qwen base (fresh NF4 quant), wraps it with
    # rank-32 LoRA per Stage 4 config, then loads heads.bin as sidecar.

Requires:
    SG_TRUST_CHECKPOINT=1  # opt-in to weights_only=False torch.load on our own ckpt
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import date
from pathlib import Path

import torch

# Make repo root importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.mimic_cxr_config import load_config_from_file  # noqa: E402
from models import SSGVQANetV2  # noqa: E402


def _safe_load(path: Path):
    """Load a checkpoint written by our own trainer (trusted source)."""
    if os.environ.get("SG_TRUST_CHECKPOINT") != "1":
        raise RuntimeError(
            "This script needs SG_TRUST_CHECKPOINT=1 in the env to allow "
            "weights_only=False torch.load of our own checkpoint. Rerun with:\n"
            "    SG_TRUST_CHECKPOINT=1 python scripts/merge_lora_for_stage4.py ..."
        )
    return torch.load(path, map_location="cpu", weights_only=False)


def _resolve_stage3_lora(cfg) -> tuple[int, int, list[str]]:
    """
    Return (rank, alpha, target_modules) as they were during Stage 3.

    Stage 3's YAML doesn't set them explicitly, so trainer defaults kicked in:
    rank=16, alpha=32 (= 2*rank), targets=['q_proj', 'k_proj', 'v_proj', 'o_proj'].
    We honour any override that is set in the config, but fall back to the
    trainer's defaults for anything absent.
    """
    rank = int(getattr(cfg.model, "lora_rank", 16))
    alpha = int(getattr(cfg.model, "lora_alpha", 2 * rank))
    targets = getattr(cfg.model, "lora_target_modules", None) or [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ]
    return rank, alpha, list(targets)


def _find_first_target_linear(peft_model, target_name: str):
    """
    Locate a LoRA-target linear inside a PEFT-wrapped model. Used for the
    pre/post-merge parity probe. Returns (module, dotted_path).
    """
    for name, module in peft_model.named_modules():
        if name.endswith(f".{target_name}") and hasattr(module, "weight"):
            return module, name
    raise RuntimeError(
        f"Could not find a '{target_name}' linear in the PEFT-wrapped model."
    )


def _probe(module, batch: torch.Tensor) -> torch.Tensor:
    """Run a fixed input through a single linear and return the fp32 output."""
    module.eval()
    with torch.no_grad():
        out = module(batch)
    return out.detach().to(torch.float32).cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage3_ckpt", required=True, type=Path,
                    help="Path to Stage-3 final_model dir (must contain pytorch_model.bin)")
    ap.add_argument("--stage3_config", required=True, type=Path,
                    help="Stage-3 YAML config (used to reconstruct the model architecture)")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output dir; will be created if missing")
    ap.add_argument("--target_lora_rank", type=int, default=32,
                    help="LoRA rank Stage 4 will use. Metadata only; recorded in the sentinel.")
    ap.add_argument("--device", default="cuda:0",
                    help="'cuda:0' (default) or 'cpu'. GPU makes merge ~5x faster.")
    ap.add_argument("--verify", action="store_true",
                    help="Run pre/post-merge forward parity check on 3 target layers.")
    ap.add_argument("--parity_tol", type=float, default=1e-2,
                    help=("Max absolute output diff allowed by --verify. Default 1e-2 for "
                          "an fp16 merge (rounding-limited). For the NF4 requantise path "
                          "(--load_quantized), realistic tolerance is 3e-1 or higher."))
    ap.add_argument("--load_quantized", action="store_true",
                    help=("Load Stage 3 in NF4 (matches training-time layout). NOT recommended: "
                          "PEFT re-quantises the merged weights back to NF4, introducing ~1-2%% "
                          "output drift and leaving the saved backbone as NF4 (which double-quants "
                          "at Stage 4 load time). Default is fp16 load → exact merge → fp16 save."))
    args = ap.parse_args()

    if not args.stage3_ckpt.is_dir():
        raise FileNotFoundError(f"--stage3_ckpt is not a directory: {args.stage3_ckpt}")
    ckpt_file = args.stage3_ckpt / "pytorch_model.bin"
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_file}")
    if not args.stage3_config.exists():
        raise FileNotFoundError(f"Missing config: {args.stage3_config}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but no CUDA available")

    # ------------------------------------------------------------
    # 1. Reconstruct Stage-3 architecture (rank 16, attn-only LoRA)
    # ------------------------------------------------------------
    cfg = load_config_from_file(str(args.stage3_config))
    s3_rank, s3_alpha, s3_targets = _resolve_stage3_lora(cfg)
    qwen_id = getattr(cfg.model, "qwen_model_id", None) or "Qwen/Qwen3-VL-8B-Instruct"

    print(f"[merge] Stage-3 LoRA:  rank={s3_rank}  alpha={s3_alpha}  targets={s3_targets}")
    print(f"[merge] Qwen backbone: {qwen_id}")

    # fp16 merge (default) vs NF4 merge (--load_quantized).
    # fp16 merge is exact: dequantize once via HF from_pretrained (fp16),
    # add the LoRA delta, save fp16. Stage 4 quantises to NF4 on load,
    # once. NF4 merge dequantises, adds LoRA, re-quantises to NF4 — the
    # requant step drops precision by ~1-2%% per layer and leaves the
    # saved backbone in a state that fights with Stage 4's own quant.
    _load_quantized = args.load_quantized
    _peak_gb_est = "~5 GB" if _load_quantized else "~16 GB"
    _mode_label = "NF4 + LoRA rank" if _load_quantized else "fp16 + LoRA rank"
    print(f"[merge] Building Stage-3 architecture ({_mode_label} {s3_rank}, "
          f"peak GPU mem est. {_peak_gb_est})...")

    model = SSGVQANetV2(
        qwen_model_id=qwen_id,
        use_quantization=_load_quantized,
        lora_rank=s3_rank,
        lora_alpha=s3_alpha,
        lora_target_modules=s3_targets,
        num_regions=cfg.model.num_regions,
        num_entities=cfg.model.num_entities,
        num_binary=cfg.model.num_binary_classes,
        num_category=cfg.model.num_category_classes,
        num_region_classes=cfg.model.num_region_classes,
        num_severity=cfg.model.num_severity_classes,
        training_mode="pretrain",
        freeze_sg_generator=True,
        torch_dtype=torch.float16,
    )
    model.eval()

    # ------------------------------------------------------------
    # 2. Load Stage-3 weights
    # ------------------------------------------------------------
    print(f"[merge] Loading state_dict from: {ckpt_file}")
    payload = _safe_load(ckpt_file)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        sd = payload["model_state_dict"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        sd = payload["state_dict"]
    elif isinstance(payload, dict) and "module" in payload:
        sd = payload["module"]
    else:
        sd = payload
    del payload

    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_lora_missing = sum(1 for k in missing if "lora_" in k)
    n_qwen_lora_ckpt = sum(1 for k in sd if k.startswith("qwen.") and "lora_" in k)
    print(f"[merge] Loaded state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    print(f"[merge]   LoRA keys present in checkpoint: {n_qwen_lora_ckpt}")
    print(f"[merge]   LoRA keys missing after load    : {n_lora_missing}")
    if n_qwen_lora_ckpt == 0:
        raise RuntimeError(
            "Stage-3 checkpoint has zero LoRA keys — refusing to merge (would be a no-op)."
        )
    if n_lora_missing > 0:
        raise RuntimeError(
            f"{n_lora_missing} LoRA keys are missing after load — Stage-3 rank/targets "
            "may not match the checkpoint. Aborting; investigate the mismatch before continuing."
        )
    del sd

    # ------------------------------------------------------------
    # 3. Optional parity probe: capture outputs BEFORE merge
    # ------------------------------------------------------------
    probe_data = None
    if args.verify:
        print(f"[merge] Moving model to {device} for parity probe + merge...")
        model.to(device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        probe_targets = ["q_proj", "v_proj", "o_proj"]
        probe_data = []
        for tname in probe_targets:
            mod, mod_path = _find_first_target_linear(model.qwen, tname)
            in_features = mod.in_features if hasattr(mod, "in_features") else model.d_llm
            torch.manual_seed(hash(tname) & 0xFFFF)
            x = torch.randn(1, 8, in_features, dtype=torch.float16, device=device)
            y_pre = _probe(mod, x)
            probe_data.append({"target": tname, "path": mod_path, "x": x, "y_pre": y_pre})
            print(f"[merge]   probe {tname:8s} at {mod_path}: y_pre shape={tuple(y_pre.shape)} "
                  f"|y|_max={y_pre.abs().max().item():.4f}")
    else:
        # Not verifying: still move to the device for a faster merge if requested.
        if device.type == "cuda":
            print(f"[merge] Moving model to {device} for merge...")
            model.to(device)

    # ------------------------------------------------------------
    # 4. Merge LoRA into base — dequantise NF4, add delta, drop PEFT wrapper
    # ------------------------------------------------------------
    print("[merge] PEFT merge_and_unload() — dequantising NF4 + folding rank-16 LoRA into base...")
    merged_qwen = model.qwen.merge_and_unload()
    model.qwen = merged_qwen  # replace so state_dict export reflects merged base
    print("[merge] Merge complete. Base is now fp16, no LoRA wrapper.")

    # ------------------------------------------------------------
    # 5. Optional parity probe: capture outputs AFTER merge, compare
    # ------------------------------------------------------------
    if args.verify and probe_data is not None:
        print("[merge] Running post-merge parity probe...")
        worst = 0.0
        for probe in probe_data:
            mod_post, _ = _find_first_target_linear(model.qwen, probe["target"])
            # Path structure changed (PEFT wrapper gone). Same underlying linear.
            y_post = _probe(mod_post, probe["x"])
            diff = (probe["y_pre"] - y_post).abs().max().item()
            worst = max(worst, diff)
            status = "OK" if diff <= args.parity_tol else "FAIL"
            print(f"[merge]   {probe['target']:8s}: max |y_pre - y_post| = {diff:.4e}  [{status}]")
        print(f"[merge] Worst-case parity diff: {worst:.4e}  (tolerance {args.parity_tol:.1e})")
        if worst > args.parity_tol:
            raise RuntimeError(
                f"Parity check FAILED. Merged model diverges from Stage 3 by {worst:.4e}. "
                "This indicates the merge corrupted weights. Aborting BEFORE writing the output."
            )
        print("[merge] Parity check PASSED.")

    # ------------------------------------------------------------
    # 6. Save merged Qwen backbone as a self-contained HF directory
    # ------------------------------------------------------------
    args.output.mkdir(parents=True, exist_ok=True)
    qwen_out = args.output / "qwen_merged_fp16"
    print(f"[merge] Saving merged Qwen backbone to: {qwen_out}")
    # Move back to CPU for the safetensors write — avoids device-map surprises.
    model.qwen.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.qwen.save_pretrained(qwen_out, safe_serialization=True, max_shard_size="4GB")
    model.processor.save_pretrained(qwen_out)

    # ------------------------------------------------------------
    # 7. Save non-Qwen weights (heads, SG, mHC, embeddings, projectors)
    # ------------------------------------------------------------
    heads_out = args.output / "heads.bin"
    heads_sd = {
        k: (v.detach().to("cpu") if isinstance(v, torch.Tensor) else v)
        for k, v in model.state_dict().items()
        if not k.startswith("qwen.")
    }
    print(f"[merge] Saving {len(heads_sd)} non-Qwen tensors to: {heads_out}")
    torch.save(heads_sd, heads_out)

    # ------------------------------------------------------------
    # 8. Provenance sentinel — trainer keys off this file's presence
    # ------------------------------------------------------------
    marker = args.output / "MERGED_BASE.json"
    marker.write_text(json.dumps({
        "source_checkpoint": str(args.stage3_ckpt.resolve()),
        "source_lora_rank": s3_rank,
        "source_lora_alpha": s3_alpha,
        "source_lora_target_modules": s3_targets,
        "target_lora_rank": args.target_lora_rank,
        "qwen_model_id": qwen_id,
        "merge_mode": ("nf4_requantised" if _load_quantized else "fp16_exact"),
        "parity_tolerance": args.parity_tol,
        "parity_verified": bool(args.verify),
        "merged_at": str(date.today()),
        "num_heads_tensors": len(heads_sd),
        "note": (
            "Stage-3 rank-16 LoRA folded into Qwen backbone "
            + ("(NF4 requantised — expect ~1-2%% per-layer drift)"
               if _load_quantized else "(fp16 exact merge)")
            + ". Stage 4 must load qwen_merged_fp16/ as base (fresh NF4 quant + "
              "rank-32 LoRA) plus heads.bin as sidecar. Do NOT strict-load "
              "pytorch_model.bin here — it does not exist in this dir by design "
              "(bytes are split base/heads)."
        ),
    }, indent=2))
    print(f"[merge] Wrote sentinel: {marker}")

    # ------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------
    total = sum(f.stat().st_size for f in args.output.rglob("*") if f.is_file())
    print(f"\n[merge] Done. Output directory total: {total / 1e9:.2f} GB")
    print(f"[merge] Contents of {args.output}:")
    for f in sorted(args.output.rglob("*")):
        if f.is_file():
            print(f"    {f.relative_to(args.output)}  ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
