#!/usr/bin/env python
"""
scripts/eval_test_set.py

Run the full v2 validation loop on the official QBA **test** split (rather
than the dev slice used at training time). Loads a Stage-4 customvqamodel
checkpoint and reports the same metric panel that `validate()` reports in
train_mimic_cxr.py, so numbers are directly comparable across dev and test.

Two ablation modes:
    default            SG pipeline active (with-SG numbers)
    --zero_sg_tokens   soft tokens zeroed before injection (Path A ablation)

Usage examples
--------------
    # Stage-4 with-SG on 10K test samples
    SG_TRUST_CHECKPOINT=1 CUDA_VISIBLE_DEVICES=0 \
    python scripts/eval_test_set.py \
        --config configs/stage4_finetune.yaml \
        --checkpoint /data/checkpoints/mimic-cxr-vqa/finetune/checkpoint-3125 \
        --model_id ./checkpoints/mimic-cxr-vqa/pretrain/final_model_merged/qwen_merged_fp16 \
        --max_samples 10000 \
        --output logs/test_eval_withsg_$(date +%Y%m%d_%H%M%S).json

    # Path A ablation on the same test slice
    SG_TRUST_CHECKPOINT=1 CUDA_VISIBLE_DEVICES=0 \
    python scripts/eval_test_set.py \
        --config configs/stage4_finetune.yaml \
        --checkpoint /data/checkpoints/mimic-cxr-vqa/finetune/checkpoint-3125 \
        --model_id ./checkpoints/mimic-cxr-vqa/pretrain/final_model_merged/qwen_merged_fp16 \
        --max_samples 10000 \
        --zero_sg_tokens \
        --output logs/test_eval_nosg_infer_$(date +%Y%m%d_%H%M%S).json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

# Make repo root importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.mimic_cxr_config import load_config_from_file  # noqa: E402
from data.mimic_cxr_dataset import MIMICCXRVQADataset, create_dataloader  # noqa: E402
from models import customvqamodel  # noqa: E402
from training.loss import MultiTaskLoss  # noqa: E402
# validate() lives in the training script alongside train loop
from train_mimic_cxr import validate  # noqa: E402


def _safe_load_state_dict(bin_path: Path):
    """Same trusted-checkpoint policy as the trainer + quantize script."""
    if os.environ.get("SG_TRUST_CHECKPOINT") != "1":
        raise RuntimeError(
            "Requires SG_TRUST_CHECKPOINT=1 in the env to allow weights_only=False "
            "torch.load of our own trainer checkpoint (numpy scalars in metrics dict "
            "break the strict loader). Rerun with:\n"
            "    SG_TRUST_CHECKPOINT=1 python scripts/eval_test_set.py ..."
        )
    return torch.load(str(bin_path), map_location="cpu", weights_only=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path,
                    help="Stage-4 (or any) YAML config used to build the model architecture")
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="Directory containing pytorch_model.bin (or the file itself)")
    ap.add_argument("--model_id", type=str, default=None,
                    help="Qwen base to construct from — usually the merged_base dir. "
                         "Falls back to config.model.qwen_model_id or the HF hub default.")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="Which official QBA split to evaluate on. Default: test.")
    ap.add_argument("--max_samples", type=int, default=10_000,
                    help="Cap on test samples. 10K -> ~1pp binomial 95%% CI. Default 10000.")
    ap.add_argument("--batch_size", type=int, default=4,
                    help="Per-GPU eval batch size (validate() is fp16 inference, batch 4 fits 48 GB).")
    ap.add_argument("--num_workers", type=int, default=2,
                    help="Dataloader worker count.")
    ap.add_argument("--zero_sg_tokens", action="store_true",
                    help="Path-A ablation: zero the SG soft tokens before injection. "
                         "SG generator + encoder + projector still run (unchanged compute) "
                         "but the LM sees zero vectors at every <|sg_token_k|> position.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Destination JSON for the metrics report.")
    ap.add_argument("--device", default="cuda:0",
                    help="cuda:N or cpu. Default cuda:0.")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # 1. Load config, mirror the trainer's LoRA / quantization decisions
    # ------------------------------------------------------------------
    cfg = load_config_from_file(str(args.config))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    lora_rank = int(getattr(cfg.model, "lora_rank", 32))
    lora_alpha = int(getattr(cfg.model, "lora_alpha", 2 * lora_rank))
    lora_targets = getattr(cfg.model, "lora_target_modules", None) or [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ]
    qwen_id = (
        args.model_id
        or getattr(cfg.model, "qwen_model_id", None)
        or "Qwen/Qwen3-VL-8B-Instruct"
    )

    print(f"[eval] Building customvqamodel:")
    print(f"       qwen_id     = {qwen_id}")
    print(f"       lora_rank   = {lora_rank}, alpha = {lora_alpha}")
    print(f"       targets     = {lora_targets}")
    print(f"       zero_sg     = {args.zero_sg_tokens}")

    model = customvqamodel(
        qwen_model_id=qwen_id,
        use_quantization=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_target_modules=lora_targets,
        num_regions=cfg.model.num_regions,
        num_entities=cfg.model.num_entities,
        num_binary=cfg.model.num_binary_classes,
        num_category=cfg.model.num_category_classes,
        num_region_classes=cfg.model.num_region_classes,
        num_severity=cfg.model.num_severity_classes,
        training_mode="finetune",
        freeze_sg_generator=True,
        torch_dtype=torch.float16,
    )

    # ------------------------------------------------------------------
    # 2. Load Stage-4 checkpoint
    # ------------------------------------------------------------------
    ckpt_path = args.checkpoint
    if ckpt_path.is_dir():
        for candidate in ("pytorch_model.bin", "model.bin", "checkpoint.bin"):
            if (ckpt_path / candidate).exists():
                ckpt_path = ckpt_path / candidate
                break
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"No checkpoint at {args.checkpoint}")

    print(f"[eval] Loading state_dict from: {ckpt_path}")
    payload = _safe_load_state_dict(ckpt_path)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        sd = payload["model_state_dict"]
    else:
        sd = payload
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[eval] Loaded: missing={len(missing)} unexpected={len(unexpected)}")
    del sd

    # ------------------------------------------------------------------
    # 3. Path A ablation — module-level flag (validate() re-reads per call)
    # ------------------------------------------------------------------
    if args.zero_sg_tokens:
        model.zero_sg_tokens = True
        print("[eval] Ablation ACTIVE: zero_sg_tokens=True")

    model.to(device).eval()

    # ------------------------------------------------------------------
    # 4. Build the test-split dataset + dataloader
    # ------------------------------------------------------------------
    print(f"[eval] Building {args.split}-split dataset (max_samples={args.max_samples})...")
    ds = MIMICCXRVQADataset(
        config=cfg.data,
        split=args.split,
        max_samples=args.max_samples,
        model_config=cfg.model,
    )
    print(f"[eval]   built {len(ds)} samples for split={args.split}")

    dl = create_dataloader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 5. Run validate() using the same loss module the trainer uses
    # ------------------------------------------------------------------
    criterion = MultiTaskLoss(
        vqa_weight=float(getattr(cfg.training, "vqa_loss_weight", 0.15)),
        generation_weight=float(getattr(cfg.training, "generation_loss_weight", 1.0)),
        chexpert_weight=float(getattr(cfg.training, "chexpert_loss_weight", 0.3)),
        scene_graph_weight=float(getattr(cfg.training, "scene_graph_loss_weight", 0.0)),
        grounding_weight=float(getattr(cfg.training, "grounding_loss_weight", 3.0)),
        phase="finetune",
    )

    print("[eval] Running validate() on test split...")
    metrics = validate(model, dl, criterion, device, cfg)
    print("[eval] validate() complete.")

    # ------------------------------------------------------------------
    # 6. Dump JSON report
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "run_kind": "no_sg_infer_ablation" if args.zero_sg_tokens else "with_sg",
        "split": args.split,
        "max_samples": args.max_samples,
        "actual_samples": len(ds),
        "checkpoint": str(args.checkpoint.resolve()),
        "model_id": qwen_id,
        "config": str(args.config.resolve()),
        "zero_sg_tokens": bool(args.zero_sg_tokens),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metrics": {k: (float(v) if hasattr(v, "item") or isinstance(v, (int, float)) else v)
                    for k, v in metrics.items()},
    }
    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(f"[eval] Wrote metrics report: {args.output}")

    # ------------------------------------------------------------------
    # 7. Human-readable summary on stdout
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Test-split results ({'WITH SG' if not args.zero_sg_tokens else 'SG ZEROED'})")
    print("=" * 70)
    for k in ("classification_accuracy", "binary_accuracy", "grounding_mean_iou",
              "grounding_acc_iou50", "chexpert_auroc",
              "generation_bleu", "generation_rouge_l"):
        v = metrics.get(k)
        if v is not None:
            print(f"  {k:32s} = {float(v):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
