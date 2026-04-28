#!/usr/bin/env python3
"""
SSG-VQA-Net v2 smoke test.

Purpose: confirm SSGVQANetV2 can be instantiated, run forward+backward+
optimizer step, and that all the safety checks (vision-path verification,
SG token round-trip, label masking) pass — on every visible GPU. No real
dataset required: builds a 1-sample dummy batch in memory.

Failures here *will* prevent training. Pass these and Stage 1 is unblocked.

Exit codes:
  0  every visible GPU passed
  1  at least one GPU failed
  2  setup error (model load / import) — no GPU was even attempted

Usage:
  python scripts/smoke_test_v2.py
  python scripts/smoke_test_v2.py --model_id Qwen/Qwen2.5-VL-7B-Instruct
  python scripts/smoke_test_v2.py --no_quantization   # full-precision LoRA
  python scripts/smoke_test_v2.py --gpus 0,2          # subset of GPUs
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from typing import List

import torch
from PIL import Image

# Make the project importable regardless of where this is launched from.
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Defer heavy imports until after argparse so --help is fast.
def _build_dummy_batch(batch_size: int, image_size: int = 448):
    """One-sample (or N-sample) batch of in-memory dummies — no disk I/O."""
    pil_images = [
        Image.new("RGB", (image_size, image_size), color=(128, 128, 128))
        for _ in range(batch_size)
    ]
    questions = ["Is there consolidation in the right lower lobe?"] * batch_size
    answer_texts = [
        "<think>Reviewing the chest radiograph for consolidation.</think>"
        "<box>0.620,0.540,0.880,0.810</box>"
        "<answer>Yes, right lower lobe consolidation is present.</answer>"
    ] * batch_size
    return pil_images, questions, answer_texts


def _run_one_gpu(
    gpu_idx: int,
    model_id: str,
    use_quantization: bool,
    batch_size: int,
    max_new_tokens: int,
    skip_inference: bool,
) -> bool:
    """Returns True on pass, False on fail. Stays self-contained per GPU."""
    print()
    print(f"================ GPU {gpu_idx} ================")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(gpu_idx)
        free, total = torch.cuda.mem_get_info(gpu_idx)
        print(f"  device:     {props.name}  cc={props.major}.{props.minor}")
        print(f"  vram:       {props.total_memory/1024**3:.1f} GiB total, "
              f"{free/1024**3:.1f} GiB free")
        torch.cuda.set_device(gpu_idx)

        # Compute capability decides bf16 vs fp16; QLoRA forced below cc 8.0
        cc = (props.major, props.minor)
        torch_dtype = torch.bfloat16 if cc >= (8, 0) else torch.float16
        force_qlora = cc < (8, 0)
    else:
        print("  ⚠ no CUDA — running on CPU (extremely slow)")
        cc = (0, 0)
        torch_dtype = torch.float32
        force_qlora = False

    use_quant_eff = use_quantization or force_qlora
    print(f"  dtype:      {torch_dtype}")
    print(f"  quant:      4-bit NF4 (QLoRA)" if use_quant_eff else f"  quant:      OFF (full precision LoRA)")

    # Heavy import only inside the per-GPU loop so a single GPU's failure
    # doesn't cascade into module-level errors.
    try:
        from models import SSGVQANetV2
    except Exception as e:
        print(f"  ✗ cannot import SSGVQANetV2: {e}")
        traceback.print_exc()
        return False

    # ---- 1. Build model -------------------------------------------------
    t0 = time.time()
    try:
        model = SSGVQANetV2(
            qwen_model_id=model_id,
            use_quantization=use_quant_eff,
            num_sg_tokens=4,        # smaller than default 8 for faster smoke
            training_mode="pretrain",
            torch_dtype=torch_dtype,
            max_answer_length=max_new_tokens,
        )
    except Exception as e:
        print(f"  ✗ model construction failed: {e}")
        traceback.print_exc()
        return False
    print(f"  ✓ model built in {time.time()-t0:.1f}s "
          f"(d_llm={model.d_llm}, sg_tokens={model.num_sg_tokens})")

    # Move new modules onto GPU (Qwen quantized weights already on device).
    # SG generator stays in fp32 on the GPU — its BatchNorms misbehave in fp16.
    device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
    for mod_name in ("sg_encoder", "sg_projector", "grounding_head", "aux_heads"):
        mod = getattr(model, mod_name)
        mod.to(device=device, dtype=torch_dtype)
    model.sg_generator.to(device=device)  # fp32 on GPU
    if not use_quant_eff:
        # Non-quantized Qwen needs explicit move. Quantized weights are
        # placed by bitsandbytes and should not be `.to()`'d.
        try:
            model.qwen.to(device)
        except Exception as e:
            print(f"  ⚠ model.qwen.to() failed (often expected for QLoRA): {e}")

    # ---- 2. Trainable param count ---------------------------------------
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  ✓ trainable params: {trainable/1e6:.1f}M / {total/1e6:.0f}M  "
          f"({100*trainable/total:.2f}%)")

    # ---- 3. Build dummy batch ------------------------------------------
    pil_images, questions, answer_texts = _build_dummy_batch(batch_size)
    gt_grounding_bboxes = torch.tensor(
        [[0.62, 0.54, 0.88, 0.81]] * batch_size, dtype=torch.float, device=device,
    )

    # ---- 4. Training forward + backward + optimizer step ----------------
    print("  • training step (forward → loss → backward → step)")
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4
    )
    model.train()
    try:
        t1 = time.time()
        out = model(
            images=None,                # legacy slot, unused in v2
            pil_images=pil_images,
            questions=questions,
            answer_texts=answer_texts,
            gt_grounding_bboxes=gt_grounding_bboxes,
        )
        fwd_t = time.time() - t1

        loss = out["lm_loss"]
        if loss is None:
            print("  ✗ lm_loss is None — Qwen forward did not produce a loss")
            return False
        if not torch.isfinite(loss):
            print(f"  ✗ non-finite loss: {loss.item()}")
            return False
        print(f"  ✓ forward ok: lm_loss={loss.item():.4f}  ({fwd_t:.1f}s)")

        # Verify the v2 output contract
        for k in (
            "vqa_logits", "chexpert_logits", "pooled_output",
            "grounding_outputs", "generated_scene_graphs",
        ):
            if k not in out:
                print(f"  ✗ output missing key: {k}")
                return False
        bb = out["grounding_outputs"]["bbox_pred"]
        if bb.shape != (batch_size, 4):
            print(f"  ✗ unexpected grounding bbox shape: {bb.shape}")
            return False
        print(f"  ✓ output contract ok: bbox_pred={bb.shape}, "
              f"vqa heads={list(out['vqa_logits'].keys())}")

        t2 = time.time()
        loss.backward()
        bwd_t = time.time() - t2

        # Check at least one trainable param has a non-zero gradient
        any_grad = False
        for p in model.parameters():
            if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0:
                any_grad = True
                break
        if not any_grad:
            print("  ✗ backward produced no gradients on any trainable parameter")
            return False
        print(f"  ✓ backward ok ({bwd_t:.1f}s, gradients present)")

        t3 = time.time()
        optim.step()
        optim.zero_grad(set_to_none=True)
        print(f"  ✓ optimizer step ok ({time.time()-t3:.2f}s)")

        # The vision-path verification fires on the *first* training forward.
        # If we got here without it raising, it passed.
        print("  ✓ vision-path verification passed "
              "(pixel_values changed Qwen's logits)")

    except Exception as e:
        print(f"  ✗ training step failed: {e}")
        traceback.print_exc()
        return False

    # ---- 5. Inference (generate) ---------------------------------------
    if not skip_inference:
        print("  • inference step (generate)")
        model.eval()
        try:
            t4 = time.time()
            with torch.no_grad():
                out_gen = model(
                    images=None,
                    pil_images=pil_images,
                    questions=questions,
                    # answer_texts=None → free generation
                )
            gen_t = time.time() - t4
            txt = out_gen["generated_answer_text"]
            if not txt or not isinstance(txt, list):
                print(f"  ✗ generation returned no text: {txt}")
                return False
            print(f"  ✓ generation ok ({gen_t:.1f}s)")
            print(f"      sample[0]: {txt[0][:120].replace(chr(10), ' ')!r}")
        except Exception as e:
            print(f"  ✗ inference step failed: {e}")
            traceback.print_exc()
            return False

    # ---- 6. Tear down ---------------------------------------------------
    del model, optim, out
    if not skip_inference:
        try:
            del out_gen
        except NameError:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(gpu_idx)
        print(f"  ✓ teardown ok (vram free after: {free/1024**3:.1f} GiB)")

    print(f"  GPU {gpu_idx}: PASS")
    return True


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SSG-VQA-Net v2 per-GPU smoke test")
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct",
                   help="Qwen2.5-VL HF id (default: 3B for fast iteration)")
    p.add_argument("--no_quantization", action="store_true",
                   help="Disable 4-bit QLoRA (only use on Ampere+ with VRAM headroom)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=16,
                   help="Generation length cap for the inference smoke step")
    p.add_argument("--gpus", type=str, default="",
                   help='Comma-separated GPU indices (default: all visible). '
                        'Example: "0,2"')
    p.add_argument("--skip_inference", action="store_true",
                   help="Only test forward+backward+step; skip the generate test")
    args = p.parse_args(argv)

    # Resolve GPU list
    if torch.cuda.is_available():
        if args.gpus:
            try:
                gpu_list = [int(g) for g in args.gpus.split(",") if g.strip()]
            except ValueError:
                print(f"--gpus must be a comma-separated int list, got: {args.gpus!r}")
                return 2
        else:
            gpu_list = list(range(torch.cuda.device_count()))
    else:
        gpu_list = [0]  # CPU "device 0"

    print(f"smoke_test_v2.py — model={args.model_id} "
          f"quant={'OFF' if args.no_quantization else 'ON'} "
          f"gpus={gpu_list} bs={args.batch_size}")

    results: dict[int, bool] = {}
    for gi in gpu_list:
        try:
            results[gi] = _run_one_gpu(
                gpu_idx=gi,
                model_id=args.model_id,
                use_quantization=not args.no_quantization,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                skip_inference=args.skip_inference,
            )
        except Exception as e:
            print(f"  ✗ uncaught exception on GPU {gi}: {e}")
            traceback.print_exc()
            results[gi] = False

    # Summary
    print()
    print("================ SUMMARY ================")
    any_pass = False
    for gi, ok in results.items():
        print(f"  GPU {gi}: {'PASS' if ok else 'FAIL'}")
        any_pass = any_pass or ok

    if all(results.values()):
        print("\n✓ All GPUs passed. Ready for Stage 1.")
        return 0
    if not any_pass:
        print("\n✗ All GPUs failed — no GPU is usable for v2 training.")
        return 1
    print("\n⚠ Some GPUs failed. Check logs above before launching multi-GPU training.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
