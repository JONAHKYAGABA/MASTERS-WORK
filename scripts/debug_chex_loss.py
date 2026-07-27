#!/usr/bin/env python
"""
One-batch debug: build the SAME dataset the trainer uses, fetch one batch,
run the model forward, run the loss function, and print every relevant value
to localize why chex_loss=0.0000 in training.

Usage:
    PYTHONPATH=$PWD python scripts/debug_chex_loss.py
"""

from __future__ import annotations
import os
import sys
import pickle
from pathlib import Path

import torch

sys.path.insert(0, '.')
from configs.mimic_cxr_config import load_config_from_file
from data.mimic_cxr_dataset import MIMICCXRVQADataset, collate_fn
from models.ssg_vqa_net_v2 import customvqamodel
from training.loss import MultiTaskLoss


def main():
    print("=" * 70)
    print("ONE-BATCH CHEX LOSS DEBUG")
    print("=" * 70)

    cfg = load_config_from_file('configs/pretrain_config.yaml')
    cfg.data.quality_grade = 'all'

    # Use the actual paths the trainer's launch uses, not the config defaults
    # (which are /path/to/... placeholders).
    mimic_cxr = 'data/mimic-cxr-jpg'
    mimic_qa = 'data/mimic-ext-cxr-qba'
    print(f"\n[config] mimic_cxr = {mimic_cxr}")
    print(f"[config] mimic_qa  = {mimic_qa}")
    print(f"[config] chexpert_labels_path = {getattr(cfg.data, 'chexpert_labels_path', '(default)')}")
    print(f"[config] quality_grade = {cfg.data.quality_grade}")

    # Build a tiny dataset directly — uses .cache/dataset_samples/*.pkl if present.
    print(f"\n[1] Building dataset (4 samples)...")
    ds = MIMICCXRVQADataset(
        mimic_cxr_path=mimic_cxr,
        mimic_qa_path=mimic_qa,
        split='train',
        max_samples=4,
        quality_grade='all',
        use_cache=True,
    )
    print(f"  → dataset built: {len(ds)} samples")

    # Get one batch (4 samples, collated)
    print("\n[2] Fetching batch of 4 samples...")
    items = [ds[i] for i in range(min(4, len(ds)))]
    batch = collate_fn(items)
    chex_lbl = batch['chexpert_labels']
    chex_msk = batch['chexpert_mask']
    print(f"  chexpert_labels.shape = {chex_lbl.shape}  dtype={chex_lbl.dtype}")
    print(f"  chexpert_labels.sum() = {chex_lbl.sum().item()}")
    print(f"  chexpert_mask.shape   = {chex_msk.shape}  dtype={chex_msk.dtype}")
    print(f"  chexpert_mask.sum()   = {chex_msk.sum().item()}  ← if 0 = root cause")
    print(f"  per-sample mask sums: {chex_msk.sum(dim=1).tolist()}")

    if chex_msk.sum().item() == 0:
        print("\n❌ ROOT CAUSE FOUND: batch chexpert_mask is all zeros.")
        print("   The dataset is loading samples but their CheXpert lookup fails.")
        print("   Likely the dataset isn't passing the CSV path correctly.")
        return 1

    # Run model forward + loss
    print("\n[3] Building model (this takes ~60s for Qwen3-VL 8B in 4-bit)...")
    device = torch.device('cuda:0')
    model = customvqamodel(
        qwen_model_id='Qwen/Qwen3-VL-8B-Instruct',
        use_quantization=True,
        training_mode='sg_only',
    ).to(device)
    model.eval()

    print("\n[4] Forward pass...")
    with torch.no_grad():
        outputs = model(
            images=batch['images'].to(device),
            pil_images=batch.get('pil_images'),
            questions=batch.get('questions'),
            answer_texts=batch.get('answer_texts'),
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            scene_graphs=batch['scene_graphs'],
            chexpert_labels=batch['chexpert_labels'].to(device),
        )

    chex_logits = outputs.get('chexpert_logits')
    print(f"  chexpert_logits is None?  {chex_logits is None}")
    if chex_logits is not None:
        print(f"  chexpert_logits.shape   = {chex_logits.shape}")
        print(f"  chexpert_logits.dtype   = {chex_logits.dtype}")
        print(f"  chexpert_logits[0]      = {chex_logits[0].float().tolist()}")
        print(f"  has NaN?                = {bool(torch.isnan(chex_logits).any())}")

    print("\n[5] Compute loss...")
    criterion = MultiTaskLoss(training_mode='standard')
    loss, loss_dict = criterion(
        outputs=outputs,
        vqa_targets={'binary': batch['answer_idx'].to(device),
                     'category': batch['answer_idx'].to(device),
                     'region': batch['answer_idx'].to(device),
                     'severity': batch['answer_idx'].to(device)},
        question_types=batch['question_types'],
        chexpert_labels=batch['chexpert_labels'].to(device),
        chexpert_mask=batch['chexpert_mask'].to(device),
    )
    chex_loss = loss_dict.get('chexpert_loss', 'missing')
    if torch.is_tensor(chex_loss):
        chex_loss_val = chex_loss.item()
    else:
        chex_loss_val = chex_loss
    print(f"  chexpert_loss in dict   = {chex_loss_val}")
    print(f"  total loss              = {loss.item():.4f}")
    print(f"  all loss keys           = {list(loss_dict.keys())}")

    print("\n" + ("✅ chex_loss is non-zero — pipeline OK"
                  if isinstance(chex_loss_val, float) and chex_loss_val > 0
                  else "❌ chex_loss is zero — bug is downstream of mask"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
