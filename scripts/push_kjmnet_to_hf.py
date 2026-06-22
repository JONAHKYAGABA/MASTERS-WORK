"""Push 4-stage KJM-Net curriculum checkpoints to Hugging Face Hub.

One public repo (KYAGABA/kjmnet) with each stage as a sub-folder.

Usage:
    # 1) Set token (get one at https://huggingface.co/settings/tokens with WRITE scope)
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

    # 2) Run
    .venv/bin/python scripts/push_kjmnet_to_hf.py

The script:
  * Creates the repo if absent (public).
  * Uploads each stage's checkpoint folder via upload_large_folder
    (parallel + resumable + dedup via SHA — re-runs skip already-uploaded files).
  * Writes a top-level README with the results table.
  * Skips a stage if its checkpoint folder is missing on disk.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, login


REPO_ID = "kyagabajonah352/kjmnet"
REPO_TYPE = "model"

# (local checkpoint dir, sub-folder name in the HF repo, short description)
STAGES = [
    (
        "checkpoints/mimic-cxr-vqa/best_model",
        "stage1_sg_only",
        "Stage 1 sg_only (B-grade) — Val SG IoU 0.320, Val Acc 0.048",
    ),
    (
        "checkpoints/mimic-cxr-vqa/checkpoint-15000",
        "stage2_alignment",
        "Stage 2 alignment (B-grade) — gen loss 3.0 -> 0.5",
    ),
    (
        "checkpoints/mimic-cxr-vqa/pretrain/best_model",
        "stage3_pretrain",
        "Stage 3 pretrain (B-grade) — Val Acc 0.280, Val Grd IoU 0.280",
    ),
    (
        "checkpoints/mimic-cxr-vqa/finetune/checkpoint-13262",
        "stage4_finetune_e1",
        "Stage 4 finetune (A-grade) Epoch 1 — Val Acc 0.464, Val Grd IoU 0.413  [BEST ACC]",
    ),
    (
        "checkpoints/mimic-cxr-vqa/finetune/best_model",
        "stage4_finetune_e2",
        "Stage 4 finetune (A-grade) Epoch 2 — Val BLEU 0.058, ROUGE-L 0.157  [BEST GEN]",
    ),
]

README = """# KJM-Net

4-stage curriculum for chest X-ray visual question answering on MIMIC-CXR.

Backbone: **Qwen3-VL-8B** (QLoRA NF4) + scene-graph conditioning.
Dataset: **MIMIC-Ext-CXR-QBA** (PhysioNet).

## Results

| Stage | Folder | Val Acc | Val Grounding IoU | Val SG IoU | Val BLEU | Val ROUGE-L |
|---|---|---:|---:|---:|---:|---:|
| 1 sg_only (B) | `stage1_sg_only/` | 0.048 | 0.032 | **0.320** | — | — |
| 2 alignment (B) | `stage2_alignment/` | — | — | — | — | — |
| 3 pretrain (B) | `stage3_pretrain/` | 0.280 | 0.280 | 0.320 | 0.078 | 0.231 |
| 4 finetune E1 (A) | `stage4_finetune_e1/` | **0.464** | **0.413** | 0.328 | 0.034 | 0.107 |
| 4 finetune E2 (A) | `stage4_finetune_e2/` | 0.384 | 0.368 | 0.330 | **0.058** | **0.157** |

Compute: 2x Quadro RTX 8000 (96 GB), ~95 h.

Each `stageX/` folder contains `pytorch_model.bin` and `config.json`.
"""


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
        print(f"Authenticated via HF_TOKEN")
    else:
        print(
            "ERROR: HF_TOKEN not set. Get one (WRITE scope) at "
            "https://huggingface.co/settings/tokens then:\n"
            "  export HF_TOKEN=hf_xxx...\n"
            "and re-run."
        )
        return 1

    # Enable xet high-performance transfer (replaces deprecated hf_transfer)
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    api = HfApi(token=token)

    # Create repo if absent (idempotent — exist_ok=True)
    print(f"\nCreating / verifying repo {REPO_ID} (PRIVATE)...")
    create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=True,
        exist_ok=True,
        token=token,
    )
    print(f"  Repo URL: https://huggingface.co/{REPO_ID}")

    # Push README first so the repo page shows the table right away
    readme_path = Path("/tmp/kjmnet_README.md")
    readme_path.write_text(README)
    print(f"\nUploading README...")
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Add README with curriculum results table",
    )
    print("  README uploaded")

    # Push each stage
    skipped = []
    for local_dir, hf_subfolder, desc in STAGES:
        p = Path(local_dir)
        if not p.exists():
            print(f"\n[SKIP] {local_dir} does not exist on disk -> {hf_subfolder}")
            skipped.append((local_dir, hf_subfolder))
            continue
        size_gb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9
        print(f"\nUploading {local_dir}  ({size_gb:.1f} GB)  -> {hf_subfolder}/")
        print(f"  {desc}")
        # upload_large_folder = parallel multi-file upload with resume
        api.upload_large_folder(
            folder_path=str(p),
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            path_in_repo=hf_subfolder,
            # don't upload the optimizer state — saves ~30% upload bandwidth
            # and isn't useful for inference-only consumers
            ignore_patterns=["optimizer.bin", "scheduler.bin", "training_state.json"],
        )
        print(f"  Done. Visit: https://huggingface.co/{REPO_ID}/tree/main/{hf_subfolder}")

    print(f"\n{'=' * 60}")
    print(f"ALL UPLOADS COMPLETE")
    print(f"Repo: https://huggingface.co/{REPO_ID}")
    if skipped:
        print(f"\nSkipped (not found on disk):")
        for local, sub in skipped:
            print(f"  {sub}: {local}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
