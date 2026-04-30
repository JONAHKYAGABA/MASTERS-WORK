"""
scripts/push_lora.py — push the fine-tuned LoRA adapter to HuggingFace Hub.

Reads HF_TOKEN and HF_PUSH_REPO from environment (or .env loaded by run_all.sh).
Creates the repo if it doesn't exist. Skips silently when HF_PUSH_REPO is empty
so the pipeline doesn't fail on optional pushes.

Usage:
    HF_TOKEN=hf_xxx HF_PUSH_REPO=youruser/telco-qwen-lora-v1 \\
        python scripts/push_lora.py --lora_dir training/checkpoints/run_v1/best_lora
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_dir", default="training/checkpoints/run_v1/best_lora")
    ap.add_argument("--repo", default=os.environ.get("HF_PUSH_REPO", ""))
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--private", default=os.environ.get("HF_PUSH_PRIVATE", "true"))
    args = ap.parse_args()

    if not args.repo:
        print("[push] HF_PUSH_REPO not set — skipping push.")
        return 0
    if not args.token:
        print("[push] HF_TOKEN not set — cannot push. Set HF_TOKEN in .env.", file=sys.stderr)
        return 1

    lora_dir = Path(args.lora_dir)
    if not (lora_dir / "adapter_config.json").exists():
        print(f"[push] No adapter at {lora_dir} — nothing to push.", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except ImportError:
        print("[push] huggingface_hub not installed", file=sys.stderr)
        return 1

    private = str(args.private).lower() in {"1", "true", "yes", "y"}
    print(f"[push] repo    = {args.repo}")
    print(f"[push] private = {private}")
    print(f"[push] from    = {lora_dir}  ({sum(p.stat().st_size for p in lora_dir.rglob('*') if p.is_file()) / 1e6:.1f} MB)")

    try:
        create_repo(args.repo, token=args.token, private=private, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"[push] create_repo error: {e}", file=sys.stderr)
        return 1

    try:
        upload_folder(
            folder_path=str(lora_dir),
            repo_id=args.repo,
            token=args.token,
            commit_message="Telco Track A LoRA fine-tune",
            ignore_patterns=["*.tmp", "checkpoint-*"],
        )
    except Exception as e:
        print(f"[push] upload_folder error: {e}", file=sys.stderr)
        return 1

    print(f"[push] ✓ pushed: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
