#!/usr/bin/env python
"""
Pre-download every HF model the training pipeline touches, with a clear
progress bar per model, BEFORE launching multi-GPU training.

Why this matters on marconi:
  - Without pre-download, both DDP ranks call from_pretrained() simultaneously
    and either (a) duplicate the 16GB Qwen2.5-VL-7B download, or (b) one rank
    races ahead and the other crashes with a partial cache.
  - Multi-process tqdm bars interleave into garbage that looks like a "freeze".
  - Pre-downloading once = both ranks instantly load from local cache.

Usage:
    python scripts/predownload_models.py
    python scripts/predownload_models.py --model Qwen/Qwen3-VL-4B-Instruct
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

DEFAULT_MODELS = [
    # Primary VLM backbone — picked up by SSGVQANetV2 default.
    "Qwen/Qwen3-VL-8B-Instruct",
]


def _human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}PB"


def _cache_size(local_dir: Path) -> int:
    if not local_dir.exists():
        return 0
    return sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file())


def download_model(model_id: str) -> None:
    """Download a single model and print a clean before/after summary."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    local_dir = cache_root / ("models--" + model_id.replace("/", "--"))
    before = _cache_size(local_dir)

    print()
    print("=" * 70)
    print(f"  Downloading: {model_id}")
    print(f"  Cache dir:   {local_dir}")
    print(f"  Already on disk: {_human_size(before)}")
    print("=" * 70)

    t0 = time.time()

    # snapshot_download streams a tqdm progress bar to stderr by default.
    # We're single-process here so the bar renders cleanly.
    snapshot_download(
        repo_id=model_id,
        # Resume partial downloads instead of restarting (important if the
        # previous DDP launch crashed mid-fetch).
        resume_download=True,
        # Avoid symlinks — Windows-mounted volumes and some FUSE mounts
        # don't support them; the trainer reads files directly anyway.
        local_dir_use_symlinks=False,
    )

    after = _cache_size(local_dir)
    elapsed = time.time() - t0

    print()
    print(f"  ✓ done in {elapsed:.1f}s — pulled {_human_size(after - before)} new bytes")
    print(f"  ✓ total cached for this model: {_human_size(after)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", action="append", default=None,
        help="HF model id to pre-download. Repeatable. "
             "Defaults to the full pipeline set."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Just print the default model list and exit."
    )
    args = parser.parse_args()

    if args.list:
        for m in DEFAULT_MODELS:
            print(m)
        return 0

    models = args.model or DEFAULT_MODELS
    print(f"\nPre-downloading {len(models)} model(s) to local HF cache:")
    for m in models:
        print(f"  - {m}")

    # HF token resolution — needed for gated repos. Qwen2.5-VL is public so
    # this is usually optional, but we surface it for clarity.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        print(f"  HF_TOKEN: set ({token[:8]}...)")
    else:
        print("  HF_TOKEN: not set (fine for public Qwen models)")

    t0 = time.time()
    for m in models:
        try:
            download_model(m)
        except Exception as e:
            print(f"\nERROR downloading {m}: {e}", file=sys.stderr)
            return 1

    total = time.time() - t0
    print()
    print("=" * 70)
    print(f"  ALL DOWNLOADS COMPLETE in {total:.1f}s ({total / 60:.1f} min)")
    print("=" * 70)
    print()
    print("You can now launch training — both DDP ranks will load from cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
