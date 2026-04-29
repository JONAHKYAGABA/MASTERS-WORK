"""
scripts/prefetch_model.py — robust HF model prefetch via snapshot_download.

Bypasses the huggingface-cli / hf binary churn. Uses the underlying Python API
which has been stable for years. Resumable: if the download was interrupted,
half-downloaded shards are reused.

Usage:
    python scripts/prefetch_model.py
    MODEL_NAME=Qwen/Qwen3-30B-A3B python scripts/prefetch_model.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Speed: 3-4x faster downloads via Rust hf_transfer if installed.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def cache_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            p = Path(root) / fn
            try:
                if p.is_file() and not p.is_symlink():
                    total += p.stat().st_size
            except OSError:
                continue
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B"))
    ap.add_argument(
        "--cache-dir",
        default=os.environ.get("TRANSFORMERS_CACHE")
        or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"),
    )
    ap.add_argument("--max-workers", type=int, default=int(os.environ.get("HF_DOWNLOAD_WORKERS", "4")))
    ap.add_argument("--max-attempts", type=int, default=8)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = "models--" + args.repo.replace("/", "--")
    snap_dir = cache_dir / safe

    print(f"[prefetch] repo      = {args.repo}")
    print(f"[prefetch] cache dir = {cache_dir}")
    print(f"[prefetch] hf_transfer = {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '0')}")
    print(f"[prefetch] before    : {_human(cache_size(snap_dir))} in {snap_dir}")

    # Try the modern API. snapshot_download has been the stable workhorse for years.
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError:
        print("[prefetch] ERROR: huggingface_hub not installed", file=sys.stderr)
        return 2

    last_err: Exception | None = None
    for attempt in range(1, args.max_attempts + 1):
        try:
            print(f"[prefetch] attempt {attempt}/{args.max_attempts}: snapshot_download(...)")
            t0 = time.time()
            path = snapshot_download(
                repo_id=args.repo,
                cache_dir=str(cache_dir),
                max_workers=args.max_workers,
            )
            secs = time.time() - t0
            after = cache_size(snap_dir)
            print(f"[prefetch] ✓ done in {secs:.1f}s")
            print(f"[prefetch] resolved snapshot: {path}")
            print(f"[prefetch] after     : {_human(after)} in {snap_dir}")
            if after < 5 * 1024 * 1024 * 1024:  # < 5 GB is suspicious for a 30B+ model
                print("[prefetch] WARNING: cache is suspiciously small for a 30B+ model.")
                print("[prefetch]          The repo might be partial or use a different shard layout.")
            return 0
        except Exception as e:
            last_err = e
            print(f"[prefetch] attempt {attempt} failed: {type(e).__name__}: {e}")
            # Most failures are transient (network, HF server). Back off and retry.
            time.sleep(min(60, 5 * attempt))

    print(f"[prefetch] FATAL after {args.max_attempts} attempts: {last_err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
