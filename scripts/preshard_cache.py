#!/usr/bin/env python3
"""
Pre-shard a large cache file for DeepSpeed/DDP distributed training.

This creates per-rank shard files that each process can load independently,
avoiding the OOM issue where all ranks load the full cache simultaneously.

Usage:
    python scripts/preshard_cache.py --cache .cache/dataset_samples/samples_train_25pct_snapshot.pkl --world_size 4
"""

import argparse
import pickle
import gc
import sys
from pathlib import Path

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install tqdm for progress bars: pip install tqdm")


class ProgressFileWrapper:
    """Wrapper to show progress while reading a large file."""
    
    def __init__(self, file_obj, total_size, desc="Loading"):
        self.file_obj = file_obj
        self.total_size = total_size
        self.bytes_read = 0
        
        if TQDM_AVAILABLE:
            self.pbar = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=desc,
                ncols=80
            )
        else:
            self.pbar = None
            self.last_percent = 0
    
    def read(self, size=-1):
        data = self.file_obj.read(size)
        self.bytes_read += len(data)
        
        if self.pbar:
            self.pbar.update(len(data))
        else:
            # Simple percentage fallback
            percent = int(100 * self.bytes_read / self.total_size)
            if percent >= self.last_percent + 5:
                print(f"  {percent}% loaded...", flush=True)
                self.last_percent = percent
        
        return data
    
    def readline(self):
        line = self.file_obj.readline()
        self.bytes_read += len(line)
        if self.pbar:
            self.pbar.update(len(line))
        return line
    
    def close(self):
        if self.pbar:
            self.pbar.close()
        self.file_obj.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def preshard_cache(cache_path: str, world_size: int = 4):
    """Split cache into per-rank shards."""
    p = Path(cache_path)
    
    if not p.exists():
        print(f"ERROR: Cache file not found: {p}")
        return False
    
    file_size = p.stat().st_size
    print(f"=" * 60)
    print(f"PRE-SHARDING CACHE FOR DISTRIBUTED TRAINING")
    print(f"=" * 60)
    print(f"Cache file: {p}")
    print(f"File size:  {file_size / 1e9:.2f} GB")
    print(f"World size: {world_size} GPUs")
    print(f"=" * 60)
    print()
    
    # Load with progress bar
    print("Step 1/2: Loading cache file...")
    with open(p, 'rb') as f:
        wrapped = ProgressFileWrapper(f, file_size, desc="Loading cache")
        samples = pickle.load(wrapped)
        wrapped.close()
    
    total = len(samples)
    print(f"\n✓ Loaded {total:,} samples")
    print()
    
    # Create shards with progress
    print(f"Step 2/2: Creating {world_size} shard files...")
    
    if TQDM_AVAILABLE:
        shard_iter = tqdm(range(world_size), desc="Writing shards", ncols=80)
    else:
        shard_iter = range(world_size)
    
    for rank in shard_iter:
        shard = samples[rank::world_size]
        shard_path = p.parent / f"{p.stem}.shard{rank}of{world_size}.pkl"
        
        if not TQDM_AVAILABLE:
            print(f"  Shard {rank}/{world_size}: {len(shard):,} samples...", end=" ", flush=True)
        
        with open(shard_path, 'wb') as f:
            pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        shard_size = shard_path.stat().st_size / 1e9
        
        if not TQDM_AVAILABLE:
            print(f"({shard_size:.2f} GB) ✓")
        elif TQDM_AVAILABLE:
            tqdm.write(f"  ✓ {shard_path.name}: {len(shard):,} samples ({shard_size:.2f} GB)")
    
    # Cleanup
    del samples
    gc.collect()
    
    print()
    print(f"=" * 60)
    print(f"✓ SUCCESS! Created {world_size} shard files")
    print(f"=" * 60)
    print(f"  Each rank will load ~{total//world_size:,} samples")
    print(f"  Memory per rank: ~{file_size / world_size / 1e9:.2f} GB (was {file_size/1e9:.2f} GB)")
    print()
    print("Now run DeepSpeed - shards will be auto-detected!")
    print(f"=" * 60)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-shard cache for distributed training")
    parser.add_argument("--cache", type=str, required=True, help="Path to cache .pkl file")
    parser.add_argument("--world_size", type=int, default=4, help="Number of GPUs/ranks")
    
    args = parser.parse_args()
    preshard_cache(args.cache, args.world_size)
