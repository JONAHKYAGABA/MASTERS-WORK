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
from pathlib import Path


def preshard_cache(cache_path: str, world_size: int = 4):
    """Split cache into per-rank shards."""
    p = Path(cache_path)
    
    if not p.exists():
        print(f"ERROR: Cache file not found: {p}")
        return False
    
    print(f"Loading cache: {p}")
    print(f"File size: {p.stat().st_size / 1e9:.2f} GB")
    
    with open(p, 'rb') as f:
        samples = pickle.load(f)
    
    total = len(samples)
    print(f"Total samples: {total:,}")
    print(f"Creating {world_size} shards...")
    
    for rank in range(world_size):
        shard = samples[rank::world_size]
        shard_path = p.parent / f"{p.stem}.shard{rank}of{world_size}.pkl"
        
        print(f"  Shard {rank}: {len(shard):,} samples -> {shard_path.name}")
        
        with open(shard_path, 'wb') as f:
            pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Show shard size
        shard_size = shard_path.stat().st_size / 1e9
        print(f"    Size: {shard_size:.2f} GB")
    
    # Cleanup
    del samples
    gc.collect()
    
    print(f"\n✓ Created {world_size} shard files")
    print(f"  Each rank will now load ~{total//world_size:,} samples ({100//world_size}% of data)")
    print(f"\nRun DeepSpeed training normally - shards will be auto-detected!")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-shard cache for distributed training")
    parser.add_argument("--cache", type=str, required=True, help="Path to cache .pkl file")
    parser.add_argument("--world_size", type=int, default=4, help="Number of GPUs/ranks")
    
    args = parser.parse_args()
    preshard_cache(args.cache, args.world_size)
