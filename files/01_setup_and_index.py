#!/usr/bin/env python3
"""
PATH C - PHASE 1: SETUP, INDEX, AND RETRIEVALS

What this does:
1. Loads + analyzes training data (length stats per subset)
2. Embeds all 29,815 training questions with BGE-M3 (multilingual)
3. Builds a FAISS index
4. Precomputes top-K retrievals for every training sample (leave-one-out)
   and every val sample (retrieve from training set)
5. Saves everything for later use

Time: ~45-60 minutes
Cost: ~$0.50

Outputs (all in /workspace/path_c/):
  data_analysis.json
  train_embeddings.npy
  faiss_index.bin
  train_metadata.parquet
  train_retrievals.json   (top-3 per train sample, excluding self)
  val_retrievals.json     (top-3 per val sample)
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

HF_TOKEN = "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"
os.environ["HF_TOKEN"] = HF_TOKEN

import torch
from huggingface_hub import login

EMBEDDING_MODEL_ID = "BAAI/bge-m3"
WORKSPACE = os.environ.get("WORKSPACE", os.path.expanduser("~/zindi"))
OUT_DIR = f"{WORKSPACE}/path_c"
TOP_K = 5  # save top-5, can use top-2 or top-3 as needed later


def load_csv(path, has_output=True):
    df = pd.read_csv(path)
    cols = ['input', 'subset']
    if has_output:
        cols.append('output')
    df = df.dropna(subset=cols).reset_index(drop=True)
    return df


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    print("="*70)
    print("PATH C - PHASE 1: SETUP AND INDEX")
    print("="*70)
    login(token=HF_TOKEN, add_to_git_credential=False)

    # ====== Load data ======
    print("\n[1/5] Loading data...")
    train_df = load_csv(f"{WORKSPACE}/Train.csv", has_output=True)
    val_df = load_csv(f"{WORKSPACE}/Val.csv", has_output=True)
    test_df = load_csv(f"{WORKSPACE}/Test.csv", has_output=False)
    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")

    # Save train metadata for fast lookup later
    train_df_meta = train_df[['input', 'output', 'subset']].copy()
    train_df_meta.to_parquet(f"{OUT_DIR}/train_metadata.parquet")
    print(f"  Saved train_metadata.parquet")

    # ====== Length analysis ======
    print("\n[2/5] Output length analysis by subset...")
    train_df['out_len'] = train_df['output'].astype(str).str.len()
    print(f"{'Subset':<15} {'count':<8} {'mean':<10} {'median':<10} {'p90':<10} {'max':<10}")
    analysis = {}
    for subset in sorted(train_df['subset'].unique()):
        sub_df = train_df[train_df['subset'] == subset]
        lens = sub_df['out_len']
        stats = {
            'count': int(len(sub_df)),
            'mean_chars': float(lens.mean()),
            'median_chars': float(lens.median()),
            'p90_chars': float(lens.quantile(0.90)),
            'max_chars': int(lens.max()),
        }
        analysis[subset] = stats
        print(f"{subset:<15} {stats['count']:<8} {stats['mean_chars']:<10.0f} "
              f"{stats['median_chars']:<10.0f} {stats['p90_chars']:<10.0f} "
              f"{stats['max_chars']:<10}")

    # Recommended max_new_tokens per subset
    token_multiplier = {
        'Eng': 0.30, 'Aka': 0.60, 'Amh': 0.80, 'Lug': 0.50, 'Swa': 0.40,
    }
    recommended_tokens = {}
    for subset in sorted(train_df['subset'].unique()):
        lang_code = subset.split('_')[0]
        p90_chars = analysis[subset]['p90_chars']
        mult = token_multiplier.get(lang_code, 0.5)
        rec = int(p90_chars * mult * 1.2)
        rec = ((rec + 31) // 32) * 32
        rec = max(96, min(384, rec))
        recommended_tokens[subset] = rec
    analysis['recommended_tokens'] = recommended_tokens
    analysis['overall_mean_chars'] = float(train_df['out_len'].mean())
    analysis['overall_median_chars'] = float(train_df['out_len'].median())

    with open(f"{OUT_DIR}/data_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  Saved data_analysis.json")

    # ====== Load embedding model ======
    print(f"\n[3/5] Loading {EMBEDDING_MODEL_ID}...")
    from sentence_transformers import SentenceTransformer
    import faiss
    embedder = SentenceTransformer(EMBEDDING_MODEL_ID, device='cuda')
    print(f"  Loaded. Embedding dim: {embedder.get_sentence_embedding_dimension()}")

    # ====== Embed all train questions ======
    print(f"\n[4/5] Embedding {len(train_df):,} train questions (~10-15 min)...")
    train_embeddings = embedder.encode(
        train_df['input'].tolist(),
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"  Embeddings shape: {train_embeddings.shape}")
    np.save(f"{OUT_DIR}/train_embeddings.npy", train_embeddings.astype('float32'))
    print(f"  Saved train_embeddings.npy ({train_embeddings.nbytes / 1e6:.1f} MB)")

    # Build FAISS index
    print("  Building FAISS index...")
    index = faiss.IndexFlatIP(train_embeddings.shape[1])
    index.add(train_embeddings.astype('float32'))
    faiss.write_index(index, f"{OUT_DIR}/faiss_index.bin")
    print(f"  Saved faiss_index.bin ({index.ntotal} vectors)")

    # ====== Compute retrievals ======
    print(f"\n[5/5] Computing retrievals (top-{TOP_K})...")
    print("  Train leave-one-out retrievals...")
    # Search for top K+1 (since top-1 is always self)
    D_train, I_train = index.search(train_embeddings.astype('float32'), TOP_K + 1)
    train_retrievals = {}
    for i in range(len(train_df)):
        retrieved_idxs = [int(idx) for idx in I_train[i] if int(idx) != i][:TOP_K]
        train_retrievals[i] = retrieved_idxs
    with open(f"{OUT_DIR}/train_retrievals.json", "w") as f:
        json.dump(train_retrievals, f)
    print(f"  Saved train_retrievals.json")

    print("  Val retrievals (from train index)...")
    val_embeddings = embedder.encode(
        val_df['input'].tolist(),
        batch_size=32, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    D_val, I_val = index.search(val_embeddings.astype('float32'), TOP_K)
    val_retrievals = {}
    for i in range(len(val_df)):
        val_retrievals[i] = [int(idx) for idx in I_val[i]]
    with open(f"{OUT_DIR}/val_retrievals.json", "w") as f:
        json.dump(val_retrievals, f)
    np.save(f"{OUT_DIR}/val_embeddings.npy", val_embeddings.astype('float32'))
    print(f"  Saved val_retrievals.json + val_embeddings.npy")

    print("  Test retrievals (from train index)...")
    test_embeddings = embedder.encode(
        test_df['input'].tolist(),
        batch_size=32, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    D_test, I_test = index.search(test_embeddings.astype('float32'), TOP_K)
    test_retrievals = {}
    for i in range(len(test_df)):
        test_retrievals[i] = [int(idx) for idx in I_test[i]]
    with open(f"{OUT_DIR}/test_retrievals.json", "w") as f:
        json.dump(test_retrievals, f)
    np.save(f"{OUT_DIR}/test_embeddings.npy", test_embeddings.astype('float32'))
    print(f"  Saved test_retrievals.json + test_embeddings.npy")

    # ====== Verify retrieval quality on a sample ======
    print(f"\n[Verification] Sample retrieval check:")
    sample_idx = 100
    sample = val_df.iloc[sample_idx]
    print(f"  Val Q: {sample['input'][:120]}")
    print(f"  Val A: {str(sample['output'])[:120]}")
    print(f"  Top-3 retrieved from training:")
    for j, idx in enumerate(val_retrievals[sample_idx][:3]):
        retrieved = train_df.iloc[idx]
        sim = float(D_val[sample_idx][j])
        print(f"    {j+1}. (sim={sim:.3f}) Q: {retrieved['input'][:80]}")
        print(f"       A: {str(retrieved['output'])[:80]}")

    print("\n" + "="*70)
    print("PHASE 1 COMPLETE")
    print("="*70)
    print(f"\nNext step: python3 02_train.py")
    print(f"\nFiles in {OUT_DIR}:")
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(f"{OUT_DIR}/{f}") / 1e6
        print(f"  {f} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
