"""
scripts/build_kb_index.py

Chunks the scraped raw .txt files in knowledge/raw/ and embeds each chunk
with sentence-transformers/all-MiniLM-L6-v2 (~80 MB, runs on CPU). Saves:

    knowledge/processed/chunks.json     # [{source, chunk_id, text}, ...]
    knowledge/processed/embeddings.npy  # float32, shape (n_chunks, 384)

Run after scrape_5g_kb.py. Idempotent — overwrites previous index.

Usage:
    pip install sentence-transformers numpy
    unset HF_HUB_OFFLINE          # so the embedding model can download
    python scripts/build_kb_index.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# If HF_HUB_OFFLINE was set by an earlier run_all.sh, the embedding model
# can't download. Clear it for this script.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.stderr.write(
        "Missing deps. Install with:\n"
        "  pip install sentence-transformers numpy\n"
    )
    sys.exit(1)


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
RAW_DIR = PROJECT_DIR / "knowledge" / "raw"
PROC_DIR = PROJECT_DIR / "knowledge" / "processed"

EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunk parameters: paragraphs > 100 chars, hard-split anything > 1500 chars at 1200.
MIN_PARA = 100
MAX_CHUNK = 1500
SPLIT_AT = 1200


def chunk_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    head, _, body = text.partition("\n\n")
    source = head.replace("SOURCE: ", "").strip() if head.startswith("SOURCE:") else str(path.name)
    if not body:
        body = text  # no header

    chunks: list[dict] = []
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if len(p.strip()) > MIN_PARA]
    for i, p in enumerate(paras):
        if len(p) <= MAX_CHUNK:
            chunks.append({
                "source": source,
                "chunk_id": f"{path.stem}_{i}",
                "text": p,
            })
        else:
            for j in range(0, len(p), SPLIT_AT):
                chunks.append({
                    "source": source,
                    "chunk_id": f"{path.stem}_{i}_{j}",
                    "text": p[j:j + MAX_CHUNK],
                })
    return chunks


def main() -> int:
    if not RAW_DIR.exists():
        print(f"FATAL: {RAW_DIR} not found. Run scripts/scrape_5g_kb.py first.",
              file=sys.stderr)
        return 1
    raw_files = sorted(RAW_DIR.glob("doc_*.txt"))
    if not raw_files:
        print(f"FATAL: no doc_*.txt in {RAW_DIR}", file=sys.stderr)
        return 1

    print(f"[index] reading {len(raw_files)} raw files from {RAW_DIR}")
    all_chunks: list[dict] = []
    for f in raw_files:
        cs = chunk_file(f)
        all_chunks.extend(cs)
        print(f"  {f.name:>14s}  ->  {len(cs):>3d} chunks")
    print(f"[index] total chunks: {len(all_chunks)}")

    if not all_chunks:
        print("FATAL: produced 0 chunks; check raw text contents", file=sys.stderr)
        return 1

    print(f"[index] loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in all_chunks]
    print(f"[index] embedding {len(texts)} chunks (CPU is fine, ~30-90s)")
    embs = model.encode(texts, show_progress_bar=True, batch_size=32, convert_to_numpy=True)
    embs = embs.astype("float32")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROC_DIR / "embeddings.npy", embs)
    (PROC_DIR / "chunks.json").write_text(
        json.dumps(all_chunks, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[index] saved:")
    print(f"  {PROC_DIR / 'chunks.json'}     ({len(all_chunks)} chunks)")
    print(f"  {PROC_DIR / 'embeddings.npy'}  ({embs.shape})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
