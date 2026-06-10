#!/usr/bin/env python3
"""
scripts/explore_dataset_riches.py — extract everything the MIMIC-Ext-CXR-QBA
dataset publishers gave us that we're NOT currently using.

Six sections:
  1) verify the SceneGraphProcessor fix (old vs new on real samples)
  2) parquet metadata inventory (12 parquet files in metadata/)
  3) stats/ CSV gold (precomputed class frequencies — use for class weights!)
  4) exports/A_frontal vs B_frontal sizes (pre-curated quality subsets)
  5) splits from metadata.json (the canonical train/val/test)
  6) per-image dimensions distribution (needed for proper bbox normalisation)

Usage:
    .venv/bin/python scripts/explore_dataset_riches.py

Output: docs/dataset_riches.md  +  prints summary to stdout.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

QA_ROOT = _ROOT / "data" / "mimic-ext-cxr-qba"
SCENE_DIR = QA_ROOT / "scene_data"
METADATA_DIR = QA_ROOT / "metadata"
STATS_DIR = QA_ROOT / "stats"
EXPORTS_DIR = QA_ROOT / "exports"

OUT = []  # lines for the markdown report

def emit(s=""):
    print(s)
    OUT.append(s)


# ═══════════════════════════════════════════════════════════════════════
# Section 1 — verify the SceneGraphProcessor fix
# ═══════════════════════════════════════════════════════════════════════
def section1_verify_processor_fix():
    emit("# Dataset Riches Report\n")
    emit("## §1 — SceneGraphProcessor fix verification (old vs new)\n")

    from data.mimic_cxr_dataset import SceneGraphProcessor

    proc = SceneGraphProcessor()
    # Try to load vocab if it exists
    di = METADATA_DIR / "dataset_info.json"
    if di.exists():
        proc.load_vocab(str(di))

    sgs = sorted(SCENE_DIR.glob("p*/p*/s*.scene_graph.json"))[:5]
    if not sgs:
        emit("No scene_graph.json files found — skipping.\n")
        return

    for sg_path in sgs:
        emit(f"### {sg_path.name}")
        meta_path = sg_path.parent / sg_path.name.replace(".scene_graph.json", ".metadata.json")
        sg = json.loads(sg_path.read_text())
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        # Pick the first image to test against
        images = meta.get("images") or {}
        if not images:
            emit("  (no metadata.json — skipping)")
            continue
        image_id = next(iter(images))
        img_info = images[image_id]
        W, H = img_info.get("size", [1024, 1024])
        emit(f"- image_id: `{image_id}`")
        emit(f"- size: {W}×{H}  view: {img_info.get('view_position', '?')}")
        emit(f"- raw observations in scene_graph.json: **{len(sg.get('observations', {}))}**")

        # Run old processor
        proc.use_legacy_processor = True
        old = proc.process(sg, W, H, image_id=image_id)
        # Run new processor
        proc.use_legacy_processor = False
        new = proc.process(sg, W, H, image_id=image_id)

        emit(f"- **OLD processor** → {old['num_objects']} samples")
        emit(f"- **NEW processor** → {new['num_objects']} samples")

        # Show unique (entity_id, region_id) pairs each version produces
        old_pairs = set(zip(old["entity_ids"].tolist(), old["region_ids"].tolist()))
        new_pairs = set(zip(new["entity_ids"].tolist(), new["region_ids"].tolist()))
        emit(f"- OLD unique (entity, region) pairs: {len(old_pairs)}")
        emit(f"- NEW unique (entity, region) pairs: {len(new_pairs)}")

        # Show bbox-area distribution
        import numpy as np
        def areas(b): return ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).tolist()
        if old["num_objects"] > 0:
            oa = areas(old["bboxes"])
            emit(f"- OLD bbox-area stats: min={min(oa):.4f} max={max(oa):.4f} mean={sum(oa)/len(oa):.4f}")
        if new["num_objects"] > 0:
            na = areas(new["bboxes"])
            emit(f"- NEW bbox-area stats: min={min(na):.4f} max={max(na):.4f} mean={sum(na)/len(na):.4f}")
        emit("")


# ═══════════════════════════════════════════════════════════════════════
# Section 2 — parquet metadata inventory
# ═══════════════════════════════════════════════════════════════════════
def section2_parquet_inventory():
    emit("## §2 — Parquet metadata files (much faster than JSON loading)\n")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        emit("`pyarrow` not installed. Run: `pip install pyarrow`\n")
        return

    for pq_path in sorted(METADATA_DIR.glob("*.parquet")):
        size_mb = pq_path.stat().st_size / 1e6
        try:
            md = pq.read_metadata(str(pq_path))
            n_rows = md.num_rows
            schema = pq.read_schema(str(pq_path))
            cols = [f"{f.name}({f.type})" for f in schema][:15]
            emit(f"### `{pq_path.name}` ({size_mb:.1f} MB, **{n_rows:,} rows**)")
            emit(f"- columns ({len(schema)}): `{cols}{'...' if len(schema) > 15 else ''}`")
            # Read just 3 rows for a preview
            tbl = pq.read_table(str(pq_path), columns=[f.name for f in schema][:8])
            preview = tbl.slice(0, 2).to_pydict()
            emit(f"- sample row[0]: ```json\n  {{")
            for k in list(preview.keys())[:8]:
                v = preview[k][0]
                v_str = str(v)[:120]
                emit(f'    "{k}": {v_str!r},')
            emit("  }\n  ```")
        except Exception as e:
            emit(f"### `{pq_path.name}` — could not inspect: {e}")
        emit("")


# ═══════════════════════════════════════════════════════════════════════
# Section 3 — stats/ CSV gold (for class weights)
# ═══════════════════════════════════════════════════════════════════════
def section3_stats_csv():
    emit("## §3 — stats/ files (precomputed class frequencies for free)\n")
    try:
        import pandas as pd
    except ImportError:
        emit("pandas not installed. Run: `pip install pandas`\n")
        return

    # The SMALL CSVs (the big *.csv.gz files are huge — load on demand only)
    small_csvs = [p for p in sorted(STATS_DIR.glob("*.csv")) if p.stat().st_size < 5_000_000]
    for csv_path in small_csvs:
        emit(f"### `{csv_path.name}` ({csv_path.stat().st_size/1024:.1f} KB)")
        try:
            df = pd.read_csv(csv_path)
            emit(f"- shape: {df.shape}  columns: `{list(df.columns)}`")
            emit("- head:")
            emit("  ```")
            for line in df.head(5).to_string().splitlines():
                emit(f"  {line}")
            emit("  ```")
        except Exception as e:
            emit(f"  (load failed: {e})")
        emit("")

    # The BIG gzipped CSVs — just inventory, don't load
    emit("### Big gzipped CSVs (for class-frequency weights — load on demand):")
    for csv_path in sorted(STATS_DIR.glob("*.csv.gz")):
        size_mb = csv_path.stat().st_size / 1e6
        emit(f"- `{csv_path.name}` ({size_mb:.1f} MB) — `pd.read_csv(path, compression='gzip', nrows=5)` to preview")
    emit("")


# ═══════════════════════════════════════════════════════════════════════
# Section 4 — exports/ pre-curated subsets
# ═══════════════════════════════════════════════════════════════════════
def section4_exports():
    emit("## §4 — exports/A_frontal vs B_frontal (pre-curated subsets)\n")
    if not EXPORTS_DIR.exists():
        emit("No exports/ directory.\n")
        return

    for sub in sorted(EXPORTS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        sg_zip = sub / "scene_data.zip"
        meta_subdir = sub / "metadata"
        emit(f"### `exports/{sub.name}/`")
        if sg_zip.exists():
            sz_gb = sg_zip.stat().st_size / 1e9
            emit(f"- `scene_data.zip` ({sz_gb:.2f} GB) — extract or stream to use this subset")
        if meta_subdir.exists():
            parquets = sorted(meta_subdir.glob("*.parquet"))
            emit(f"- `metadata/` ({len(parquets)} parquet files):")
            for p in parquets:
                emit(f"  - `{p.name}` ({p.stat().st_size/1e6:.1f} MB)")
            dataset_info = meta_subdir / "dataset_info.json"
            if dataset_info.exists():
                try:
                    di = json.loads(dataset_info.read_text())
                    emit(f"- `metadata/dataset_info.json` keys: `{list(di.keys())}`")
                except Exception:
                    pass
        emit("")


# ═══════════════════════════════════════════════════════════════════════
# Section 5 — official splits from metadata.json (NOT hash-partitioning)
# ═══════════════════════════════════════════════════════════════════════
def section5_splits():
    emit("## §5 — Official splits from metadata.json (the canonical assignment)\n")
    split_counts: Counter = Counter()
    procedure_counts: Counter = Counter()
    view_counts: Counter = Counter()
    sample_paths_per_split = defaultdict(list)
    n_scanned = 0
    n_to_scan = 5000

    metas = list(SCENE_DIR.glob("p*/p*/s*.metadata.json"))
    emit(f"Found {len(metas):,} metadata.json files total. Sampling {n_to_scan}.\n")
    import random
    random.seed(42)
    random.shuffle(metas)

    for p in metas[:n_to_scan]:
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        n_scanned += 1
        split = m.get("split", "(unknown)")
        split_counts[split] += 1
        procedure_counts[m.get("procedure", "(none)")] += 1
        for img_info in (m.get("images") or {}).values():
            view_counts[img_info.get("view_position", "?")] += 1
        if len(sample_paths_per_split[split]) < 3:
            sample_paths_per_split[split].append(str(p.relative_to(QA_ROOT)))

    emit(f"### Split distribution (n={n_scanned})")
    for split, count in split_counts.most_common():
        pct = 100 * count / max(1, n_scanned)
        emit(f"- **{split}**: {count:,} ({pct:.1f}%)  e.g. {sample_paths_per_split[split][:2]}")
    emit("")
    emit("### Procedure distribution (top 10)")
    for proc, count in procedure_counts.most_common(10):
        emit(f"- {proc}: {count:,}")
    emit("")
    emit("### View position distribution")
    for view, count in view_counts.most_common():
        emit(f"- {view}: {count:,}")
    emit("")


# ═══════════════════════════════════════════════════════════════════════
# Section 6 — per-image dimensions (for proper bbox normalisation)
# ═══════════════════════════════════════════════════════════════════════
def section6_image_dims():
    emit("## §6 — Per-image dimensions distribution (proper bbox normalisation)\n")
    widths, heights = [], []
    images_per_study: Counter = Counter()
    metas = list(SCENE_DIR.glob("p*/p*/s*.metadata.json"))
    import random
    random.seed(43)
    random.shuffle(metas)
    for p in metas[:5000]:
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        imgs = m.get("images") or {}
        images_per_study[len(imgs)] += 1
        for img_info in imgs.values():
            size = img_info.get("size")
            if isinstance(size, list) and len(size) == 2:
                widths.append(size[0])
                heights.append(size[1])

    if widths:
        import numpy as np
        w, h = np.array(widths), np.array(heights)
        emit(f"### Image dimensions (n={len(widths):,})")
        emit(f"- width:  min={w.min()} max={w.max()} mean={w.mean():.0f} median={np.median(w):.0f}")
        emit(f"- height: min={h.min()} max={h.max()} mean={h.mean():.0f} median={np.median(h):.0f}")
        emit("- **Implication**: bboxes in qa.json are in PIXELS — must normalize "
             "by these per-image sizes, NOT a single global value.")
        emit("")
    emit("### Images per study")
    for n_imgs, count in sorted(images_per_study.items()):
        emit(f"- {n_imgs} image(s): {count:,} studies")
    emit("")


def main():
    section1_verify_processor_fix()
    section2_parquet_inventory()
    section3_stats_csv()
    section4_exports()
    section5_splits()
    section6_image_dims()

    out_path = _ROOT / "docs" / "dataset_riches.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n✓ Wrote {out_path}")


if __name__ == "__main__":
    main()
