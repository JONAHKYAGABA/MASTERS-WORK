"""Dedupe an existing sample cache to one sample per unique image.

Use when you have a .pkl built without --one_question_per_image and want to
verify the dedup effect (or use the result via --prebuilt_cache_train) without
running the full multi-hour cache rebuild.

NOTE: pickle.load is used here on the user's own .cache/dataset_samples files,
which are written by this repo's training pipeline. They are not external/
untrusted inputs. Do not point this script at .pkl files from other sources.

Usage:
    python scripts/dedupe_cache_by_image.py \
        --input  .cache/dataset_samples/samples_train_41aeedb1d6f64ca5.pkl \
        --output .cache/dataset_samples/samples_train_dedup.pkl

    # Optional: cap to a max number of unique images
    python scripts/dedupe_cache_by_image.py \
        --input  .cache/dataset_samples/samples_train_41aeedb1d6f64ca5.pkl \
        --output .cache/dataset_samples/samples_train_dedup_50k.pkl \
        --max_images 50000
"""
import argparse
import os
import pickle
import sys
from collections import Counter


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input cache .pkl")
    p.add_argument("--output", required=True, help="Output deduped cache .pkl")
    p.add_argument(
        "--max_images", type=int, default=None,
        help="Optional cap on unique images to keep (after dedup)."
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Report stats only; do not write output."
    )
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1

    in_size_mb = os.path.getsize(args.input) / 1e6
    print(f"Loading: {args.input}  ({in_size_mb:.1f} MB)")
    with open(args.input, "rb") as f:
        samples = pickle.load(f)
    print(f"  Loaded {len(samples):,} samples")

    # Pre-dedup stats
    def _key(s):
        return s.get("image_path") or s.get("dicom_id") or s.get("study_id")

    pre_images = {_key(s) for s in samples if _key(s) is not None}
    pre_studies = {s.get("study_id") for s in samples if s.get("study_id") is not None}
    pre_patients = {s.get("subject_id") for s in samples if s.get("subject_id") is not None}

    print("\n--- Before dedup ---")
    print(f"  samples : {len(samples):,}")
    print(f"  images  : {len(pre_images):,}")
    print(f"  studies : {len(pre_studies):,}")
    print(f"  patients: {len(pre_patients):,}")
    print(f"  questions/image: {len(samples)/max(1, len(pre_images)):.1f}")

    # Dedupe — first occurrence per image_path wins
    seen = set()
    deduped = []
    for s in samples:
        k = _key(s)
        if k is None or k in seen:
            continue
        seen.add(k)
        deduped.append(s)

    if args.max_images and len(deduped) > args.max_images:
        deduped = deduped[: args.max_images]

    post_studies = {s.get("study_id") for s in deduped}
    post_patients = {s.get("subject_id") for s in deduped}
    qtype_counts = Counter(s.get("question_type") for s in deduped)

    print("\n--- After dedup ---")
    print(f"  samples : {len(deduped):,}  ({len(deduped)/max(1,len(samples)):.1%} of input)")
    print(f"  images  : {len(deduped):,}  (== samples by construction)")
    print(f"  studies : {len(post_studies):,}")
    print(f"  patients: {len(post_patients):,}")
    print("\nTop question types in deduped cache:")
    for qt, c in qtype_counts.most_common(10):
        print(f"  {c:>7,d}  {qt}")

    if args.dry_run:
        print("\n(dry-run; nothing written)")
        return 0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print(f"\nWriting: {args.output}")
    with open(args.output, "wb") as f:
        pickle.dump(deduped, f, protocol=pickle.HIGHEST_PROTOCOL)
    out_size_mb = os.path.getsize(args.output) / 1e6
    print(f"  Wrote {len(deduped):,} samples  ({out_size_mb:.1f} MB)")
    print("\nUse with:")
    print(f"  --prebuilt_cache_train {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
