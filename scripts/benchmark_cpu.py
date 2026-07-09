"""CPU inference benchmark harness for the quantized SSG-VQA-Net v2.

Objective (v) of the thesis: characterize latency, throughput, memory, and
quality delta of each quantized variant on CPU-only hardware.

Given a directory of variants (produced by ``scripts/quantize_and_export.py``),
this script measures:

  * end-to-end latency (median / p95 / p99),
  * time-to-first-token (median / p95),
  * per-component latency (SG generator / LM completion / refinement),
  * throughput (queries per minute),
  * peak resident-set memory (via psutil sampled every 100 ms),
  * Val Accuracy on a held-out subset,
  * Val Grounding IoU on the same subset.

Emits one row per variant to ``benchmark_manifest.json`` for the paper's
tables to be regenerated deterministically.

Usage:
    # Full benchmark of all variants against 500 test questions after 100 warmups,
    # with quality eval on 200 studies (fast; use --n_quality 1000 for the paper).
    python scripts/benchmark_cpu.py \\
        --quantized_root ./quantized_models \\
        --qba_root data/mimic-ext-cxr-qba \\
        --jpg_root data/mimic-cxr-jpg \\
        --n_warmup 100 --n_measure 500 --n_quality 200 \\
        --output benchmark_manifest.json

    # Single variant only
    python scripts/benchmark_cpu.py --variant q4_k_m ...

The harness pins the environment to a fixed thread count so numbers are
reproducible. See scripts/README_quantization.md for the exact test-bed
specification recorded in the paper.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ==========================================================================
# Memory sampler (runs in background thread, records peak RSS)
# ==========================================================================
class MemorySampler:
    """Samples the current process's RSS every ``interval_ms`` in a
    daemon thread. Peak MiB is available via ``self.peak_mib``.
    """
    def __init__(self, interval_ms: int = 100) -> None:
        self.interval_ms = interval_ms
        self.peak_mib = 0.0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def _run(self) -> None:
        try:
            import psutil
        except ImportError:
            logger.warning("psutil not installed; memory measurements will be 0.")
            return
        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss / (1024 ** 2)  # MiB
                if rss > self.peak_mib:
                    self.peak_mib = rss
            except Exception:
                pass
            self._stop.wait(self.interval_ms / 1000.0)

    def start(self) -> "MemorySampler":
        self.peak_mib = 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.peak_mib


# ==========================================================================
# Dataset sampling (uses the same walker as extract_dataset_samples.py)
# ==========================================================================
def _pick_test_studies(qba_root: Path, jpg_root: Path, n: int,
                       quality: Optional[str] = "A") -> List[Dict[str, Any]]:
    """Return the first N QBA studies (each with image_path, question,
    expected answer, and GT bbox where available)."""
    import json as _json
    from PIL import Image  # noqa: F401

    q_root = qba_root / "qa"
    if not q_root.exists():
        raise FileNotFoundError(f"qa root not found: {q_root}")

    def _find_image(subject_id: int, study_id: int) -> Optional[Path]:
        p_group = f"p{str(subject_id)[:2]}"
        study_dir = jpg_root / "files" / p_group / f"p{subject_id}" / f"s{study_id}"
        if not study_dir.exists():
            return None
        imgs = sorted(study_dir.glob("*.jpg"))
        return imgs[0] if imgs else None

    _GRADE_ORDER = {"A++": 5, "A+": 4, "A": 3, "B": 1, "C": 0, "D": -1}
    _INT_MAP = {
        'region_extract_quality': ['B', 'B', 'A', 'A', 'A++'],
        'entity_extract_quality': ['B', 'A', 'A++'],
        'localization_quality':   ['B', 'B', 'A', 'A++', 'A++'],
    }

    def _grade(q: Dict[str, Any]) -> Optional[str]:
        eq = q.get("extraction_quality") or {}
        if not isinstance(eq, dict):
            return None
        grades = []
        for k, v in eq.items():
            if isinstance(v, int) and k in _INT_MAP:
                enum = _INT_MAP[k]
                if 0 <= v < len(enum):
                    grades.append(enum[v])
            elif isinstance(v, str) and v in _GRADE_ORDER:
                grades.append(v)
        return (min(grades, key=lambda g: _GRADE_ORDER.get(g, 0))
                if grades else None)

    out = []
    for p_group in sorted(q_root.iterdir()):
        if not (p_group.is_dir() and p_group.name.startswith("p")):
            continue
        for patient_dir in sorted(p_group.iterdir()):
            if not patient_dir.is_dir():
                continue
            try:
                subject_id = int(patient_dir.name[1:])
            except ValueError:
                continue
            for qa_file in sorted(patient_dir.glob("s*.qa.json")):
                try:
                    study_id = int(qa_file.stem.split(".")[0][1:])
                except Exception:
                    continue
                img = _find_image(subject_id, study_id)
                if img is None:
                    continue
                try:
                    with open(qa_file) as f:
                        qa = _json.load(f)
                except Exception:
                    continue
                for q in qa.get("questions", []):
                    if quality:
                        g = _grade(q) or "B"
                        if _GRADE_ORDER.get(g, 0) < _GRADE_ORDER.get(quality, 0):
                            continue
                    ans = q.get("answers", [])
                    ans_text = next((a.get("text", "") for a in ans
                                     if a.get("answer_type", "main_answer") == "main_answer"),
                                    "")
                    # Extract first GT bbox if the answer carries one.
                    bbox = None
                    for a in ans:
                        loc = a.get("localization")
                        if isinstance(loc, dict) and loc:
                            b = next(iter(loc.values()), None)
                            if isinstance(b, dict):
                                bbs = b.get("bboxes") or []
                                if bbs and len(bbs[0]) == 4:
                                    bbox = list(bbs[0])
                                    break
                    out.append({
                        "subject_id": subject_id,
                        "study_id": study_id,
                        "image_path": img,
                        "question": q.get("question", ""),
                        "question_type": q.get("question_type", "?"),
                        "gt_answer": ans_text,
                        "gt_bbox_raw": bbox,
                    })
                    if len(out) >= n:
                        return out
    return out


# ==========================================================================
# Quality metrics
# ==========================================================================
def _iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _answer_matches(pred: str, gt: str) -> bool:
    """Cheap exact-match after normalisation. For binary questions matches
    "yes"/"no" prefixes; for open-ended questions matches any substring of
    length >= 4 characters in common."""
    if not pred or not gt:
        return False
    p = pred.strip().lower()
    g = gt.strip().lower()
    if p == g:
        return True
    # Binary yes/no fast path
    for token in ("yes", "no"):
        if p.startswith(token) and g.startswith(token):
            return True
    # Longer overlap: any 4-char substring of gt in pred
    if len(g) >= 4:
        for i in range(len(g) - 3):
            if g[i : i + 4] in p:
                return True
    return False


# ==========================================================================
# Benchmark one variant
# ==========================================================================
@dataclass
class VariantResult:
    variant: str
    variant_dir: str
    disk_bytes: int
    disk_human: str

    # Latency (all in milliseconds)
    latency_median_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    ttft_median_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    sg_gen_median_ms: float = 0.0
    answer_median_ms: float = 0.0
    refine_median_ms: float = 0.0

    # Throughput and memory
    throughput_qpm: float = 0.0
    peak_rss_mib: float = 0.0
    peak_rss_gb: float = 0.0

    # Quality
    val_acc: Optional[float] = None
    val_grd_iou: Optional[float] = None
    n_quality: int = 0

    # Setup metadata
    n_warmup: int = 0
    n_measure: int = 0
    n_threads: int = 0
    load_time_s: float = 0.0
    ok: bool = False
    error: Optional[str] = None


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024.0:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.2f} PB"


def benchmark_variant(variant_dir: Path, studies: List[Dict[str, Any]],
                      n_warmup: int, n_measure: int, n_quality: int,
                      num_threads: int) -> VariantResult:
    """Run the full benchmark loop for one variant."""
    from PIL import Image

    variant = variant_dir.name
    result = VariantResult(
        variant=variant,
        variant_dir=str(variant_dir),
        disk_bytes=_dir_size_bytes(variant_dir),
        disk_human=_human_size(_dir_size_bytes(variant_dir)),
        n_warmup=n_warmup,
        n_measure=n_measure,
        n_threads=num_threads,
    )

    # -------- Load pipeline --------
    logger.info(f"[{variant}] loading pipeline...")
    t0 = time.perf_counter()
    try:
        from scripts.serve_app_cpu import CPUPipeline
        pipeline = CPUPipeline.from_variant_dir(variant_dir, num_threads=num_threads)
    except Exception as e:
        logger.error(f"[{variant}] failed to load: {e}", exc_info=True)
        result.error = str(e)
        return result
    result.load_time_s = time.perf_counter() - t0
    logger.info(f"[{variant}] loaded in {result.load_time_s:.1f} s")

    # -------- Warm-up --------
    logger.info(f"[{variant}] warming up ({n_warmup} queries)...")
    for i, s in enumerate(studies[:n_warmup]):
        try:
            img = Image.open(s["image_path"]).convert("RGB")
            _ = pipeline(img, s["question"], max_new_tokens=64)
            if (i + 1) % 25 == 0:
                logger.info(f"  warmup {i+1}/{n_warmup}")
        except Exception as e:
            logger.warning(f"  warmup {i} raised: {e}")

    # -------- Measure --------
    logger.info(f"[{variant}] measuring ({n_measure} queries)...")
    lat_total, lat_ttft = [], []
    lat_sg, lat_ans, lat_ref = [], [], []
    sampler = MemorySampler(interval_ms=100).start()
    t_window = time.perf_counter()
    err_count = 0
    for i, s in enumerate(studies[n_warmup : n_warmup + n_measure]):
        try:
            img = Image.open(s["image_path"]).convert("RGB")
            t = time.perf_counter()
            out = pipeline(img, s["question"], max_new_tokens=64)
            total_ms = 1000.0 * (time.perf_counter() - t)
            timing = out.get("timing", {})
            lat_total.append(total_ms)
            if "ttft_ms" in timing:
                lat_ttft.append(timing["ttft_ms"])
            if "sg_gen_ms" in timing:
                lat_sg.append(timing["sg_gen_ms"])
            if "answer_ms" in timing:
                lat_ans.append(timing["answer_ms"])
            if "refine_ms" in timing:
                lat_ref.append(timing["refine_ms"])
            if (i + 1) % 50 == 0:
                logger.info(f"  measure {i+1}/{n_measure}  median so far: {statistics.median(lat_total):.1f} ms")
        except Exception as e:
            err_count += 1
            logger.warning(f"  measure {i} raised: {e}")
    total_window_s = time.perf_counter() - t_window
    peak_mib = sampler.stop()
    if err_count:
        logger.warning(f"[{variant}] {err_count}/{n_measure} queries raised errors")

    if not lat_total:
        result.error = "no successful queries during measurement"
        del pipeline
        gc.collect()
        return result

    lat_total.sort()
    result.latency_median_ms = statistics.median(lat_total)
    result.latency_p95_ms = lat_total[int(0.95 * len(lat_total)) - 1]
    result.latency_p99_ms = lat_total[int(0.99 * len(lat_total)) - 1]
    result.latency_min_ms = lat_total[0]
    result.latency_max_ms = lat_total[-1]
    if lat_ttft:
        lat_ttft.sort()
        result.ttft_median_ms = statistics.median(lat_ttft)
        result.ttft_p95_ms = lat_ttft[int(0.95 * len(lat_ttft)) - 1]
    result.sg_gen_median_ms = statistics.median(lat_sg) if lat_sg else 0.0
    result.answer_median_ms = statistics.median(lat_ans) if lat_ans else 0.0
    result.refine_median_ms = statistics.median(lat_ref) if lat_ref else 0.0
    result.throughput_qpm = 60_000.0 / statistics.mean(lat_total)
    result.peak_rss_mib = peak_mib
    result.peak_rss_gb = peak_mib / 1024.0

    # -------- Quality --------
    if n_quality > 0:
        logger.info(f"[{variant}] quality eval on {n_quality} studies...")
        correct = 0
        ious: List[float] = []
        n_done = 0
        for s in studies[n_warmup + n_measure : n_warmup + n_measure + n_quality]:
            try:
                img = Image.open(s["image_path"]).convert("RGB")
                out = pipeline(img, s["question"], max_new_tokens=128)
                n_done += 1
                if _answer_matches(out.get("answer", ""), s.get("gt_answer", "")):
                    correct += 1
                bb = out.get("bbox_refined")
                gt = s.get("gt_bbox_raw")
                if bb and gt and len(bb) == 4 and len(gt) == 4:
                    # Normalize GT to [0, 1] using image size
                    W, H = img.size
                    gt_n = [gt[0] / W, gt[1] / H, gt[2] / W, gt[3] / H]
                    if gt_n[2] > gt_n[0] and gt_n[3] > gt_n[1]:
                        ious.append(_iou_xyxy(bb, gt_n))
                if n_done % 50 == 0:
                    acc = correct / n_done
                    m_iou = statistics.mean(ious) if ious else 0.0
                    logger.info(f"  quality {n_done}/{n_quality}  acc={acc:.3f}  iou={m_iou:.3f}")
            except Exception as e:
                logger.warning(f"  quality query raised: {e}")
        result.n_quality = n_done
        result.val_acc = correct / max(n_done, 1)
        result.val_grd_iou = statistics.mean(ious) if ious else 0.0

    result.ok = True
    del pipeline
    gc.collect()
    return result


# ==========================================================================
# CLI
# ==========================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quantized_root", type=Path, required=True,
                   help="Root dir containing one subdir per variant.")
    p.add_argument("--variant", type=str, default=None,
                   help="If set, benchmark only this variant.")
    p.add_argument("--qba_root", type=Path, required=True)
    p.add_argument("--jpg_root", type=Path, required=True)
    p.add_argument("--quality_grade", type=str, default="A")
    p.add_argument("--n_warmup", type=int, default=100)
    p.add_argument("--n_measure", type=int, default=500)
    p.add_argument("--n_quality", type=int, default=200,
                   help="Number of studies for Val Acc / Grd IoU (in ADDITION to warmup+measure).")
    p.add_argument("--num_threads", type=int,
                   default=int(os.environ.get("OMP_NUM_THREADS", 8)))
    p.add_argument("--output", type=Path, default=Path("benchmark_manifest.json"))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Force CPU + pin threads for reproducibility.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_threads)

    # ---- Pick test studies once, shared across variants ----
    n_total = args.n_warmup + args.n_measure + args.n_quality
    logger.info(f"walking QBA for {n_total} A-grade studies...")
    studies = _pick_test_studies(args.qba_root, args.jpg_root, n_total,
                                  quality=args.quality_grade)
    if len(studies) < n_total:
        logger.warning(
            f"only found {len(studies)} studies; will use what's available "
            f"(needed {n_total})"
        )

    # ---- Which variants ----
    if args.variant:
        variant_dirs = [args.quantized_root / args.variant]
    else:
        variant_dirs = sorted(
            d for d in args.quantized_root.iterdir()
            if d.is_dir() and (d / "heads.safetensors").exists() or
            any(p.suffix == ".gguf" for p in d.iterdir() if p.is_file())
        )
    logger.info(f"benchmarking variants: {[d.name for d in variant_dirs]}")

    # ---- Benchmark each ----
    results: List[VariantResult] = []
    for vd in variant_dirs:
        if not vd.exists():
            logger.warning(f"skip: {vd} does not exist")
            continue
        logger.info("=" * 70)
        logger.info(f"VARIANT: {vd.name}")
        logger.info("=" * 70)
        r = benchmark_variant(
            vd, studies,
            n_warmup=args.n_warmup, n_measure=args.n_measure,
            n_quality=args.n_quality, num_threads=args.num_threads,
        )
        results.append(r)

    # ---- Emit manifest ----
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "test_bed": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "num_threads": args.num_threads,
            "n_warmup": args.n_warmup,
            "n_measure": args.n_measure,
            "n_quality": args.n_quality,
            "quality_grade": args.quality_grade,
        },
        "variants": [r.__dict__ for r in results],
    }
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"wrote {args.output}")

    # ---- Summary table ----
    print()
    print("=" * 118)
    print(f"{'variant':<12}  {'disk':>10}  {'load_s':>7}  "
          f"{'lat_med':>8}  {'p95':>7}  {'p99':>7}  {'ttft':>7}  "
          f"{'qpm':>6}  {'peak_gb':>8}  {'val_acc':>8}  {'grd_iou':>8}")
    print("-" * 118)
    for r in results:
        if not r.ok:
            print(f"{r.variant:<12}  {r.disk_human:>10}  ERROR: {r.error}")
            continue
        print(
            f"{r.variant:<12}  "
            f"{r.disk_human:>10}  "
            f"{r.load_time_s:>7.1f}  "
            f"{r.latency_median_ms:>8.1f}  "
            f"{r.latency_p95_ms:>7.1f}  "
            f"{r.latency_p99_ms:>7.1f}  "
            f"{r.ttft_median_ms:>7.1f}  "
            f"{r.throughput_qpm:>6.1f}  "
            f"{r.peak_rss_gb:>8.2f}  "
            f"{(r.val_acc or 0):>8.3f}  "
            f"{(r.val_grd_iou or 0):>8.3f}"
        )
    print("=" * 118)
    logger.info(f"summary written to {args.output}")


if __name__ == "__main__":
    main()
