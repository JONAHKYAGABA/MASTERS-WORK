#!/usr/bin/env python3
"""
scripts/test_real_data.py — verify SSGVQANetV2 on real QBA + MIMIC-CXR-JPG samples.

Loads N real (image, question, scene graph) triplets from
  - data/mimic-ext-cxr-qba/exports/A_frontal/metadata/q1M/   (question metadata)
  - data/mimic-ext-cxr-qba/scene_data/                       (per-study scene graphs)
  - data/mimic-cxr-jpg/                                       (the actual JPG images)

Then runs:
  1. Forward + backward + optimizer step (training path)
  2. Generation (inference path)
  3. Prints loss, gt-bbox vs predicted-bbox, generated answer per sample

Usage:
  python scripts/test_real_data.py                   # 3 samples, GPU 0
  python scripts/test_real_data.py --n 5 --gpu 1
  python scripts/test_real_data.py --model_id Qwen/Qwen2.5-VL-7B-Instruct
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def load_paths() -> Dict[str, str]:
    """Read configs/paths.yaml written by setup_marconi.sh."""
    import yaml
    with open(_ROOT / "configs" / "paths.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg["data"]


def _read_metadata_table(path_no_ext: Path):
    """Read either .parquet or .csv.gz, whichever exists. Always reset_index
    so index columns (patient_id, study_id, ...) become regular columns."""
    import pandas as pd
    parquet = path_no_ext.with_suffix(".parquet")
    if parquet.exists():
        df = pd.read_parquet(parquet)
    else:
        csv = path_no_ext.with_suffix(".csv.gz")
        if not csv.exists():
            raise FileNotFoundError(
                f"Neither {parquet} nor {csv} found. Did the QBA download finish?"
            )
        df = pd.read_csv(csv)
    # Promote any index columns to regular columns so our column lookups work
    if df.index.name is not None or isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    return df


def _find_col(df, *candidates: str) -> str:
    """Return the first column name from `candidates` that exists in df.
    Handles QBA's mixed naming: bare ('patient_id'), prefixed
    ('question.patient_id'), or aliases ('subject_id')."""
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
        # Also try with common prefixes
        for prefix in ("question.", "image.", "answer.", "study."):
            if (prefix + c) in cols:
                return prefix + c
    raise KeyError(
        f"None of {candidates} (or prefixed variants) found in columns. "
        f"Available: {sorted(cols)[:20]}..."
    )


def _prefix(subject_id: int | str) -> str:
    """Subject-id directory prefix: 12345678 → 'p1', 'p12345678'."""
    s = str(subject_id)
    if not s.startswith("p"):
        s = f"p{s}"
    return s


def _study_prefix(study_id: int | str) -> str:
    s = str(study_id)
    if not s.startswith("s"):
        s = f"s{s}"
    return s


def load_scene_graph(qba_root: Path, subject: str, study: str) -> Optional[Dict[str, Any]]:
    """Load a single per-study scene graph JSON."""
    p1x = subject[:3]  # 'p1' + first digit → 'p10', 'p11', etc.
    sg_path = qba_root / "scene_data" / p1x / subject / f"{study}.scene_graph.json"
    if not sg_path.exists():
        return None
    with open(sg_path) as f:
        return json.load(f)


def load_qa(qba_root: Path, subject: str, study: str, question_id: str) -> Optional[Dict[str, Any]]:
    """Load a single QA pair from the per-study qa.json (zipped or unzipped)."""
    p1x = subject[:3]
    # Prefer unzipped
    unzipped = qba_root / "qa" / p1x / subject / f"{study}.qa.json"
    if unzipped.exists():
        with open(unzipped) as f:
            data = json.load(f)
    else:
        # Fall back to reading from qa.zip
        import zipfile
        zip_path = qba_root / "qa.zip"
        if not zip_path.exists():
            return None
        with zipfile.ZipFile(zip_path) as zf:
            inner = f"{p1x}/{subject}/{study}.qa.json"
            try:
                with zf.open(inner) as f:
                    data = json.load(f)
            except KeyError:
                return None

    for q in data.get("questions", []):
        if str(q.get("question_id")) == str(question_id):
            return q
    return None


# ---------------------------------------------------------------------------
# Scene graph → model input
# ---------------------------------------------------------------------------


_NAME_ID_CACHE: Dict[str, Dict[str, int]] = {"region": {}, "entity": {}}


def name_to_id(name: str, kind: str, mod: int) -> int:
    """Stable name → int id. Caches first-seen names in a per-kind dict so
    repeated names get the same id across samples in this run."""
    table = _NAME_ID_CACHE[kind]
    if name not in table:
        table[name] = (hash(name) & 0x7FFFFFFF) % mod
    return table[name]


def sg_to_model_dict(
    sg: Dict[str, Any],
    image_id: str,
    image_w: int,
    image_h: int,
    num_regions: int = 310,
    num_entities: int = 237,
) -> Dict[str, Any]:
    """Convert QBA per-study scene graph to per-image SceneGraphEncoderV2 input."""
    bboxes: List[List[float]] = []
    entity_ids: List[int] = []
    region_ids: List[int] = []
    positiveness: List[int] = []

    for obs_id, obs in sg.get("observations", {}).items():
        loc = obs.get("localization", {}).get(image_id, {})
        obs_bboxes = loc.get("bboxes", []) if isinstance(loc, dict) else []
        if not obs_bboxes:
            continue

        # First bbox per observation, normalize pixel coords → [0, 1]
        x1, y1, x2, y2 = obs_bboxes[0]
        bbox_norm = [
            max(0.0, min(1.0, float(x1) / max(image_w, 1))),
            max(0.0, min(1.0, float(y1) / max(image_h, 1))),
            max(0.0, min(1.0, float(x2) / max(image_w, 1))),
            max(0.0, min(1.0, float(y2) / max(image_h, 1))),
        ]
        # Sanity: x2 > x1, y2 > y1
        if bbox_norm[2] <= bbox_norm[0] or bbox_norm[3] <= bbox_norm[1]:
            continue

        region_names = obs.get("regions", [])
        region_name = (
            region_names[0]["region"] if region_names and isinstance(region_names[0], dict)
            else (region_names[0] if region_names else "unknown")
        )
        entities = obs.get("obs_entities", [])
        entity_name = entities[0] if entities else "unknown"
        pos = obs.get("positiveness", "unknown")
        pos_id = {"pos": 1, "neg": 0}.get(pos, 2)

        bboxes.append(bbox_norm)
        region_ids.append(name_to_id(str(region_name), "region", num_regions))
        entity_ids.append(name_to_id(str(entity_name), "entity", num_entities))
        positiveness.append(pos_id)

    import numpy as np
    return {
        "bboxes": np.asarray(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32),
        "entity_ids": np.asarray(entity_ids, dtype=np.int64),
        "region_ids": np.asarray(region_ids, dtype=np.int64),
        "positiveness": np.asarray(positiveness, dtype=np.int64),
        "num_objects": len(bboxes),
    }


def format_answer(q: Dict[str, Any], image_id: str, image_w: int, image_h: int) -> str:
    """Format QBA answer triplet into '<think>...</think><box>...</box><answer>...</answer>'."""
    main_answers = [a for a in q.get("answers", []) if a.get("answer_type") == "main_answer"]
    if not main_answers:
        return (
            "<think>No findings to report.</think>"
            "<box>0.000,0.000,1.000,1.000</box>"
            "<answer>The chest X-ray appears unremarkable.</answer>"
        )

    a = main_answers[0]
    text = a.get("text", "Unknown.")
    regions = ", ".join(str(r) for r in a.get("regions", []) if r)
    entities = ", ".join(str(e) for e in a.get("obs_entities", []) if e)

    # First valid bbox for this image, normalized
    bbox_str = "0.000,0.000,1.000,1.000"
    loc = a.get("localization", {}).get(image_id, {})
    if isinstance(loc, dict):
        bboxes = loc.get("bboxes", [])
        if bboxes:
            x1, y1, x2, y2 = bboxes[0]
            bbox_str = (
                f"{x1/max(image_w,1):.3f},{y1/max(image_h,1):.3f},"
                f"{x2/max(image_w,1):.3f},{y2/max(image_h,1):.3f}"
            )

    cot_parts = []
    if regions:
        cot_parts.append(f"Region(s) of interest: {regions}.")
    if entities:
        cot_parts.append(f"Observed: {entities}.")
    cot = " ".join(cot_parts) if cot_parts else "Reviewing the chest radiograph."
    return f"<think>{cot}</think><box>{bbox_str}</box><answer>{text}</answer>"


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------


def pick_real_samples(paths: Dict[str, str], n: int, verbose: bool = True) -> List[Dict[str, Any]]:
    """Walk QBA q1M metadata, return N triplets where everything exists on disk."""
    from PIL import Image

    qba_root = Path(paths["mimic_ext_cxr_qba_path"])
    jpg_root = Path(paths["mimic_cxr_jpg_path"])

    q1m_dir = qba_root / "exports" / "A_frontal" / "metadata" / "q1M"
    if not q1m_dir.exists():
        # Fall back to top-level metadata if q1M isn't present
        q1m_dir = qba_root / "metadata"

    q_meta = _read_metadata_table(q1m_dir / "question_metadata")
    qi_meta = _read_metadata_table(qba_root / "metadata" / "question_image_metadata")
    img_meta = _read_metadata_table(qba_root / "metadata" / "image_metadata")

    if verbose:
        print(f"q_meta rows : {len(q_meta):>10,}  cols: {list(q_meta.columns)[:6]}...")
        print(f"qi_meta rows: {len(qi_meta):>10,}  cols: {list(qi_meta.columns)[:6]}...")
        print(f"img_meta rows: {len(img_meta):>10,}  cols: {list(img_meta.columns)[:6]}...")

    # Resolve column names once (QBA uses patient_id, sometimes prefixed)
    q_pat  = _find_col(q_meta,  "patient_id", "subject_id")
    q_stud = _find_col(q_meta,  "study_id")
    q_qid  = _find_col(q_meta,  "question_id")
    qi_qid = _find_col(qi_meta, "question_id")
    qi_iid = _find_col(qi_meta, "image_id", "dicom_id")
    im_iid = _find_col(img_meta, "image_id", "dicom_id")
    # Optional image-dim columns (different QBA versions name them differently)
    try:
        im_w_col = _find_col(img_meta, "image_width", "size_x", "Columns", "cols")
    except KeyError:
        im_w_col = None
    try:
        im_h_col = _find_col(img_meta, "image_height", "size_y", "Rows", "rows")
    except KeyError:
        im_h_col = None

    if verbose:
        print(f"\n[cols] q_meta: pat={q_pat!r}  stud={q_stud!r}  qid={q_qid!r}")
        print(f"[cols] qi_meta: qid={qi_qid!r}  iid={qi_iid!r}")
        print(f"[cols] img_meta: iid={im_iid!r}  w={im_w_col!r}  h={im_h_col!r}\n")

    # ------------------------------------------------------------------
    # IMPORTANT: QBA's `question_id` is NOT unique — it's a template name
    # like 'B09_describe_abnormal_subcat_007' reused across every study.
    # The unique key for a question is the TRIPLE (patient_id, study_id,
    # question_id). Joining on question_id alone produced a 28M-row
    # cross-product. Build a composite key instead.
    # ------------------------------------------------------------------
    head_size = max(n * 50, 50)
    q_head = q_meta.head(head_size).reset_index(drop=True)
    print(f"[trim] candidate questions: {len(q_head)} "
          f"(oversampling {head_size} to find {n} valid)")

    qi_pat = _find_col(qi_meta, "patient_id", "subject_id")
    qi_stud = _find_col(qi_meta, "study_id")

    # Step 1: filter qi_meta by study_id only (cheap, removes ~99.9% of rows)
    study_ids = set(q_head[q_stud].astype(str).tolist())
    print(f"[trim] filtering qi_meta (70M rows) to {len(study_ids)} candidate studies ...")
    t = time.time()
    qi_by_study_mask = qi_meta[qi_stud].astype(str).isin(study_ids)
    qi_subset = qi_meta[qi_by_study_mask].copy()
    print(f"[trim] after study filter: {len(qi_subset):,} rows in {time.time()-t:.1f}s")

    # Step 2: build the unique (patient|study|question) composite key on both sides
    def _make_key(df, pat_col, stud_col, qid_col):
        return (df[pat_col].astype(str) + "|"
                + df[stud_col].astype(str) + "|"
                + df[qid_col].astype(str))

    q_head["_key"] = _make_key(q_head, q_pat, q_stud, q_qid)
    qi_subset["_key"] = _make_key(qi_subset, qi_pat, qi_stud, qi_qid)

    # Step 3: filter again on the composite key — now precise
    qi_small = qi_subset[qi_subset["_key"].isin(set(q_head["_key"]))].copy()
    print(f"[trim] after composite-key filter: {len(qi_small):,} rows")
    if len(qi_small) == 0:
        print("✗ no qi_meta rows matched the composite key — verify column names match across files")
        return []
    if len(qi_small) > len(q_head) * 20:
        print(f"⚠ {len(qi_small)/len(q_head):.0f}× qi rows per question — still suspect")

    qi_by_qid = qi_small.set_index("_key", drop=False)
    im_by_iid = img_meta.set_index(im_iid, drop=False)

    # Per-step skip counters + first-failure printouts for diagnosis
    skip_reasons: Dict[str, int] = {
        "qi_lookup": 0, "img_meta": 0, "img_file": 0,
        "scene_graph": 0, "qa": 0, "no_objects": 0,
    }
    first_examples: Dict[str, str] = {}

    samples: List[Dict[str, Any]] = []
    for idx, q_row in q_head.iterrows():
        if len(samples) >= n:
            break

        subject = _prefix(q_row[q_pat])
        study = _study_prefix(q_row[q_stud])
        question_id = q_row[q_qid]
        composite = q_row["_key"]

        try:
            qi_match = qi_by_qid.loc[[composite]]
        except (KeyError, TypeError):
            skip_reasons["qi_lookup"] += 1
            first_examples.setdefault("qi_lookup", f"key={composite!r}")
            continue
        if len(qi_match) == 0:
            skip_reasons["qi_lookup"] += 1
            continue
        image_id = qi_match.iloc[0][qi_iid]

        try:
            ir_row = im_by_iid.loc[image_id]
            if hasattr(ir_row, "iloc"):
                ir_row = ir_row.iloc[0]
        except (KeyError, TypeError):
            skip_reasons["img_meta"] += 1
            first_examples.setdefault("img_meta", f"image_id={image_id!r}")
            continue

        iw = int(ir_row[im_w_col]) if im_w_col else 0
        ih = int(ir_row[im_h_col]) if im_h_col else 0
        image_id = str(image_id)

        p1x = subject[:3]
        img_path = jpg_root / p1x / subject / study / f"{image_id}.jpg"
        if not img_path.exists():
            skip_reasons["img_file"] += 1
            first_examples.setdefault("img_file", str(img_path))
            continue

        if iw == 0 or ih == 0:
            with Image.open(img_path) as im:
                iw, ih = im.size

        sg = load_scene_graph(qba_root, subject, study)
        if sg is None:
            skip_reasons["scene_graph"] += 1
            sg_path = qba_root / "scene_data" / p1x / subject / f"{study}.scene_graph.json"
            first_examples.setdefault("scene_graph", str(sg_path))
            continue

        qa = load_qa(qba_root, subject, study, question_id)
        if qa is None:
            skip_reasons["qa"] += 1
            first_examples.setdefault("qa", f"qid={question_id!r} study={study}")
            continue

        sg_dict = sg_to_model_dict(sg, image_id, iw, ih)
        if sg_dict["num_objects"] == 0:
            skip_reasons["no_objects"] += 1
            first_examples.setdefault("no_objects", f"image_id={image_id} (no bboxes for this image)")
            continue

        ans_text = format_answer(qa, image_id, iw, ih)
        question = qa.get("question", "Describe the chest X-ray.")

        samples.append({
            "subject": subject,
            "study": study,
            "image_id": image_id,
            "img_path": img_path,
            "image_size": (iw, ih),
            "question": question,
            "answer_text": ans_text,
            "scene_graph": sg_dict,
        })

        if verbose:
            print(f"  ✓ sample #{len(samples)}: {study}/{image_id}  "
                  f"size={iw}x{ih}  sg_objects={sg_dict['num_objects']}")
            print(f"    Q: {question[:90]}")
            print(f"    A: {ans_text[:120].replace(chr(10), ' ')}")

    if len(samples) < n:
        print(f"\n⚠ only found {len(samples)}/{n} samples with full metadata + image + scene graph")
        print(f"[skips] candidates iterated: {len(q_head)}")
        for reason, count in skip_reasons.items():
            if count > 0:
                ex = first_examples.get(reason, "")
                print(f"  - {reason:14s}: {count:>4d}   first ex: {ex}")
    return samples


# ---------------------------------------------------------------------------
# Model run
# ---------------------------------------------------------------------------


def run_real_data_test(samples: List[Dict[str, Any]], model_id: str, gpu: int) -> int:
    """Build model, run training step + generation on the batch. Returns exit code."""
    from PIL import Image
    from models import SSGVQANetV2

    if not samples:
        print("\n✗ no samples to run on — abort")
        return 1

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        cc = torch.cuda.get_device_capability(gpu)
        torch_dtype = torch.bfloat16 if cc >= (8, 0) else torch.float16
        device = torch.device(f"cuda:{gpu}")
        force_qlora = cc < (8, 0)
    else:
        torch_dtype = torch.float32
        device = torch.device("cpu")
        force_qlora = False

    print(f"\n[model] {model_id}  device=cuda:{gpu}  dtype={torch_dtype}  qlora={force_qlora}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    t0 = time.time()
    model = SSGVQANetV2(
        qwen_model_id=model_id,
        use_quantization=force_qlora,
        num_sg_tokens=4,
        training_mode="pretrain",
        torch_dtype=torch_dtype,
        max_answer_length=32,
    )
    for name in ("sg_encoder", "sg_projector", "grounding_head", "aux_heads"):
        getattr(model, name).to(device=device, dtype=torch_dtype)
    model.sg_generator.to(device=device)
    print(f"[model] built in {time.time()-t0:.1f}s (d_llm={model.d_llm})")

    # Load PIL images now (not earlier, because PIL holds file handles)
    pil_images = [Image.open(s["img_path"]).convert("RGB") for s in samples]
    questions = [s["question"] for s in samples]
    answer_texts = [s["answer_text"] for s in samples]
    scene_graphs = [s["scene_graph"] for s in samples]
    gt_bboxes = torch.stack([
        torch.tensor(
            [
                float(s["answer_text"].split("<box>")[1].split("</box>")[0].split(",")[i])
                if "<box>" in s["answer_text"] else 0.0
                for i in range(4)
            ],
            dtype=torch.float, device=device,
        )
        for s in samples
    ])

    # ---- Training step ----
    print(f"\n[train] forward + loss + backward + step on {len(samples)} real samples")
    model.train()
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4
    )
    t1 = time.time()
    out = model(
        images=None,
        pil_images=pil_images,
        questions=questions,
        answer_texts=answer_texts,
        scene_graphs=scene_graphs,
        gt_grounding_bboxes=gt_bboxes,
    )
    fwd_t = time.time() - t1
    loss = out["lm_loss"]
    if loss is None or not torch.isfinite(loss):
        print(f"  ✗ non-finite lm_loss: {loss}")
        return 1
    print(f"  ✓ forward: lm_loss={loss.item():.4f}  ({fwd_t:.1f}s)")

    t2 = time.time()
    loss.backward()
    print(f"  ✓ backward ({time.time()-t2:.1f}s)")

    t3 = time.time()
    optim.step()
    optim.zero_grad(set_to_none=True)
    print(f"  ✓ optimizer step ({time.time()-t3:.2f}s)")

    bb = out["grounding_outputs"]["bbox_pred"]
    print(f"\n[train] predicted bboxes vs ground truth (normalized):")
    for i, s in enumerate(samples):
        gt = gt_bboxes[i].tolist()
        pr = bb[i].tolist()
        print(f"  sample {i+1}  gt=[{gt[0]:.3f},{gt[1]:.3f},{gt[2]:.3f},{gt[3]:.3f}]"
              f"  pred=[{pr[0]:.3f},{pr[1]:.3f},{pr[2]:.3f},{pr[3]:.3f}]")

    # ---- Inference (generate) ----
    print(f"\n[infer] generation")
    model.eval()
    t4 = time.time()
    with torch.no_grad():
        gen = model(
            images=None,
            pil_images=pil_images,
            questions=questions,
            scene_graphs=scene_graphs,
        )
    print(f"  ✓ generation ({time.time()-t4:.1f}s)")

    texts = gen["generated_answer_text"]
    for i, (s, t) in enumerate(zip(samples, texts or [])):
        print(f"\n  sample {i+1}")
        print(f"    Q  : {s['question'][:100]}")
        print(f"    gt : {s['answer_text'].split('<answer>')[-1].split('</answer>')[0][:100]}")
        print(f"    gen: {t[:140].replace(chr(10), ' ') if t else '(empty)'}")

    print(f"\n✓ Real-data test PASSED on {len(samples)} samples")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description="SSGVQANetV2 real-data smoke test")
    p.add_argument("--n", type=int, default=3, help="Number of samples (default 3)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    args = p.parse_args(argv)

    print(f"=== SSG-VQA-Net v2 real-data test (n={args.n}) ===\n")

    paths = load_paths()
    print(f"MIMIC-CXR-JPG : {paths['mimic_cxr_jpg_path']}")
    print(f"QBA           : {paths['mimic_ext_cxr_qba_path']}\n")

    samples = pick_real_samples(paths, args.n)
    if not samples:
        print("\n✗ no samples loaded — check that scene_data/ is unzipped and JPG paths resolve.")
        return 1

    return run_real_data_test(samples, args.model_id, args.gpu)


if __name__ == "__main__":
    sys.exit(main())
