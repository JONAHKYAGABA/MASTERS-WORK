# Quantization + CPU Deployment Guide

This folder contains the implementation of Objectives (iv) and (v) of the
SSG-VQA-Net v2 thesis: post-training quantization and CPU-only inference.

Three scripts + one canonical run book:

| Script | Purpose | Objective |
|---|---|---|
| `quantize_and_export.py` | Produce 6 quantized variants from a Stage-4 checkpoint | iv |
| `serve_app_cpu.py` | Load one variant on CPU and answer a single question | v |
| `benchmark_cpu.py` | Measure latency / throughput / memory / quality per variant | v |

The end-to-end pipeline: **train → quantize → benchmark → paper table**.

---

## 1. Prerequisites

### Python dependencies (in your training venv)

```bash
source .venv/bin/activate

# Already installed for training:
#   torch, transformers, peft, bitsandbytes, safetensors, torchxrayvision, psutil

# Additional for the CPU + GGUF path:
pip install "llama-cpp-python>=0.3.0"

# Optional (used for a smoother HF -> GGUF conversion):
pip install "gguf>=0.14.0" "sentencepiece>=0.2.0"
```

### llama.cpp checkout (required for GGUF variants only)

```bash
cd /opt   # or wherever you keep source trees
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

# Register the location so quantize_and_export.py can find it
export LLAMA_CPP=/opt/llama.cpp

# Test
$LLAMA_CPP/build/bin/llama-quantize --help | head -3
```

You can skip llama.cpp entirely if you only want the FP16 / INT8 / NF4
variants — those use bitsandbytes.

### CPU-only environment (for benchmarking)

Force CUDA off and pin threads for reproducibility:

```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# On marconi: pin to CPU cores 0-7 so the numbers reproduce
taskset -c 0-7 python scripts/benchmark_cpu.py ...
```

---

## 2. Quantize the trained model

Point at your Stage-4 (or Stage-3) `best_model` directory. The script
auto-follows a `checkpoint_step.txt` pointer if present.

```bash
python scripts/quantize_and_export.py \
    --checkpoint ./checkpoints/mimic-cxr-vqa/finetune/best_model \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --output_dir ./quantized_models \
    --variants fp16 int8 nf4 q5_k_m q4_k_m q3_k_m \
    --llama_cpp_path /opt/llama.cpp \
    --verbose
```

If your checkpoint has numpy metadata that the safe torch loader
rejects, add `SG_TRUST_CHECKPOINT=1` in front (only do this for
checkpoints you produced yourself):

```bash
SG_TRUST_CHECKPOINT=1 python scripts/quantize_and_export.py ...
```

**Output tree:**

```
./quantized_models/
├── disk_manifest.json        # per-variant status + disk sizes
├── fp16/
│   ├── model.safetensors     # merged Qwen backbone at FP16
│   ├── heads.safetensors     # SG generator + heads (FP16 sidecar)
│   ├── config.json           # HF config with hidden_size
│   └── tokenizer / processor files
├── int8/
│   ├── model.safetensors     # bnb 8-bit dynamic
│   ├── heads.safetensors     # (same sidecar copied verbatim)
│   └── ...
├── nf4/
│   ├── model.safetensors     # bnb 4-bit NF4 + double-quant
│   └── ...
├── q4_k_m/
│   ├── model-q4_k_m.gguf     # GGUF quantized backbone
│   ├── heads.safetensors     # copied from fp16
│   └── ...
├── q5_k_m/
│   └── model-q5_k_m.gguf
└── q3_k_m/
    └── model-q3_k_m.gguf
```

**Expected disk footprints** (Stage-4 Qwen3-VL-8B with LoRA rank-32 merged):

| Variant | Size | Ratio vs FP16 |
|---|--:|--:|
| fp16    | ~16.4 GB | 1.00× |
| int8    | ~8.6 GB  | 1.90× |
| nf4     | ~4.6 GB  | 3.57× |
| q5_k_m  | ~5.7 GB  | 2.88× |
| q4_k_m  | ~4.1 GB  | 4.00× |
| q3_k_m  | ~3.4 GB  | 4.82× |

---

## 3. Sanity-check a single variant

Confirm one variant loads and generates before running the 500-query
benchmark. Test on any chest X-ray from your dataset:

```bash
CUDA_VISIBLE_DEVICES="" python scripts/serve_app_cpu.py \
    --variant_dir ./quantized_models/q4_k_m \
    --image ./dataset_samples/sample_01_image.png \
    --question "Is there a pleural effusion?" \
    --num_threads 8
```

Expected output:

```
======================================================================
variant : q4_k_m
answer  : No focal consolidation or pleural effusion is seen.
think   : (chain of thought from the model)
bbox    : [0.36, 0.42, 0.68, 0.72]
timing  : {'sg_gen_ms': 152.3, 'ttft_ms': 924.1, 'answer_ms': 2418.7,
           'refine_ms': 47.9} (total 3543.0 ms)
======================================================================
```

If this errors out (missing GGUF file, tokenizer mismatch, etc.), fix
before proceeding — the benchmark will silently drop failing queries and
give you meaningless numbers.

---

## 4. Run the full benchmark

The paper's tables come from this command. Locks the test-bed at 8 CPU
cores, no GPU, 100 warm-up + 500 measured queries per variant, plus 200
studies for Val Acc + Grd IoU.

```bash
CUDA_VISIBLE_DEVICES="" \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
taskset -c 0-7 \
python scripts/benchmark_cpu.py \
    --quantized_root ./quantized_models \
    --qba_root data/mimic-ext-cxr-qba \
    --jpg_root data/mimic-cxr-jpg \
    --n_warmup 100 --n_measure 500 --n_quality 200 \
    --output benchmark_manifest.json \
    --verbose \
    2>&1 | tee logs/benchmark_$(date +%Y%m%d_%H%M%S).log
```

Wall clock: roughly **6-10 hours total** (all six variants, ~1-2 hours
per variant depending on backend).

If you only want one variant (e.g. Q4_K_M for a quick check):

```bash
python scripts/benchmark_cpu.py \
    --quantized_root ./quantized_models \
    --variant q4_k_m \
    --qba_root data/mimic-ext-cxr-qba \
    --jpg_root data/mimic-cxr-jpg \
    --n_warmup 20 --n_measure 100 --n_quality 50 \
    --output benchmark_q4_only.json
```

**Output**: `benchmark_manifest.json` — one JSON object per variant with
all timing statistics, memory peak, and quality metrics. Also a summary
table printed at the end.

---

## 5. Paper-ready tables from the manifest

`benchmark_manifest.json` is designed to be trivially post-processed into
the LaTeX tables the paper needs. Two example converters:

```python
# Extract Table X (disk + quality) as CSV
import json, csv
data = json.load(open("benchmark_manifest.json"))
with open("table_disk_quality.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["variant", "disk_gb", "peak_ram_gb",
                "val_acc", "val_grd_iou"])
    for v in data["variants"]:
        w.writerow([v["variant"],
                    round(v["disk_bytes"] / 1e9, 2),
                    round(v["peak_rss_gb"], 2),
                    round((v.get("val_acc") or 0), 3),
                    round((v.get("val_grd_iou") or 0), 3)])
```

```python
# Extract Table Y (latency) as CSV
import json, csv
data = json.load(open("benchmark_manifest.json"))
with open("table_latency.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["variant", "median_s", "p95_s", "p99_s", "qpm",
                "ttft_ms", "sg_gen_ms", "answer_ms", "refine_ms"])
    for v in data["variants"]:
        w.writerow([v["variant"],
                    round(v["latency_median_ms"] / 1000, 2),
                    round(v["latency_p95_ms"] / 1000, 2),
                    round(v["latency_p99_ms"] / 1000, 2),
                    round(v["throughput_qpm"], 1),
                    round(v["ttft_median_ms"], 1),
                    round(v["sg_gen_median_ms"], 1),
                    round(v["answer_median_ms"], 1),
                    round(v["refine_median_ms"], 1)])
```

---

## 6. Canonical test-bed (record this in the paper)

The methodology section should record the exact environment the numbers
were taken on. Copy this block:

```
CPU:      Intel Xeon Silver 4310 (12 cores, 24 threads, 2.10 GHz)  # OR your actual CPU
          -- benchmark restricted to 8 cores via `taskset -c 0-7`
RAM:      16 GB DDR4 3200 MHz
OS:       Ubuntu 22.04 LTS,  kernel 6.5.x
Software: torch 2.7.0, transformers 4.57.6, peft 0.11.x,
          bitsandbytes 0.43.1, llama-cpp-python 0.3.x,
          llama.cpp commit <SHA>
Threads:  OMP_NUM_THREADS=8, MKL_NUM_THREADS=8, llama.cpp n_threads=8
CPU gov:  performance (pinned)
GPU:      disabled via CUDA_VISIBLE_DEVICES=""
Batch:    1 (interactive single-question)
KV cache: enabled (llama.cpp mmap on)
```

To grab your actual CPU model + kernel:

```bash
lscpu | grep -E "Model name|CPU\(s\)"
uname -r
free -g
```

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `torch.load(weights_only=True) rejected …` | numpy pickled in checkpoint metrics | `SG_TRUST_CHECKPOINT=1` prefix (checkpoints you own only) |
| `convert_hf_to_gguf.py: not found` | llama.cpp checkout missing | See §1; set `LLAMA_CPP=/path/to/llama.cpp` |
| `heads.safetensors: not found` (in variant dir) | Ran `quantize_and_export.py` without FP16 variant | Re-run with `--variants fp16 <target>` (FP16 is always needed as the source for other variants) |
| OOM at model load in `serve_app_cpu.py` | 16 GB RAM budget too tight | Use `q4_k_m` or smaller; enable `LLAMA_N_CTX=1024` |
| `llama-cpp-python` doesn't recognise Qwen3-VL vision fusion | GGUF converter version too old | Update `pip install llama-cpp-python --upgrade`; fall back to `bnb` (nf4) if unresolved |
| Benchmark reports 0 for `val_grd_iou` on all variants | Test studies had no GT bboxes | Increase `--n_quality` or drop `--quality_grade` to `B` |

---

## 8. Reproducibility artifacts to publish alongside the paper

For the supplementary materials, commit these to your HF repo:

- `quantized_models/disk_manifest.json`
- `benchmark_manifest.json`
- `benchmark_<timestamp>.log` (raw command output)
- The quantized variant weights themselves (public HF repos:
  `KYAGABA/ssg-vqa-net-v2-{fp16,int8,nf4,q4_k_m,q5_k_m,q3_k_m}`)
- This README

That gives reviewers everything they need to reproduce the tables byte-
for-byte on a matching CPU.
