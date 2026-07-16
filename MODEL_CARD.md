---
license: other
license_name: mimic-cxr-dua
license_link: https://physionet.org/content/mimic-cxr/2.0.0/
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - medical
  - radiology
  - chest-xray
  - visual-question-answering
  - scene-graph
  - qwen3-vl
  - qlora
  - grounded-vqa
base_model: Qwen/Qwen3-VL-8B-Instruct
datasets:
  - MIMIC-Ext-CXR-QBA
language:
  - en
---

# SSG-VQA-Net v2 — Stage 4 fine-tuned

Scene-graph-guided grounded VQA over MIMIC-CXR. Qwen3-VL-8B-Instruct backbone (4-bit NF4, double-quant, fp16 compute, SDPA) + rank-32 MLP-inclusive LoRA + standalone TorchXRayVision-DETR scene-graph generator (22 entities × 30 regions) + mHC grounding refinement head. Trained through the four-stage curriculum on MIMIC-Ext-CXR-QBA.

Full methodology in the accompanying thesis. This card reports **results only**.

- Code: [https://github.com/JONAHKYAGABA/MASTERS-WORK](https://github.com/JONAHKYAGABA/MASTERS-WORK)
- Wandb: `kyagabajonah/mimic-cxr-vqa`
- Sibling checkpoints: `KYAGABA/mimic-cxr-vqa-stage1-sg-only`, `KYAGABA/mimic-cxr-vqa-stage3-pretrain`

## Repositories

| Variant | HuggingFace repo | Size | How to load |
|---|---|---:|---|
| Stage 4 final (baseline) | `KYAGABA/mimic-cxr-vqa-stage4-finetune` | 9 GB | HF `AutoModelForImageTextToText`, NF4 quant config |
| FP16 (dequantised baseline) | `KYAGABA/mimic-cxr-vqa-stage4-fp16` | 16.4 GB | HF `AutoModelForImageTextToText`, no quant config |
| INT8 (bitsandbytes) | `KYAGABA/mimic-cxr-vqa-stage4-int8` | 8.6 GB | HF + `BitsAndBytesConfig(load_in_8bit=True)` |
| NF4 (bitsandbytes, QLoRA format) | `KYAGABA/mimic-cxr-vqa-stage4-nf4` | 4.6 GB | HF + `BitsAndBytesConfig(load_in_4bit=True, nf4)` |
| GGUF Q5_K_M | `KYAGABA/mimic-cxr-vqa-stage4-q5_k_m` | 5.7 GB | `llama-cpp-python` |
| GGUF Q4_K_M | `KYAGABA/mimic-cxr-vqa-stage4-q4_k_m` | 4.1 GB | `llama-cpp-python` |
| GGUF Q3_K_M | `KYAGABA/mimic-cxr-vqa-stage4-q3_k_m` | 3.5 GB | `llama-cpp-python` |

Every repo also ships a `heads.safetensors` sidecar (~300 MB) containing the scene-graph generator, encoder/projector, mHC grounding head, and auxiliary heads. The plain HF snippets below use only the Qwen backbone (fast baseline). For the full `<think><box><answer>` pipeline with SG-token injection and mHC grounding refinement, use `scripts/serve_app.py` from the training repo (§ Full pipeline below).

## Quick inference — HuggingFace variants (fp16 / int8 / nf4 / Stage 4 baseline)

Works for the top 4 repos in the table above. Change `REPO` and `QUANT_CFG`; everything else is identical.

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from PIL import Image

# ---- pick one ----
REPO = "KYAGABA/mimic-cxr-vqa-stage4-finetune"   # NF4 baseline (default)
# REPO = "KYAGABA/mimic-cxr-vqa-stage4-fp16"     # FP16
# REPO = "KYAGABA/mimic-cxr-vqa-stage4-int8"     # bnb INT8
# REPO = "KYAGABA/mimic-cxr-vqa-stage4-nf4"      # bnb NF4 (same as baseline)

QUANT_CFG = BitsAndBytesConfig(         # <-- omit for fp16
    load_in_4bit=True,                  # <-- load_in_8bit=True for int8
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

processor = AutoProcessor.from_pretrained(REPO)
model = AutoModelForImageTextToText.from_pretrained(
    REPO,
    quantization_config=QUANT_CFG,       # set to None for the fp16 repo
    torch_dtype=torch.float16,
    attn_implementation="sdpa",          # Turing / no FlashAttention-2
    device_map="auto",
).eval()

# ---- inference ----
image = Image.open("chest_xray.jpg").convert("RGB")
question = "Is there a pleural effusion in this chest X-ray?"

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": question},
    ],
}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True).to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

# Strip the prompt tokens, decode just the answer
answer = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(answer)
```

## Quick inference — GGUF variants (Q3_K_M / Q4_K_M / Q5_K_M)

CPU-only inference via `llama-cpp-python`. Install first: `pip install "llama-cpp-python>=0.3"`. Requires llama.cpp with Qwen3-VL support.

```python
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler   # or Qwen3VL when available
from PIL import Image
import base64, io

# ---- pick one ----
REPO = "KYAGABA/mimic-cxr-vqa-stage4-q4_k_m"    # 4.1 GB
# REPO = "KYAGABA/mimic-cxr-vqa-stage4-q5_k_m"  # 5.7 GB
# REPO = "KYAGABA/mimic-cxr-vqa-stage4-q3_k_m"  # 3.5 GB

model_path = hf_hub_download(REPO, filename=f"model-{REPO.split('-')[-1]}.gguf")

llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=8,            # match your CPU
    verbose=False,
)

# Encode the image as a data URI for llama.cpp's multimodal API
with open("chest_xray.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
image_uri = f"data:image/jpeg;base64,{b64}"

output = llm.create_chat_completion(
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_uri}},
            {"type": "text",      "text": "Is there a pleural effusion in this chest X-ray?"},
        ],
    }],
    max_tokens=256,
)
print(output["choices"][0]["message"]["content"])
```

Note: llama.cpp's Qwen3-VL multimodal path is still stabilising. If `create_chat_completion` errors on the image, fall back to the HF NF4 variant (`KYAGABA/mimic-cxr-vqa-stage4-nf4`) — same disk footprint, mature loader.

## Full pipeline (SG injection + mHC grounding refinement)

The snippets above use the Qwen backbone directly and skip the scene-graph soft-token injection and grounding-refinement head that make this model different from stock Qwen3-VL. To use the full pipeline you need the training repo:

```bash
git clone https://github.com/JONAHKYAGABA/MASTERS-WORK
cd MASTERS-WORK
pip install -r requirements.txt

# GPU (any HF variant)
python scripts/serve_app.py \
    --checkpoint KYAGABA/mimic-cxr-vqa-stage4-finetune \
    --image chest_xray.jpg \
    --question "Is there a pleural effusion?"

# CPU (any GGUF variant)
CUDA_VISIBLE_DEVICES="" python scripts/serve_app_cpu.py \
    --variant_dir <local_dir_of_gguf_repo> \
    --image chest_xray.jpg \
    --question "Is there a pleural effusion?" \
    --num_threads 8
```

`serve_app.py` returns the parsed `<think>`, `<box>`, `<answer>` fields plus the mHC-refined bounding box and pointing score.

## Per-stage configuration (as run)

Reproduces Table `tab:setup` from the methodology. Effective batch size = per-GPU 2 × 2 GPUs × grad-accum 4 = 16.

| Stage | Phase | Grade | Loss weights (vqa/gen/chex/sg/grd) | LR | Train cap | Notes |
|---:|---|:-:|---|---:|---:|---|
| 1 | SG generator | A | 0.05 / 0.05 / 0.05 / 1.0 / 0.05 | 5e-5 | 100,000 | standalone detector, EMA off |
| 2 | alignment | B | 0.0 / 1.0 / 0.05 / 0.0 / 0.0 | 5e-5 | 250,000 | EMA off |
| 3 | pretrain | B | 0.05 / 1.0 / 0.05 / 0.0 / 2.0 | 2e-5 | 250,000 | EMA 0.999 |
| 4 | finetune | A | 0.05 / 1.0 / 0.05 / 0.0 / 3.0 | 5.0e-6 | 100,000 | rank-32 MLP LoRA, EMA 0.999 |

## Achieved vs target metrics

Reproduces the target column from the methodology's Evaluation Metrics table. Validation used a 1,000-study held-out slice with **generated** scene graphs (Stage 3 onward) to match inference.

| Stage | Primary metric | Target (methodology) | Achieved |
|---:|---|---|---|
| 1 | SG mAP, bbox IoU | mAP > 0.6, IoU > 0.5 | Stage-1 detector converged on 22×30 vocab; downstream Val Grd IoU peaked at **0.276** at Stage 3 |
| 2 | CheXpert AUROC | AUROC > 0.75 | **0.55** (below target; chexpert head starved at loss weight 0.05) |
| 3 | VQA F1, grounding IoU | F1 > 0.70, IoU > 0.60 | classification_accuracy peak **0.516** (final **0.436**), Val Grd IoU **0.276**, pointing_accuracy path validated |
| 4 | Grounding IoU@0.5 | sharper than Stage 3 | **not achieved** — IoU@0.5 = **0.068**, mean IoU 0.214 (softer than Stage 3's mean 0.276); classification-family metrics improved instead |

## Full Stage-4 validation metrics (final step)

Best-model checkpoint saved at the peak-accuracy validation. Full metric panel reported per methodology's Evaluation Metrics section.

| Family | Metric | Value |
|---|---|---:|
| Classification | classification_accuracy | **0.504** |
| Classification | binary_accuracy | 0.522 |
| Classification | binary_f1 | 0.489 |
| Classification | category_accuracy | 0.296 |
| Classification | category_f1 | 0.109 |
| Classification | region_accuracy | 0.724 |
| Classification | region_f1 | 0.296 |
| Classification | severity_accuracy | 0.619 |
| Classification | severity_f1 | 0.361 |
| Grounding | grounding_mean_iou | 0.214 |
| Grounding | grounding_acc_iou25 | 0.356 |
| Grounding | grounding_acc_iou50 | 0.068 |
| Grounding | pointing_accuracy | 0.932 |
| CheXpert | chexpert_auroc | 0.553 |
| Generation | generation_bleu | 0.048 |
| Generation | generation_rouge_l | 0.150 |
| Generation | generation_exact_match | 0.000 |
| Generation | generation_word_overlap | 0.271 |
| Generation | template_bleu | 0.148 |
| Generation | template_word_overlap | 0.365 |
| SG (Hungarian-matched) | sg_mean_iou | 8.1e-6 |
| SG (Hungarian-matched) | sg_iou_50 | 0.000 |
| SG (Hungarian-matched) | sg_match_count | 869 |
| Loss | val_loss | 0.695 |

## Stage-to-stage progression (headline metric)

| Metric | Stage 3 (peak) | Stage 3 (final) | Stage 4 (final) | Δ vs Stage 3 final |
|---|---:|---:|---:|---:|
| classification_accuracy | 0.516 | 0.436 | **0.504** | +15.6% |
| binary_accuracy | — | 0.411 | **0.522** | +27% |
| chexpert_auroc | — | 0.548 | 0.553 | +0.9% |
| grounding_mean_iou | — | 0.276 | 0.214 | −22% |
| generation_bleu | — | 0.103 | 0.048 | −53% |
| generation_rouge_l | — | 0.221 | 0.150 | −32% |

**Trade-off:** Stage 4's grounding_loss_weight = 3.0 combined with a higher VQA / CheXpert / per-head weighting put ≈80% of the gradient budget on non-generation losses. Classification-family metrics rose sharply; free-text metrics regressed. Pointing accuracy remained strong (0.932) but IoU@0.5 stayed soft (0.068) — the model points at the right region but boxes are loose.

## Quantization policy (asymmetric)

Reproduces Table `tab:qpolicy` from the methodology. Only the Qwen backbone (99.4% of parameters) is quantised. LoRA adapters, SG generator, mHC grounding head, and aux heads are kept in FP16/FP32 and saved as a `heads.safetensors` sidecar (~300 MB).

| Component | Params | Precision (PTQ) | Rationale |
|---|---:|---|---|
| Qwen3-VL-8B backbone | 8.03 B | 4-bit or 8-bit | Dominates memory; token throughput is the bottleneck |
| LoRA adapters (rank 32) | ~50 M | FP16 (unchanged) | Small, task-specific, noise-sensitive |
| SG generator (DenseNet + DETR) | ~30 M | FP16 (unchanged) | Small; spatial precision matters for IoU |
| Grounding head (mHC) | ~4 M | FP32 (unchanged) | Sinkhorn-Knopp is fp16-unstable |
| Auxiliary heads | ~1 M | FP16 (unchanged) | Negligible size |

## Quantized variants

Reproduces Table `tab:qvariants` from the methodology. Six variants produced by `scripts/quantize_and_export.py` from the Stage-4 best checkpoint.

| Variant | Toolchain | Method | Disk | Val Acc | Median CPU latency (8-core) |
|---|---|---|---:|---:|---:|
| FP16 (baseline) | HuggingFace | none | 16.4 GB | *TBD* | *TBD* |
| INT8 | bitsandbytes | row-wise dynamic INT8 | 8.6 GB | *TBD* | *TBD* |
| NF4 | bitsandbytes | 4-bit NormalFloat + double-quant | 4.6 GB | *TBD* | *TBD* |
| GGUF Q5_K_M | llama.cpp | 5-bit block-wise K-quants | 5.7 GB | *TBD* | *TBD* |
| GGUF Q4_K_M | llama.cpp | 4-bit block-wise K-quants | 4.1 GB | *TBD* | *TBD* |
| GGUF Q3_K_M | llama.cpp | 3-bit block-wise K-quants | 3.5 GB | *TBD* | *TBD* |

Val Acc and Median CPU latency to be filled in from `benchmark_manifest.json` once `scripts/benchmark_cpu.py` completes on the canonical test-bed (Intel Xeon 12/24 cores 2.1 GHz, 8-thread pin, `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, `CUDA_VISIBLE_DEVICES=""`, batch 1, 100 warm-up + 500 measured queries per variant).

## Training environment (as run)

- 2 × NVIDIA Quadro RTX 8000 (48 GB each, Turing cc 7.5) — no bf16, no FlashAttention-2
- DDP, per-GPU micro-batch 2, gradient accumulation 4 → effective batch 16
- AdamW, weight decay 0.01 (Stages 1-3) / 0.05 (Stage 4), warmup 5%, grad-clip 1.0
- fp16 dynamic loss scaler; FP32 islands on mHC Sinkhorn-Knopp / GIoU / geometric features / SG projector attention / multi-task loss total; trainable params kept in FP32 under autocast

## Bias, risks, limitations

- **Not a medical device.** Research prototype only. Do not use for clinical decision-making, triage, or patient care.
- **Label ceiling.** Scene graphs are LLM+atlas-generated (prior analysis reports mean max-IoU ≈ 0.37); reported IoU numbers are bounded by this noise floor.
- **Loose boxes at Stage 4.** IoU@0.5 = 0.068 despite pointing_accuracy = 0.932 — coarse ROI hint, not diagnostic-grade localisation.
- **Free-text regression at Stage 4.** BLEU and ROUGE-L drop materially vs Stage 3 due to the loss re-balance toward grounding + classification.
- **Stage 2 CheXpert target not met.** Achieved AUROC ≈ 0.55 vs target > 0.75 (head starved at loss weight 0.05; addressed later at Stages 3-4 by bumping to 0.3 but only recovered to 0.553).
- **PhysioNet DUA.** Downstream users must hold credentialed PhysioNet access and abide by the MIMIC-CXR DUA (no re-identification, no redistribution to non-credentialed users).

## License

Released under the PhysioNet MIMIC-CXR Data Use Agreement (research-only, credentialed access). Downstream users must confirm PhysioNet access before use.

## Citation

```bibtex
@misc{ssg-vqa-net-v2-stage4,
  title  = {SSG-VQA-Net v2: Scene-Graph-Guided VQA on MIMIC-CXR},
  author = {Kyagaba, Jonah},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/KYAGABA/mimic-cxr-vqa-stage4-finetune}},
  note   = {MASTERS-WORK thesis project}
}
```
