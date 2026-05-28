# Path C: RAG-Enhanced Ensemble Pipeline

Research-backed pipeline for the Zindi MSRH challenge. Based on cited approaches:
RAG-BioQA (medical QA with RAG), AfroBench (Gemma 2 27B is best open model for African
languages), self-consistency decoding (Wang 2022, Malon 2024), and the winning pattern
from Zindi's LLM Telecom Networks competition (RAG + ensemble).

## Hardware target: 2x Quadro RTX 8000 (Turing, 48GB each)

IMPORTANT: These cards are Turing (compute 7.5), which has NO native bf16. All scripts
use fp16 with loss scaling. Do not change fp16 back to bf16, it will be slow or unstable.
No Flash Attention 2 either (Turing), so attn_implementation="eager" is used throughout.

Compute is local and free, so the only cost is wall-clock time. Turing is roughly half
the fp16 throughput of an A40, so runs take ~1.5-2x longer than typical cloud estimates.

## Files

| File | Purpose | Time (on RTX 8000) |
|------|---------|--------------------|
| `smoke_test.py` | Verify entire pipeline on small data | 30-40 min |
| `01_setup_and_index.py` | Data analysis + BGE-M3 embeddings + FAISS index + retrievals | 45-60 min |
| `02_train.py` | QLoRA fine-tune with RAG-aware prompts | 25-35 hrs (or ~15h at 1 epoch) |
| `03_ensemble_inference.py` | Generate predictions with self-consistency + ensemble | 8-16 hrs |

### Pin each job to a specific GPU

GPU 0 runs your display (Xorg/gnome/firefox), GPU 1 is free. Always pin with
CUDA_VISIBLE_DEVICES so a job uses exactly one card (avoids splitting the model across
both GPUs and conflicting with the desktop):

```bash
# Primary training on the free GPU 1
CUDA_VISIBLE_DEVICES=1 python3 02_train.py

# Second model in parallel on GPU 0 (close firefox first to free VRAM)
CUDA_VISIBLE_DEVICES=0 MODEL_ID=google/medgemma-27b-text-it python3 02_train.py
```

Since compute is free, training BOTH models for the ensemble is now worth it (the earlier
"skip the second model to save money" advice was budget-driven and no longer applies).
You can run them in parallel, one per GPU, finishing both in ~30h wall-clock.

## Pipeline diagram

```
Train.csv ─┐
Val.csv   ─├─> 01_setup_and_index.py ─> embeddings + FAISS + retrievals
Test.csv  ─┘                              │
                                          ▼
                                       02_train.py (Gemma 2 27B + LoRA, RAG-aware)
                                          │
                                          ▼
                       (optional) 02_train.py with MODEL_ID=medgemma ─> Model B
                                          │
                                          ▼
                              03_ensemble_inference.py
                                          │
                                          ▼
                                   submission_v3.csv
```

## Dependencies

Use a fresh venv to avoid clashing with your MIMIC-CXR / Telco work on this machine:
```bash
python3 -m venv ~/zindi_venv && source ~/zindi_venv/bin/activate
pip install transformers==4.55.4 accelerate==1.5.2 peft==0.14.0 \
            bitsandbytes==0.45.0 datasets==3.0.0 huggingface_hub \
            sentence-transformers==3.0.1 faiss-cpu pandas pyarrow
```

Use `faiss-cpu`, not `faiss-gpu`: your CUDA is 13.0 and faiss-gpu pip wheels lag behind
CUDA releases. Exact search over 30K vectors on CPU is effectively instant, so there's no
benefit to the GPU build here.

bitsandbytes 0.45.0 supports Turing (compute 7.5), so 4-bit NF4 quantization works on the
RTX 8000. Confirm your install sees both GPUs: `python3 -c "import torch; print(torch.cuda.device_count())"` should print 2.

## Execution order

### Step 0: Put data in place
Place `Train.csv`, `Val.csv`, `Test.csv` in `~/zindi/` (or set `WORKSPACE`).
Check you have ~120GB free for the two 27B models: `df -h ~`.

### Step 1: Smoke test (always run first)
```bash
CUDA_VISIBLE_DEVICES=1 python3 smoke_test.py
```
This downloads BGE-M3 (~600MB) and Gemma 2 27B (~50GB). If both succeed and training
loss drops (no `nan`), the full pipeline will work. If anything fails, stop and debug.
On Turing this takes ~30-40 min.

### Step 2: Build retrievals
```bash
CUDA_VISIBLE_DEVICES=1 python3 01_setup_and_index.py
```
Outputs everything to `~/zindi/path_c/`. Don't proceed until this finishes cleanly.

### Step 3: Train Model A (Gemma 2 27B) on the free GPU 1
```bash
CUDA_VISIBLE_DEVICES=1 nohup python3 02_train.py > training_a.log 2>&1 &
tail -f training_a.log
```
Watch the first ~50 steps. Loss should be 1.5-3.0 and decreasing. If you see `nan`,
kill it and lower learning_rate to 2e-5 in 02_train.py (Gemma 2 fp16 overflow on Turing).
Early stopping triggers if eval_loss plateaus for 3 evals.

### Step 4: Train Model B (Med-Gemma 27B) in parallel on GPU 0
Compute is free, so do this for the ensemble. Close firefox first to free GPU 0 VRAM.
```bash
CUDA_VISIBLE_DEVICES=0 MODEL_ID=google/medgemma-27b-text-it \
    nohup python3 02_train.py > training_b.log 2>&1 &
tail -f training_b.log
```
Both run simultaneously, one per card, finishing in ~30h wall-clock.

### Step 5: Inference + ensemble (use the free GPU 1)
```bash
# Single model
CUDA_VISIBLE_DEVICES=1 python3 03_ensemble_inference.py \
    ~/zindi/path_c/models/gemma-2-27b-it-rag

# Two-model ensemble (run after both trainings finish)
CUDA_VISIBLE_DEVICES=1 python3 03_ensemble_inference.py \
    ~/zindi/path_c/models/gemma-2-27b-it-rag \
    ~/zindi/path_c/models/medgemma-27b-text-it-rag
```

### Step 6: Submit
- Download `~/zindi/path_c/submission_v3.csv` (or `submission_model_a.csv` if single model)
- Upload to Zindi
- Manually SELECT this submission AND your previous best (5Ux9TL9k, 0.369) as your two
  final picks for the private leaderboard.

## Data location

Scripts read Train.csv, Val.csv, Test.csv from `$WORKSPACE` (default `~/zindi`) and write
all outputs under `$WORKSPACE/path_c`. Either put your CSVs in `~/zindi/`, or point
WORKSPACE elsewhere:
```bash
export WORKSPACE=/home/you/code/zindi_data   # wherever your CSVs live
```

## What's research-backed (citations)

| Choice | Source |
|--------|--------|
| Base model = Gemma 2 27B | AfroBench (ACL 2025 Findings): best open LLM for African languages on 8 of 14 languages tested |
| Embedding model = BGE-M3 | BAAI multilingual retrieval model, supports FLORES-200 languages including Akan, Amharic, Luganda, Swahili |
| RAG-aware fine-tuning | RAG-BioQA paper (Oct 2025): "Fine-tuning improves BERTScore by 81% over the base model" with FAISS + LoRA |
| Detailed task-specific prompts | Patient Discharge study (NCBI 2026): ROUGE-1 16.59% → 42.72% with detailed instructions |
| Self-consistency decoding | Wang et al. 2022 (Self-Consistency improves CoT) and Malon & Zhu 2024 (Sample & Select for open generation) |
| RAG + ensemble for Zindi LLM | Winner of Zindi Specializing LLMs for Telecom Networks: used ColBERT + Falcon + Phi-2 |

## Realistic expectations

The leader is at 0.688. With this pipeline:
- Single model (Gemma 2 27B + RAG + self-consistency): plausibly 0.50-0.60
- Two-model ensemble: plausibly 0.55-0.65 with luck

These are estimates extrapolating from related research. I cannot guarantee specific scores
because I haven't tested this exact pipeline on this exact data. Treat as informed bets,
not promises.

## Things I'm not certain about

1. **Gemma 2 27B vs Med-Gemma 27B**: AfroBench tested Gemma 2 27B and rated it best open
   model for African languages, but didn't include Med-Gemma. Med-Gemma is on Gemma 3
   (newer architecture) but unknown African language coverage. Best to train both and
   compare via the validation set.

2. **BGE-M3 quality on Akan**: BGE-M3 covers FLORES-200 languages which includes
   Akan (twi_Latn), but I don't have benchmarks showing retrieval quality specifically
   for medical Akan. The retrieval might be weak for Akan and strong for English.

3. **Gated model access**: `google/gemma-2-27b-it` requires accepting Google's terms
   on HuggingFace. If access fails, your existing `google/medgemma-27b-text-it` access
   should still work (you used it before). Set `MODEL_ID=google/medgemma-27b-text-it`.

4. **Max sequence length**: I set max_length=1024 to accommodate retrieved examples in
   the prompt. If you see OOM errors, drop to 768 in `02_train.py`.

5. **Selection criterion (ROUGE-L vs retrieved exemplars)**: Picking the candidate with
   highest ROUGE-L against retrieved exemplars favors lexically-similar outputs. This
   should help ROUGE scores, but I haven't tested it on this dataset.

## If something fails

- **Smoke test fails**: Don't proceed. The full pipeline won't work either. Send me the
  error.
- **Phase 1 fails (embeddings)**: Try `pip install sentence-transformers --upgrade`.
  If still failing, swap to LaBSE: change `EMBEDDING_MODEL_ID = "sentence-transformers/LaBSE"`.
- **Phase 2 OOM**: Set `max_length=768` in `02_train.py` and re-run.
- **Phase 3 too slow**: Reduce `N_CANDIDATES` from 3 to 2 in `03_ensemble_inference.py`.
- **Gemma 2 access denied**: Accept terms at https://huggingface.co/google/gemma-2-27b-it
  or use Med-Gemma instead (`MODEL_ID=google/medgemma-27b-text-it`).
