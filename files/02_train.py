#!/usr/bin/env python3
"""
PATH C - PHASE 2: FINE-TUNE WITH RAG-AWARE PROMPTS

Trains a model with QLoRA where each training example includes 2 retrieved
similar Q&A pairs in the prompt. This is the RAG-BioQA pattern (cited in research).

Default: google/gemma-2-27b-it (best open model for African languages per AfroBench)
Alternative: google/medgemma-27b-text-it (set MODEL_ID env var)

Time: ~17-20 hours
Cost: ~$8-10 on A40

Usage:
  python3 02_train.py                                    # default Gemma 2 27B
  MODEL_ID=google/medgemma-27b-text-it python3 02_train.py   # use Med-Gemma instead
  SMOKE=1 python3 02_train.py                            # mini run (200 samples, 30 min)

Requires Phase 1 outputs in /workspace/path_c/
"""

import os
import sys
import json
import warnings
from pathlib import Path

HF_TOKEN = "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"
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from huggingface_hub import login
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed,
    Trainer, TrainingArguments, EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

warnings.filterwarnings('ignore')

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-2-27b-it")
SMOKE = os.environ.get("SMOKE", "0") == "1"
WORKSPACE = os.environ.get("WORKSPACE", os.path.expanduser("~/zindi"))
PATH_C_DIR = f"{WORKSPACE}/path_c"
OUTPUT_BASE = f"{WORKSPACE}/path_c/models"

# Derive output dir from model name
model_short = MODEL_ID.split("/")[-1].replace(".", "-")
OUTPUT_DIR = f"{OUTPUT_BASE}/{model_short}-rag"

LANGUAGE_MAP = {
    'Aka': 'Akan', 'Amh': 'Amharic', 'Lug': 'Luganda',
    'Swa': 'Swahili', 'Eng': 'English',
}
REGION_MAP = {
    'Aka_Gha': 'Ghana', 'Amh_Eth': 'Ethiopia',
    'Lug_Uga': 'Uganda', 'Swa_Ken': 'Kenya',
    'Eng_Uga': 'Uganda', 'Eng_Gha': 'Ghana',
    'Eng_Eth': 'Ethiopia', 'Eng_Ken': 'Kenya',
}

# Number of retrieved examples to include in the prompt during training.
# 2 is a balance: enough context, not too long.
N_RETRIEVED_TRAIN = 2


def make_rag_prompt(question, subset, retrieved_examples):
    """RAG-aware prompt. MUST match inference prompt exactly."""
    lang_code = subset.split('_')[0]
    language = LANGUAGE_MAP.get(lang_code, lang_code)
    region = REGION_MAP.get(subset, '')

    examples_text = ""
    for i, ex in enumerate(retrieved_examples, 1):
        examples_text += f"\nExample {i}:\nQuestion: {ex['input']}\nAnswer: {ex['output']}\n"

    return (
        f"You are answering a maternal, sexual, and reproductive health (MSRH) "
        f"question for community education in {region}. "
        f"Provide a clear, accurate, and culturally appropriate response in {language}.\n"
        f"\nReference examples:{examples_text}\n"
        f"Now answer this question:\n"
        f"Question: {question}\n"
        f"Answer:"
    )


class RAGTrainDataset(Dataset):
    def __init__(self, df, retrievals_idx, train_df_ref, tokenizer, max_length=1024,
                 n_retrieved=N_RETRIEVED_TRAIN):
        """
        df: the working dataframe (either train or a subset)
        retrievals_idx: dict mapping df row index -> list of train_df row indices
        train_df_ref: the reference train dataframe for looking up retrieved examples
        """
        self.df = df.reset_index(drop=True)
        self.retrievals_idx = retrievals_idx
        self.train_df_ref = train_df_ref
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_retrieved = n_retrieved

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        retrieved_idxs = self.retrievals_idx.get(str(idx), self.retrievals_idx.get(idx, []))
        retrieved = []
        for ridx in retrieved_idxs[:self.n_retrieved]:
            ref_row = self.train_df_ref.iloc[int(ridx)]
            retrieved.append({
                'input': ref_row['input'],
                'output': str(ref_row['output']),
            })

        user_msg = make_rag_prompt(row['input'], row['subset'], retrieved)
        answer = str(row['output'])

        prompt_messages = [{"role": "user", "content": user_msg}]
        prompt_ids = list(self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True
        ))
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        answer_ids = answer_ids + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        attention_mask = [1] * len(input_ids)

        # If too long, truncate from start (keeps recent context + answer)
        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]
            attention_mask = attention_mask[overflow:]

        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": attention_mask}


class DataCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features):
        max_length = max(len(f["input_ids"]) for f in features)
        max_length = ((max_length + 7) // 8) * 8
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            pad_len = max_length - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
        }


def main():
    set_seed(42)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    mode_str = "SMOKE" if SMOKE else "FULL"
    print("="*70)
    print(f"PATH C - PHASE 2: TRAINING ({mode_str})")
    print(f"Model: {MODEL_ID}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*70)
    login(token=HF_TOKEN, add_to_git_credential=False)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ====== Load Phase 1 outputs ======
    print("\nLoading Phase 1 outputs...")
    required = ['train_metadata.parquet', 'train_retrievals.json', 'val_retrievals.json']
    for f in required:
        path = f"{PATH_C_DIR}/{f}"
        if not os.path.exists(path):
            print(f"ERROR: Missing {path}. Run 01_setup_and_index.py first.")
            return
    train_df = pd.read_parquet(f"{PATH_C_DIR}/train_metadata.parquet")
    print(f"  Train: {len(train_df):,}")

    val_df = pd.read_csv(f"{WORKSPACE}/Val.csv").dropna(
        subset=['input', 'output', 'subset']).reset_index(drop=True)
    print(f"  Val: {len(val_df):,}")

    with open(f"{PATH_C_DIR}/train_retrievals.json") as f:
        train_retrievals = json.load(f)
    with open(f"{PATH_C_DIR}/val_retrievals.json") as f:
        val_retrievals = json.load(f)
    print(f"  Loaded retrievals")

    # ====== Subset for SMOKE mode ======
    if SMOKE:
        # 200 stratified samples
        small_idxs = []
        for s in train_df['subset'].unique():
            idxs = train_df[train_df['subset'] == s].index[:25].tolist()
            small_idxs.extend(idxs)
        # Remap retrievals to keep them valid
        train_df_subset = train_df.iloc[small_idxs].reset_index(drop=True)
        # For smoke mode, we use simple retrieval index mapping (use train_df_ref's original index)
        # Map the new indices to original retrievals
        train_retrievals_smoke = {}
        for new_idx, orig_idx in enumerate(small_idxs):
            orig_retrievals = train_retrievals.get(str(orig_idx), [])
            train_retrievals_smoke[new_idx] = orig_retrievals
        active_train_df = train_df_subset
        active_train_retrievals = train_retrievals_smoke

        val_idxs = []
        for s in val_df['subset'].unique():
            val_idxs.extend(val_df[val_df['subset'] == s].index[:2].tolist())
        val_df_subset = val_df.iloc[val_idxs].reset_index(drop=True)
        val_retrievals_smoke = {}
        for new_idx, orig_idx in enumerate(val_idxs):
            val_retrievals_smoke[new_idx] = val_retrievals.get(str(orig_idx), [])
        active_val_df = val_df_subset
        active_val_retrievals = val_retrievals_smoke
    else:
        active_train_df = train_df
        active_train_retrievals = train_retrievals
        # Use 500 val samples for faster eval
        active_val_df = val_df.sample(min(500, len(val_df)),
                                      random_state=42).reset_index(drop=True)
        # Map sampled val indices to retrievals
        active_val_retrievals = {}
        for new_idx, orig_idx in enumerate(active_val_df.index):
            # If sample preserved order, just use the val_retrievals directly
            # But sample reorders, so get original index BEFORE reset_index
            pass
        # Simpler: just re-sample and look up by original position
        sampled_val = val_df.sample(min(500, len(val_df)), random_state=42)
        active_val_df = sampled_val.reset_index(drop=False).rename(
            columns={'index': '_orig_idx'})
        active_val_retrievals = {}
        for i in range(len(active_val_df)):
            orig_idx = int(active_val_df.iloc[i]['_orig_idx'])
            active_val_retrievals[i] = val_retrievals.get(str(orig_idx), [])
        active_val_df = active_val_df.drop(columns=['_orig_idx'])

    print(f"\nActive train: {len(active_train_df):,}, Active val: {len(active_val_df):,}")

    # ====== Verify prompt sample ======
    sample_idx = 0
    sample_row = active_train_df.iloc[sample_idx]
    sample_retrievals_idxs = active_train_retrievals.get(
        sample_idx, active_train_retrievals.get(str(sample_idx), []))
    sample_retrieved = []
    for ridx in sample_retrievals_idxs[:N_RETRIEVED_TRAIN]:
        ref = train_df.iloc[int(ridx)]
        sample_retrieved.append({'input': ref['input'], 'output': str(ref['output'])})
    sample_prompt = make_rag_prompt(sample_row['input'], sample_row['subset'], sample_retrieved)
    print("\nSample RAG prompt (first 800 chars):")
    print("-"*70)
    print(sample_prompt[:800])
    print("-"*70)
    print(f"Expected answer (first 200): {str(sample_row['output'])[:200]}")

    # ====== Load model ======
    print(f"\nLoading {MODEL_ID} (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_storage=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()
    print(f"  Loaded ({torch.cuda.memory_allocated() / 1e9:.1f} GB)")

    peft_config = LoraConfig(
        lora_alpha=32, lora_dropout=0.05, r=16, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    # ====== Build datasets ======
    train_dataset = RAGTrainDataset(
        active_train_df, active_train_retrievals, train_df, tokenizer, max_length=1024)
    val_dataset = RAGTrainDataset(
        active_val_df, active_val_retrievals, train_df, tokenizer, max_length=1024)
    data_collator = DataCollator(tokenizer)

    # ====== Training args ======
    if SMOKE:
        epochs, save_steps, eval_steps, logging_steps = 2, 50, 25, 5
    else:
        epochs, save_steps, eval_steps, logging_steps = 2, 200, 200, 20

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=eval_steps,
        learning_rate=3e-5,
        bf16=False, fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        seed=42,
    )

    callbacks = []
    if not SMOKE:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=data_collator, tokenizer=tokenizer,
        callbacks=callbacks,
    )

    total_steps = len(train_dataset) * epochs // 16
    est_h = total_steps * 35 / 3600
    print(f"\nSteps: {total_steps}, ETA: {est_h:.1f}h, Cost: ${est_h * 0.47:.2f}")
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nSaved to {OUTPUT_DIR}")
    print("Next: python3 03_ensemble_inference.py")


if __name__ == "__main__":
    main()
