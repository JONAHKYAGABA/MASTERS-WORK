#!/usr/bin/env python3
"""
SMOKE TEST FOR PATH C PIPELINE
Verifies in ~15-20 minutes that the entire pipeline works:
1. CSV loading + data analysis
2. BGE-M3 embedding model loads
3. FAISS index builds + retrieval works
4. Gemma 2 27B loads with QLoRA
5. RAG-aware training works (loss drops)
6. RAG-aware inference produces valid output

Cost: ~$0.30 on A40.

Usage:
  python3 smoke_test.py
"""

import os
import sys

HF_TOKEN = "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"
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

import warnings
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from huggingface_hub import login
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed,
    Trainer, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

warnings.filterwarnings('ignore')

# Configuration
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-2-27b-it")
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
WORKSPACE = os.environ.get("WORKSPACE", os.path.expanduser("~/zindi"))
OUTPUT_DIR = f"{WORKSPACE}/smoke_outputs"

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


def make_rag_prompt(question, subset, retrieved_examples):
    """RAG-aware prompt with few-shot retrieved examples"""
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


class RAGDataset(Dataset):
    def __init__(self, df, retrievals, tokenizer, max_length=1024):
        self.df = df.reset_index(drop=True)
        self.retrievals = retrievals
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        retrieved = self.retrievals.get(idx, [])
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
    print("="*70)
    print("SMOKE TEST: PATH C PIPELINE")
    print("="*70)
    login(token=HF_TOKEN, add_to_git_credential=False)

    # ====== STEP 1: Load data ======
    print("\n[1/6] Loading data...")
    train_df = pd.read_csv(f"{WORKSPACE}/Train.csv").dropna(
        subset=['input', 'output', 'subset']).reset_index(drop=True)
    val_df = pd.read_csv(f"{WORKSPACE}/Val.csv").dropna(
        subset=['input', 'output', 'subset']).reset_index(drop=True)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}")
    assert len(train_df) > 25000, f"Too few rows: {len(train_df)}"

    small_train = pd.concat([
        train_df[train_df['subset'] == s].head(25)
        for s in train_df['subset'].unique()
    ]).reset_index(drop=True)
    small_val = pd.concat([
        val_df[val_df['subset'] == s].head(2)
        for s in val_df['subset'].unique()
    ]).reset_index(drop=True)
    print(f"  Smoke: train={len(small_train)}, val={len(small_val)}")

    # ====== STEP 2: Embed + FAISS ======
    print("\n[2/6] Loading BGE-M3 embedding model...")
    from sentence_transformers import SentenceTransformer
    import faiss

    embedder = SentenceTransformer(EMBEDDING_MODEL_ID, device='cuda')
    print(f"  Embedder ready")

    print("  Embedding smoke set...")
    smoke_embeddings = embedder.encode(
        small_train['input'].tolist(),
        batch_size=32, show_progress_bar=False, normalize_embeddings=True,
    )
    smoke_index = faiss.IndexFlatIP(smoke_embeddings.shape[1])
    smoke_index.add(smoke_embeddings.astype('float32'))
    print(f"  FAISS index: {smoke_index.ntotal} vectors, dim {smoke_embeddings.shape[1]}")

    # ====== STEP 3: Test retrieval ======
    print("\n[3/6] Testing retrieval...")
    test_q = small_val.iloc[0]['input']
    test_emb = embedder.encode([test_q], normalize_embeddings=True)
    D, I = smoke_index.search(test_emb.astype('float32'), 3)
    print(f"  Query: {test_q[:80]}")
    print(f"  Top-3 retrieved (similarities: {[round(x, 3) for x in D[0].tolist()]}):")
    for i, idx in enumerate(I[0]):
        print(f"    {i+1}. {small_train.iloc[idx]['input'][:80]}")

    # ====== STEP 4: Precompute retrievals ======
    print("\n[4/6] Precomputing retrievals (leave-one-out)...")
    D, I = smoke_index.search(smoke_embeddings.astype('float32'), 3)
    retrievals = {}
    for i in range(len(small_train)):
        retrieved_idxs = [int(idx) for idx in I[i] if int(idx) != i][:2]
        retrievals[i] = [
            {'input': small_train.iloc[idx]['input'],
             'output': str(small_train.iloc[idx]['output'])}
            for idx in retrieved_idxs
        ]

    val_embeddings = embedder.encode(
        small_val['input'].tolist(),
        batch_size=32, show_progress_bar=False, normalize_embeddings=True,
    )
    D, I = smoke_index.search(val_embeddings.astype('float32'), 2)
    val_retrievals = {}
    for i in range(len(small_val)):
        retrieved_idxs = [int(idx) for idx in I[i][:2]]
        val_retrievals[i] = [
            {'input': small_train.iloc[idx]['input'],
             'output': str(small_train.iloc[idx]['output'])}
            for idx in retrieved_idxs
        ]

    # Free embedder memory
    del embedder
    torch.cuda.empty_cache()

    sample_prompt = make_rag_prompt(
        small_train.iloc[0]['input'],
        small_train.iloc[0]['subset'],
        retrievals[0]
    )
    print(f"\n  Sample prompt (first 600 chars):")
    print("  " + "-"*68)
    for line in sample_prompt[:600].split("\n"):
        print(f"  {line}")
    print("  " + "-"*68)

    # ====== STEP 5: Load model ======
    print(f"\n[5/6] Loading {MODEL_ID} (4-bit)...")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

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
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ====== STEP 6: Train + inference ======
    print("\n[6/6] Training on smoke subset...")
    train_dataset = RAGDataset(small_train, retrievals, tokenizer, max_length=1024)
    val_dataset = RAGDataset(small_val, val_retrievals, tokenizer, max_length=1024)
    data_collator = DataCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=20,
        learning_rate=3e-5,
        bf16=False, fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        seed=42,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=data_collator, tokenizer=tokenizer,
    )

    trainer.train()

    initial_loss = None
    final_loss = None
    final_eval_loss = None
    for log in trainer.state.log_history:
        if 'loss' in log:
            if initial_loss is None:
                initial_loss = log['loss']
            final_loss = log['loss']
        if 'eval_loss' in log:
            final_eval_loss = log['eval_loss']

    print(f"\n  Initial loss: {initial_loss:.3f}")
    print(f"  Final loss:   {final_loss:.3f}")
    print(f"  Eval loss:    {final_eval_loss:.3f}")
    print(f"  GPU peak: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    # Inference test
    print("\n  Testing inference on 3 samples...")
    model.eval()
    for i, row in small_val.head(3).iterrows():
        prompt_text = make_rag_prompt(row['input'], row['subset'], val_retrievals[i])
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors='pt', max_length=1024,
                          truncation=True).to(model.device)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model.generate(
                    **inputs, max_new_tokens=200, do_sample=False, num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )
        gen = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
        )
        print(f"\n  [{row['subset']}]")
        print(f"  Q: {row['input'][:100]}")
        print(f"  Expected: {str(row['output'])[:120]}")
        print(f"  Generated: {gen[:200]}")

    passed = (initial_loss is not None and final_loss is not None
              and final_loss < initial_loss)
    print("\n" + "="*70)
    print("SMOKE TEST PASSED" if passed else "SMOKE TEST FAILED")
    print("="*70)


if __name__ == "__main__":
    main()
