"""
scripts/finetune.py — LoRA fine-tune Qwen on the distilled corpus (Stage D).

Loads the base model in 4-bit, attaches a LoRA adapter via peft, trains
on traces/train_traces.jsonl with HuggingFace Trainer. Single-GPU
(CUDA_VISIBLE_DEVICES=0 by default — leave GPU 1 free for the LLM server).

Saves the best adapter to training/checkpoints/run_v1/best_lora/.

Run:
    CUDA_VISIBLE_DEVICES=0 python scripts/finetune.py \\
        --traces traces/train_traces.jsonl \\
        --output_dir training/checkpoints/run_v1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
sys.path.insert(0, str(PROJECT_DIR))


SYSTEM_PROMPT_PATH = PROJECT_DIR / "prompts" / "system_prompt.md"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_traces(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def build_text(rec: Dict[str, Any], system_prompt: str) -> Dict[str, str]:
    """Build a single training example. Returns {'prompt': str, 'completion': str}."""
    options_text = "\n".join(f"{o['id']}: {o['label']}" for o in rec.get("options", []))
    user_msg = (
        f"{rec.get('input', '').strip()}\n\n"
        f"## Task\n{rec.get('task', '').strip()}\n\n"
        f"## Options\n{options_text}"
    )
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    completion = f"{rec.get('trace', '').strip()}<|im_end|>"
    return {"prompt": prompt, "completion": completion}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="traces/train_traces.jsonl")
    ap.add_argument("--output_dir", default="training/checkpoints/run_v1")
    ap.add_argument("--base_model", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B"))
    ap.add_argument("--val_size", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--per_device_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="5-step smoke test (verify the pipeline before committing GPU-hours)")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------- data ----------
    traces_path = Path(args.traces)
    if not traces_path.exists():
        print(f"FATAL: {traces_path} not found", file=sys.stderr)
        return 1
    raw = load_traces(traces_path)
    print(f"[finetune] loaded {len(raw)} traces from {traces_path}")
    if len(raw) < 50:
        print("FATAL: fewer than 50 traces — distillation hasn't produced enough yet", file=sys.stderr)
        return 1

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8") if SYSTEM_PROMPT_PATH.exists() else ""
    examples = [build_text(r, system_prompt) for r in raw]

    # Stratified-ish split: 200 (or args.val_size) val, rest train
    rng = random.Random(args.seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    val_size = min(args.val_size, max(20, len(examples) // 10))
    val_idx = set(indices[:val_size])
    train_recs = [examples[i] for i in indices if i not in val_idx]
    val_recs = [examples[i] for i in indices if i in val_idx]
    print(f"[finetune] split: train={len(train_recs)} val={len(val_recs)}")

    # ---------- tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(rec: Dict[str, str]) -> Dict[str, Any]:
        prompt_ids = tokenizer(rec["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(rec["completion"], add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids[:]
        if len(full_ids) > args.max_seq_length:
            full_ids = full_ids[: args.max_seq_length]
            labels = labels[: args.max_seq_length]
        attn = [1] * len(full_ids)
        return {"input_ids": full_ids, "attention_mask": attn, "labels": labels}

    train_ds = Dataset.from_list(train_recs).map(tokenize, remove_columns=["prompt", "completion"])
    val_ds = Dataset.from_list(val_recs).map(tokenize, remove_columns=["prompt", "completion"])

    # ---------- model ----------
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print(f"[finetune] loading {args.base_model} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---------- collator ----------
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, attn, labels = [], [], []
        pad_id = tokenizer.pad_token_id or 0
        for b in batch:
            pad_n = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad_n)
            attn.append(b["attention_mask"] + [0] * pad_n)
            labels.append(b["labels"] + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    # ---------- training args ----------
    targs = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs if not args.smoke else 1,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=True,
        bf16=False,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch" if not args.smoke else "no",
        eval_strategy="epoch" if not args.smoke else "no",
        save_total_limit=2,
        load_best_model_at_end=not args.smoke,
        metric_for_best_model="eval_loss" if not args.smoke else None,
        report_to=[],
        seed=args.seed,
        max_steps=5 if args.smoke else -1,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
    )

    print("[finetune] starting training...")
    trainer.train()

    if not args.smoke:
        best_dir = out / "best_lora"
        best_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))
        print(f"[finetune] best LoRA saved to {best_dir}")
    else:
        print("[finetune] smoke test passed (5 steps).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
