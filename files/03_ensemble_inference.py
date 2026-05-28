#!/usr/bin/env python3
"""
PATH C - PHASE 3: ENSEMBLE INFERENCE WITH RAG + SELF-CONSISTENCY

For each test question:
1. Retrieve top-3 similar Q&A pairs from training data
2. Build RAG-aware prompt with retrieved few-shot examples
3. Generate N candidates with temperature (self-consistency)
4. Select best candidate by avg ROUGE-L against retrieved exemplars
   (this favors outputs that share vocabulary with similar examples)
5. Optionally ensemble across two models (if MODEL_B provided)
6. Final pick: candidate with highest score

Time: ~6h for one model, ~12h for two models
Cost: ~$3 / $6

Usage:
  # Single model
  python3 03_ensemble_inference.py /workspace/path_c/models/gemma-2-27b-it-rag

  # Two-model ensemble
  python3 03_ensemble_inference.py \
      /workspace/path_c/models/gemma-2-27b-it-rag \
      /workspace/path_c/models/medgemma-27b-text-it-rag

Outputs:
  /workspace/path_c/submission_v3.csv          (final ensembled submission)
  /workspace/path_c/predictions_model_a.json   (model A raw outputs)
  /workspace/path_c/predictions_model_b.json   (model B raw outputs, if 2 models)
"""

import os
import sys
import json
import re
from pathlib import Path

HF_TOKEN = "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"
os.environ["HF_TOKEN"] = HF_TOKEN

import numpy as np
import pandas as pd
import torch

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

from tqdm import tqdm
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

WORKSPACE = os.environ.get("WORKSPACE", os.path.expanduser("~/zindi"))
PATH_C_DIR = f"{WORKSPACE}/path_c"
N_CANDIDATES = 3  # self-consistency: generate N candidates per question
TOP_K_RETRIEVE = 3  # number of retrieved examples in prompt

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
    """MUST match training prompt exactly"""
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


def clean(text):
    text = re.sub(r'<pad>|</s>|<unk>|<s>|<start_of_turn>|<end_of_turn>', '', text)
    text = re.sub(r'^Answer in \w+:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^Answer:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_for_rouge(s):
    """Simple whitespace + punctuation tokenization for ROUGE-style scoring"""
    return re.findall(r'\w+', s.lower())


def rouge_l_f1(candidate, reference):
    """Simple ROUGE-L F1 (longest common subsequence)"""
    c_tokens = tokenize_for_rouge(candidate)
    r_tokens = tokenize_for_rouge(reference)
    if not c_tokens or not r_tokens:
        return 0.0
    # LCS dynamic programming
    n, m = len(c_tokens), len(r_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if c_tokens[i-1] == r_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[n][m]
    if lcs == 0:
        return 0.0
    p = lcs / n
    r = lcs / m
    return 2 * p * r / (p + r)


def score_candidate(candidate, retrieved_examples):
    """Score = mean ROUGE-L against retrieved exemplar answers."""
    if not retrieved_examples:
        return 0.0
    scores = [rouge_l_f1(candidate, ex['output']) for ex in retrieved_examples]
    return float(np.mean(scores))


def load_model(adapter_path, base_model_id=None):
    """Load a fine-tuned model. Auto-detects base model from adapter_config.json"""
    print(f"\nLoading model from {adapter_path}...")

    # Auto-detect base model
    if base_model_id is None:
        config_path = os.path.join(adapter_path, "adapter_config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        base_model_id = cfg.get("base_model_name_or_path")
        if not base_model_id:
            raise ValueError(f"Could not find base_model in {config_path}")
    print(f"  Base model: {base_model_id}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_storage=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"  Loaded ({torch.cuda.memory_allocated() / 1e9:.1f} GB)")
    return model, tokenizer, base_model_id


def generate_candidates(model, tokenizer, prompt, max_new_tokens, n_candidates):
    """Generate N candidates: 1 greedy + (N-1) sampled"""
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors='pt', max_length=1024,
                       truncation=True).to(model.device)

    candidates = []

    # 1 greedy
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1, use_cache=True,
            )
    gen = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
    )
    candidates.append(clean(gen))

    # N-1 sampled with diversity
    for _ in range(n_candidates - 1):
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=True, temperature=0.7, top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1, use_cache=True,
                )
        gen = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
        )
        candidates.append(clean(gen))

    return candidates


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 03_ensemble_inference.py <model_a_path> [model_b_path]")
        return

    model_a_path = sys.argv[1]
    model_b_path = sys.argv[2] if len(sys.argv) > 2 else None

    print("="*70)
    print("PATH C - PHASE 3: ENSEMBLE INFERENCE")
    print(f"Model A: {model_a_path}")
    if model_b_path:
        print(f"Model B: {model_b_path}")
    else:
        print("Single-model mode (self-consistency only)")
    print("="*70)
    login(token=HF_TOKEN, add_to_git_credential=False)

    # ====== Load Phase 1 outputs ======
    print("\nLoading Phase 1 outputs...")
    train_df = pd.read_parquet(f"{PATH_C_DIR}/train_metadata.parquet")
    with open(f"{PATH_C_DIR}/test_retrievals.json") as f:
        test_retrievals = json.load(f)
    with open(f"{PATH_C_DIR}/data_analysis.json") as f:
        analysis = json.load(f)
    recommended_tokens = analysis.get('recommended_tokens', {})
    print(f"  Train ref: {len(train_df):,}")
    print(f"  Test retrievals: {len(test_retrievals)}")

    test_df = pd.read_csv(f"{WORKSPACE}/Test.csv").dropna(
        subset=['input', 'subset']).reset_index(drop=True)
    print(f"  Test: {len(test_df)}")

    # ====== Model A inference ======
    print("\n" + "="*70)
    print("MODEL A INFERENCE")
    print("="*70)
    model_a, tokenizer_a, _ = load_model(model_a_path)

    predictions_a = []
    candidates_a_all = []
    errors_a = 0

    for idx in tqdm(range(len(test_df)), desc="Model A"):
        row = test_df.iloc[idx]
        subset = row['subset']
        max_tokens = recommended_tokens.get(subset, 200)

        retrieved_idxs = test_retrievals.get(str(idx), [])[:TOP_K_RETRIEVE]
        retrieved = [
            {'input': train_df.iloc[int(i)]['input'],
             'output': str(train_df.iloc[int(i)]['output'])}
            for i in retrieved_idxs
        ]

        prompt = make_rag_prompt(row['input'], subset, retrieved)

        try:
            candidates = generate_candidates(
                model_a, tokenizer_a, prompt, max_tokens, N_CANDIDATES
            )
            scores = [score_candidate(c, retrieved) for c in candidates]
            best_idx = int(np.argmax(scores))
            best = candidates[best_idx]
            if not best or len(best) < 5:
                best = "Please consult a healthcare professional for accurate information."
                errors_a += 1
        except Exception as e:
            errors_a += 1
            if errors_a <= 3:
                print(f"\nModel A error at {idx}: {str(e)[:80]}")
            best = "Please consult a healthcare professional for accurate information."
            candidates = [best]
            scores = [0.0]

        predictions_a.append(best)
        candidates_a_all.append({
            'idx': idx, 'subset': subset,
            'candidates': candidates, 'scores': scores,
            'selected': best
        })

        if (idx + 1) % 100 == 0:
            with open(f"{PATH_C_DIR}/predictions_model_a.json", "w") as f:
                json.dump(candidates_a_all, f, ensure_ascii=False)
            temp = pd.DataFrame({
                'ID': test_df['ID'][:len(predictions_a)],
                'TargetRLF1': predictions_a,
                'TargetR1F1': predictions_a,
                'TargetLLM': predictions_a,
            })
            temp.to_csv(f"{PATH_C_DIR}/submission_model_a_checkpoint.csv", index=False)

    with open(f"{PATH_C_DIR}/predictions_model_a.json", "w") as f:
        json.dump(candidates_a_all, f, ensure_ascii=False)

    print(f"\nModel A done. Errors: {errors_a}")

    # Save single-model submission
    submission_a = pd.DataFrame({
        'ID': test_df['ID'],
        'TargetRLF1': predictions_a,
        'TargetR1F1': predictions_a,
        'TargetLLM': predictions_a,
    })
    submission_a.to_csv(f"{PATH_C_DIR}/submission_model_a.csv", index=False)
    print(f"Saved submission_model_a.csv")

    # Free Model A
    del model_a, tokenizer_a
    torch.cuda.empty_cache()

    # ====== Model B inference (if provided) ======
    if not model_b_path:
        print("\nSingle-model mode. Final submission:")
        print(f"  {PATH_C_DIR}/submission_model_a.csv")
        return

    print("\n" + "="*70)
    print("MODEL B INFERENCE")
    print("="*70)
    model_b, tokenizer_b, _ = load_model(model_b_path)

    predictions_b = []
    candidates_b_all = []
    errors_b = 0

    for idx in tqdm(range(len(test_df)), desc="Model B"):
        row = test_df.iloc[idx]
        subset = row['subset']
        max_tokens = recommended_tokens.get(subset, 200)

        retrieved_idxs = test_retrievals.get(str(idx), [])[:TOP_K_RETRIEVE]
        retrieved = [
            {'input': train_df.iloc[int(i)]['input'],
             'output': str(train_df.iloc[int(i)]['output'])}
            for i in retrieved_idxs
        ]
        prompt = make_rag_prompt(row['input'], subset, retrieved)

        try:
            candidates = generate_candidates(
                model_b, tokenizer_b, prompt, max_tokens, N_CANDIDATES
            )
            scores = [score_candidate(c, retrieved) for c in candidates]
            best_idx = int(np.argmax(scores))
            best = candidates[best_idx]
            if not best or len(best) < 5:
                best = "Please consult a healthcare professional for accurate information."
                errors_b += 1
        except Exception as e:
            errors_b += 1
            best = "Please consult a healthcare professional for accurate information."
            candidates = [best]
            scores = [0.0]

        predictions_b.append(best)
        candidates_b_all.append({
            'idx': idx, 'subset': subset,
            'candidates': candidates, 'scores': scores,
            'selected': best
        })

        if (idx + 1) % 100 == 0:
            with open(f"{PATH_C_DIR}/predictions_model_b.json", "w") as f:
                json.dump(candidates_b_all, f, ensure_ascii=False)

    with open(f"{PATH_C_DIR}/predictions_model_b.json", "w") as f:
        json.dump(candidates_b_all, f, ensure_ascii=False)
    print(f"\nModel B done. Errors: {errors_b}")

    # ====== Ensemble: pick best across both models ======
    print("\n" + "="*70)
    print("ENSEMBLE: SELECTING BEST PER QUESTION")
    print("="*70)

    final_predictions = []
    a_wins = 0
    b_wins = 0
    for i in range(len(test_df)):
        retrieved_idxs = test_retrievals.get(str(i), [])[:TOP_K_RETRIEVE]
        retrieved = [
            {'input': train_df.iloc[int(idx)]['input'],
             'output': str(train_df.iloc[int(idx)]['output'])}
            for idx in retrieved_idxs
        ]
        pred_a = predictions_a[i]
        pred_b = predictions_b[i]
        score_a = score_candidate(pred_a, retrieved)
        score_b = score_candidate(pred_b, retrieved)

        if score_a >= score_b:
            final_predictions.append(pred_a)
            a_wins += 1
        else:
            final_predictions.append(pred_b)
            b_wins += 1

    print(f"Model A won: {a_wins} / {len(test_df)}")
    print(f"Model B won: {b_wins} / {len(test_df)}")

    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'TargetRLF1': final_predictions,
        'TargetR1F1': final_predictions,
        'TargetLLM': final_predictions,
    })
    out_path = f"{PATH_C_DIR}/submission_v3.csv"
    submission.to_csv(out_path, index=False)

    lens = [len(p) for p in final_predictions]
    print(f"\nFinal submission stats:")
    print(f"  Total: {len(submission)}")
    print(f"  Avg length: {sum(lens)/len(lens):.0f} chars")
    print(f"  Min/Max length: {min(lens)}/{max(lens)}")
    print(f"\nFile: {out_path}")
    print("\nUpload submission_v3.csv to Zindi.")


if __name__ == "__main__":
    main()
