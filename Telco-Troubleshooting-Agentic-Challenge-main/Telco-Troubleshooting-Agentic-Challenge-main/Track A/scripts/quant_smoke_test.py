"""
scripts/quant_smoke_test.py

Verifies Qwen3.5-35B-A3B will load in 4-bit (NF4 + double-quant + fp16
compute) and generate on this hardware before we commit to the strategy.

Run AFTER scripts/setup_environment.sh, with the 'telco' conda env active.

    conda activate telco
    python scripts/quant_smoke_test.py

Exit code: 0 on success, 1 on failure.
"""
from __future__ import annotations

import gc
import os
import sys
import time
import traceback

# Silence tokenizer fork warnings before transformers import
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3.5-35B-A3B")
PROMPT = "Hello, please respond with exactly: TEST OK"
MAX_NEW_TOKENS = 256
HEADER = "=" * 64


def banner(title: str) -> None:
    print()
    print(HEADER)
    print(title)
    print(HEADER)


def gpu_mem_lines(prefix: str) -> str:
    if not torch.cuda.is_available():
        return f"{prefix}: no CUDA"
    parts = [prefix + ":"]
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        used_gb = (total - free) / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        parts.append(f"  GPU{i}: {used_gb:6.2f} GB used / {total_gb:6.2f} GB total")
    return "\n".join(parts)


def per_gpu_used_gb() -> list[float]:
    if not torch.cuda.is_available():
        return []
    out: list[float] = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        out.append((total - free) / (1024 ** 3))
    return out


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def main() -> int:
    banner("Qwen3.5-35B-A3B 4-bit smoke test")
    print(f"Model:           {MODEL_ID}")
    print(f"Torch:           {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices:    {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024 ** 3)
            print(f"  GPU{i}: {props.name} ({total_gb:.1f} GB)")
    else:
        print("FAILED: no CUDA device available")
        return 1

    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        print(f"FAILED: missing dependency: {exc}")
        print("Run scripts/setup_environment.sh first.")
        return 1

    banner("[1/4] Build BitsAndBytesConfig (4-bit NF4, fp16 compute, double-quant)")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print("  load_in_4bit              = True")
    print("  bnb_4bit_compute_dtype    = torch.float16")
    print("  bnb_4bit_quant_type       = 'nf4'")
    print("  bnb_4bit_use_double_quant = True")

    banner("[2/4] Load tokenizer")
    tok_t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  tokenizer loaded in {time.time() - tok_t0:.1f}s")

    banner("[3/4] Load model in 4-bit (device_map='auto')")
    print(gpu_mem_lines("BEFORE LOAD"))
    load_t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"FAILED loading model: {exc}")
        traceback.print_exc()
        cleanup()
        return 1
    load_secs = time.time() - load_t0
    print(f"  model loaded in {load_secs:.1f}s")
    print(gpu_mem_lines("AFTER LOAD"))
    model.eval()

    banner(f"[4/4] Generate {MAX_NEW_TOKENS} tokens on the test prompt")
    print(f"Prompt: {PROMPT!r}")
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    gen_t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    except Exception as exc:
        print(f"FAILED during generate: {exc}")
        traceback.print_exc()
        del model
        cleanup()
        return 1
    gen_secs = time.time() - gen_t0
    new_tokens = int(out.shape[-1] - inputs["input_ids"].shape[-1])
    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )

    used_per_gpu = per_gpu_used_gb()
    max_used = max(used_per_gpu) if used_per_gpu else 0.0
    sum_used = sum(used_per_gpu)

    banner("Generated text")
    print(response.strip() or "(empty response)")

    banner("RESULT")
    print(
        f"SMOKE TEST PASSED: model loaded in 4-bit, generated {new_tokens} tokens "
        f"in {gen_secs:.2f} seconds, {max_used:.2f} GB used per GPU "
        f"(peak across GPUs; total across {len(used_per_gpu)} GPUs = {sum_used:.2f} GB)"
    )
    print(f"Load time: {load_secs:.1f}s | Tokens/sec: {new_tokens / gen_secs:.2f}")

    del model
    cleanup()
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception as exc:  # noqa: BLE001
        print(f"SMOKE TEST FAILED with unhandled exception: {exc}")
        traceback.print_exc()
    finally:
        cleanup()
    sys.exit(rc)
