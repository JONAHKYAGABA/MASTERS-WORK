"""
scripts/llm_server.py

Minimal OpenAI-compatible chat-completions server backed by transformers
+ bitsandbytes 4-bit quantization. Designed for the RTX 8000 / Turing setup
where vLLM v1 is broken (it requires Ampere+ and trips on Python 3.13's
multiprocessing).

Endpoints:
    GET  /health                 — readiness probe
    GET  /v1/models              — OpenAI-compatible model list
    POST /v1/chat/completions    — OpenAI-compatible non-streaming chat

Tool-call handling: Qwen-style <tool_call>{...}</tool_call> blocks are parsed
out of the generation and returned as OpenAI tool_calls. If the request
includes a `tools` field, the chat template renders them; otherwise tool
calls are passthrough text.

This is NOT a vLLM replacement. No continuous batching, no streaming, no
PagedAttention. It's a small, reliable shim adequate for our experiment.

Run via:  python scripts/llm_server.py --model Qwen/Qwen3.5-35B-A3B --port 8001
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [llm_server] %(message)s",
)
log = logging.getLogger("llm_server")

# Globals filled by load_model()
_MODEL: Optional[Any] = None
_TOKENIZER: Optional[Any] = None
_MODEL_ID: Optional[str] = None


# --------------------------------------------------------------------------- API

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    stream: Optional[bool] = False
    n: Optional[int] = 1
    stop: Optional[Any] = None


app = FastAPI(title="Telco LLM Shim", version="0.1.0")


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok" if _MODEL is not None else "loading",
        "model": _MODEL_ID,
    }


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    if _MODEL_ID is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "object": "list",
        "data": [{
            "id": _MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }],
    }


_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    calls: List[Dict[str, Any]] = []
    for m in _TOOL_RE.finditer(text):
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = obj.get("name") or ""
        args = obj.get("arguments")
        if isinstance(args, dict):
            args_str = json.dumps(args, ensure_ascii=False)
        elif isinstance(args, str):
            args_str = args
        else:
            args_str = "{}"
        if not name:
            continue
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {"name": name, "arguments": args_str},
        })
    cleaned = _TOOL_RE.sub("", text).strip()
    return cleaned, calls


def msgs_to_dicts(msgs: List[ChatMessage]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in msgs:
        d: Dict[str, Any] = {"role": m.role}
        # Force a string content; some chat templates choke on None.
        d["content"] = m.content if m.content is not None else ""
        if m.tool_calls:
            d["tool_calls"] = m.tool_calls
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.name:
            d["name"] = m.name
        out.append(d)
    return out


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest) -> Dict[str, Any]:
    if _MODEL is None or _TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    msgs = msgs_to_dicts(req.messages)

    # apply_chat_template with tools=... handles tool descriptors for Qwen.
    try:
        prompt_ids = _TOKENIZER.apply_chat_template(
            msgs,
            tools=req.tools,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except Exception as exc:
        log.warning(f"apply_chat_template with tools failed: {exc}; falling back without tools")
        prompt_ids = _TOKENIZER.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    prompt_ids = prompt_ids.to(_MODEL.device)
    prompt_len = int(prompt_ids.shape[-1])

    do_sample = (req.temperature or 0.0) > 0.0
    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=min(int(req.max_tokens or 1024), 4096),
        do_sample=do_sample,
        pad_token_id=_TOKENIZER.pad_token_id or _TOKENIZER.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = max(float(req.temperature or 0.01), 0.01)
        gen_kwargs["top_p"] = float(req.top_p or 1.0)

    t0 = time.time()
    try:
        with torch.no_grad():
            out = _MODEL.generate(prompt_ids, **gen_kwargs)
    except Exception as exc:
        log.error(f"generate() failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"generate failed: {exc}")
    gen_secs = time.time() - t0

    new_tokens = out[0][prompt_len:]
    completion_tokens = int(new_tokens.shape[-1])
    text = _TOKENIZER.decode(new_tokens, skip_special_tokens=True)

    if req.tools:
        cleaned_text, tool_calls = parse_tool_calls(text)
    else:
        cleaned_text, tool_calls = text, []

    message: Dict[str, Any] = {"role": "assistant", "content": cleaned_text or None}
    finish_reason = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    log.info(
        f"chat in={prompt_len}t out={completion_tokens}t "
        f"{gen_secs:.1f}s tool_calls={len(tool_calls)}"
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _MODEL_ID,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_len + completion_tokens,
        },
    }


# --------------------------------------------------------------------------- model

def gpu_mem_summary() -> str:
    if not torch.cuda.is_available():
        return "no CUDA"
    parts = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        used = (total - free) / (1024 ** 3)
        tot = total / (1024 ** 3)
        parts.append(f"GPU{i}: {used:.1f}/{tot:.1f} GB")
    return "  ".join(parts)


def load_model(model_id: str) -> None:
    global _MODEL, _TOKENIZER, _MODEL_ID
    log.info(f"Loading tokenizer: {model_id}")
    _TOKENIZER = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if _TOKENIZER.pad_token_id is None and _TOKENIZER.eos_token_id is not None:
        _TOKENIZER.pad_token_id = _TOKENIZER.eos_token_id

    log.info(f"Loading model in 4-bit (NF4 + double-quant, fp16 compute): {model_id}")
    log.info(f"BEFORE LOAD: {gpu_mem_summary()}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    _MODEL.eval()
    _MODEL_ID = model_id
    log.info(f"Model loaded in {time.time() - t0:.1f}s")
    log.info(f"AFTER LOAD:  {gpu_mem_summary()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("LLM_PORT", "8001")))
    args = ap.parse_args()
    load_model(args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
