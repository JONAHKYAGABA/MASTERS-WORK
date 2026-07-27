"""CPU-only inference wrapper for a quantized SSG-VQA-Net v2.

Objective (v) of the thesis: evaluate the quantized model's inference
performance and latency in CPU-only environments.

This is a stripped-down version of ``scripts/serve_app.py`` that:
  * loads ONE of the six quantized variants produced by
    ``scripts/quantize_and_export.py``,
  * runs the full VQA pipeline (SG generator + Qwen answer generation +
    grounding refinement) on CPU with no GPU dependency,
  * exposes a callable ``CPUPipeline`` class the benchmark harness can
    instantiate directly (no HTTP server) plus an optional CLI for
    interactive single-query inference.

Component precision policy matches ``quantize_and_export.py``: the Qwen
backbone runs at whatever precision the variant defines; the SG generator,
grounding head, and auxiliary heads run in FP16 (grounding's mHC block
in FP32).

Usage (interactive CLI):
    python scripts/serve_app_cpu.py \\
        --variant_dir ./quantized_models/q4_k_m \\
        --image path/to/xray.jpg \\
        --question "Is there a pleural effusion?"

Usage (as a library, from benchmark_cpu.py):
    from scripts.serve_app_cpu import CPUPipeline
    p = CPUPipeline.from_variant_dir(Path("./quantized_models/q4_k_m"))
    result = p(image=pil_img, question="Is there pneumothorax?")
    # result: {"answer": str, "bbox": [x1,y1,x2,y2], "think": str, ...}
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


_BOX_RE = re.compile(
    r"<box>\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*</box>"
)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


# ==========================================================================
# Helpers
# ==========================================================================
def _detect_variant_kind(variant_dir: Path) -> str:
    """Return one of {"gguf", "bnb", "fp16"} based on files in variant_dir."""
    if any(p.suffix == ".gguf" for p in variant_dir.iterdir()):
        return "gguf"
    # BitsAndBytes-quantized models save via HF with a quantization_config
    # entry in config.json.
    cfg_path = variant_dir / "config.json"
    if cfg_path.exists():
        try:
            import json
            with open(cfg_path) as f:
                cfg = json.load(f)
            if "quantization_config" in cfg:
                return "bnb"
        except Exception:
            pass
    return "fp16"


def _load_heads_sidecar(variant_dir: Path) -> Dict[str, torch.Tensor]:
    """Load the FP16 heads sidecar written by quantize_and_export.py."""
    st_path = variant_dir / "heads.safetensors"
    bin_path = variant_dir / "heads.bin"
    if st_path.exists():
        from safetensors.torch import load_file
        return load_file(str(st_path), device="cpu")
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu", weights_only=True)
    raise FileNotFoundError(
        f"No heads sidecar in {variant_dir} (looked for heads.safetensors "
        "and heads.bin). Re-run quantize_and_export.py against your Stage-4 "
        "checkpoint."
    )


def _parse_structured_output(text: str) -> Dict[str, Any]:
    """Extract <think>, <box>, <answer> from generated text."""
    think = _THINK_RE.search(text)
    box = _BOX_RE.search(text)
    ans = _ANSWER_RE.search(text)
    bbox = None
    if box is not None:
        try:
            coords = [float(box.group(i)) for i in (1, 2, 3, 4)]
            coords = [max(0.0, min(1.0, c)) for c in coords]
            if coords[2] > coords[0] and coords[3] > coords[1]:
                bbox = coords
        except (ValueError, IndexError):
            bbox = None
    return {
        "think": think.group(1).strip() if think else None,
        "bbox_raw": bbox,
        "answer": ans.group(1).strip() if ans else text.strip(),
        "raw_text": text,
    }


# ==========================================================================
# CPUPipeline: the top-level object the benchmark uses
# ==========================================================================
@dataclass
class CPUPipeline:
    """A CPU-only inference pipeline for SSG-VQA-Net v2.

    Instantiate via ``CPUPipeline.from_variant_dir(path)``; call as a
    function with (image, question) -> result dict.

    Attributes captured after each call (accessible via ``last_timing``):
        sg_gen_ms   -- SG generator forward pass
        ttft_ms     -- time to first LM token
        answer_ms   -- LM completion (excluding TTFT)
        refine_ms   -- grounding-head refinement
        total_ms    -- sum of the above (approximates end-to-end)
    """

    variant_dir: Path
    variant_kind: str
    num_threads: int
    sg_generator: Any = None       # models.sg_generators.SGGenerator
    sg_encoder: Any = None
    sg_projector: Any = None
    grounding_head: Any = None
    processor: Any = None
    qwen: Any = None               # HF causal LM (bnb / fp16) OR llama_cpp.Llama
    _placeholder_ids: List[int] = field(default_factory=list)
    last_timing: Dict[str, float] = field(default_factory=dict)

    # ----------------------------------------------------------------------
    @classmethod
    def from_variant_dir(cls, variant_dir: Path, num_threads: int = 8) -> "CPUPipeline":
        variant_dir = Path(variant_dir)
        if not variant_dir.exists():
            raise FileNotFoundError(f"variant_dir does not exist: {variant_dir}")

        kind = _detect_variant_kind(variant_dir)
        logger.info(f"loading variant kind={kind} from {variant_dir}")

        # Global torch thread pinning for CPU perf reproducibility.
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(min(2, num_threads))

        pipeline = cls(variant_dir=variant_dir, variant_kind=kind,
                       num_threads=num_threads)
        pipeline._load_heads()
        pipeline._load_qwen()
        pipeline._add_sg_placeholder_tokens(num_sg_tokens=8)
        return pipeline

    # ----------------------------------------------------------------------
    def _load_heads(self):
        """Build the SG generator + encoder + projector + grounding head +
        aux heads on CPU, then overlay weights from the sidecar."""
        from models.sg_generators import get_sg_generator
        from models.ssg_vqa_net_v2 import (
            AuxiliaryHeads, GroundingRefinementHead,
            SceneGraphEncoderV2, SGTokenProjector,
        )

        d_llm = self._discover_d_llm_from_config()
        logger.info(f"d_llm={d_llm} (from variant config)")

        # SG generator (image -> scene-graph dicts + raw tensors).
        # Default matches the Stage-1 config: TXRV DenseNet + DETR head.
        self.sg_generator = get_sg_generator(
            "txrv_detr", num_entities=22, num_regions=30
        ).eval()

        # SG encoder + projector: on-CPU FP16 modules.
        self.sg_encoder = SceneGraphEncoderV2(
            num_regions=30, num_entities=22, num_relations=10,
        ).eval()
        self.sg_projector = SGTokenProjector(
            d_node=128, d_llm=d_llm, num_tokens=8,
        ).eval()
        self.grounding_head = GroundingRefinementHead(
            d_llm=d_llm, d_sg=128,
        ).eval()
        self.aux_heads = AuxiliaryHeads(d_llm=d_llm).eval()

        # Overlay weights from sidecar.
        heads = _load_heads_sidecar(self.variant_dir)
        loaded_from_sidecar = 0
        for module_name in ("sg_generator", "sg_encoder", "sg_projector",
                            "grounding_head", "aux_heads"):
            sub = getattr(self, module_name)
            sub_state = {}
            prefix = f"{module_name}."
            for k, v in heads.items():
                if k.startswith(prefix):
                    sub_state[k[len(prefix):]] = v
            if sub_state:
                missing, unexpected = sub.load_state_dict(sub_state, strict=False)
                loaded_from_sidecar += len(sub_state)
                logger.info(
                    f"  {module_name}: loaded {len(sub_state)} keys "
                    f"(missing {len(missing)}, unexpected {len(unexpected)})"
                )
        logger.info(f"heads sidecar total keys loaded: {loaded_from_sidecar}")

        # Cast heads to FP16 except mHC (must stay FP32).
        for m in (self.sg_encoder, self.sg_projector, self.aux_heads):
            m.to(dtype=torch.float16)
        if hasattr(self.grounding_head, "mhc"):
            self.grounding_head.to(dtype=torch.float16)
            self.grounding_head.mhc.to(dtype=torch.float32)

    # ----------------------------------------------------------------------
    def _discover_d_llm_from_config(self) -> int:
        """Read hidden_size from variant's config.json (works for HF + GGUF)."""
        import json
        cfg_path = self.variant_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            for key in ("hidden_size", "n_embd", "d_model"):
                if key in cfg:
                    return int(cfg[key])
            tc = cfg.get("text_config", {})
            if isinstance(tc, dict) and "hidden_size" in tc:
                return int(tc["hidden_size"])
        # GGUF-only variants: fall back to known Qwen3-VL-8B hidden size.
        logger.warning("could not find hidden_size in config; assuming Qwen3-VL-8B (4096)")
        return 4096

    # ----------------------------------------------------------------------
    def _load_qwen(self):
        if self.variant_kind in ("fp16", "bnb"):
            self._load_qwen_hf()
        elif self.variant_kind == "gguf":
            self._load_qwen_gguf()
        else:
            raise ValueError(f"unsupported variant kind: {self.variant_kind}")

    def _load_qwen_hf(self):
        """Load HF-format Qwen (fp16 or bnb-quantized) on CPU."""
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(str(self.variant_dir))
        # Cap image pixels the same way the trainer does -- radiology
        # inputs are large and default limits OOM the LM's KV-cache on CPU.
        try:
            self.processor.image_processor.max_pixels = int(
                os.environ.get("QWEN_MAX_PIXELS", 448 * 448)
            )
            self.processor.image_processor.min_pixels = int(
                os.environ.get("QWEN_MIN_PIXELS", 256 * 256)
            )
        except AttributeError:
            pass

        logger.info(f"loading Qwen from {self.variant_dir} on CPU...")
        self.qwen = AutoModelForImageTextToText.from_pretrained(
            str(self.variant_dir),
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        self.qwen.eval()

    def _load_qwen_gguf(self):
        """Load a GGUF-quantized Qwen via llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for GGUF variants. Install with:\n"
                "    pip install 'llama-cpp-python[server]>=0.3.0'\n"
                "See scripts/README_quantization.md for build tips."
            ) from e

        # Find the .gguf file
        gguf_files = list(self.variant_dir.glob("*.gguf"))
        if not gguf_files:
            raise FileNotFoundError(f"no .gguf file in {self.variant_dir}")
        gguf_path = gguf_files[0]
        logger.info(f"loading GGUF: {gguf_path.name}")

        self.qwen = Llama(
            model_path=str(gguf_path),
            n_ctx=int(os.environ.get("LLAMA_N_CTX", 2048)),
            n_threads=self.num_threads,
            n_batch=int(os.environ.get("LLAMA_N_BATCH", 512)),
            use_mmap=True,        # bound RAM
            use_mlock=False,
            logits_all=False,
            embedding=False,
            verbose=False,
        )
        # For GGUF we can't use HF's AutoProcessor for the image path directly.
        # We keep a lightweight processor from the FP16 sidecar dir if present
        # (so the vision pre-processing lives in torch alongside the SG gen).
        fp16_neighbor = self.variant_dir.parent / "fp16"
        if fp16_neighbor.exists():
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(str(fp16_neighbor))
            try:
                self.processor.image_processor.max_pixels = 448 * 448
                self.processor.image_processor.min_pixels = 256 * 256
            except AttributeError:
                pass
            logger.info(f"borrowed processor from {fp16_neighbor}")
        else:
            raise FileNotFoundError(
                "GGUF variant needs the FP16 sibling dir for the vision processor. "
                f"Expected at {fp16_neighbor}. Re-run quantize_and_export.py with "
                "at least --variants fp16 <gguf_variant>."
            )

    # ----------------------------------------------------------------------
    def _add_sg_placeholder_tokens(self, num_sg_tokens: int):
        """Add <|sg_token_N|> placeholders to the tokenizer (parallels
        customvqamodel._add_sg_placeholder_tokens). For HF-backed variants
        we also resize the embedding matrix. For GGUF this is a no-op --
        llama.cpp cannot resize embeddings so we skip actual injection
        and rely on prompt-level SG-token summary instead (documented
        limitation)."""
        self._num_sg_tokens = num_sg_tokens
        tokens = [f"<|sg_token_{i}|>" for i in range(num_sg_tokens)]
        if self.variant_kind in ("fp16", "bnb"):
            tokenizer = self.processor.tokenizer
            n = tokenizer.add_special_tokens(
                {"additional_special_tokens": tokens}
            )
            if n > 0:
                base = (self.qwen.get_base_model()
                        if hasattr(self.qwen, "get_base_model") else self.qwen)
                base.resize_token_embeddings(len(tokenizer))
            self._placeholder_ids = [
                tokenizer.convert_tokens_to_ids(t) for t in tokens
            ]

    # ----------------------------------------------------------------------
    def __call__(self, image, question: str, indications: Optional[str] = None,
                 max_new_tokens: int = 256) -> Dict[str, Any]:
        """Single-query inference. Records per-component timing."""
        from PIL import Image
        self.last_timing = {}

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")

        prompt = self._build_prompt(question, indications)

        # ---- 1. SG generator forward (image -> {raw, dicts}) --------------
        t0 = time.perf_counter()
        sg_result = self._run_sg_generator(image)
        self.last_timing["sg_gen_ms"] = 1000 * (time.perf_counter() - t0)

        # ---- 2. LM generate -----------------------------------------------
        if self.variant_kind == "gguf":
            gen = self._run_gguf(prompt, sg_result, max_new_tokens)
        else:
            gen = self._run_hf(image, prompt, sg_result, max_new_tokens)

        # ---- 3. Grounding refinement (parse box, refine via mHC head) -----
        parsed = _parse_structured_output(gen["text"])
        t_refine = time.perf_counter()
        parsed["bbox_refined"] = self._refine_bbox(
            parsed.get("bbox_raw"), gen.get("pooled_hidden"), sg_result,
        )
        self.last_timing["refine_ms"] = 1000 * (time.perf_counter() - t_refine)

        parsed["timing"] = dict(self.last_timing)
        parsed["variant"] = str(self.variant_dir.name)
        return parsed

    # ----------------------------------------------------------------------
    def _build_prompt(self, question: str, indications: Optional[str]) -> str:
        if indications:
            question = f"Clinical context: {indications}. Question: {question}"
        sg_block = "".join(
            f"<|sg_token_{i}|>" for i in range(self._num_sg_tokens)
        )
        # Structured target format (matches customvqamodel._build_prompts).
        # Note: this is a simplified single-turn form; the full model uses
        # Qwen's chat template.
        return (
            f"<|im_start|>user\n"
            f"[image]\n{sg_block}\n{question}"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )

    # ----------------------------------------------------------------------
    def _run_sg_generator(self, image) -> Dict[str, Any]:
        """Run TXRV DETR on a single PIL image -> scene-graph dicts."""
        import numpy as np
        arr = np.asarray(image.convert("RGB").resize((224, 224)),
                         dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            out = self.sg_generator(tensor, return_dicts=True)
        return out

    # ----------------------------------------------------------------------
    def _run_hf(self, image, prompt: str, sg_result: Dict[str, Any],
                max_new_tokens: int) -> Dict[str, Any]:
        """HF-format LM generation with SG token injection."""
        from PIL import Image as _Image
        inputs = self.processor(text=prompt, images=[image],
                                return_tensors="pt", padding=True)
        with torch.no_grad():
            # Encode SG tokens and inject at placeholder positions.
            t_ttft = time.perf_counter()
            base_embed_fn = self.qwen.get_input_embeddings()
            inputs_embeds = base_embed_fn(inputs["input_ids"])
            sg_nodes, sg_mask = self.sg_encoder(sg_result["dicts"],
                                                inputs["input_ids"].device)
            sg_tokens = self.sg_projector(sg_nodes, sg_mask)
            inputs_embeds = self._inject_sg_tokens(
                inputs_embeds, inputs["input_ids"], sg_tokens
            )
            # First-token latency: run one forward to get logits for the
            # first output token, then continue with generate for the rest.
            first = self.qwen(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                return_dict=True,
                output_hidden_states=True,
            )
            self.last_timing["ttft_ms"] = 1000 * (time.perf_counter() - t_ttft)
            pooled = first.hidden_states[-1].mean(dim=1)

            # Continue generation greedily.
            t_ans = time.perf_counter()
            out = self.qwen.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
            self.last_timing["answer_ms"] = 1000 * (time.perf_counter() - t_ans)
        text = self.processor.tokenizer.decode(out[0], skip_special_tokens=True)
        return {"text": text, "pooled_hidden": pooled}

    def _inject_sg_tokens(self, inputs_embeds, input_ids, sg_tokens):
        """Scatter projected SG tokens into placeholder positions."""
        sg_ids = torch.as_tensor(self._placeholder_ids, device=input_ids.device)
        matches = input_ids.unsqueeze(1) == sg_ids.view(1, -1, 1)  # (B, K, L)
        positions = matches.long().argmax(dim=-1)                  # (B, K)
        B, K, D = sg_tokens.shape
        b_idx = torch.arange(B, device=input_ids.device).unsqueeze(1).expand(B, K)
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[b_idx, positions] = sg_tokens.to(inputs_embeds.dtype)
        return inputs_embeds

    # ----------------------------------------------------------------------
    def _run_gguf(self, prompt: str, sg_result: Dict[str, Any],
                  max_new_tokens: int) -> Dict[str, Any]:
        """GGUF backend (llama-cpp-python). SG token injection is done as
        a natural-language SG summary appended to the prompt because
        llama.cpp cannot ingest external embedding vectors mid-prompt.

        This is the documented tradeoff for the GGUF variant: we lose the
        soft-token pathway but gain 5-10x CPU speedup. The SG summary
        contains the same discrete information (entity, region,
        positiveness, bbox) so the LM can still reason over it.
        """
        # Build a textual SG summary from the scene-graph dicts.
        sg_text = self._sg_dicts_to_text(sg_result["dicts"])
        prompt_with_sg = prompt.replace(
            "".join(f"<|sg_token_{i}|>" for i in range(self._num_sg_tokens)),
            f"[scene_graph]\n{sg_text}\n[/scene_graph]",
        )

        t_ttft = time.perf_counter()
        # First token
        stream = self.qwen(
            prompt_with_sg,
            max_tokens=1,
            temperature=0.0,
            stream=False,
            echo=False,
        )
        self.last_timing["ttft_ms"] = 1000 * (time.perf_counter() - t_ttft)
        first_token = stream["choices"][0]["text"]

        # Continue
        t_ans = time.perf_counter()
        rest = self.qwen(
            prompt_with_sg + first_token,
            max_tokens=max_new_tokens - 1,
            temperature=0.0,
            stream=False,
            echo=False,
        )
        self.last_timing["answer_ms"] = 1000 * (time.perf_counter() - t_ans)
        rest_text = rest["choices"][0]["text"]
        return {"text": first_token + rest_text, "pooled_hidden": None}

    def _sg_dicts_to_text(self, dicts: List[Dict[str, Any]]) -> str:
        """Render a scene-graph dict as compact text for GGUF prompt."""
        lines = []
        for b, sg in enumerate(dicts):
            for i in range(sg.get("num_objects", 0)):
                ent = sg["entity_ids"][i] if "entity_ids" in sg else None
                reg = sg["region_ids"][i] if "region_ids" in sg else None
                pos = sg["positiveness"][i] if "positiveness" in sg else None
                bb = sg["bboxes"][i].tolist() if "bboxes" in sg else None
                lines.append(f"- entity={ent} region={reg} pos={pos} bbox={bb}")
        return "\n".join(lines) or "- (no observations detected)"

    # ----------------------------------------------------------------------
    def _refine_bbox(self, init_bbox, pooled_hidden, sg_result) -> Optional[List[float]]:
        """Run the mHC grounding refinement head; return refined bbox or None."""
        if pooled_hidden is None:
            # GGUF path currently emits no pooled hidden state; skip refinement.
            return init_bbox
        try:
            sg_nodes, sg_mask = self.sg_encoder(sg_result["dicts"],
                                                pooled_hidden.device)
            init_t = None
            if init_bbox is not None:
                init_t = torch.tensor([init_bbox], dtype=torch.float16,
                                       device=pooled_hidden.device)
            with torch.no_grad():
                out = self.grounding_head(
                    pooled_hidden.to(torch.float16),
                    sg_nodes.to(torch.float16),
                    sg_mask,
                    init_bbox=init_t,
                )
            return out["bbox_pred"][0].detach().cpu().tolist()
        except Exception as e:
            logger.warning(f"refinement failed: {e}; returning init_bbox")
            return init_bbox


# ==========================================================================
# CLI
# ==========================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant_dir", type=Path, required=True,
                   help="Path to one variant produced by quantize_and_export.py.")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--question", type=str, required=True)
    p.add_argument("--indications", type=str, default=None,
                   help="Optional clinical indication prefix.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--num_threads", type=int,
                   default=int(os.environ.get("OMP_NUM_THREADS", 8)))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s [%(levelname)s] %(message)s")

    # Force CPU-only
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    pipeline = CPUPipeline.from_variant_dir(args.variant_dir,
                                             num_threads=args.num_threads)

    from PIL import Image
    img = Image.open(args.image).convert("RGB")

    t_total = time.perf_counter()
    result = pipeline(img, args.question, indications=args.indications,
                      max_new_tokens=args.max_new_tokens)
    total_ms = 1000 * (time.perf_counter() - t_total)

    print("=" * 70)
    print(f"variant : {result.get('variant')}")
    print(f"answer  : {result['answer']}")
    if result.get("think"):
        print(f"think   : {result['think']}")
    print(f"bbox    : {result.get('bbox_refined')}")
    print(f"timing  : {result['timing']} (total {total_ms:.1f} ms)")
    print("=" * 70)


if __name__ == "__main__":
    main()
