"""
SSG-VQA-Net v2 — VLM-Backed Scene-Graph Guided VQA for Chest X-Ray

Drop-in replacement for the ConvNeXt + Bio_ClinicalBERT + SIM + from-scratch-decoder
architecture in mimic_vqa_model.py.

Key changes vs v1:
------------------
  - Qwen2.5-VL-7B (or 3B) as unified vision-language backbone, loaded in 4-bit
    NF4 (QLoRA) for Turing-generation GPUs (RTX 8000, V100) that lack bf16.
  - LoRA adapters on attention projections only. Base model frozen + quantized.
  - Scene graph injected as N_SG soft tokens spliced into Qwen's embedding
    stream, analogous to how QoQ-Med injects ECG-JEPA features.
  - Dedicated grounding refinement head with optional mHC fusion — produces
    higher-IoU bboxes than the LLM's native <box> output alone.
  - Aux classification heads (CheXpert + multi-head VQA) kept as low-weight
    auxiliary losses on pooled LLM hidden states.
  - Scene graph generator injected as a dependency (trained separately in
    Stage 1, frozen in Stages 2-4).

Forward output contract:
------------------------
Preserves the keys consumed by your existing training.loss.MultiTaskLoss and
training.metrics.VQAMetrics, so train_mimic_cxr.py and evaluate.py need only
minor edits (see MIGRATION NOTES at the bottom of this file).

Dataset changes required:
-------------------------
The batch must additionally include:
  - 'questions':     List[str]              — raw question text (for Qwen processor)
  - 'pil_images':    List[PIL.Image.Image]  — raw images for Qwen processor
  - 'answer_texts':  List[str]              — raw answer text in structured format
                                              "<think>...</think><box>x1,y1,x2,y2</box>
                                              <answer>...</answer>"
  - Existing 'images', 'input_ids', 'scene_graphs', 'chexpert_labels', etc.
    remain — 'images' is still used by the scene-graph generator pipeline.
The Qwen processor tokenizes questions+answers internally; the BERT input_ids
in your batches are ignored by v2.

Authors: migration spec 2026-04-24
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Optional dependencies — graceful failure with clear error messages
# -----------------------------------------------------------------------------

try:
    from transformers import (
        AutoProcessor,
        AutoModelForImageTextToText,
        BitsAndBytesConfig,
    )
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False

try:
    import bitsandbytes as bnb  # noqa: F401
    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False

# Legacy v1 components (mHC + SceneGraphGenerator) are inlined further down
# in this file (see "INLINED LEGACY V1 COMPONENTS" section). They were
# previously imported from a separate file; consolidating into one file keeps
# the project to a single model module.
_HAS_LEGACY = True


# Forward declarations — populated by the INLINED LEGACY V1 COMPONENTS block
# at the bottom of this file. Listed here so static-analysis tools can see the
# names that classes higher up in the file (e.g. GroundingRefinementHead) refer
# to. Python resolves these at instantiation time, not class-definition time.
SceneGraphGenerator = None  # type: ignore  # noqa: E305
mHCBlock = None  # type: ignore
RMSNorm = None  # type: ignore
HyperConnection = None  # type: ignore
ManifoldProjection = None  # type: ignore
sinkhorn_knopp = None  # type: ignore


# =============================================================================
# SCENE GRAPH ENCODER (v2) — relation-aware GAT
# =============================================================================


class RelationAwareGAT(nn.Module):
    """
    Relation-typed graph attention layer.

    One attention head per relation type plus a shared untyped head. Aggregates
    messages from neighbours weighted by learned per-relation attention.

    This is inline (no torch-geometric dependency). Objects are treated as a
    fully-connected graph; the relation-type tensor gates which pairs actually
    exchange messages.
    """

    def __init__(
        self,
        d_node: int = 128,
        num_relations: int = 10,
        num_shared_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_node = d_node
        self.num_relations = num_relations
        self.num_shared_heads = num_shared_heads
        self.total_heads = num_relations + num_shared_heads
        assert d_node % self.total_heads == 0, (
            f"d_node ({d_node}) must be divisible by total heads ({self.total_heads})"
        )
        self.d_head = d_node // self.total_heads

        self.q_proj = nn.Linear(d_node, d_node)
        self.k_proj = nn.Linear(d_node, d_node)
        self.v_proj = nn.Linear(d_node, d_node)
        self.out_proj = nn.Linear(d_node, d_node)

        self.norm = nn.LayerNorm(d_node)
        self.ffn = nn.Sequential(
            nn.Linear(d_node, d_node * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_node * 2, d_node),
        )
        self.ffn_norm = nn.LayerNorm(d_node)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        nodes: torch.Tensor,          # (B, N, d_node)
        relation_mask: torch.Tensor,  # (B, N, N, num_relations) one-hot or soft
        node_mask: torch.Tensor,      # (B, N) 1 where valid
    ) -> torch.Tensor:
        B, N, _ = nodes.shape

        q = self.q_proj(nodes).view(B, N, self.total_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(nodes).view(B, N, self.total_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(nodes).view(B, N, self.total_heads, self.d_head).transpose(1, 2)
        # q/k/v: (B, heads, N, d_head)

        # Relation-gated attention: for the first num_relations heads, mask
        # each head to the corresponding relation type. Shared heads attend
        # over all pairs.
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        # scores: (B, heads, N, N)

        # Apply relation gating to relation-typed heads (first num_relations)
        if self.num_relations > 0:
            # (B, N, N, num_relations) -> (B, num_relations, N, N)
            rel_gate = relation_mask.permute(0, 3, 1, 2)
            # Treat zero relation-mask entries as -inf to kill those edges
            typed_scores = scores[:, : self.num_relations]
            typed_scores = typed_scores.masked_fill(rel_gate < 1e-4, float("-inf"))
            scores = torch.cat(
                [typed_scores, scores[:, self.num_relations:]], dim=1
            )

        # Node padding mask — key side
        if node_mask is not None:
            key_mask = node_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N)
            scores = scores.masked_fill(~key_mask.bool(), float("-inf"))

        # Softmax with safe handling of all-masked rows (isolated nodes)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                           # (B, heads, N, d_head)
        out = out.transpose(1, 2).reshape(B, N, self.d_node)  # (B, N, d_node)
        out = self.out_proj(out)

        # Residual + FFN
        h = self.norm(nodes + self.dropout(out))
        h = self.ffn_norm(h + self.dropout(self.ffn(h)))
        return h


class SceneGraphEncoderV2(nn.Module):
    """
    Encodes scene-graph dicts into per-node features suitable for projection
    into the LLM's embedding space.

    Input dict keys per graph:
      bboxes:        List/ndarray of (x1, y1, x2, y2) normalized in [0, 1]
      entity_ids:    ints in [0, num_entities)
      region_ids:    ints in [0, num_regions)
      positiveness:  optional ints in {0, 1}
      relations:     optional (N, N, num_relations) tensor (soft or one-hot)
      num_objects:   int

    Output:
      node_features: (B, N_max, d_node)
      node_mask:     (B, N_max)
    """

    def __init__(
        self,
        num_regions: int = 310,
        num_entities: int = 237,
        num_relations: int = 10,
        d_node: int = 128,
        num_gat_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.d_node = d_node

        # Embeddings
        self.region_embed = nn.Embedding(num_regions + 1, d_node // 3, padding_idx=num_regions)
        self.entity_embed = nn.Embedding(num_entities + 1, d_node // 3, padding_idx=num_entities)
        self.pos_embed = nn.Embedding(3, d_node // 12)  # 0=neg, 1=pos, 2=unknown

        # Geometric bbox features: [x1, y1, x2, y2, w, h, area, aspect] -> d_node/4
        self.bbox_proj = nn.Sequential(
            nn.Linear(8, d_node // 4),
            nn.GELU(),
            nn.LayerNorm(d_node // 4),
        )

        # Combine all into d_node
        combined_dim = (d_node // 3) + (d_node // 3) + (d_node // 12) + (d_node // 4)
        self.combiner = nn.Sequential(
            nn.Linear(combined_dim, d_node),
            nn.LayerNorm(d_node),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Relation-aware GAT stack
        self.gat_layers = nn.ModuleList([
            RelationAwareGAT(d_node=d_node, num_relations=num_relations, dropout=dropout)
            for _ in range(num_gat_layers)
        ])

    def forward(
        self,
        scene_graphs: List[Dict[str, Any]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not scene_graphs:
            # Return a single empty-token batch so downstream doesn't crash
            return (
                torch.zeros(1, 1, self.d_node, device=device),
                torch.zeros(1, 1, device=device),
            )

        n_max = max(int(sg.get("num_objects", 0)) for sg in scene_graphs)
        n_max = max(n_max, 1)
        B = len(scene_graphs)

        # Allocate with padding indices
        region_ids = torch.full((B, n_max), self.num_regions, dtype=torch.long, device=device)
        entity_ids = torch.full((B, n_max), self.num_entities, dtype=torch.long, device=device)
        pos_ids = torch.full((B, n_max), 2, dtype=torch.long, device=device)
        bbox_feats = torch.zeros(B, n_max, 8, device=device)
        node_mask = torch.zeros(B, n_max, device=device)
        relations = torch.zeros(B, n_max, n_max, self.num_relations, device=device)

        for b, sg in enumerate(scene_graphs):
            n = int(sg.get("num_objects", 0))
            if n == 0:
                continue
            n = min(n, n_max)

            bboxes = torch.as_tensor(sg["bboxes"][:n], dtype=torch.float, device=device)
            x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            w = (x2 - x1).clamp(min=1e-6)
            h = (y2 - y1).clamp(min=1e-6)
            bbox_feats[b, :n] = torch.stack(
                [x1, y1, x2, y2, w, h, w * h, w / h], dim=-1
            )

            ent = torch.as_tensor(sg["entity_ids"][:n], dtype=torch.long, device=device)
            reg = torch.as_tensor(sg["region_ids"][:n], dtype=torch.long, device=device)
            entity_ids[b, :n] = ent.clamp(max=self.num_entities - 1)
            region_ids[b, :n] = reg.clamp(max=self.num_regions - 1)

            if "positiveness" in sg and sg["positiveness"] is not None:
                p = torch.as_tensor(sg["positiveness"][:n], dtype=torch.long, device=device)
                pos_ids[b, :n] = p.clamp(max=2)

            if "relations" in sg and sg["relations"] is not None:
                rel = torch.as_tensor(sg["relations"], dtype=torch.float, device=device)
                # rel might be (N, N, R) — crop to n
                relations[b, :n, :n, :] = rel[:n, :n, : self.num_relations]

            node_mask[b, :n] = 1.0

        # Combine features
        region_e = self.region_embed(region_ids)
        entity_e = self.entity_embed(entity_ids)
        pos_e = self.pos_embed(pos_ids)
        bbox_e = self.bbox_proj(bbox_feats)
        combined = torch.cat([region_e, entity_e, pos_e, bbox_e], dim=-1)
        nodes = self.combiner(combined)  # (B, N, d_node)

        # Apply GAT layers
        for gat in self.gat_layers:
            nodes = gat(nodes, relations, node_mask)

        return nodes, node_mask


# =============================================================================
# SG TOKEN PROJECTOR — compress node features to a fixed token budget
# =============================================================================


class SGTokenProjector(nn.Module):
    """
    Cross-attention pooling: K learned queries attend over scene-graph nodes,
    producing K soft tokens in the LLM's hidden space.
    """

    def __init__(
        self,
        d_node: int = 128,
        d_llm: int = 3584,
        num_tokens: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_llm = d_llm

        self.queries = nn.Parameter(torch.randn(num_tokens, d_node) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            d_node, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_node)
        self.ffn = nn.Sequential(
            nn.Linear(d_node, d_node * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_node * 4, d_node),
        )
        self.norm2 = nn.LayerNorm(d_node)

        # Project to LLM hidden dim
        self.out_proj = nn.Linear(d_node, d_llm)
        self.out_norm = nn.LayerNorm(d_llm)

    def forward(
        self,
        node_features: torch.Tensor,  # (B, N, d_node)
        node_mask: torch.Tensor,      # (B, N)
    ) -> torch.Tensor:
        B = node_features.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, K, d_node)

        key_padding_mask = ~node_mask.bool() if node_mask is not None else None
        attended, _ = self.cross_attn(
            q, node_features, node_features,
            key_padding_mask=key_padding_mask,
        )
        h = self.norm1(q + attended)
        h = self.norm2(h + self.ffn(h))
        tokens = self.out_norm(self.out_proj(h))   # (B, K, d_llm)
        return tokens


# =============================================================================
# GROUNDING REFINEMENT HEAD — priority #1 component
# =============================================================================


class GroundingRefinementHead(nn.Module):
    """
    Produces a refined bbox from:
      - LLM hidden state at the <box> token position (or at a reserved location)
      - Scene-graph node features (for anatomical priors)
      - Optional initial bbox from the LLM's native <box> output

    Optionally routes the fused signal through a single mHCBlock (Birkhoff
    manifold, n=4 paths) — the v1 mHC contribution is relocated here, where
    heterogeneous-source fusion actually benefits from manifold constraints.
    """

    def __init__(
        self,
        d_llm: int,
        d_sg: int,
        d_hidden: int = 512,
        num_heads: int = 8,
        use_mhc: bool = True,
        mhc_manifold: str = "birkhoff",
        num_mhc_paths: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.use_mhc = use_mhc and _HAS_LEGACY

        self.llm_proj = nn.Linear(d_llm, d_hidden)
        self.sg_proj = nn.Linear(d_sg, d_hidden)

        self.cross_attn = nn.MultiheadAttention(
            d_hidden, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_hidden)

        if self.use_mhc:
            self.mhc = mHCBlock(
                hidden_size=d_hidden,
                num_heads=num_heads,
                ff_dim=d_hidden * 4,
                num_hc_paths=num_mhc_paths,
                manifold_type=mhc_manifold,
                dropout=dropout,
                sinkhorn_iters=20,
            )
        else:
            self.mhc = None

        # Delta regression: input = [fused (d_hidden) | init_bbox (4)]
        self.delta_head = nn.Sequential(
            nn.Linear(d_hidden + 4, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 4),
        )

        # Pointing score (does a relevant bbox exist at all?)
        self.pointing_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(
        self,
        llm_hidden: torch.Tensor,          # (B, d_llm) pooled or at <box> position
        sg_features: torch.Tensor,         # (B, N, d_sg) per-node
        sg_mask: torch.Tensor,             # (B, N)
        init_bbox: Optional[torch.Tensor] = None,  # (B, 4) in [0,1] or None
    ) -> Dict[str, torch.Tensor]:
        B = llm_hidden.size(0)
        device = llm_hidden.device

        q = self.llm_proj(llm_hidden).unsqueeze(1)   # (B, 1, d_hidden)
        kv = self.sg_proj(sg_features)                # (B, N, d_hidden)

        key_padding_mask = ~sg_mask.bool() if sg_mask is not None else None
        attended, attn_weights = self.cross_attn(
            q, kv, kv,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        fused = self.cross_norm(q + attended).squeeze(1)  # (B, d_hidden)

        if self.mhc is not None:
            fused = self.mhc(fused.unsqueeze(1)).squeeze(1)

        # Initial bbox: if not provided, use a learned center anchor
        if init_bbox is None:
            init_bbox = torch.tensor([0.25, 0.25, 0.75, 0.75], device=device)
            init_bbox = init_bbox.unsqueeze(0).expand(B, -1)

        delta = self.delta_head(torch.cat([fused, init_bbox], dim=-1))
        refined = torch.sigmoid(init_bbox + delta * 0.3)  # bounded delta
        pointing = torch.sigmoid(self.pointing_head(fused))

        return {
            "bbox_pred": refined,            # (B, 4) normalized
            "pointing_score": pointing,      # (B, 1)
            "spatial_attention": attn_weights.squeeze(1),  # (B, N)
            "grounding_features": fused,     # (B, d_hidden)
        }


# =============================================================================
# AUXILIARY HEADS — kept as low-weight losses for stability & fallback
# =============================================================================


class AuxiliaryHeads(nn.Module):
    """
    CheXpert (14-class multi-label) + VQA multi-head (binary/category/region/
    severity), both fed from pooled LLM hidden states.
    """

    def __init__(
        self,
        d_llm: int,
        num_chexpert: int = 14,
        num_binary: int = 2,
        num_category: int = 14,
        num_region: int = 26,
        num_severity: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        def mlp(d_out: int, d_mid: int = 512) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_llm, d_mid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_mid, d_out),
            )

        self.chexpert = mlp(num_chexpert)
        self.binary = mlp(num_binary, d_mid=256)
        self.category = mlp(num_category)
        self.region = mlp(num_region)
        self.severity = mlp(num_severity, d_mid=128)

    def forward(self, pooled: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        chexpert_logits = self.chexpert(pooled)
        vqa_logits = {
            "binary": self.binary(pooled),
            "category": self.category(pooled),
            "region": self.region(pooled),
            "severity": self.severity(pooled),
        }
        return chexpert_logits, vqa_logits


# =============================================================================
# STRUCTURED OUTPUT PARSER — extract <think>/<box>/<answer> from text
# =============================================================================


_BOX_RE = re.compile(
    r"<box>\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*</box>"
)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def parse_structured_output(text: str) -> Dict[str, Any]:
    """Extract reasoning / bbox / answer from a model-generated string."""
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
        "bbox": bbox,
        "answer": ans.group(1).strip() if ans else text.strip(),
        "raw": text,
    }


# =============================================================================
# MAIN MODEL
# =============================================================================


class SSGVQANetV2(nn.Module):
    """
    Scene-Graph-Guided VQA model built around Qwen2.5-VL.

    Parameters
    ----------
    qwen_model_id : str
        HuggingFace model ID. Default Qwen/Qwen2.5-VL-7B-Instruct.
        Use Qwen/Qwen2.5-VL-3B-Instruct for smaller GPU budgets.
    use_quantization : bool
        If True, load base model in 4-bit NF4 (QLoRA). Recommended for 48GB
        Turing cards (RTX 8000, Titan RTX, Quadro 6000). Disable on A100/H100
        where full-precision LoRA is feasible.
    lora_rank : int
        LoRA rank for attention projections. 16 is a good default.
    num_sg_tokens : int
        Number of scene-graph soft tokens injected per sample. 8 is typical;
        larger values give more SG bandwidth but compete with image tokens.
    scene_graph_generator : nn.Module | None
        Pretrained SceneGraphGenerator (from v1). If None, a default is built
        and must be trained in Stage 1 before this model is useful. Frozen by
        default in all stages after Stage 1.
    freeze_sg_generator : bool
        Whether to freeze the SG generator. Should be True for Stages 2-4.
    training_mode : str
        One of {'sg_only', 'alignment', 'pretrain', 'finetune', 'rl'}.
        Controls which parameter groups require grad.
    """

    _SG_PLACEHOLDER_PREFIX = "<|sg_token_"

    def __init__(
        self,
        qwen_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        use_quantization: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        num_sg_tokens: int = 8,
        num_regions: int = 310,
        num_entities: int = 237,
        num_relations: int = 10,
        sg_node_dim: int = 128,
        sg_gat_layers: int = 2,
        num_chexpert: int = 14,
        num_binary: int = 2,
        num_category: int = 14,
        num_region_classes: int = 26,
        num_severity: int = 4,
        use_mhc_in_grounding: bool = True,
        mhc_manifold: str = "birkhoff",
        scene_graph_generator: Optional[nn.Module] = None,
        freeze_sg_generator: bool = True,
        training_mode: str = "pretrain",
        max_answer_length: int = 256,
        torch_dtype: torch.dtype = torch.float16,  # Turing: no bf16
    ):
        super().__init__()

        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for SSGVQANetV2. Install with:\n"
                "    pip install 'transformers>=4.45'"
            )
        if not _HAS_PEFT:
            raise ImportError(
                "peft is required for LoRA adapters. Install with:\n"
                "    pip install 'peft>=0.11'"
            )
        if use_quantization and not _HAS_BNB:
            raise ImportError(
                "bitsandbytes is required for 4-bit QLoRA. Install with:\n"
                "    pip install 'bitsandbytes>=0.43'"
            )

        self.qwen_model_id = qwen_model_id
        self.use_quantization = use_quantization
        self.num_sg_tokens = num_sg_tokens
        self.training_mode = training_mode
        self.max_answer_length = max_answer_length
        self.torch_dtype = torch_dtype
        self._lora_target_modules = lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj"
        ]

        # ---- 1. Load Qwen (quantized) + processor -----------------------------
        self.processor = AutoProcessor.from_pretrained(qwen_model_id)

        quant_config = None
        if use_quantization:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )

        self.qwen = AutoModelForImageTextToText.from_pretrained(
            qwen_model_id,
            quantization_config=quant_config,
            torch_dtype=torch_dtype,
            # Turing lacks FlashAttention-2 — use SDPA instead
            attn_implementation="sdpa",
            device_map=None,  # let the trainer handle placement
        )

        if use_quantization:
            self.qwen = prepare_model_for_kbit_training(
                self.qwen, use_gradient_checkpointing=True
            )

        # ---- 2. Apply LoRA ---------------------------------------------------
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=self._lora_target_modules,
            task_type="CAUSAL_LM",
        )
        self.qwen = get_peft_model(self.qwen, lora_cfg)

        # Discover LLM hidden size dynamically (7B: 3584, 3B: 2048)
        self.d_llm = self._discover_hidden_size()

        # ---- 3. Add SG placeholder tokens to the tokenizer -------------------
        self.sg_token_ids = self._add_sg_placeholder_tokens(num_sg_tokens)

        # Round-trip check: verify our placeholder strings really do tokenize
        # to the special-token ids we just registered. Qwen's processor
        # normalizes text before tokenization in some versions, which can
        # silently break SG injection. Catch that at startup, not at step 1.
        _round_trip_ids = self.processor.tokenizer(
            self._sg_placeholder_block(),
            add_special_tokens=False,
            return_tensors=None,
        )["input_ids"]
        for sg_id in self.sg_token_ids:
            if sg_id not in _round_trip_ids:
                raise RuntimeError(
                    f"SG placeholder id {sg_id} did not survive tokenization. "
                    f"Got token ids {_round_trip_ids} from string "
                    f"'{self._sg_placeholder_block()}'. The Qwen tokenizer is "
                    "probably normalising/splitting the marker. Fix "
                    "_SG_PLACEHOLDER_PREFIX or use a different sentinel."
                )

        # Cache the assistant-turn delimiter as a TOKEN SEQUENCE, not a
        # single id (Qwen BPE splits 'assistant' into multiple subtokens, and
        # the previous single-token compare silently produced cut=0 → loss
        # was computed on the entire prompt including image+SG tokens).
        # Stored once here so _mask_prompt_labels doesn't retokenize per batch.
        self._assistant_delim_ids: List[int] = self.processor.tokenizer(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
            return_tensors=None,
        )["input_ids"]
        if not self._assistant_delim_ids:
            raise RuntimeError(
                "Tokenizer produced an empty delimiter sequence for "
                "'<|im_start|>assistant\\n'. Label masking would mask every "
                "row. Verify Qwen's chat template format."
            )

        # Vision-path verification flag — flipped to True after the first
        # forward pass confirms pixel_values changed Qwen's logits.
        self._vision_path_verified = False

        # ---- 4. Scene graph pipeline -----------------------------------------
        if scene_graph_generator is None:
            # Discover Qwen ViT hidden size so the SG generator's RPN conv
            # accepts ViT feature maps directly (1280 for Qwen2.5-VL-7B/3B).
            base = (
                self.qwen.get_base_model()
                if hasattr(self.qwen, "get_base_model")
                else self.qwen
            )
            vit_cfg = getattr(base.config, "vision_config", None)
            vit_hidden = (
                int(vit_cfg.hidden_size) if vit_cfg is not None and hasattr(vit_cfg, "hidden_size")
                else 1280
            )
            self.sg_generator = SceneGraphGenerator(
                visual_dim=vit_hidden,
                hidden_size=768,
                num_entity_classes=num_entities,
                num_region_classes=num_regions,
                num_relationships=num_relations,
                max_objects=20,
                dropout=0.1,
            )
        else:
            self.sg_generator = scene_graph_generator

        self.freeze_sg_generator = freeze_sg_generator

        self.sg_encoder = SceneGraphEncoderV2(
            num_regions=num_regions,
            num_entities=num_entities,
            num_relations=num_relations,
            d_node=sg_node_dim,
            num_gat_layers=sg_gat_layers,
        )

        self.sg_projector = SGTokenProjector(
            d_node=sg_node_dim,
            d_llm=self.d_llm,
            num_tokens=num_sg_tokens,
        )

        # ---- 5. Grounding refinement head -----------------------------------
        self.grounding_head = GroundingRefinementHead(
            d_llm=self.d_llm,
            d_sg=sg_node_dim,
            use_mhc=use_mhc_in_grounding,
            mhc_manifold=mhc_manifold,
        )

        # ---- 6. Aux heads ----------------------------------------------------
        self.aux_heads = AuxiliaryHeads(
            d_llm=self.d_llm,
            num_chexpert=num_chexpert,
            num_binary=num_binary,
            num_category=num_category,
            num_region=num_region_classes,
            num_severity=num_severity,
        )

        # Apply training-mode freezing
        self.set_training_mode(training_mode)

    # ----------------------------------------------------------------------
    # Setup helpers
    # ----------------------------------------------------------------------

    def _discover_hidden_size(self) -> int:
        """Qwen2.5-VL-7B=3584, 3B=2048. Read from the loaded config."""
        cfg = self.qwen.config
        # PEFT wraps the config; walk through common names
        for attr in ("hidden_size", "d_model"):
            v = getattr(cfg, attr, None)
            if v is not None:
                return int(v)
        # Try nested text_config
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
            return int(text_cfg.hidden_size)
        raise RuntimeError("Could not discover LLM hidden size from Qwen config.")

    def _add_sg_placeholder_tokens(self, n: int) -> List[int]:
        """Add N placeholder special tokens and resize embeddings."""
        tokenizer = self.processor.tokenizer
        new_tokens = [f"{self._SG_PLACEHOLDER_PREFIX}{i}|>" for i in range(n)]
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": new_tokens}
        )
        if num_added > 0:
            # Resize embeddings on the base model (underneath PEFT wrapper)
            base = self.qwen.get_base_model() if hasattr(self.qwen, "get_base_model") else self.qwen
            base.resize_token_embeddings(len(tokenizer))
        return [tokenizer.convert_tokens_to_ids(t) for t in new_tokens]

    def _sg_placeholder_block(self) -> str:
        """Return the concatenated placeholder string used in chat templates."""
        return "".join(
            f"{self._SG_PLACEHOLDER_PREFIX}{i}|>" for i in range(self.num_sg_tokens)
        )

    # ----------------------------------------------------------------------
    # Training-mode control
    # ----------------------------------------------------------------------

    def set_training_mode(self, mode: str):
        """
        Modes:
          sg_only    — train SG generator + encoder + projector only
          alignment  — train SG encoder + projector + aux heads; Qwen frozen
          pretrain   — train LoRA + all new components; SG generator frozen
          finetune   — same as pretrain (the LR schedule differentiates them)
          rl         — same trainable set as finetune, used by GRPO outer loop
        """
        self.training_mode = mode

        def set_grad(module: nn.Module, flag: bool):
            for p in module.parameters():
                p.requires_grad = flag

        # Always freeze SG generator unless in sg_only mode
        set_grad(self.sg_generator, mode == "sg_only")

        # Qwen LoRA — active in pretrain/finetune/rl
        qwen_trainable = mode in {"pretrain", "finetune", "rl"}
        for name, p in self.qwen.named_parameters():
            if "lora_" in name:
                p.requires_grad = qwen_trainable
            else:
                p.requires_grad = False  # base weights never trained

        # SG encoder + projector
        train_sg_path = mode in {"sg_only", "alignment", "pretrain", "finetune", "rl"}
        set_grad(self.sg_encoder, train_sg_path and mode != "sg_only")
        set_grad(self.sg_projector, train_sg_path and mode != "sg_only")

        # Grounding head — active once we have a real signal
        set_grad(self.grounding_head, mode in {"pretrain", "finetune", "rl"})

        # Aux heads
        set_grad(self.aux_heads, mode in {"alignment", "pretrain", "finetune", "rl"})

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing on Qwen AND on the mHC sub-block."""
        if hasattr(self.qwen, "gradient_checkpointing_enable"):
            self.qwen.gradient_checkpointing_enable()
        # mHC has its own per-block flag (it doesn't honour torch.utils.checkpoint
        # globally). Without this propagation the grounding head missed out on
        # ~20% of recoverable activation memory.
        if getattr(self.grounding_head, "mhc", None) is not None:
            self.grounding_head.mhc._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Symmetric to gradient_checkpointing_enable."""
        if hasattr(self.qwen, "gradient_checkpointing_disable"):
            self.qwen.gradient_checkpointing_disable()
        if getattr(self.grounding_head, "mhc", None) is not None:
            self.grounding_head.mhc._gradient_checkpointing = False

    # ----------------------------------------------------------------------
    # Scene graph pipeline
    # ----------------------------------------------------------------------

    def _extract_qwen_vit_feature_maps(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run Qwen's ViT once and reshape its packed patch output into per-sample
        2D spatial grids that SceneGraphGenerator's Conv2d RPN can consume.

        Qwen2.5-VL ViT returns features with `spatial_merge_size` already
        applied (default 2), so for an input whose patchified grid is (H, W)
        the output token grid is (H // 2, W // 2).
        """
        base = self.qwen.get_base_model() if hasattr(self.qwen, "get_base_model") else self.qwen
        visual = base.visual

        # The processor emits pixel_values in fp32. Qwen ViT runs in fp16
        # (or whatever bnb_4bit_compute_dtype was set to). Some transformers
        # versions auto-upcast, others crash, and bitsandbytes-quantized
        # paths have been inconsistent across releases. Explicit cast = no
        # surprises.
        pixel_values = pixel_values.to(dtype=self.torch_dtype)

        ctx = torch.no_grad() if self.freeze_sg_generator else torch.enable_grad()
        with ctx:
            vit_out = visual(pixel_values, grid_thw=image_grid_thw)
            # vit_out: (total_merged_tokens, hidden_size)

        spatial_merge = getattr(base.config.vision_config, "spatial_merge_size", 2)

        feature_maps: List[torch.Tensor] = []
        offset = 0
        for b in range(image_grid_thw.size(0)):
            # Qwen2.5-VL usually returns (B, 3) per [T, H, W] but a couple of
            # versions return (B, 2) for still-image inputs (T implicit = 1).
            # Guard the unpack so a version skew doesn't crash here.
            row = image_grid_thw[b].tolist()
            if len(row) == 3:
                T, H, W = row
            elif len(row) == 2:
                T, H, W = 1, row[0], row[1]
            else:
                raise RuntimeError(
                    f"Unexpected image_grid_thw row length {len(row)} (value={row}); "
                    "Qwen ViT integration was written for (B, 3) or (B, 2)."
                )
            H_out = H // spatial_merge
            W_out = W // spatial_merge
            n_tokens = T * H_out * W_out
            sample = vit_out[offset : offset + n_tokens]
            offset += n_tokens

            if T == 1:
                fmap = sample.reshape(H_out, W_out, -1).permute(2, 0, 1)
            else:
                fmap = sample.reshape(T, H_out, W_out, -1)[T // 2].permute(2, 0, 1)
            feature_maps.append(fmap)

        # Pad to common spatial size (dynamic resolution gives variable grids)
        max_h = max(f.shape[1] for f in feature_maps)
        max_w = max(f.shape[2] for f in feature_maps)
        C = feature_maps[0].shape[0]
        padded = torch.zeros(
            len(feature_maps), C, max_h, max_w,
            dtype=feature_maps[0].dtype,
            device=feature_maps[0].device,
        )
        for b, f in enumerate(feature_maps):
            padded[b, :, : f.shape[1], : f.shape[2]] = f
        return padded

    def _sg_outputs_to_dicts(
        self,
        sg_outputs: Dict[str, torch.Tensor],
        objectness_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Convert SceneGraphGenerator raw outputs to dicts for SceneGraphEncoderV2."""
        bbox_preds = sg_outputs["bbox_preds"]
        entity_logits = sg_outputs["entity_logits"]
        region_logits = sg_outputs["region_logits"]
        positiveness_logits = sg_outputs["positiveness_logits"]
        relationship_logits = sg_outputs["relationship_logits"]
        objectness = sg_outputs["objectness_scores"]

        B, N = bbox_preds.shape[:2]
        results: List[Dict[str, Any]] = []
        for b in range(B):
            scores = objectness[b]
            keep = scores >= objectness_threshold
            n_keep = int(keep.sum().item())
            if n_keep == 0:
                # Fallback: keep the single highest-scoring proposal
                keep = torch.zeros(N, dtype=torch.bool, device=bbox_preds.device)
                keep[scores.argmax()] = True
                n_keep = 1

            kept_idx = keep.nonzero(as_tuple=True)[0]
            results.append({
                "bboxes": bbox_preds[b, keep].detach().cpu().numpy(),
                "entity_ids": entity_logits[b, keep].argmax(dim=-1).detach().cpu().numpy(),
                "region_ids": region_logits[b, keep].argmax(dim=-1).detach().cpu().numpy(),
                "positiveness": positiveness_logits[b, keep].argmax(dim=-1).detach().cpu().numpy(),
                "relations": torch.softmax(
                    relationship_logits[b, kept_idx][:, kept_idx], dim=-1
                ).detach(),
                "num_objects": n_keep,
            })
        return results

    def _run_sg_generator(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
        """
        Extract Qwen ViT features and run the SG generator.

        Returns ``(raw_outputs, sg_dicts)`` — the raw outputs (RPN logits +
        bbox preds + entity/region logits + relationship logits) are required
        by ``MultiTaskLoss._compute_scene_graph_loss`` in ``sg_only`` mode;
        the dicts are what ``SceneGraphEncoderV2`` consumes for the soft
        token path.

        ``ctx`` follows ``self.freeze_sg_generator`` so this method can also
        be used in Stage 1 with grads enabled.
        """
        feature_maps = self._extract_qwen_vit_feature_maps(pixel_values, image_grid_thw)
        ctx = torch.no_grad() if self.freeze_sg_generator else torch.enable_grad()
        with ctx:
            sg_raw = self.sg_generator(feature_maps)
        return sg_raw, self._sg_outputs_to_dicts(sg_raw)

    # ----------------------------------------------------------------------
    # SG token injection
    # ----------------------------------------------------------------------

    def _inject_sg_tokens(
        self,
        inputs_embeds: torch.Tensor,  # (B, L, D)
        input_ids: torch.Tensor,      # (B, L)
        sg_tokens: torch.Tensor,      # (B, K, D)
    ) -> torch.Tensor:
        """
        Replace SG placeholder positions in inputs_embeds with projected SG
        tokens. Vectorised across the (B, K) grid — no Python-level loops.

        Raises if any placeholder position is missing for any sample, since
        that means the chat template stripped or rewrote our markers and the
        SG signal would silently vanish.
        """
        B, L, _ = inputs_embeds.shape
        K = sg_tokens.size(1)
        device = input_ids.device

        # (K,) tensor of placeholder ids, broadcastable against (B, L)
        sg_id_tensor = torch.as_tensor(self.sg_token_ids, device=device)

        # matches[b, k, l] = True iff input_ids[b, l] == sg_token_ids[k]
        matches = input_ids.unsqueeze(1) == sg_id_tensor.view(1, K, 1)  # (B, K, L)

        has_match = matches.any(dim=-1)                       # (B, K)
        if not bool(has_match.all().item()):
            missing = (~has_match).nonzero(as_tuple=False)
            raise RuntimeError(
                f"SG placeholder tokens missing from {missing.size(0)} "
                f"(batch, slot) positions. First few: {missing[:5].tolist()}. "
                "The chat template likely stripped or rewrote the markers in "
                "_sg_placeholder_block(). Verify processor.apply_chat_template "
                "preserves <|sg_token_*|> tokens verbatim."
            )

        # First occurrence index per (b, k); argmax returns first True
        positions = matches.long().argmax(dim=-1)             # (B, K)

        # Scatter SG tokens into the matched positions (advanced indexing)
        out = inputs_embeds.clone()
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, K)
        out[batch_idx, positions] = sg_tokens.to(dtype=out.dtype)
        return out

    # ----------------------------------------------------------------------
    # Prompt construction
    # ----------------------------------------------------------------------

    def _build_prompts(
        self,
        questions: List[str],
        answers: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Format each sample as a Qwen chat with image, scene-graph block, and
        question. During training, answer text is appended to supervise LM loss.
        """
        sg_block = self._sg_placeholder_block()
        texts: List[str] = []
        for i, q in enumerate(questions):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"[scene_graph]{sg_block}[/scene_graph]\n\n{q}"},
                ],
            }]
            if answers is not None:
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": answers[i]}],
                })
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=answers is None,
            )
            texts.append(prompt)
        return texts

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,                   # (B, 3, H, W) — for SG generator
        pil_images: List[Any],                  # raw images for Qwen processor
        questions: List[str],                   # raw text
        scene_graphs: Optional[List[Dict[str, Any]]] = None,  # precomputed (training) or None → generated from Qwen ViT
        answer_texts: Optional[List[str]] = None,   # training target text
        question_types: Optional[List[str]] = None,
        # Grounding GT — if supplied during training the refinement head is
        # initialised from a noised version of the ground-truth bbox so that
        # at inference (where init comes from the LLM's parsed <box>) the
        # head sees an init_bbox distribution it has actually trained on.
        gt_grounding_bboxes: Optional[torch.Tensor] = None,  # (B, 4) in [0,1]
        gt_pointing_valid: Optional[torch.Tensor] = None,    # (B, 1) or (B,)
        # Legacy tensor inputs (ignored in v2; kept for signature compatibility)
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        answer_ids: Optional[torch.Tensor] = None,
        **_unused,
    ) -> Dict[str, Any]:
        device = next(self.parameters()).device

        # Defensive checks: the v2 path needs raw PIL images + raw question
        # strings for Qwen's processor. If a caller still passes only the
        # legacy tensor inputs, fail fast with a clear message.
        if pil_images is None or questions is None:
            raise ValueError(
                "SSGVQANetV2.forward requires `pil_images` (List[PIL.Image]) "
                "and `questions` (List[str]) — produced by collate_fn in "
                "data/mimic_cxr_dataset.py. Got "
                f"pil_images={'set' if pil_images is not None else 'None'}, "
                f"questions={'set' if questions is not None else 'None'}."
            )

        # ---- 1. Build chat prompts and run Qwen processor --------------------
        # Tokenization runs first because the SG path (step 2) consumes the
        # processor's pixel_values + image_grid_thw outputs.
        prompts = self._build_prompts(questions, answer_texts)
        proc_inputs = self.processor(
            text=prompts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        # ---- 2. Scene graphs: reuse caller-provided dicts or generate fresh --
        # During training (Stages 2-4) the dataset's GT scene graphs are
        # usually passed in. In sg_only mode (Stage 1) and at inference,
        # ``scene_graphs is None`` and we run the SG generator. We also force
        # generation when training_mode == 'sg_only' so the loss has raw RPN
        # outputs to supervise — even if the caller mistakenly passed dicts.
        sg_raw_outputs: Optional[Dict[str, torch.Tensor]] = None
        if scene_graphs is None or self.training_mode == "sg_only":
            sg_raw_outputs, scene_graphs = self._run_sg_generator(
                proc_inputs["pixel_values"],
                proc_inputs["image_grid_thw"],
            )

        # ---- 3. Encode SG dicts → node features → soft tokens -----------------
        sg_nodes, sg_mask = self.sg_encoder(scene_graphs, device)
        sg_tokens = self.sg_projector(sg_nodes, sg_mask)  # (B, K, d_llm)

        # ---- 4. Compute inputs_embeds and splice SG tokens into the stream ---
        # Qwen handles image-token substitution inside its forward when
        # pixel_values are passed; we only need to substitute SG placeholders
        # at the positions added by _build_prompts → _sg_placeholder_block.
        base_embed_fn = self.qwen.get_input_embeddings()
        inputs_embeds = base_embed_fn(proc_inputs["input_ids"])
        inputs_embeds = self._inject_sg_tokens(
            inputs_embeds, proc_inputs["input_ids"], sg_tokens
        )

        # ---- 5. LM forward with labels (training) or generate (inference) ----
        if answer_texts is not None:
            # Mask prompt tokens so LM loss only fires on the assistant turn.
            labels = self._mask_prompt_labels(proc_inputs["input_ids"].clone())

            outputs = self.qwen(
                inputs_embeds=inputs_embeds,
                attention_mask=proc_inputs.get("attention_mask"),
                pixel_values=proc_inputs.get("pixel_values"),
                image_grid_thw=proc_inputs.get("image_grid_thw"),
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
            )
            lm_loss = outputs.loss
            last_hidden = outputs.hidden_states[-1]  # (B, L, D)
            generated_ids = proc_inputs["input_ids"]
            generated_text = None

            # ---- 5a. One-time vision-path verification ---------------------
            # Qwen's forward signature accepts inputs_embeds + pixel_values
            # together, but the documented behaviour varies across versions:
            # some substitute vision features into inputs_embeds, others
            # silently ignore pixel_values. A text-only training run would
            # converge to a usable language model and look fine on loss
            # curves — until eval IoU is at chance. Catch this once.
            if not self._vision_path_verified:
                with torch.no_grad():
                    out_no_img = self.qwen(
                        inputs_embeds=inputs_embeds,
                        attention_mask=proc_inputs.get("attention_mask"),
                        labels=labels,
                        return_dict=True,
                    )
                if torch.allclose(
                    out_no_img.logits.float(),
                    outputs.logits.float(),
                    atol=1e-3,
                ):
                    raise RuntimeError(
                        "Qwen produced identical logits with and without "
                        "pixel_values — the vision path is inactive. Either "
                        "(a) substitute Qwen ViT features into inputs_embeds "
                        "manually before this call, or (b) pass input_ids "
                        "instead of inputs_embeds so Qwen's own substitution "
                        "kicks in. Check transformers/peft versions; this "
                        "interaction broke between several 2024-2025 releases."
                    )
                self._vision_path_verified = True
        else:
            # Inference: greedy generate, then re-run a forward pass over the
            # full generated sequence to get correctly-shaped hidden states
            # for downstream pooling (the per-step hidden_states tuple from
            # ``generate`` has irregular shapes — step 0 is (B, prompt_len, D)
            # and subsequent steps are (B, 1, D), so naive `[-1][-1]` only
            # captures the final token).
            with torch.no_grad():
                gen_out = self.qwen.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=proc_inputs.get("attention_mask"),
                    pixel_values=proc_inputs.get("pixel_values"),
                    image_grid_thw=proc_inputs.get("image_grid_thw"),
                    max_new_tokens=self.max_answer_length,
                    do_sample=False,
                    return_dict_in_generate=True,
                )
            generated_ids = gen_out.sequences
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            # Re-forward over generated_ids to get correctly-shaped hidden
            # states for downstream pooling. We deliberately did NOT pass
            # output_hidden_states=True to generate(): on 48GB cards the
            # per-step hidden cache balloons, and a single second forward is
            # cheaper than holding it for the full max_new_tokens window.
            with torch.no_grad():
                pad_id = self.processor.tokenizer.pad_token_id
                attn_mask = (
                    (generated_ids != pad_id) if pad_id is not None
                    else torch.ones_like(generated_ids)
                )
                fwd = self.qwen(
                    input_ids=generated_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            last_hidden = fwd.hidden_states[-1]
            lm_loss = None

        # ---- 6. Pool hidden states for aux heads & grounding -----------------
        if last_hidden is not None:
            attn_mask = proc_inputs.get("attention_mask")
            if attn_mask is not None:
                mask = attn_mask.unsqueeze(-1).to(last_hidden.dtype)
                pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                pooled = last_hidden.mean(1)
        else:
            pooled = torch.zeros(len(questions), self.d_llm, device=device)

        # ---- 7. Grounding refinement head ------------------------------------
        # Distribution-matched init_bbox to avoid a train/inference mismatch:
        #   * Training (gt_grounding_bboxes provided): start from the GT box
        #     plus small Gaussian noise. The head learns "given a roughly
        #     correct box, predict the precise delta" — exactly the regime it
        #     will see at inference, where init comes from the LLM's <box>.
        #   * Inference (generated_text available): start from the LLM's
        #     parsed <box>, fall back to a centre anchor if parsing fails.
        #   * Otherwise (e.g. validation without supervision): centre anchor.
        init_bboxes: Optional[torch.Tensor] = None
        if self.training and gt_grounding_bboxes is not None:
            init_bboxes = gt_grounding_bboxes.to(device=device, dtype=torch.float)
            # Noise scale tuned so the head sees ~ the same dispersion as a
            # decent LLM <box> at inference. ~5% box-side stddev is empirical.
            noise = torch.randn_like(init_bboxes) * 0.05
            init_bboxes = (init_bboxes + noise).clamp(0.0, 1.0)
        elif generated_text is not None:
            init_list = [
                parse_structured_output(t)["bbox"] or [0.25, 0.25, 0.75, 0.75]
                for t in generated_text
            ]
            init_bboxes = torch.tensor(init_list, device=device, dtype=torch.float)

        grounding_out = self.grounding_head(
            pooled, sg_nodes, sg_mask, init_bbox=init_bboxes
        )

        # ---- 8. Aux heads (CheXpert + VQA multi-head) ------------------------
        chexpert_logits, vqa_logits = self.aux_heads(pooled)

        # ---- 9. Parse generated text into answer strings for metrics ---------
        template_answers: List[str] = []
        if generated_text is not None:
            for t in generated_text:
                template_answers.append(parse_structured_output(t)["answer"])

        # ---- 10. Assemble output dict (matches v1 keys consumed by ---------
        #          training.loss.MultiTaskLoss and training.metrics.VQAMetrics)
        return {
            # Classification
            "vqa_logits": vqa_logits,
            "chexpert_logits": chexpert_logits,
            "pooled_output": pooled,
            # Generation
            "generated_answer_ids": generated_ids,
            "generated_answer_logits": (
                outputs.logits if answer_texts is not None else None
            ),
            "generated_answer_text": generated_text,
            "template_answer": template_answers,
            "lm_loss": lm_loss,  # Qwen-computed; preferred over manual CE
            # Explainability (from grounding cross-attention)
            "attention_weights": {
                "grounding_to_sg": grounding_out["spatial_attention"],
            },
            # Scene graph: raw RPN/entity/region/relationship logits are
            # exposed when the SG generator was actually run this forward
            # (sg_only mode or scene_graphs=None). MultiTaskLoss._compute_
            # scene_graph_loss requires this to be non-None during Stage 1.
            "scene_graph_outputs": sg_raw_outputs,
            "generated_scene_graphs": scene_graphs,
            # Surface gt_pointing_valid back to the loss so it can weight the
            # pointing-score BCE without re-fetching from the batch dict.
            "_gt_pointing_valid": gt_pointing_valid,
            # Grounding
            "grounding_outputs": grounding_out,
            # mHC telemetry (path weights, gate values, amax_gain) — only
            # populated when GroundingRefinementHead.use_mhc=True.
            "mhc_metrics": (
                self.grounding_head.mhc.get_metrics()
                if self.grounding_head.mhc is not None else {}
            ),
        }

    def _mask_prompt_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Mask everything up to and including the LAST occurrence of the
        '<|im_start|>assistant\\n' delimiter so LM loss is computed only on
        the assistant turn. Uses the cached, multi-token delimiter sequence
        (``self._assistant_delim_ids``) — the previous single-token compare
        was broken because Qwen's BPE tokenizer splits 'assistant' into
        multiple subtokens.

        Edge cases:
          * No delimiter found in a row → mask the entire row (loss=0 for
            that sample). Safer than training on the user prompt.
          * Sequence shorter than delimiter → row is masked entirely.
        """
        delim = torch.as_tensor(self._assistant_delim_ids, device=labels.device)
        K = int(delim.numel())
        B, L = labels.shape

        if K == 0 or L < K:
            return torch.full_like(labels, -100)

        # Sliding K-token window over labels, vectorised:
        # windows: (B, L - K + 1, K) — each window is one possible match.
        windows = labels.unfold(dimension=1, size=K, step=1)
        # matches[b, w] = True iff windows[b, w] == delim
        matches = (windows == delim.view(1, 1, K)).all(dim=-1)
        # matches: (B, W) where W = L - K + 1

        has_match = matches.any(dim=-1)  # (B,)

        # Last True position per row via reverse-argmax trick
        reversed_matches = matches.flip(-1)
        last_match_pos = (matches.size(1) - 1) - reversed_matches.long().argmax(dim=-1)

        # cut = first token AFTER the matched delimiter sequence; everything
        # at index < cut is part of the user/system prompt → mask.
        # Where no match, cut = L (mask the whole row).
        cut = torch.where(
            has_match,
            last_match_pos + K,
            torch.full_like(last_match_pos, L),
        )

        pos_idx = torch.arange(L, device=labels.device).unsqueeze(0).expand(B, -1)
        prompt_mask = pos_idx < cut.unsqueeze(1)

        masked = labels.clone()
        masked[prompt_mask] = -100
        return masked

    # ----------------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------------

    def save_pretrained(self, save_directory: str):
        import os, json
        os.makedirs(save_directory, exist_ok=True)
        # 1. Qwen + LoRA adapters
        self.qwen.save_pretrained(f"{save_directory}/qwen_lora")
        self.processor.save_pretrained(f"{save_directory}/qwen_lora")
        # 2. Custom components
        torch.save({
            "sg_generator": self.sg_generator.state_dict(),
            "sg_encoder": self.sg_encoder.state_dict(),
            "sg_projector": self.sg_projector.state_dict(),
            "grounding_head": self.grounding_head.state_dict(),
            "aux_heads": self.aux_heads.state_dict(),
            "sg_token_ids": self.sg_token_ids,
            "num_sg_tokens": self.num_sg_tokens,
        }, f"{save_directory}/ssg_components.pt")
        # 3. Config
        with open(f"{save_directory}/config.json", "w") as f:
            json.dump({
                "qwen_model_id": self.qwen_model_id,
                "use_quantization": self.use_quantization,
                "num_sg_tokens": self.num_sg_tokens,
                "training_mode": self.training_mode,
                "d_llm": self.d_llm,
            }, f, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory: str, **override_kwargs) -> "SSGVQANetV2":
        """
        Load a checkpoint saved by ``save_pretrained``.

        Steps:
          1. Read config.json and instantiate a fresh model (this loads Qwen
             from HF Hub and applies a fresh LoRA adapter).
          2. Overwrite the LoRA adapter with the saved weights via PEFT's
             standard adapter-load API.
          3. Load custom components (SG generator/encoder/projector,
             grounding head, aux heads) from ssg_components.pt.

        ``override_kwargs`` lets callers change runtime-only options (e.g.
        ``training_mode='finetune'``, ``torch_dtype=torch.bfloat16``) without
        editing the saved config.
        """
        import json
        from pathlib import Path

        save_path = Path(save_directory)
        with open(save_path / "config.json") as f:
            cfg = json.load(f)

        # d_llm is discovered from the loaded Qwen config — never passed in.
        cfg.pop("d_llm", None)
        cfg.update(override_kwargs)

        instance = cls(**cfg)

        # Load LoRA adapter weights into the freshly-built PEFT wrapper.
        # PEFT writes adapter_model.{bin,safetensors} under the save dir.
        lora_dir = save_path / "qwen_lora"
        if lora_dir.exists():
            try:
                # Preferred path: PEFT >= 0.6 accepts a directory containing
                # an adapter_config.json + adapter_model file.
                instance.qwen.load_adapter(str(lora_dir), adapter_name="default")
            except Exception:
                # Fallback: hand-load the adapter state dict.
                adapter_bin = lora_dir / "adapter_model.bin"
                adapter_st = lora_dir / "adapter_model.safetensors"
                if adapter_st.exists():
                    try:
                        from safetensors.torch import load_file as _st_load  # type: ignore[import-not-found]
                    except ImportError as _e:
                        raise ImportError(
                            "Adapter saved as safetensors but `safetensors` "
                            "is not installed. `pip install safetensors`."
                        ) from _e
                    adapter_state = _st_load(str(adapter_st))
                elif adapter_bin.exists():
                    adapter_state = torch.load(str(adapter_bin), map_location="cpu")
                else:
                    raise FileNotFoundError(
                        f"No adapter weights found under {lora_dir}"
                    )
                missing, unexpected = instance.qwen.load_state_dict(
                    adapter_state, strict=False
                )
                if missing or unexpected:
                    warnings.warn(
                        f"LoRA load: {len(missing)} missing, "
                        f"{len(unexpected)} unexpected keys."
                    )

        # Load custom components.
        ck_path = save_path / "ssg_components.pt"
        if ck_path.exists():
            state = torch.load(str(ck_path), map_location="cpu")
            instance.sg_generator.load_state_dict(state["sg_generator"])
            instance.sg_encoder.load_state_dict(state["sg_encoder"])
            instance.sg_projector.load_state_dict(state["sg_projector"])
            instance.grounding_head.load_state_dict(state["grounding_head"])
            instance.aux_heads.load_state_dict(state["aux_heads"])
            # sg_token_ids must match — the placeholder strings are
            # deterministic from num_sg_tokens, so this should always pass.
            saved_ids = state.get("sg_token_ids")
            if saved_ids is not None and saved_ids != instance.sg_token_ids:
                warnings.warn(
                    f"sg_token_ids mismatch on load: saved={saved_ids}, "
                    f"current={instance.sg_token_ids}. Embedding indices may "
                    "have shifted; verify the tokenizer revision."
                )

        return instance


# =============================================================================
# INLINED LEGACY V1 COMPONENTS
# =============================================================================
# Manifold-Constrained Hyper-Connections (mHC) + SceneGraphGenerator from v1.
# Reused by v2's GroundingRefinementHead (mHC) and SG pipeline.
# Defined at module bottom so they can shadow the forward-declarations at the
# top — this works because v2 classes only reference these names at __init__
# time, by which point the module is fully loaded.
# =============================================================================


def sinkhorn_knopp(x: torch.Tensor, t_max: int = 20, eps: float = 1e-8) -> torch.Tensor:
    """Sinkhorn-Knopp projection onto the Birkhoff polytope (mHC paper Eq. 8-9)."""
    x_pos = torch.exp(x - x.max(dim=-1, keepdim=True)[0])
    for _ in range(t_max):
        x_pos = x_pos / (x_pos.sum(dim=-1, keepdim=True) + eps)
        if x_pos.dim() >= 2:
            x_pos = x_pos / (x_pos.sum(dim=-2, keepdim=True) + eps)
    return x_pos


class RMSNorm(nn.Module):
    """RMSNorm as used in the mHC paper (Eq. 5)."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class ManifoldProjection(nn.Module):
    """Projects residual onto a manifold (default: Birkhoff polytope)."""

    def __init__(
        self,
        dim: int,
        manifold_type: str = "birkhoff",
        use_qr: bool = True,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.dim = dim
        self.manifold_type = manifold_type
        self.use_qr = use_qr
        self.sinkhorn_iters = sinkhorn_iters
        self.alpha = nn.Parameter(torch.ones(1) * 0.01)

        if manifold_type == "sphere":
            self.radius = nn.Parameter(torch.ones(1))
        elif manifold_type in ("grassmann", "stiefel"):
            self.rank = max(dim // 4, 1)
        elif manifold_type == "birkhoff":
            self.hres_weight = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        if self.manifold_type == "birkhoff":
            ds_matrix = sinkhorn_knopp(self.hres_weight, self.sinkhorn_iters)
            projected = F.linear(residual, ds_matrix)
        elif self.manifold_type == "sphere":
            norm = residual.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            projected = self.radius * residual / norm
        elif self.manifold_type == "oblique":
            norm = residual.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            projected = residual / norm
        elif self.manifold_type == "grassmann":
            projected = self._grassmann_project(residual)
        elif self.manifold_type == "stiefel":
            projected = self._stiefel_project(residual)
        else:
            projected = residual
        return self.alpha * projected

    def _grassmann_project(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
        if x_2d.shape[0] < self.rank:
            return x
        try:
            if self.use_qr:
                Q, _ = torch.linalg.qr(x_2d.T)
                projected = x_2d @ Q[:, : self.rank] @ Q[:, : self.rank].T
            else:
                U, S, Vh = torch.linalg.svd(x_2d, full_matrices=False)
                projected = U[:, : self.rank] @ torch.diag(S[: self.rank]) @ Vh[: self.rank, :]
            return projected.reshape(orig_shape)
        except Exception:
            return x

    def _stiefel_project(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
        try:
            U, _, Vh = torch.linalg.svd(x_2d, full_matrices=False)
            return (U @ Vh).reshape(orig_shape)
        except Exception:
            return x


class HyperConnection(nn.Module):
    """Manifold-Constrained Hyper-Connection block (mHC paper Eq. 5-9)."""

    def __init__(
        self,
        dim: int,
        num_paths: int = 4,
        manifold_type: str = "birkhoff",
        dropout: float = 0.1,
        use_qr: bool = True,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.dim = dim
        self.num_paths = num_paths
        self.path_weights = nn.Parameter(torch.ones(num_paths) / num_paths)
        self.manifold_projs = nn.ModuleList([
            ManifoldProjection(dim, manifold_type, use_qr=use_qr, sinkhorn_iters=sinkhorn_iters)
            for _ in range(num_paths)
        ])
        self.path_rms_norms = nn.ModuleList([RMSNorm(dim) for _ in range(num_paths)])
        self.path_dynamic_projs = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(num_paths)]
        )
        self.path_static_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(dim)) for _ in range(num_paths)]
        )
        self.path_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_paths)])
        self.gate_alpha = nn.Parameter(torch.ones(1) * 0.01)
        self.gate_proj = nn.Linear(dim * 2, dim)

        self._last_gate_values = None
        self._last_path_weights = None
        self._last_amax_gain = None
        self._input_amax = None
        self._output_amax = None

    def forward(self, x: torch.Tensor, f_x: torch.Tensor) -> torch.Tensor:
        try:
            dt = next(self.gate_proj.parameters()).dtype
            if x.dtype != dt:
                x = x.to(dtype=dt)
            if f_x.dtype != dt:
                f_x = f_x.to(dtype=dt)
        except StopIteration:
            pass

        residual = f_x - x
        self._input_amax = x.abs().max().detach()
        weights = F.softmax(self.path_weights, dim=0)
        self._last_path_weights = weights.detach()

        path_outputs = []
        for i, manifold_proj in enumerate(self.manifold_projs):
            normed = self.path_rms_norms[i](residual)
            dynamic = torch.tanh(self.path_dynamic_projs[i](normed))
            path_residual = self.path_dropouts[i](dynamic + self.path_static_biases[i])
            path_outputs.append(weights[i] * manifold_proj(path_residual))

        combined = sum(path_outputs)
        gate = torch.sigmoid(self.gate_proj(torch.cat([x, combined], dim=-1)))
        self._last_gate_values = gate.mean().detach()

        output = x + self.gate_alpha * gate * combined
        self._output_amax = output.abs().max().detach()
        if self._input_amax is not None and self._input_amax > 0:
            self._last_amax_gain = (self._output_amax / self._input_amax).item()
        return output

    def get_metrics(self) -> Dict[str, float]:
        m: Dict[str, float] = {}
        if self._last_path_weights is not None:
            for i, w in enumerate(self._last_path_weights):
                m[f"path_{i}_weight"] = w.item()
        if self._last_gate_values is not None:
            m["gate_mean"] = self._last_gate_values.item()
        if self._last_amax_gain is not None:
            m["amax_gain"] = self._last_amax_gain
        if self._input_amax is not None:
            m["input_amax"] = self._input_amax.item()
        if self._output_amax is not None:
            m["output_amax"] = self._output_amax.item()
        return m


class mHCBlock(nn.Module):
    """mHC-enhanced transformer block. Used by GroundingRefinementHead in v2."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        ff_dim: int = 3072,
        num_hc_paths: int = 4,
        manifold_type: str = "birkhoff",
        dropout: float = 0.1,
        use_qr: bool = True,
        min_seq_len: int = 4,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.min_seq_len = min_seq_len

        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(hidden_size)

        self.attn_mhc = HyperConnection(
            hidden_size, num_hc_paths, manifold_type, dropout, use_qr, sinkhorn_iters
        )
        self.ff_mhc = HyperConnection(
            hidden_size, num_hc_paths, manifold_type, dropout, use_qr, sinkhorn_iters
        )

        self.pos_embed = nn.Parameter(torch.randn(1, min_seq_len, hidden_size) * 0.02)
        self._gradient_checkpointing = False

    def _get_param_dtype(self) -> torch.dtype:
        try:
            return next(self.ff.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _attention_block(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        dt = self._get_param_dtype()
        if x.dtype != dt:
            x = x.to(dtype=dt)
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        if attn_out.dtype != dt:
            attn_out = attn_out.to(dtype=dt)
        return self.attn_mhc(x, self.attn_norm(x + attn_out))

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        dt = self._get_param_dtype()
        if x.dtype != dt:
            x = x.to(dtype=dt)
        ff_out = self.ff(x)
        return self.ff_mhc(x, self.ff_norm(x + ff_out))

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L, _ = x.shape
        original_len = L
        if L < self.min_seq_len:
            pad_len = self.min_seq_len - L
            x = F.pad(x, (0, 0, 0, pad_len), value=0)
            x[:, original_len:, :] = x[:, original_len:, :] + self.pos_embed[:, :pad_len, :]
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, pad_len), value=True)
            else:
                key_padding_mask = torch.zeros(
                    B, self.min_seq_len, dtype=torch.bool, device=x.device
                )
                key_padding_mask[:, original_len:] = True

        if self._gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self._attention_block, x, key_padding_mask, use_reentrant=False
            )
            x = torch.utils.checkpoint.checkpoint(
                self._ff_block, x, use_reentrant=False
            )
        else:
            x = self._attention_block(x, key_padding_mask)
            x = self._ff_block(x)

        if original_len < self.min_seq_len:
            x = x[:, :original_len, :]
        return x

    def get_metrics(self) -> Dict[str, float]:
        m: Dict[str, float] = {}
        for k, v in self.attn_mhc.get_metrics().items():
            m[f"attn_mhc_{k}"] = v
        for k, v in self.ff_mhc.get_metrics().items():
            m[f"ff_mhc_{k}"] = v
        return m


class SceneGraphGenerator(nn.Module):
    """
    Region-proposal scene graph generator. Consumes 2D feature maps
    (B, C, H, W); for v2 these come from Qwen's ViT (C=1280 for 7B).
    """

    def __init__(
        self,
        visual_dim: int = 1024,
        hidden_size: int = 768,
        num_entity_classes: int = 237,
        num_region_classes: int = 310,
        num_relationships: int = 10,
        max_objects: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_objects = max_objects
        self.roi_pool_size = 7

        self.rpn_conv = nn.Sequential(
            nn.Conv2d(visual_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.rpn_cls = nn.Conv2d(256, num_entity_classes + 1, 1)
        self.rpn_reg = nn.Conv2d(256, 4, 1)
        self.rpn_centerness = nn.Conv2d(256, 1, 1)

        roi_feat_dim = visual_dim * self.roi_pool_size * self.roi_pool_size
        self.entity_classifier = nn.Sequential(
            nn.Linear(roi_feat_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_entity_classes),
        )
        self.region_classifier = nn.Sequential(
            nn.Linear(roi_feat_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_region_classes),
        )
        self.positiveness_classifier = nn.Sequential(
            nn.Linear(roi_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

        self.obj_proj = nn.Linear(roi_feat_dim, hidden_size)
        self.rel_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 + hidden_size // 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_relationships),
        )
        self.spatial_encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_size // 4),
        )

    def _get_param_dtype(self) -> torch.dtype:
        try:
            return next(self.entity_classifier.parameters()).dtype
        except StopIteration:
            return torch.float32

    def forward(
        self,
        visual_features: torch.Tensor,
        gt_bboxes=None,
        gt_entities=None,
        gt_regions=None,
    ) -> Dict[str, Any]:
        B, C, H, W = visual_features.shape
        device = visual_features.device
        param_dtype = self._get_param_dtype()
        if visual_features.dtype != param_dtype:
            visual_features = visual_features.to(dtype=param_dtype)

        rpn_feat = self.rpn_conv(visual_features)
        if rpn_feat.dtype != param_dtype:
            rpn_feat = rpn_feat.to(dtype=param_dtype)

        rpn_cls = self.rpn_cls(rpn_feat)
        rpn_reg = self.rpn_reg(rpn_feat)
        centerness = self.rpn_centerness(rpn_feat)

        centerness_flat = centerness.view(B, -1)
        scores, indices = centerness_flat.topk(
            min(self.max_objects, centerness_flat.shape[1]), dim=1
        )
        scores = torch.sigmoid(scores)

        bbox_flat = rpn_reg.view(B, 4, -1).permute(0, 2, 1)
        boxes = torch.zeros(B, self.max_objects, 4, dtype=param_dtype, device=device)
        for b in range(B):
            selected = bbox_flat[b, indices[b] % bbox_flat.shape[1]]
            boxes[b, : selected.shape[0]] = torch.sigmoid(selected)

        roi_features = torch.zeros(
            B, self.max_objects, C * self.roi_pool_size * self.roi_pool_size,
            dtype=param_dtype, device=device,
        )
        for b in range(B):
            for n in range(self.max_objects):
                box = boxes[b, n]
                x1 = max(0, min(int(box[0].item() * W), W - 1))
                y1 = max(0, min(int(box[1].item() * H), H - 1))
                x2 = max(x1 + 1, min(int(box[2].item() * W) + 1, W))
                y2 = max(y1 + 1, min(int(box[3].item() * H) + 1, H))
                roi = visual_features[b : b + 1, :, y1:y2, x1:x2]
                pooled = F.adaptive_avg_pool2d(roi, (self.roi_pool_size, self.roi_pool_size))
                roi_features[b, n] = pooled.flatten().to(dtype=param_dtype)

        entity_logits = self.entity_classifier(roi_features)
        region_logits = self.region_classifier(roi_features)
        positiveness_logits = self.positiveness_classifier(roi_features)

        obj_features = self.obj_proj(roi_features)
        N = self.max_objects
        subj_exp = obj_features.unsqueeze(2).expand(B, N, N, -1)
        obj_exp = obj_features.unsqueeze(1).expand(B, N, N, -1)
        subj_bbox = boxes.unsqueeze(2).expand(B, N, N, 4)
        obj_bbox = boxes.unsqueeze(1).expand(B, N, N, 4)
        spatial_input = torch.cat([subj_bbox, obj_bbox], dim=-1).to(dtype=param_dtype)
        spatial = self.spatial_encoder(spatial_input)
        rel_input = torch.cat([subj_exp, obj_exp, spatial], dim=-1)
        relationship_logits = self.rel_classifier(rel_input)

        return {
            "bbox_preds": boxes,
            "entity_logits": entity_logits,
            "region_logits": region_logits,
            "positiveness_logits": positiveness_logits,
            "relationship_logits": relationship_logits,
            "objectness_scores": scores,
            "rpn_cls_logits": rpn_cls,
            "rpn_bbox_preds": rpn_reg,
            "rpn_centerness": centerness,
        }
