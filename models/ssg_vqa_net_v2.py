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

import logging
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

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
        num_shared_heads: Optional[int] = None,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.d_node = d_node

        # Auto-pick num_shared_heads so total_heads = num_relations +
        # num_shared_heads divides d_node. Without this, the default combo
        # (num_relations=10 + num_shared_heads=2 = 12) fails for d_node=128
        # because 128 % 12 != 0. We pick the smallest valid total_heads that
        # gives at least one shared head.
        if num_shared_heads is None:
            valid = [
                t for t in range(num_relations + 1, d_node + 1)
                if d_node % t == 0
            ]
            if not valid:
                raise ValueError(
                    f"Cannot find total_heads >= {num_relations + 1} that "
                    f"divides d_node={d_node}. Increase d_node or pass "
                    "num_shared_heads explicitly."
                )
            num_shared_heads = valid[0] - num_relations
        self.num_shared_heads = num_shared_heads

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
            RelationAwareGAT(
                d_node=d_node,
                num_relations=num_relations,
                num_shared_heads=num_shared_heads,
                dropout=dropout,
            )
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

        # Pick a float dtype that matches this encoder's own parameters so
        # subsequent F.linear / F.matmul calls don't hit "Float vs Half"
        # mismatches. When the trainer casts the SG encoder to fp16 (Turing)
        # or bf16 (Ampere+), bbox_feats must follow.
        try:
            pdtype = next(self.bbox_proj.parameters()).dtype
        except StopIteration:
            pdtype = torch.float32

        # Allocate with padding indices (longs stay long for embedding lookup,
        # floats use pdtype so they survive the linear projections).
        region_ids = torch.full((B, n_max), self.num_regions, dtype=torch.long, device=device)
        entity_ids = torch.full((B, n_max), self.num_entities, dtype=torch.long, device=device)
        pos_ids = torch.full((B, n_max), 2, dtype=torch.long, device=device)
        bbox_feats = torch.zeros(B, n_max, 8, device=device, dtype=pdtype)
        node_mask = torch.zeros(B, n_max, device=device, dtype=pdtype)
        relations = torch.zeros(B, n_max, n_max, self.num_relations, device=device, dtype=pdtype)

        for b, sg in enumerate(scene_graphs):
            n = int(sg.get("num_objects", 0))
            if n == 0:
                continue
            n = min(n, n_max)

            # CRITICAL: build bbox features in FP32 with sane clamps before
            # casting to pdtype. In fp16:
            #   - `w / h` overflows when h is tiny (1 / 1e-6 = 1e6 > 65504 fp16 max → inf)
            #   - `w * h` underflows when both are tiny (1e-6 * 1e-6 = 1e-12 → 0)
            # Either produces NaN downstream in bbox_proj. We saw this on real
            # MIMIC bboxes that legitimately have small heights/widths.
            bboxes_f = torch.as_tensor(sg["bboxes"][:n], dtype=torch.float32, device=device)
            x1, y1, x2, y2 = bboxes_f[:, 0], bboxes_f[:, 1], bboxes_f[:, 2], bboxes_f[:, 3]
            # 1e-3 clamp: aspect ratio max becomes 1000, area min becomes 1e-6
            # — both safely inside fp16 range
            w = (x2 - x1).clamp(min=1e-3)
            h = (y2 - y1).clamp(min=1e-3)
            area = (w * h).clamp(min=1e-6, max=1.0)
            aspect = (w / h).clamp(min=1e-3, max=1e3)
            feats_per_obs = torch.stack(
                [x1, y1, x2, y2, w, h, area, aspect], dim=-1
            )
            bbox_feats[b, :n] = feats_per_obs.to(dtype=pdtype)

            ent = torch.as_tensor(sg["entity_ids"][:n], dtype=torch.long, device=device)
            reg = torch.as_tensor(sg["region_ids"][:n], dtype=torch.long, device=device)
            entity_ids[b, :n] = ent.clamp(max=self.num_entities - 1)
            region_ids[b, :n] = reg.clamp(max=self.num_regions - 1)

            if "positiveness" in sg and sg["positiveness"] is not None:
                p = torch.as_tensor(sg["positiveness"][:n], dtype=torch.long, device=device)
                pos_ids[b, :n] = p.clamp(max=2)

            if "relations" in sg and sg["relations"] is not None:
                rel = torch.as_tensor(sg["relations"], dtype=pdtype, device=device)
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

        # Apply GAT layers — scrub any NaN/Inf between layers so a single
        # bad attention row doesn't poison the whole graph.
        for gat in self.gat_layers:
            nodes = gat(nodes, relations, node_mask)
            if torch.isnan(nodes).any() or torch.isinf(nodes).any():
                nodes = torch.nan_to_num(nodes, nan=0.0, posinf=0.0, neginf=0.0)

        # Final safety: encoder output must be finite. If we still see NaN
        # here, it's a real model bug worth surfacing — but don't crash;
        # masked positions (where num_objects=0) legitimately have garbage
        # values that just shouldn't contribute.
        if torch.isnan(nodes).any() or torch.isinf(nodes).any():
            nodes = torch.nan_to_num(nodes, nan=0.0, posinf=0.0, neginf=0.0)

        return nodes, node_mask


# =============================================================================
# SG TOKEN PROJECTOR — compress node features to a fixed token budget
# =============================================================================


def _mha_fp32(
    mha: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Run an ``nn.MultiheadAttention`` module in fp32 by calling
    ``F.multi_head_attention_forward`` with weights cast to fp32 on the fly.

    Needed because the module's params may be fp16 (Turing) or bf16 (Ampere+),
    and the softmax inside attention is fp16-unsafe when scores have large
    magnitudes — which happens in the SG projector because the d_node→d_llm
    out-projection downstream can be sensitive to small NaN seeds.

    Returns the attended output (B, T_q, d) in fp32. Caller is responsible
    for casting back to its desired output dtype.
    """
    # nn.MultiheadAttention.batch_first=True means inputs come in as (B, T, d).
    # F.multi_head_attention_forward expects (T, B, d) when batch_first is
    # not honored at the functional level — so transpose ourselves.
    if mha.batch_first:
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

    in_proj_weight = mha.in_proj_weight.float() if mha.in_proj_weight is not None else None
    in_proj_bias = mha.in_proj_bias.float() if mha.in_proj_bias is not None else None
    out_proj_weight = mha.out_proj.weight.float()
    out_proj_bias = mha.out_proj.bias.float() if mha.out_proj.bias is not None else None

    attn_out, _ = F.multi_head_attention_forward(
        query=query.float(),
        key=key.float(),
        value=value.float(),
        embed_dim_to_check=mha.embed_dim,
        num_heads=mha.num_heads,
        in_proj_weight=in_proj_weight,
        in_proj_bias=in_proj_bias,
        bias_k=mha.bias_k.float() if mha.bias_k is not None else None,
        bias_v=mha.bias_v.float() if mha.bias_v is not None else None,
        add_zero_attn=mha.add_zero_attn,
        dropout_p=0.0,  # eval-style for stability; training dropout applied outside
        out_proj_weight=out_proj_weight,
        out_proj_bias=out_proj_bias,
        training=mha.training,
        key_padding_mask=key_padding_mask,
        need_weights=False,
        attn_mask=None,
        use_separate_proj_weight=False,
    )

    if mha.batch_first:
        attn_out = attn_out.transpose(0, 1)
    return attn_out


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
        # Output dtype = LLM dtype (matches what _inject_sg_tokens splices into).
        try:
            out_dtype = next(self.out_proj.parameters()).dtype
        except StopIteration:
            out_dtype = node_features.dtype

        # Run the whole projector in fp32. On Turing (fp16 only, no bf16) the
        # softmax in nn.MultiheadAttention + the d_node→d_llm=3584 linear can
        # easily overflow fp16's 65504 ceiling when the SG encoder produces
        # large-magnitude node features, producing NaN tokens that then poison
        # the LLM input embedding. d_node=128 is tiny so fp32 internal cost
        # is negligible. We cast params + activations to fp32 for this call
        # using a functional path rather than mutating the module dtype.
        node_features = node_features.float()
        if node_mask is not None:
            node_mask = node_mask.float()

        B = node_features.size(0)
        # Force queries to fp32 here so the matmul inside cross_attn is fp32.
        q = self.queries.float().unsqueeze(0).expand(B, -1, -1)

        # Guard against all-masked rows (sample with num_objects=0): replace
        # with a single-True mask so softmax has at least one valid key and
        # doesn't produce NaN. We zero those tokens after the fact.
        all_masked = None
        key_padding_mask = None
        if node_mask is not None:
            key_padding_mask = ~node_mask.bool()
            all_masked = key_padding_mask.all(dim=-1)  # (B,)
            if all_masked.any():
                # Unmask the first key for those rows just to keep softmax sane
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False

        # Temporarily run cross_attn in fp32 by casting its parameters' compute.
        # MultiheadAttention is stateless in dtype; PyTorch picks dtype from
        # inputs and parameters. We do it by calling functional F.multi_head_*.
        # Simpler: use the module after upcasting its parameters in-place would
        # break the module — instead, use F.scaled_dot_product_attention path.
        # Easier and correct: clone weights to fp32 on the fly.
        attended = _mha_fp32(
            self.cross_attn,
            q, node_features, node_features,
            key_padding_mask=key_padding_mask,
        )

        # The rest of the projector — cast layer params to fp32 on call.
        h = F.layer_norm(
            q + attended,
            self.norm1.normalized_shape,
            weight=self.norm1.weight.float() if self.norm1.weight is not None else None,
            bias=self.norm1.bias.float() if self.norm1.bias is not None else None,
            eps=self.norm1.eps,
        )

        ffn_h = F.linear(h, self.ffn[0].weight.float(), self.ffn[0].bias.float())
        ffn_h = F.gelu(ffn_h)
        ffn_h = F.linear(ffn_h, self.ffn[3].weight.float(), self.ffn[3].bias.float())

        h = F.layer_norm(
            h + ffn_h,
            self.norm2.normalized_shape,
            weight=self.norm2.weight.float() if self.norm2.weight is not None else None,
            bias=self.norm2.bias.float() if self.norm2.bias is not None else None,
            eps=self.norm2.eps,
        )

        tokens = F.linear(h, self.out_proj.weight.float(), self.out_proj.bias.float())
        tokens = F.layer_norm(
            tokens,
            self.out_norm.normalized_shape,
            weight=self.out_norm.weight.float() if self.out_norm.weight is not None else None,
            bias=self.out_norm.bias.float() if self.out_norm.bias is not None else None,
            eps=self.out_norm.eps,
        )

        # Zero out tokens that came from all-masked rows (they fed off a fake
        # unmasked key and are meaningless).
        if all_masked is not None and all_masked.any():
            tokens = tokens.clone()
            tokens[all_masked] = 0.0

        # Final NaN/Inf scrub — should be unreachable now, but cheap insurance.
        if not torch.isfinite(tokens).all():
            tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)

        return tokens.to(dtype=out_dtype)


# =============================================================================
# GROUNDING REFINEMENT HEAD — priority #1 component
# =============================================================================


class GroundingRefinementHead(nn.Module):
    """
    Produces a refined bbox from:
      - LLM hidden state at the <box> token position (or at a reserved location)
      - Scene-graph node features (for anatomical priors)
      - Optional initial bbox from the LLM's native <box> output

    Routes the fused signal through a single mHCBlock (Birkhoff manifold,
    n=4 paths). mHC is mandatory — the v1 manifold-constrained fusion is the
    architectural contribution that justifies this head over a vanilla MLP.
    """

    def __init__(
        self,
        d_llm: int,
        d_sg: int,
        d_hidden: int = 512,
        num_heads: int = 8,
        mhc_manifold: str = "birkhoff",
        num_mhc_paths: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_hidden = d_hidden

        # mHC is mandatory. If the legacy mHCBlock isn't available, fail loudly
        # at construction time rather than silently degrading to a vanilla
        # residual — silent degradation invalidates the architectural claim
        # and produces a different model than the config promises.
        if not _HAS_LEGACY or mHCBlock is None:
            raise RuntimeError(
                "GroundingRefinementHead requires mHCBlock, but the inlined "
                "legacy v1 components were not loaded. Check that the "
                "'INLINED LEGACY V1 COMPONENTS' section at the bottom of "
                "ssg_vqa_net_v2.py executed (mHCBlock must be defined)."
            )

        self.llm_proj = nn.Linear(d_llm, d_hidden)
        self.sg_proj = nn.Linear(d_sg, d_hidden)

        self.cross_attn = nn.MultiheadAttention(
            d_hidden, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_hidden)

        self.mhc = mHCBlock(
            hidden_size=d_hidden,
            num_heads=num_heads,
            ff_dim=d_hidden * 4,
            num_hc_paths=num_mhc_paths,
            manifold_type=mhc_manifold,
            dropout=dropout,
            sinkhorn_iters=20,
        )

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

        # Defensive dtype alignment: callers (the main forward + the SG
        # encoder) may hand us tensors in fp32 even when this head's
        # weights got cast to fp16/bf16 by the trainer. Without this, the
        # first F.linear errors with "Float vs Half".
        try:
            pdtype = next(self.llm_proj.parameters()).dtype
        except StopIteration:
            pdtype = llm_hidden.dtype
        if llm_hidden.dtype != pdtype:
            llm_hidden = llm_hidden.to(dtype=pdtype)
        if sg_features.dtype != pdtype:
            sg_features = sg_features.to(dtype=pdtype)

        q = self.llm_proj(llm_hidden).unsqueeze(1)   # (B, 1, d_hidden)
        kv = self.sg_proj(sg_features)                # (B, N, d_hidden)

        key_padding_mask = ~sg_mask.bool() if sg_mask is not None else None
        attended, attn_weights = self.cross_attn(
            q, kv, kv,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        fused = self.cross_norm(q + attended).squeeze(1)  # (B, d_hidden)

        # mHC fusion is mandatory — manifold-constrained hyper-connection over
        # the cross-attended bbox/SG signal.
        fused = self.mhc(fused.unsqueeze(1)).squeeze(1)

        # Initial bbox: if not provided, use a learned center anchor.
        # Must match this head's pdtype so torch.cat below doesn't upcast
        # the whole concat to fp32 and then mismatch delta_head's fp16 weights.
        if init_bbox is None:
            init_bbox = torch.tensor(
                [0.25, 0.25, 0.75, 0.75], device=device, dtype=pdtype,
            )
            init_bbox = init_bbox.unsqueeze(0).expand(B, -1)
        elif init_bbox.dtype != pdtype:
            init_bbox = init_bbox.to(dtype=pdtype)

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
        # Same defensive cast as GroundingRefinementHead: callers may hand us
        # fp32 even when our weights are fp16/bf16.
        try:
            pdtype = next(self.chexpert.parameters()).dtype
        except StopIteration:
            pdtype = pooled.dtype
        if pooled.dtype != pdtype:
            pooled = pooled.to(dtype=pdtype)
        # Defensive: scrub NaN/Inf at the boundary BEFORE the MLPs propagate
        # them into all four head outputs. Without this, a single NaN element
        # in Qwen's pooled hidden state (rare but observed on Turing fp16)
        # contaminates the softmax over all classes, silently producing bad
        # gradients that slowly destabilize the optimizer over ~1000 steps.
        if not torch.isfinite(pooled).all():
            pooled = torch.nan_to_num(pooled, nan=0.0, posinf=1e4, neginf=-1e4)
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
    Scene-Graph-Guided VQA model built around Qwen3-VL.

    Parameters
    ----------
    qwen_model_id : str
        HuggingFace model ID. Default Qwen/Qwen3-VL-8B-Instruct.
        Use Qwen/Qwen3-VL-4B-Instruct (or -2B-Instruct) for smaller GPU budgets.
        NOTE: requires transformers >= 4.57 (Qwen3-VL release). The Qwen2.5-VL
        port lived in transformers 4.45+; if you previously had it pinned to
        4.55.x, bump it before running this code.
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
        qwen_model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
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

        # CRITICAL: cap the image pixel budget. Qwen2.5-VL's default
        # max_pixels = 12.8M lets full-resolution chest X-rays (2544×3056 ≈
        # 7.8M pixels) generate ~40,000 visual tokens, blowing GPU memory at
        # the first forward (5-6GB activation per sample). For radiology
        # 448-512 px is plenty — anatomical structures remain clearly visible.
        # min_pixels also prevents tiny inputs from degrading too far.
        # Override by setting max_image_pixels env var if needed.
        try:
            _max_px = int(os.environ.get("QWEN_MAX_PIXELS", 448 * 448))
            _min_px = int(os.environ.get("QWEN_MIN_PIXELS", 256 * 256))
            if hasattr(self.processor, "image_processor"):
                self.processor.image_processor.max_pixels = _max_px
                self.processor.image_processor.min_pixels = _min_px
                warnings.warn(
                    f"Qwen image processor capped at min={_min_px} max={_max_px} pixels "
                    "(default would be 12.8M, causing OOM on radiology images). "
                    "Set QWEN_MAX_PIXELS / QWEN_MIN_PIXELS env vars to override.",
                    stacklevel=2,
                )
        except Exception as _e:
            warnings.warn(f"Could not cap Qwen image processor pixels: {_e}", stacklevel=2)

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

        # Discover LLM hidden size dynamically.
        # Known sizes: Qwen2.5-VL-7B=3584, 2.5-VL-3B=2048, Qwen3-VL-8B=4096,
        # Qwen3-VL-4B=2560, Qwen3-VL-2B=1536. The discovery walks both the
        # top-level config and text_config; Qwen3-VL nests differently than
        # 2.5-VL did but both attrs are covered.
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
            # IMPORTANT: Qwen-VL's `base.visual(...)` returns features AFTER
            # the PatchMerger, which projects raw ViT hidden up to the LLM
            # hidden dim (Qwen2.5-VL: 2048 for 3B, 3584 for 7B; Qwen3-VL:
            # 1536 for 2B, 2560 for 4B, 4096 for 8B). The earlier version
            # used vision_config.hidden_size here and crashed with
            # "expected 1280 channels, got 2048" because the feature maps had
            # been merged. _extract_qwen_vit_feature_maps now also handles
            # Qwen3-VL DeepStack tuple/list returns; see its docstring.
            #
            # Using self.d_llm matches what _extract_qwen_vit_feature_maps
            # actually returns. If you ever change the extraction to bypass
            # the merger (return raw ViT features), revert this to vit_hidden.
            self.sg_generator = SceneGraphGenerator(
                visual_dim=self.d_llm,
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
        # mHC's RMSNorm, ManifoldProjection, and sinkhorn_knopp all now
        # compute internally in fp32 (see classes at file bottom), so the
        # earlier Turing/fp16 auto-disable is no longer necessary —
        # Birkhoff/Sinkhorn is stable on every GPU.
        self.grounding_head = GroundingRefinementHead(
            d_llm=self.d_llm,
            d_sg=sg_node_dim,
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

        # ---- 7. View-position projector -------------------------------------
        # Projects the (B, 4) one-hot view encoding [PA, AP, LATERAL, OTHER]
        # from the dataset into LLM hidden space. Added to the pooled
        # representation before aux heads + grounding so PA/AP/LATERAL is a
        # first-class signal — without this, the view_encoding tensor is
        # already in the batch dict but the model ignored it entirely.
        # init_zero=True so an untrained projector contributes 0 at step 0
        # (doesn't perturb the baseline).
        self.view_proj = nn.Linear(4, self.d_llm)
        nn.init.zeros_(self.view_proj.weight)
        nn.init.zeros_(self.view_proj.bias)

        # Apply training-mode freezing
        self.set_training_mode(training_mode)

    # ----------------------------------------------------------------------
    # Setup helpers
    # ----------------------------------------------------------------------

    def _discover_hidden_size(self) -> int:
        """Read the LLM hidden size from the loaded config.

        Qwen2.5-VL-7B=3584, 2.5-VL-3B=2048.
        Qwen3-VL-8B=4096, 3-VL-4B=2560, 3-VL-2B=1536.
        Qwen3-VL nests `hidden_size` under `text_config` in some transformers
        versions; the fallback below handles that.
        """
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

        # Grounding head — active once we have a real signal.
        # mHC sub-block runs in fp32 (re-cast by trainer after the bulk
        # grounding_head→fp16 cast) so its Sinkhorn-Knopp gradients are
        # numerically stable on Turing without bf16 support. The rest of
        # grounding_head trains in fp16 like everything else.
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

    def _describe_vit_struct(self, x: Any, depth: int = 0, max_depth: int = 4) -> str:
        """One-line dump of a possibly-nested ViT return value. Used for the
        first-call diagnostic so we can SEE what Qwen3-VL handed us."""
        indent = "  " * depth
        if depth > max_depth:
            return f"{indent}<...>"
        if isinstance(x, torch.Tensor):
            return f"{indent}Tensor{tuple(x.shape)} dtype={x.dtype}"
        if isinstance(x, (list, tuple)):
            head = f"{indent}{type(x).__name__}(len={len(x)})"
            inner = "\n".join(self._describe_vit_struct(t, depth + 1, max_depth)
                              for t in x[:6])
            tail = f"\n{indent}  ...(+{len(x)-6} more)" if len(x) > 6 else ""
            return f"{head}\n{inner}{tail}"
        if isinstance(x, dict):
            head = f"{indent}dict(keys={list(x.keys())})"
            inner = "\n".join(
                f"{indent}  [{k}]:\n{self._describe_vit_struct(v, depth + 2, max_depth)}"
                for k, v in x.items()
            )
            return f"{head}\n{inner}"
        if hasattr(x, "last_hidden_state"):
            return (f"{indent}{type(x).__name__} ModelOutput\n"
                    f"{self._describe_vit_struct(x.last_hidden_state, depth + 1, max_depth)}")
        return f"{indent}{type(x).__name__}={x!r}"

    def _unwrap_vit_features(self, vit_out: Any) -> torch.Tensor:
        """Recursively find a 2D tensor of shape (*, d_llm) inside Qwen-VL's
        ``base.visual(...)`` return value.

        Qwen3-VL DeepStack returns nested structures whose exact shape varies
        by transformers version: sometimes a plain tensor, sometimes a tuple
        ``(merged, [level1, level2, ...])``, sometimes a dict. Rather than
        guess the index, we walk the structure looking for a 2D tensor whose
        last-dim matches self.d_llm — that uniquely identifies the
        post-merger feature tensor in every shape we've seen.

        On the first call we PRINT the structure (not log — print bypasses
        the smoke-test's logger config) so a future version skew gives us a
        legible structural dump instead of an AttributeError.
        """
        # First-call structural dump — visible in every smoke / training log.
        if not getattr(self, "_logged_vit_return_shape", False):
            dump = self._describe_vit_struct(vit_out)
            print(
                f"[Qwen-VL ViT] base.visual(...) returned:\n{dump}\n"
                f"[Qwen-VL ViT] Searching for 2D tensor with last-dim == d_llm = {self.d_llm} ...",
                flush=True,
            )
            self._logged_vit_return_shape = True

        d_llm = self.d_llm

        def _walk(x: Any, path: str = "$"):
            """Yield (tensor, path) for every torch.Tensor reachable from x."""
            if isinstance(x, torch.Tensor):
                yield x, path
                return
            if hasattr(x, "last_hidden_state"):
                yield from _walk(x.last_hidden_state, f"{path}.last_hidden_state")
                return
            if isinstance(x, dict):
                for k in ("last_hidden_state", "hidden_states", "merged_features",
                          "features", "image_embeds"):
                    if k in x:
                        yield from _walk(x[k], f"{path}[{k!r}]")
                for k, v in x.items():
                    yield from _walk(v, f"{path}[{k!r}]")
                return
            if isinstance(x, (list, tuple)):
                for i, v in enumerate(x):
                    yield from _walk(v, f"{path}[{i}]")
                return

        # Pass 1: exact match (2D, last-dim == d_llm). This is what we want.
        for t, path in _walk(vit_out):
            if t.dim() == 2 and t.size(-1) == d_llm:
                if not getattr(self, "_logged_vit_match_path", False):
                    print(f"[Qwen-VL ViT] Picked {path} with shape {tuple(t.shape)}",
                          flush=True)
                    self._logged_vit_match_path = True
                return t

        # Pass 2: 3D match (B, N, d_llm) — flatten leading dims. Some
        # transformers builds return per-batch instead of packed.
        for t, path in _walk(vit_out):
            if t.dim() == 3 and t.size(-1) == d_llm:
                if not getattr(self, "_logged_vit_match_path", False):
                    print(f"[Qwen-VL ViT] Picked {path} with shape {tuple(t.shape)} "
                          f"(3D — flattening to 2D)", flush=True)
                    self._logged_vit_match_path = True
                return t.reshape(-1, d_llm)

        # No match — dump everything we saw for debugging and raise.
        seen = [(tuple(t.shape), str(t.dtype), path) for t, path in _walk(vit_out)]
        raise RuntimeError(
            f"Could not find a tensor with last-dim == d_llm ({d_llm}) inside "
            f"base.visual(...) output. Saw the following tensors:\n  "
            + "\n  ".join(f"{path}: shape={shape} dtype={dtype}" for shape, dtype, path in seen)
            + "\nPaste this list back to debug. The fix is either to (a) pick the "
              "right tensor here, or (b) lower the SG generator's visual_dim to "
              "match a real ViT level."
        )

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
        the output token grid is (H // 2, W // 2). Return is a single tensor
        of shape (total_merged_tokens, d_llm).

        Qwen3-VL adds DeepStack — internally the ViT consumes multi-level
        features and may return EITHER a single packed tensor (same as 2.5-VL,
        deepest level only — the LLM consumes the rest internally) OR a
        tuple/list of per-level packed tensors. We can only feed ONE 2D grid
        into the SG generator's RPN, so when we get a tuple we take the
        LAST element (deepest, post-merger, matches d_llm). A one-shot log
        on the first call documents what we actually got so you can spot
        a version skew without re-reading the model file.
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

        # DeepStack tolerance — Qwen3-VL's `.visual(...)` can return a NESTED
        # structure (tuple/list/dict/ModelOutput, sometimes nested 2+ deep).
        # Earlier attempts to grab `[-1]` blindly hit "list of lists" cases.
        # Instead: recursively search the structure for a 2D tensor whose
        # last-dim matches d_llm — that's the post-merger feature tensor we
        # actually want. We print (not log) the structure on first call so
        # smoke-test stdout shows it regardless of how logging is configured.
        vit_out = self._unwrap_vit_features(vit_out)

        # Qwen3-VL renamed vision_config.spatial_merge_size in some builds;
        # try both before defaulting.
        vis_cfg = getattr(base.config, "vision_config", None) or base.config
        spatial_merge = (
            getattr(vis_cfg, "spatial_merge_size", None)
            or getattr(vis_cfg, "merge_size", None)
            or getattr(base.config, "spatial_merge_size", None)
            or 2
        )

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

        Returns ``(raw_outputs, sg_dicts)`` — raw outputs are required by
        ``MultiTaskLoss._compute_scene_graph_loss`` in ``sg_only`` mode;
        the dicts feed ``SceneGraphEncoderV2``.

        IMPORTANT: when the SG generator is frozen we force it into eval()
        for the forward, regardless of whether model.train() was called on
        the outer module. The generator contains BatchNorm2d layers which
        produce NaN in fp16 with batch_size=1 (0 variance → div by ~sqrt(eps)
        → blow-up → NaN propagates through SG tokens into Qwen → lm_loss
        nan). Eval mode uses the running stats (initially mean=0/var=1) and
        sidesteps the batch-stat path entirely.
        """
        feature_maps = self._extract_qwen_vit_feature_maps(pixel_values, image_grid_thw)
        ctx = torch.no_grad() if self.freeze_sg_generator else torch.enable_grad()
        prev_mode = self.sg_generator.training
        if self.freeze_sg_generator:
            self.sg_generator.eval()
        try:
            with ctx:
                sg_raw = self.sg_generator(feature_maps)
        finally:
            self.sg_generator.train(prev_mode)
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

        # NaN guard: if upstream (SG encoder/projector/mHC) produced NaN
        # in any soft token, replace with zeros. Otherwise the NaN poisons
        # Qwen's inputs_embeds and lm_loss returns nan with no diagnosis.
        if torch.isnan(sg_tokens).any() or torch.isinf(sg_tokens).any():
            warnings.warn(
                "_inject_sg_tokens: NaN/Inf detected in sg_tokens; "
                "replacing with zeros. Inspect SG encoder/projector init.",
                RuntimeWarning,
            )
            sg_tokens = torch.nan_to_num(sg_tokens, nan=0.0, posinf=0.0, neginf=0.0)

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
        indications: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Format each sample as a Qwen chat with image, scene-graph block, and
        question. During training, the answer text is appended to supervise LM
        loss.

        If `indications` is supplied (per-sample QBA clinical-indication
        strings, e.g. "Female with HIV, chest pain and dyspnea; evaluate for
        infiltrate and effusion."), it's prepended as a `Clinical context:`
        prefix. This gives the LLM the "why was this X-ray ordered" framing
        that QBA encodes in its scene graph's `indication` section — without
        it the model has to infer intent from the question alone.
        """
        sg_block = self._sg_placeholder_block()
        texts: List[str] = []
        for i, q in enumerate(questions):
            user_text_parts: List[str] = [f"[scene_graph]{sg_block}[/scene_graph]"]
            if indications is not None and i < len(indications):
                ind = (indications[i] or "").strip()
                if ind:
                    user_text_parts.append(f"Clinical context: {ind}")
            user_text_parts.append(f"Question: {q}")
            user_text = "\n\n".join(user_text_parts)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
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
        # NEW v2 signals from the dataset's collate_fn:
        view_encodings: Optional[torch.Tensor] = None,       # (B, 4) one-hot [PA, AP, LATERAL, OTHER]
        indications: Optional[List[str]] = None,             # QBA clinical-indication strings, one per sample (or None)
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
        # If QBA indications are provided we prepend them as clinical context
        # ("Clinical context: <indication>. Question: <q>") — this is QBA's
        # `answer_for_indication` source sentence, which gives the LLM the
        # "why was this X-ray ordered" framing it would otherwise miss.
        prompts = self._build_prompts(questions, answer_texts, indications=indications)
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
            # === REPORT LOSS: LM CE restricted to <think>...</think> tokens ===
            # When use_reports is on (dataset side), the <think> block contains
            # the radiologist's FINDINGS+IMPRESSION text and is the main "report
            # generation" supervision target. Expose a separately-monitored
            # report_loss so the training loop can log it independently from
            # the full lm_loss (which also covers <box> and <answer>).
            report_loss = self._compute_report_loss(outputs.logits, labels)
            last_hidden = outputs.hidden_states[-1]  # (B, L, D)
            last_hidden_mask = proc_inputs.get("attention_mask")  # matches prompt length
            generated_ids = proc_inputs["input_ids"]
            generated_text = None

            # ---- 5a. One-time vision-path verification ---------------------
            # Qwen's forward signature accepts inputs_embeds + pixel_values
            # together, but the documented behaviour varies across versions:
            # some substitute vision features into inputs_embeds, others
            # silently ignore pixel_values. A text-only training run would
            # converge to a usable language model and look fine on loss
            # curves — until eval IoU is at chance. Catch this once.
            #
            # NOTE: this runs ONE forward pass with image and ONE without,
            # transiently DOUBLING memory at step 0. On tight-memory GPUs
            # (Turing 48GB with QLoRA fp16) this can OOM on the first step
            # before training really begins. Set SKIP_VISION_PATH_CHECK=1
            # to bypass (you trade a one-time safety check for headroom).
            _skip_vision_check = bool(int(os.environ.get("SKIP_VISION_PATH_CHECK", "0")))
            if _skip_vision_check:
                # Pretend we already verified — silences future iterations.
                self._vision_path_verified = True
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
            # Use the mask we built for generated_ids — NOT proc_inputs'
            # attention_mask, which describes the original prompt length and
            # would broadcast-mismatch against last_hidden.
            last_hidden_mask = attn_mask
            lm_loss = None
            report_loss = None

        # ---- 6. Pool hidden states for aux heads & grounding -----------------
        # Each branch above set `last_hidden_mask` to the mask that matches
        # `last_hidden`'s sequence dimension. Don't fall back to
        # proc_inputs.attention_mask here — that's only correct in the
        # training branch, and using it in inference (where last_hidden
        # corresponds to generated_ids, not the prompt) crashes with a
        # broadcast mismatch.
        if last_hidden is not None:
            if last_hidden_mask is not None and last_hidden_mask.shape[1] == last_hidden.shape[1]:
                mask = last_hidden_mask.unsqueeze(-1).to(last_hidden.dtype)
                pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                pooled = last_hidden.mean(1)
        else:
            pooled = torch.zeros(len(questions), self.d_llm, device=device)

        # ---- 6a. Inject view-position encoding into the pooled rep ----------
        # The dataset emits a (B, 4) one-hot encoding of view position
        # [PA, AP, LATERAL, OTHER]. Project it into the LLM hidden space and
        # ADD to pooled — a residual that an AP/LATERAL-relevant aux head can
        # learn to use without dominating the LLM-derived signal. The
        # projector is zero-initialised so day-1 training matches the
        # pre-fix behaviour exactly.
        if view_encodings is not None:
            try:
                pdtype = next(self.view_proj.parameters()).dtype
            except StopIteration:
                pdtype = pooled.dtype
            ve = view_encodings.to(device=device, dtype=pdtype)
            view_feat = self.view_proj(ve)
            if view_feat.dtype != pooled.dtype:
                view_feat = view_feat.to(dtype=pooled.dtype)
            pooled = pooled + view_feat

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
            "report_loss": report_loss,  # LM CE restricted to <think> tokens (radiologist report)
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
            # mHC telemetry (path weights, gate values, amax_gain). mHC is
            # mandatory, so this is always populated.
            "mhc_metrics": self.grounding_head.mhc.get_metrics(),
        }

    def _mask_prompt_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Mask everything up to (and including) the LAST '<|im_start|>' token,
        leaving the assistant-role tokens + answer content as the only
        supervised positions. Format-agnostic — doesn't depend on the exact
        post-prefix sequence (whitespace, role spelling, BPE quirks).

        Why this version: the earlier multi-token-subsequence approach
        required '<|im_start|>assistant\\n' to tokenize identically inside
        the chat template's rendered text vs. when tokenized in isolation.
        When Qwen's processor disagreed by even one byte (extra space,
        different newline), NO match was found, the row was masked
        entirely, and CrossEntropyLoss with all -100 returned NaN.

        Safety net: any row that ends up fully masked still gets its
        last 16 tokens unmasked so the LM has something to fit and we
        never silently NaN.
        """
        tokenizer = self.processor.tokenizer
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")

        B, L = labels.shape
        device = labels.device

        # Position of every <|im_start|> token per row.
        is_marker = labels == im_start_id  # (B, L)
        has_marker = is_marker.any(dim=-1)

        # Last True per row via reverse-argmax.
        reversed_pos = is_marker.flip(-1).long().argmax(dim=-1)
        last_pos = (L - 1) - reversed_pos  # (B,)

        # Mask everything up to AND INCLUDING the last <|im_start|> (so the
        # role tag itself isn't a label) — then add 1 more to skip the role
        # token that immediately follows ('assistant').
        # Result: labels from (last_pos + 2) onward are kept as targets.
        cut = torch.where(
            has_marker,
            (last_pos + 2).clamp(max=L),
            torch.full_like(last_pos, L),
        )

        pos_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        prompt_mask = pos_idx < cut.unsqueeze(1)

        masked = labels.clone()
        masked[prompt_mask] = -100

        # Safety net: if a row got fully masked (e.g., no <|im_start|>
        # found at all), restore the last 16 labels so loss isn't NaN.
        fully_masked = (masked == -100).all(dim=-1)
        if bool(fully_masked.any().item()):
            for b in range(B):
                if bool(fully_masked[b].item()):
                    keep_n = min(16, L)
                    masked[b, -keep_n:] = labels[b, -keep_n:]

        return masked

    def _compute_report_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """LM cross-entropy restricted to tokens INSIDE <think>...</think>.

        Memory-conscious implementation:
          1) Find <think>...</think> spans on CPU (cheap, no GPU alloc).
          2) Index-select ONLY those positions from logits before CE
             (typically ~50-500 tokens instead of full ~3000-seq length).
          3) Wrap in torch.no_grad() — report_loss is MONITORED ONLY,
             not added to total_loss (see training/loss.py), so we can
             skip storing activations entirely.

        This avoids the OOM that the naive (B,L,V)-wide CE would cause on
        Turing 48GB with QLoRA — Qwen3 vocab is ~150k tokens, so the full
        per-token CE intermediate would be ~B*L*V floats (>1 GiB at B=4,
        L=3k, V=150k).

        Returns None when no <think> spans exist in this batch (the
        trainer skips logging on None to avoid plotting noise).
        """
        if logits is None or labels is None:
            return None

        # Lazily cache the tag token IDs (multi-token tag handled as a
        # full subsequence match, not single-token-special).
        if not hasattr(self, "_think_open_ids"):
            tok = self.processor.tokenizer
            self._think_open_ids = tok.encode("<think>", add_special_tokens=False)
            self._think_close_ids = tok.encode("</think>", add_special_tokens=False)

        open_ids = self._think_open_ids
        close_ids = self._think_close_ids
        if not open_ids or not close_ids:
            return None

        with torch.no_grad():
            # Shift for causal LM: predict labels[t+1] from logits[t]
            shift_labels = labels[:, 1:]
            B, Lm1 = shift_labels.shape
            ids_cpu = shift_labels.detach().cpu().tolist()

            # Collect (batch_idx, time_idx) pairs that fall inside any
            # <think>...</think> span and are not prompt-masked (-100).
            batch_idx: List[int] = []
            time_idx: List[int] = []
            ol, cl = len(open_ids), len(close_ids)
            for b in range(B):
                row = ids_cpu[b]
                i = 0
                while i <= len(row) - ol:
                    if row[i : i + ol] == open_ids:
                        j = i + ol
                        while j <= len(row) - cl and row[j : j + cl] != close_ids:
                            j += 1
                        end_inner = j
                        for k in range(i + ol, min(end_inner, Lm1)):
                            if row[k] != -100:
                                batch_idx.append(b)
                                time_idx.append(k)
                        i = max(end_inner + cl, i + 1)
                    else:
                        i += 1

            if not batch_idx:
                return None

            device = logits.device
            b_idx_t = torch.tensor(batch_idx, device=device, dtype=torch.long)
            t_idx_t = torch.tensor(time_idx, device=device, dtype=torch.long)

            # Index-select ONLY the (N,) report positions from the
            # logits (N, V) and labels (N,). N is typically 100-500,
            # not the full ~3000-token sequence — keeps memory low.
            # logits indexing: shift = logits[:, :-1, :], select pos t
            # is equivalent to selecting from raw logits at position t.
            selected_logits = logits[b_idx_t, t_idx_t, :].float()  # (N, V)
            selected_labels = shift_labels[b_idx_t, t_idx_t].long()  # (N,)

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            return loss_fct(selected_logits, selected_labels).detach()

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
    """
    Sinkhorn-Knopp projection onto the Birkhoff polytope (mHC paper Eq. 8-9).

    NumericaI safety: the iteration uses `torch.exp` followed by
    repeated divisions, both of which underflow/overflow easily in fp16 —
    one of the smoke-test GPUs hit `nan` loss because `exp()` returned 0
    everywhere and the subsequent division produced inf. We compute the
    whole projection in fp32 and cast the result back to the caller's
    dtype. The cost is a few KB of intermediate fp32 storage; the gain is
    fp16/bf16 training stability.

    Stage 3 crash (DeepSpeed loss-scale floor) traced back to this fn:
    when the upstream `hres_weight` accumulates any NaN/Inf from fp16
    gradient noise, exp(NaN) = NaN, every iteration propagates NaN, and
    the doubly-stochastic output is all NaN. The grounding head then
    produces NaN logits → vqa head produces NaN → DeepSpeed sees overflow
    every batch → halves loss scale until it hits the floor → crash.

    Defense: scrub NaN/Inf at INPUT (replace with finite values that
    sinkhorn can normalize), CLAMP exp argument to prevent overflow in
    fp32 (max ~88), and scrub OUTPUT once more before returning.
    """
    orig_dtype = x.dtype
    x = x.float()

    # 1. Scrub input: replace NaN with 0, ±Inf with ±88 (clamp to fp32 exp range)
    if not torch.isfinite(x).all():
        x = torch.nan_to_num(x, nan=0.0, posinf=88.0, neginf=-88.0)

    # 2. Subtract max for numerical stability (already there), but clamp the
    #    shifted value to [-88, 88] so exp() can't overflow to Inf
    shifted = (x - x.max(dim=-1, keepdim=True)[0]).clamp(min=-88.0, max=88.0)
    x_pos = torch.exp(shifted)

    for _ in range(t_max):
        denom1 = x_pos.sum(dim=-1, keepdim=True).clamp(min=eps)
        x_pos = x_pos / denom1
        if x_pos.dim() >= 2:
            denom2 = x_pos.sum(dim=-2, keepdim=True).clamp(min=eps)
            x_pos = x_pos / denom2

    # 3. Final scrub: if any NaN/Inf slipped through (shouldn't, but cheap),
    #    replace with uniform distribution row (1/N) — the most neutral fallback
    if not torch.isfinite(x_pos).all():
        N = x_pos.shape[-1]
        x_pos = torch.where(torch.isfinite(x_pos), x_pos, torch.full_like(x_pos, 1.0 / N))

    return x_pos.to(dtype=orig_dtype)


class RMSNorm(nn.Module):
    """
    RMSNorm as used in the mHC paper (Eq. 5).

    Computes internally in fp32 — `x.pow(2)` overflows fp16 quickly and the
    subsequent `1/rms` division underflows the other way. Both produce NaN.
    The fp32 round-trip costs nothing for small tensors and keeps the
    Birkhoff/Sinkhorn manifold stable on Turing-class GPUs.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        rms = x_f.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        out = x_f / rms * self.weight.float()
        return out.to(dtype=orig_dtype)


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
        # Compute the projection in fp32 then cast back. The Birkhoff path
        # (exp + repeated divisions) and norm-based projections (sphere /
        # oblique) all underflow or overflow in fp16, which produced silent
        # NaNs on Turing GPUs. Cost is a couple of cast ops on a (d, d)
        # weight + (B, *, d) activations — negligible.
        orig_dtype = residual.dtype
        residual_f = residual.float()
        if self.manifold_type == "birkhoff":
            ds_matrix = sinkhorn_knopp(self.hres_weight.float(), self.sinkhorn_iters)
            projected = F.linear(residual_f, ds_matrix)
        elif self.manifold_type == "sphere":
            norm = residual_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            projected = self.radius.float() * residual_f / norm
        elif self.manifold_type == "oblique":
            norm = residual_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            projected = residual_f / norm
        elif self.manifold_type == "grassmann":
            projected = self._grassmann_project(residual_f)
        elif self.manifold_type == "stiefel":
            projected = self._stiefel_project(residual_f)
        else:
            projected = residual_f
        return (self.alpha.float() * projected).to(dtype=orig_dtype)

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
        # Option C dtype boundary: this block's params are fp32 (re-cast by
        # the trainer for Sinkhorn stability on Turing). Cast input fp16→fp32
        # on entry, do all math in fp32, cast output back to caller's dtype
        # on exit. Cost: ~256KB transient fp32 storage per batch — negligible.
        caller_dtype = x.dtype
        param_dtype = self._get_param_dtype()
        if x.dtype != param_dtype:
            x = x.to(dtype=param_dtype)

        B, L, _ = x.shape
        original_len = L
        if L < self.min_seq_len:
            pad_len = self.min_seq_len - L
            x = F.pad(x, (0, 0, 0, pad_len), value=0)
            x[:, original_len:, :] = x[:, original_len:, :] + self.pos_embed[:, :pad_len, :].to(dtype=param_dtype)
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

        # Final NaN/Inf scrub on block output. If anything in this block
        # produced NaN (sinkhorn underflow, attention with all-masked rows,
        # exp overflow in fp16), don't let it propagate to the downstream
        # grounding head where DeepSpeed will see it as gradient overflow
        # and shrink the loss scale until it hits the floor.
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

        # Cast back to caller's dtype (typically fp16) so the rest of
        # grounding_head sees the expected dtype.
        if x.dtype != caller_dtype:
            x = x.to(dtype=caller_dtype)

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

        # === ROI feature extraction via torchvision.ops.roi_align ===
        # WHY THE REPLACEMENT: the old per-(B,N) python loop converted bbox
        # coords to ints via .item(), which severs the gradient from the
        # entity/region classifier loss back to bbox_preds. That meant the
        # bbox path could ONLY be updated by the bbox L1/GIoU loss term —
        # explaining why sg_loss plateaus at ~9.5 even with Hungarian
        # matching: the class heads can't push boxes to better positions.
        #
        # roi_align (Mask R-CNN paper) is fully differentiable through the
        # box coordinates (bilinear interpolation, no quantisation), so
        # gradients from entity_classifier and region_classifier flow back
        # into the RPN regression head. Also ~4× faster — no python loop,
        # no GPU→CPU syncs.
        from torchvision.ops import roi_align as _roi_align
        N = self.max_objects
        # boxes are normalised to [0,1] — scale to feature-map pixel coords
        boxes_pixel = boxes.clone()
        boxes_pixel[..., 0] = boxes_pixel[..., 0] * float(W)
        boxes_pixel[..., 1] = boxes_pixel[..., 1] * float(H)
        boxes_pixel[..., 2] = boxes_pixel[..., 2] * float(W)
        boxes_pixel[..., 3] = boxes_pixel[..., 3] * float(H)
        # Defensively ensure x2>x1 and y2>y1 (at least 1px) so roi_align
        # doesn't crash on degenerate boxes
        eps = 1.0  # 1 feature-map pixel
        boxes_pixel[..., 2] = torch.maximum(boxes_pixel[..., 2], boxes_pixel[..., 0] + eps)
        boxes_pixel[..., 3] = torch.maximum(boxes_pixel[..., 3], boxes_pixel[..., 1] + eps)
        # roi_align expects (K, 5) = [batch_idx, x1, y1, x2, y2]
        batch_idx = torch.arange(B, device=device, dtype=param_dtype).unsqueeze(1).expand(-1, N).reshape(-1, 1)
        boxes_flat_5 = boxes_pixel.reshape(B * N, 4)
        rois = torch.cat([batch_idx, boxes_flat_5], dim=1)  # (B*N, 5)
        # roi_align prefers fp32 for the input feature map; cast then back
        pooled = _roi_align(
            visual_features.float(),
            rois.float(),
            output_size=(self.roi_pool_size, self.roi_pool_size),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )  # (B*N, C, roi, roi)
        roi_features = pooled.reshape(
            B, N, C * self.roi_pool_size * self.roi_pool_size
        ).to(dtype=param_dtype)

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
