"""Standalone chest-X-ray scene-graph generator.

Backbone
--------
TorchXRayVision's DenseNet121 (``densenet121-res224-all``), pretrained on
14+ chest X-ray datasets (roughly 1M images). Auto-downloads on first
use via ``torchxrayvision.models.DenseNet``. Falls back to a torchvision
ImageNet DenseNet121 if the ``torchxrayvision`` package is not installed
(with a clear log line about the fallback).

Head
----
DETR-style set prediction. A learned query bank (N=20 by default) attends
over the flattened backbone feature map through two Transformer decoder
layers. Per-query outputs: 4-d bbox (sigmoid xyxy), entity logits,
region logits, positiveness logits, objectness score, and a pairwise
relation logit tensor.

Interface
---------
Implements ``SGGenerator`` from ``models/sg_generators/base.py``. Input
is a batch of raw RGB images (B, 3, H, W) in ``[0, 1]``; output is the
same ``{"raw": ..., "dicts": ...}`` payload that the downstream
``SceneGraphEncoderV2`` and ``MultiTaskLoss`` already consume, so
plugging this generator into ``customvqamodel`` costs nothing else.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SGGenerator, register_sg_generator

logger = logging.getLogger(__name__)


# =====================================================================
# Backbone loader -- auto-downloads TXRV DenseNet121 weights on first use
# =====================================================================
def _load_txrv_backbone() -> tuple:
    """Return ``(backbone, feature_channels)``.

    Preference order:
      1. torchxrayvision DenseNet121 ``densenet121-res224-all`` (CXR-pretrained).
      2. torchvision DenseNet121 ImageNet-pretrained (weaker domain match).

    Both cases return a ``nn.Module`` that accepts standard-format (B, 3, H, W)
    RGB float32 input in [0, 1] and emits (B, C, h, w) feature maps. The
    wrapper handles TXRV's 1-channel + [-1024, 1024] preprocessing internally
    so callers don't have to know which backbone is active.
    """
    try:
        import torchxrayvision as xrv
    except ImportError:
        logger.warning(
            "torchxrayvision not installed; falling back to ImageNet-pretrained "
            "DenseNet121 backbone. Install with `pip install torchxrayvision` "
            "for chest-X-ray-pretrained weights (better domain fit)."
        )
        return _load_torchvision_densenet121_backbone()

    logger.info("Loading torchxrayvision densenet121-res224-all (CXR-pretrained)...")
    # xrv.models.DenseNet auto-downloads weights to
    # ~/.torchxrayvision/models_data/ on first call. weights=
    # 'densenet121-res224-all' is the multi-dataset ensemble.
    net = xrv.models.DenseNet(weights="densenet121-res224-all")
    # TXRV wraps a torchvision DenseNet in `features`; the tail head after
    # `features` is a classifier we don't need. Use just the conv trunk.
    backbone_features = net.features  # (B, 1024, 7, 7) at 224x224 input
    feature_channels = 1024
    return _TXRVBackboneWrapper(backbone_features, source="txrv"), feature_channels


def _load_torchvision_densenet121_backbone() -> tuple:
    """ImageNet DenseNet121 fallback."""
    from torchvision.models import densenet121, DenseNet121_Weights
    net = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    return _TXRVBackboneWrapper(net.features, source="torchvision"), 1024


# ImageNet normalisation constants for the fallback backbone.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class _TXRVBackboneWrapper(nn.Module):
    """Adapts a standard (B, 3, H, W) [0, 1] RGB input to whatever the wrapped
    backbone expects, then emits (B, C, h, w) features.

    TXRV DenseNet expects:
      * 1 channel (chest X-rays are grayscale)
      * pixel values in [-1024, 1024]
    Torchvision DenseNet121 expects:
      * 3 channels
      * ImageNet-normalised: (x - mean) / std with the ImageNet stats

    We inspect ``source`` at construction and apply the correct preprocessing
    in ``forward`` so the outer generator code stays uniform.
    """

    def __init__(self, features: nn.Module, source: str = "txrv") -> None:
        super().__init__()
        self.features = features
        assert source in ("txrv", "torchvision"), source
        self.source = source
        # Register normalisation constants as buffers so they move with .to()
        # and DDP-shard cleanly.
        self.register_buffer("_imagenet_mean", _IMAGENET_MEAN.clone())
        self.register_buffer("_imagenet_std", _IMAGENET_STD.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) float in [0, 1]
        if self.source == "txrv":
            # RGB -> grayscale via standard luminance weights (matches
            # PIL's convert('L') and cv2's cvtColor with COLOR_RGB2GRAY).
            if x.size(1) == 3:
                r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
                x = 0.299 * r + 0.587 * g + 0.114 * b   # (B, 1, H, W)
            elif x.size(1) != 1:
                raise ValueError(
                    f"txrv backbone expects 1 or 3 input channels, got {x.size(1)}"
                )
            # [0, 1] -> [-1024, 1024]: this matches
            # torchxrayvision.datasets.normalize(img, 255) up to the same
            # rescale constant xrv uses in its data pipeline.
            x = x * 2048.0 - 1024.0
        else:
            # torchvision expects 3-channel ImageNet-normalised input.
            if x.size(1) == 1:
                x = x.expand(-1, 3, -1, -1)
            x = (x - self._imagenet_mean) / self._imagenet_std
        return self.features(x)


# =====================================================================
# DETR-style detection head
# =====================================================================
class _DetrHead(nn.Module):
    """Query bank + Transformer decoder producing per-query predictions."""

    def __init__(
        self,
        feature_channels: int,
        d_model: int,
        num_queries: int,
        num_entities: int,
        num_regions: int,
        num_relations: int,
        num_decoder_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries

        # Project backbone feature channels down to d_model.
        self.input_proj = nn.Conv2d(feature_channels, d_model, kernel_size=1)

        # 2-D sinusoidal positional embedding is overkill for a small feature
        # map; a learnable per-position table is cheaper and stable.
        # Cap at 32x32 so we don't waste params if resolution changes.
        self.pos_embed = nn.Parameter(torch.zeros(1, d_model, 32, 32))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.query_embed = nn.Embedding(num_queries, d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # Per-query heads
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 4),
        )
        self.entity_head = nn.Linear(d_model, num_entities)
        self.region_head = nn.Linear(d_model, num_regions)
        self.positiveness_head = nn.Linear(d_model, 3)  # neg, pos, unknown
        self.objectness_head = nn.Linear(d_model, 1)

        # Pairwise relation head: concat (q_i, q_j) -> num_relations.
        self.relation_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_relations),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, _, H, W = features.shape

        # Project + add positional embedding, then flatten to (B, HW, D).
        feats = self.input_proj(features)  # (B, D, H, W)
        pe = F.interpolate(
            self.pos_embed, size=(H, W), mode="bilinear", align_corners=False
        )
        feats = feats + pe
        memory = feats.flatten(2).transpose(1, 2)  # (B, HW, D)

        # Query bank: (B, N, D)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Decode. batch_first=True so tgt is (B, N, D), memory is (B, HW, D).
        decoded = self.decoder(tgt=queries, memory=memory)  # (B, N, D)

        # Per-query heads
        bbox = torch.sigmoid(self.bbox_head(decoded))  # xyxy in [0, 1]
        # Ensure x2 > x1 and y2 > y1 (functional, keeps grad).
        x1 = bbox[..., 0:1]
        y1 = bbox[..., 1:2]
        x2 = torch.maximum(bbox[..., 2:3], x1 + 1e-3)
        y2 = torch.maximum(bbox[..., 3:4], y1 + 1e-3)
        bbox = torch.cat([x1, y1, x2, y2], dim=-1)

        entity_logits = self.entity_head(decoded)
        region_logits = self.region_head(decoded)
        pos_logits = self.positiveness_head(decoded)
        objectness = torch.sigmoid(self.objectness_head(decoded)).squeeze(-1)

        # Pairwise relations: (B, N, N, 2D) -> (B, N, N, R)
        Bq, Nq, D = decoded.shape
        qi = decoded.unsqueeze(2).expand(Bq, Nq, Nq, D)
        qj = decoded.unsqueeze(1).expand(Bq, Nq, Nq, D)
        pair = torch.cat([qi, qj], dim=-1)
        relation_logits = self.relation_head(pair)

        return {
            "bbox_preds": bbox,                       # (B, N, 4)
            "entity_logits": entity_logits,           # (B, N, E)
            "region_logits": region_logits,           # (B, N, R)
            "positiveness_logits": pos_logits,        # (B, N, 3)
            "relationship_logits": relation_logits,   # (B, N, N, Rel)
            "objectness_scores": objectness,          # (B, N)
        }


# =====================================================================
# Public generator (registered)
# =====================================================================
@register_sg_generator("txrv_detr")
class TXRVDetrSGGenerator(SGGenerator):
    """CXR-pretrained backbone + DETR-style head. Standalone: takes raw
    (B, 3, H, W) images, returns SG dicts + raw outputs. No Qwen anywhere.
    """

    def __init__(
        self,
        num_entities: int,
        num_regions: int,
        num_relations: int = 10,
        num_queries: int = 20,
        d_model: int = 384,
        num_decoder_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        input_size: int = 224,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__(
            num_entities=num_entities,
            num_regions=num_regions,
            num_relations=num_relations,
        )
        self.input_size = int(input_size)
        self.backbone, feat_channels = _load_txrv_backbone()
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = _DetrHead(
            feature_channels=feat_channels,
            d_model=d_model,
            num_queries=num_queries,
            num_entities=num_entities,
            num_regions=num_regions,
            num_relations=num_relations,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        *,
        objectness_threshold: float = 0.3,
        return_dicts: bool = True,
    ) -> Dict[str, Any]:
        # Standard input contract: (B, 3, H, W) float in [0, 1]. Resize to
        # self.input_size (TXRV DenseNet was trained at 224).
        if images.dim() != 4:
            raise ValueError(f"expected (B, 3, H, W); got shape {tuple(images.shape)}")
        if images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1)
        if images.shape[-1] != self.input_size or images.shape[-2] != self.input_size:
            images = F.interpolate(
                images, size=(self.input_size, self.input_size),
                mode="bilinear", align_corners=False,
            )

        features = self.backbone(images)  # (B, C, h, w)
        raw = self.head(features)

        result: Dict[str, Any] = {"raw": raw}
        if return_dicts:
            result["dicts"] = self._to_dicts(raw, objectness_threshold)
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _to_dicts(
        raw: Dict[str, torch.Tensor],
        objectness_threshold: float,
    ) -> List[Dict[str, Any]]:
        """Threshold + argmax to produce per-image scene-graph dicts.

        Matches the payload SceneGraphEncoderV2 expects: numpy arrays for
        the discrete fields, GPU-side tensor for the soft relation matrix
        so it can retain grad in sg_only mode.
        """
        bbox = raw["bbox_preds"]
        ent = raw["entity_logits"]
        reg = raw["region_logits"]
        pos = raw["positiveness_logits"]
        rel = raw["relationship_logits"]
        obj = raw["objectness_scores"]

        B, N = bbox.shape[:2]
        results: List[Dict[str, Any]] = []
        for b in range(B):
            keep = obj[b] >= objectness_threshold
            n_keep = int(keep.sum().item())
            if n_keep == 0:
                keep = torch.zeros(N, dtype=torch.bool, device=bbox.device)
                keep[obj[b].argmax()] = True
                n_keep = 1
            kept = keep.nonzero(as_tuple=True)[0]
            results.append({
                "bboxes": bbox[b, keep].detach().cpu().numpy(),
                "entity_ids": ent[b, keep].argmax(dim=-1).detach().cpu().numpy(),
                "region_ids": reg[b, keep].argmax(dim=-1).detach().cpu().numpy(),
                "positiveness": pos[b, keep].argmax(dim=-1).detach().cpu().numpy(),
                "relations": torch.softmax(rel[b, kept][:, kept], dim=-1).detach(),
                "num_objects": n_keep,
            })
        return results
