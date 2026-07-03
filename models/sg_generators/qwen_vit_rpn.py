"""Legacy Qwen-ViT-features + RPN scene-graph generator.

Wraps the pre-existing SceneGraphGenerator (an RPN over Qwen ViT patch
features) in the SGGenerator interface so it can be used as an A/B
baseline against the standalone TorchXRayVision-DETR generator. This
is the ONLY generator that requires a Qwen backbone in memory; used
solely for comparison / ablation, not for the recommended training path.

Loading Qwen just to run this generator makes it appropriate for
evaluate-only runs of an existing checkpoint. For fresh training start
from ``txrv_detr``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .base import SGGenerator, register_sg_generator

logger = logging.getLogger(__name__)


@register_sg_generator("qwen_vit_rpn")
class QwenViTRPNSGGenerator(SGGenerator):
    """Adapter: extracts Qwen ViT features from a shared Qwen module and
    runs the legacy SceneGraphGenerator RPN over them.

    Because Qwen features are used, this generator does NOT accept a plain
    ``(B, 3, H, W)`` image tensor. Instead the caller (SSGVQANetV2) passes
    a pre-computed feature map via the ``features`` kwarg. That keeps the
    signature compatible with the base class while acknowledging that this
    generator is a wrapper around the old coupled path.
    """

    def __init__(
        self,
        num_entities: int = 237,
        num_regions: int = 310,
        num_relations: int = 10,
        visual_dim: int = 1024,
        hidden_size: int = 768,
        max_objects: int = 20,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            num_entities=num_entities,
            num_regions=num_regions,
            num_relations=num_relations,
        )
        # Delayed import so this module is safe to load even in environments
        # where the monolithic model won't build.
        from models.ssg_vqa_net_v2 import SceneGraphGenerator
        self.core = SceneGraphGenerator(
            visual_dim=visual_dim,
            hidden_size=hidden_size,
            num_entity_classes=num_entities,
            num_region_classes=num_regions,
            num_relationships=num_relations,
            max_objects=max_objects,
            dropout=dropout,
        )

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        *,
        features: Optional[torch.Tensor] = None,
        objectness_threshold: float = 0.3,
        return_dicts: bool = True,
    ) -> Dict[str, Any]:
        if features is None:
            raise ValueError(
                "qwen_vit_rpn requires pre-extracted Qwen ViT features via "
                "features=... . Use txrv_detr for a fully standalone "
                "image-in generator."
            )
        raw = self.core(features)
        result: Dict[str, Any] = {"raw": raw}
        if return_dicts:
            from .torchxrayvision_detr import TXRVDetrSGGenerator
            result["dicts"] = TXRVDetrSGGenerator._to_dicts(raw, objectness_threshold)
        return result
