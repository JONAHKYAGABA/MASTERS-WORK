"""Pluggable scene-graph generators.

Public API::

    from models.sg_generators import (
        SGGenerator,
        get_sg_generator,
        list_sg_generators,
        register_sg_generator,
    )

Built-in registered names:
  - ``txrv_detr``       : TorchXRayVision DenseNet121 + DETR head. Standalone
                          image-in / scene-graph-out. Default choice.
  - ``qwen_vit_rpn``    : Legacy RPN over Qwen ViT features. Requires a Qwen
                          module and pre-extracted features. A/B baseline only.
  - ``chest_imagenome`` : Stub. Planned wrapper around the published
                          ImaGenome anatomical detector.
  - ``detr_sg``         : Stub. Full DETR with a heavier query bank.
  - ``dino_sg``         : Stub. Grounding-DINO backbone variant.
"""
from .base import (
    SGGenerator,
    get_sg_generator,
    list_sg_generators,
    register_sg_generator,
)

__all__ = [
    "SGGenerator",
    "get_sg_generator",
    "list_sg_generators",
    "register_sg_generator",
]
