"""
models — v2 entry point.

The previous v1 model (MIMICCXRVQAModel + ConvNeXt + Bio_ClinicalBERT + custom
decoder) has been retired. v2 wraps Qwen2.5-VL with LoRA, scene-graph soft
tokens, and a manifold-constrained grounding refinement head.

Usage:
    from models import SSGVQANetV2

    model = SSGVQANetV2(
        qwen_model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        use_quantization=True,
        training_mode="pretrain",
    )

The paper-faithful mHC components (RMSNorm, sinkhorn_knopp, ManifoldProjection,
HyperConnection, mHCBlock) and the SceneGraphGenerator live in the same
ssg_vqa_net_v2.py file and are re-exported here for ablation studies and
backwards-compatible imports.
"""

from .ssg_vqa_net_v2 import (
    # v2 primary API
    SSGVQANetV2,
    SceneGraphEncoderV2,
    SGTokenProjector,
    GroundingRefinementHead,
    AuxiliaryHeads,
    parse_structured_output,
    # mHC components (re-exported for ablation / external use)
    RMSNorm,
    sinkhorn_knopp,
    ManifoldProjection,
    HyperConnection,
    mHCBlock,
    # Scene graph generator (frozen after Stage 1 in v2)
    SceneGraphGenerator,
)

__all__ = [
    "SSGVQANetV2",
    "SceneGraphEncoderV2",
    "SGTokenProjector",
    "GroundingRefinementHead",
    "AuxiliaryHeads",
    "parse_structured_output",
    "RMSNorm",
    "sinkhorn_knopp",
    "ManifoldProjection",
    "HyperConnection",
    "mHCBlock",
    "SceneGraphGenerator",
]
