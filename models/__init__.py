"""
MIMIC-CXR VQA Model with Complete MIMIC-Ext-CXR-QBA Integration

Fully integrates with MIMIC-Ext-CXR-QBA dataset features:
- Hierarchical answer generation from report-derived sentences
- Visual grounding with answer-level bounding boxes (scene graph-guided)
- Scene graph generation from 310 regions × 237 entities
- Multi-head classification (binary, category, region, severity)
- Manifold-Constrained Hyper-Connections (mHC) for enhanced feature fusion

Usage:
======
    from models import MIMICCXRVQAModel
    
    model = MIMICCXRVQAModel(
        use_mhc=True,           # Enable mHC (default: True)
        mhc_manifold='sphere',  # 'sphere', 'oblique', 'grassmann', 'stiefel'
        num_mhc_paths=3,        # Number of hyper-connection paths
    )
    model.set_training_mode('pretrain')  # 'standard', 'pretrain', 'finetune'
    
    # Forward pass (training)
    outputs = model(
        images, input_ids, attention_mask, scene_graphs,
        answer_ids=batch['answer_ids'],  # For decoder training
        question_types=batch['question_types']
    )
    
    # Classification outputs
    vqa_logits = outputs['vqa_logits']  # {binary, category, region, severity}
    
    # Answer Generation outputs (from MIMIC-Ext-CXR-QBA answer text)
    generated_text = outputs['generated_answer_text']   # Decoder: "There is consolidation..."
    template_answer = outputs['template_answer']        # Template: "Yes, Pneumonia is present..."
    generation_logits = outputs['generated_answer_logits']  # For loss computation
    
    # Scene Graph Generation outputs (from MIMIC-Ext-CXR-QBA observations)
    sg_out = outputs['scene_graph_outputs']
    sg_out['bbox_preds']      # Generated bboxes
    sg_out['entity_logits']   # 237 entity classes
    sg_out['region_logits']   # 310 region classes
    
    # Visual Grounding outputs (from MIMIC-Ext-CXR-QBA answer localization)
    grnd = outputs['grounding_outputs']
    grnd['bbox_pred']         # Predicted answer bbox
    grnd['pointing_score']    # Confidence of localization
    
    # Explainability
    attn = outputs['attention_weights']  # For attention visualization
    
    # mHC metrics (for ablation studies)
    mhc_metrics = outputs['mhc_metrics']  # path weights, gate values
    
    # Inference - generate human-readable answers
    answers = model.generate_answer(images, input_ids, attention_mask, scene_graphs)
    print(answers['decoder_answer'])   # ["There is consolidation in the right lower lobe."]
    print(answers['template_answer'])  # ["Yes, Consolidation is present in the right lower lobe."]

mHC Reference (Paper-Faithful Implementation):
==============================================
    "mHC: Manifold-Constrained Hyper-Connections" (Xie et al., 2024)
    
    Paper defaults now implemented:
    - Birkhoff polytope via Sinkhorn-Knopp (doubly stochastic matrices)
    - n=4 expansion paths
    - RMSNorm + tanh for dynamic mappings
    - Learnable α initialized to 0.01
    - Tracks Amax Gain Magnitude for stability analysis
    
    Manifold choices:
    - 'birkhoff': Doubly stochastic via Sinkhorn-Knopp (PAPER DEFAULT)
    - 'sphere': Normalizes to unit sphere (simpler alternative)
    - 'oblique': Column-wise normalization (good for sparse features)
    - 'grassmann': Low-rank subspace projection (expensive but expressive)
    - 'stiefel': Orthonormal projection (stricter than Grassmann)
    
    Key metrics for monitoring (Paper Fig. 3, 7-8):
    - amax_gain: Ratio of output/input max absolute value (should be ~1.0)
    - path_*_weight: Importance of each HC path
    - gate_mean: Average gating value
"""

from .mimic_vqa_model import (
    MIMICCXRVQAModel,
    MIMICVQAOutput,
    extract_visual_attention_map,
    attention_to_bbox,
    # Manifold-Constrained Hyper-Connections (Paper-faithful)
    RMSNorm,
    sinkhorn_knopp,
    ManifoldProjection,
    HyperConnection,
    mHCBlock,
)

__all__ = [
    'MIMICCXRVQAModel', 
    'MIMICVQAOutput', 
    'extract_visual_attention_map', 
    'attention_to_bbox',
    # mHC components (paper-faithful, can be used independently)
    'RMSNorm',
    'sinkhorn_knopp',
    'ManifoldProjection',
    'HyperConnection',
    'mHCBlock',
]
