"""
MIMIC-CXR VQA Training Module

Handles all training objectives using MIMIC-Ext-CXR-QBA data:
- VQA classification loss (multi-head: binary, category, region, severity)
- Answer generation loss (cross-entropy for decoder using answer text)
- Scene graph generation loss (entity, region, bbox from observations)
- Visual grounding loss (bbox from answer localization)
- CheXpert auxiliary loss

Usage:
======
    from training import MultiTaskLoss, VQAMetrics
    from data import create_dataloader
    
    # Create loss with automatic weight adjustment per training stage
    loss_fn = MultiTaskLoss(training_mode='pretrain')
    # Modes: 'standard', 'pretrain', 'finetune'
    
    # Training loop
    for batch in dataloader:
        outputs = model(
            batch['images'],
            batch['input_ids'],
            batch['attention_mask'],
            batch['scene_graphs'],
            question_types=batch['question_types'],
            answer_ids=batch['answer_ids'],  # From MIMIC-Ext-CXR-QBA hierarchical answers
        )
        
        # Compute multi-task loss
        loss, loss_dict = loss_fn(
            outputs,
            vqa_targets={'binary': ..., 'category': ...},
            chexpert_labels=batch['chexpert_labels'],
            chexpert_mask=batch['chexpert_mask'],
            question_types=batch['question_types'],
            # Answer generation targets (from MIMIC-Ext-CXR-QBA)
            answer_ids=batch['answer_ids'],
            # Scene graph targets (from scene_graph.json observations)
            gt_sg_bboxes=batch['gt_sg_bboxes'],
            gt_sg_entities=batch['gt_sg_entities'],
            gt_sg_regions=batch['gt_sg_regions'],
            # Grounding targets (from answer localization)
            gt_grounding_bboxes=batch['gt_grounding_bboxes'],
            gt_pointing_valid=batch['gt_pointing_valid'],
        )
        
        # Update metrics (including generation BLEU/ROUGE)
        metrics.update(outputs, vqa_targets, ..., reference_answers=batch['reference_answers'])
    
    # Compute final metrics
    results = metrics.compute()
    # results includes: classification_accuracy, generation_bleu, sg_entity_accuracy, grounding_mean_iou, etc.
"""

from .loss import (
    MultiTaskLoss,
    FocalLoss,
)

from .metrics import (
    VQAMetrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
    compute_iou,
    compute_batch_iou,
    compute_bleu,
    compute_rouge_l,
)

__all__ = [
    # Loss
    'MultiTaskLoss',
    'FocalLoss',
    # Metrics
    'VQAMetrics',
    'compute_confusion_matrix',
    'compute_per_class_metrics',
    'compute_iou',
    'compute_batch_iou',
    'compute_bleu',
    'compute_rouge_l',
]
