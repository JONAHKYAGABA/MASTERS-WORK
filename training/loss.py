"""
Multi-Task Loss Functions for MIMIC-CXR VQA

Single unified MultiTaskLoss class that handles all tasks:
- VQA multi-head classification loss
- Answer generation loss (cross-entropy for decoder)
- CheXpert auxiliary classification loss
- Scene graph generation loss
- Visual grounding loss

Training Mode Weights:
======================
'standard': vqa=1.0, generation=0.5, chexpert=0.3, sg=0.1, grounding=0.1
'pretrain': vqa=1.0, generation=0.5, chexpert=0.3, sg=0.2, grounding=0.15
'finetune': vqa=1.0, generation=0.8, chexpert=0.1, sg=0.05, grounding=0.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional, Any


class MultiTaskLoss(nn.Module):
    """
    Unified loss for VQA + all auxiliary tasks.
    
    Automatically detects which outputs are present and applies
    corresponding losses. Set training_mode to adjust weights.
    """
    
    # Preset weights for different training stages
    MODE_WEIGHTS = {
        'standard': {'vqa': 1.0, 'generation': 0.5, 'chexpert': 0.3, 'scene_graph': 0.1, 'grounding': 0.1},
        'pretrain': {'vqa': 1.0, 'generation': 0.5, 'chexpert': 0.3, 'scene_graph': 0.2, 'grounding': 0.15},
        'finetune': {'vqa': 1.0, 'generation': 0.8, 'chexpert': 0.1, 'scene_graph': 0.05, 'grounding': 0.2},
    }
    
    def __init__(
        self,
        training_mode: str = 'standard',
        # Override mode weights if needed
        vqa_weight: Optional[float] = None,
        generation_weight: Optional[float] = None,
        chexpert_weight: Optional[float] = None,
        scene_graph_weight: Optional[float] = None,
        grounding_weight: Optional[float] = None,
        # Head weights
        binary_weight: float = 1.0,
        category_weight: float = 0.5,
        region_weight: float = 0.5,
        severity_weight: float = 0.3,
        ignore_index: int = -1,
        label_smoothing: float = 0.0,
        use_giou: bool = True,
        pad_token_id: int = 0,
    ):
        super().__init__()
        
        # Get mode defaults
        mode_weights = self.MODE_WEIGHTS.get(training_mode, self.MODE_WEIGHTS['standard'])
        
        # Set weights (explicit > mode default)
        self.vqa_weight = vqa_weight if vqa_weight is not None else mode_weights['vqa']
        self.generation_weight = generation_weight if generation_weight is not None else mode_weights['generation']
        self.chexpert_weight = chexpert_weight if chexpert_weight is not None else mode_weights['chexpert']
        self.scene_graph_weight = scene_graph_weight if scene_graph_weight is not None else mode_weights['scene_graph']
        self.grounding_weight = grounding_weight if grounding_weight is not None else mode_weights['grounding']
        
        self.head_weights = {
            'binary': binary_weight,
            'category': category_weight,
            'region': region_weight,
            'severity': severity_weight,
        }
        
        self.ignore_index = ignore_index
        self.use_giou = use_giou
        self.pad_token_id = pad_token_id
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.generation_ce_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=label_smoothing)
    
    def forward(
        self,
        outputs: Any,
        vqa_targets: Dict[str, torch.Tensor],
        chexpert_labels: torch.Tensor,
        chexpert_mask: torch.Tensor,
        question_types: List[str],
        # Optional: Answer generation ground truth
        answer_ids: Optional[torch.Tensor] = None,
        # Optional: Scene graph ground truth
        gt_sg_bboxes: Optional[List[torch.Tensor]] = None,
        gt_sg_entities: Optional[List[torch.Tensor]] = None,
        gt_sg_regions: Optional[List[torch.Tensor]] = None,
        # Optional: Grounding ground truth
        gt_grounding_bboxes: Optional[torch.Tensor] = None,
        gt_pointing_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute all applicable losses.
        """
        device = chexpert_labels.device
        loss_dict = {}
        
        # Get outputs
        vqa_logits = outputs.get('vqa_logits', {}) if isinstance(outputs, dict) else \
                     (outputs.vqa_logits if hasattr(outputs, 'vqa_logits') else {})
        chexpert_logits = outputs.get('chexpert_logits') if isinstance(outputs, dict) else \
                          (outputs.chexpert_logits if hasattr(outputs, 'chexpert_logits') else None)
        sg_outputs = outputs.get('scene_graph_outputs') if isinstance(outputs, dict) else None
        grounding_outputs = outputs.get('grounding_outputs') if isinstance(outputs, dict) else None
        generation_logits = outputs.get('generated_answer_logits') if isinstance(outputs, dict) else None
        
        # =====================================================
        # VQA CLASSIFICATION LOSS
        # =====================================================
        total_vqa_loss = torch.tensor(0.0, device=device)
        head_indices = self._get_head_indices(question_types)
        
        for head_name, indices in head_indices.items():
            if not indices or head_name not in vqa_logits:
                continue
            
            indices_tensor = torch.tensor(indices, device=device)
            head_logits = vqa_logits[head_name]
            
            if head_name in vqa_targets:
                head_targets = vqa_targets[head_name]
                
                if len(indices) < head_logits.shape[0]:
                    head_logits = head_logits[indices_tensor]
                    head_targets = head_targets[indices_tensor]
                
                head_loss = self.ce_loss(head_logits, head_targets)
                
                if not torch.isnan(head_loss):
                    loss_dict[f'vqa_{head_name}_loss'] = head_loss
                    total_vqa_loss = total_vqa_loss + self.head_weights.get(head_name, 1.0) * head_loss
        
        loss_dict['vqa_loss'] = total_vqa_loss
        
        # =====================================================
        # ANSWER GENERATION LOSS
        # =====================================================
        generation_loss = torch.tensor(0.0, device=device)
        
        if generation_logits is not None and answer_ids is not None:
            # Shift logits and labels for causal LM
            # logits: (B, T, V), answer_ids: (B, T)
            shift_logits = generation_logits[:, :-1, :].contiguous()
            shift_labels = answer_ids[:, 1:].contiguous()
            
            # Flatten and compute loss
            B, T, V = shift_logits.shape
            generation_loss = self.generation_ce_loss(
                shift_logits.view(B * T, V),
                shift_labels.view(B * T)
            )
            loss_dict['generation_loss'] = generation_loss
        
        # =====================================================
        # CHEXPERT LOSS
        # =====================================================
        chexpert_loss = torch.tensor(0.0, device=device)
        
        if chexpert_logits is not None:
            raw_loss = self.bce_loss(chexpert_logits, chexpert_labels)
            masked_loss = raw_loss * chexpert_mask
            valid_count = chexpert_mask.sum()
            if valid_count > 0:
                chexpert_loss = masked_loss.sum() / valid_count
        
        loss_dict['chexpert_loss'] = chexpert_loss
        
        # =====================================================
        # SCENE GRAPH LOSS
        # =====================================================
        sg_loss = torch.tensor(0.0, device=device)
        
        if sg_outputs is not None and self.scene_graph_weight > 0:
            sg_loss, sg_loss_dict = self._compute_scene_graph_loss(
                sg_outputs, gt_sg_bboxes, gt_sg_entities, gt_sg_regions, device
            )
            loss_dict.update(sg_loss_dict)
        
        loss_dict['scene_graph_loss'] = sg_loss
        
        # =====================================================
        # GROUNDING LOSS
        # =====================================================
        grounding_loss = torch.tensor(0.0, device=device)
        
        if grounding_outputs is not None and self.grounding_weight > 0 and gt_grounding_bboxes is not None:
            grounding_loss, grnd_loss_dict = self._compute_grounding_loss(
                grounding_outputs, gt_grounding_bboxes, gt_pointing_valid, device
            )
            loss_dict.update(grnd_loss_dict)
        
        loss_dict['grounding_loss'] = grounding_loss
        
        # =====================================================
        # TOTAL LOSS
        # =====================================================
        total_loss = (
            self.vqa_weight * total_vqa_loss +
            self.generation_weight * generation_loss +
            self.chexpert_weight * chexpert_loss +
            self.scene_graph_weight * sg_loss +
            self.grounding_weight * grounding_loss
        )
        
        return total_loss, loss_dict
    
    def _compute_scene_graph_loss(
        self,
        sg_outputs: Dict[str, torch.Tensor],
        gt_bboxes: Optional[List[torch.Tensor]],
        gt_entities: Optional[List[torch.Tensor]],
        gt_regions: Optional[List[torch.Tensor]],
        device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute scene graph generation loss."""
        loss_dict = {}
        
        bbox_preds = sg_outputs['bbox_preds']
        entity_logits = sg_outputs['entity_logits']
        region_logits = sg_outputs['region_logits']
        objectness = sg_outputs['objectness_scores']
        
        B, N = bbox_preds.shape[:2]
        
        entity_loss = torch.tensor(0.0, device=device)
        region_loss = torch.tensor(0.0, device=device)
        bbox_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        num_valid = 0
        
        if gt_bboxes is None:
            return torch.tensor(0.0, device=device), {}
        
        for b in range(B):
            if b >= len(gt_bboxes) or gt_bboxes[b] is None:
                continue
            
            gt_box = gt_bboxes[b].to(device)
            M = gt_box.shape[0]
            if M == 0:
                continue
            
            K = min(N, M)
            pred_box = bbox_preds[b, :K]
            gt_box_matched = gt_box[:K]
            
            # Bbox loss
            if self.use_giou:
                bbox_loss = bbox_loss + self._giou_loss(pred_box, gt_box_matched).mean()
            else:
                bbox_loss = bbox_loss + F.smooth_l1_loss(pred_box, gt_box_matched)
            
            # Entity loss
            if gt_entities is not None and b < len(gt_entities) and gt_entities[b] is not None:
                ent_target = gt_entities[b].to(device)[:K].long()
                entity_loss = entity_loss + self.ce_loss(entity_logits[b, :K], ent_target)
            
            # Region loss
            if gt_regions is not None and b < len(gt_regions) and gt_regions[b] is not None:
                reg_target = gt_regions[b].to(device)[:K].long()
                region_loss = region_loss + self.ce_loss(region_logits[b, :K], reg_target)
            
            # Objectness loss
            obj_target = torch.zeros(N, device=device)
            obj_target[:K] = 1.0
            obj_loss = obj_loss + F.binary_cross_entropy_with_logits(objectness[b], obj_target)
            
            num_valid += 1
        
        if num_valid > 0:
            entity_loss = entity_loss / num_valid
            region_loss = region_loss / num_valid
            bbox_loss = bbox_loss / num_valid
            obj_loss = obj_loss / num_valid
        
        loss_dict['sg_entity_loss'] = entity_loss
        loss_dict['sg_region_loss'] = region_loss
        loss_dict['sg_bbox_loss'] = bbox_loss
        loss_dict['sg_objectness_loss'] = obj_loss
        
        total = entity_loss + 0.5 * region_loss + bbox_loss + 0.5 * obj_loss
        return total, loss_dict
    
    def _compute_grounding_loss(
        self,
        grounding_outputs: Dict[str, torch.Tensor],
        gt_bboxes: torch.Tensor,
        gt_pointing_valid: Optional[torch.Tensor],
        device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute grounding loss."""
        loss_dict = {}
        
        bbox_pred = grounding_outputs['bbox_pred']
        pointing_score = grounding_outputs['pointing_score']
        
        B = bbox_pred.shape[0]
        
        # Bbox loss
        if self.use_giou:
            bbox_loss = self._giou_loss(bbox_pred, gt_bboxes).mean()
        else:
            bbox_loss = F.smooth_l1_loss(bbox_pred, gt_bboxes)
        
        loss_dict['grounding_bbox_loss'] = bbox_loss
        
        # Pointing loss (ensure target shape matches score shape)
        if gt_pointing_valid is not None:
            pointing_target = gt_pointing_valid.float().view(B, 1)
        else:
            pointing_target = torch.ones(B, 1, device=device)
        
        pointing_loss = F.binary_cross_entropy(pointing_score.view(B, 1), pointing_target)
        loss_dict['grounding_pointing_loss'] = pointing_loss
        
        total = bbox_loss + 0.5 * pointing_loss
        return total, loss_dict
    
    def _giou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute GIoU loss."""
        x1 = torch.max(pred[:, 0], target[:, 0])
        y1 = torch.max(pred[:, 1], target[:, 1])
        x2 = torch.min(pred[:, 2], target[:, 2])
        y2 = torch.min(pred[:, 3], target[:, 3])
        
        inter_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union_area = pred_area + target_area - inter_area
        
        iou = inter_area / (union_area + 1e-8)
        
        enc_x1 = torch.min(pred[:, 0], target[:, 0])
        enc_y1 = torch.min(pred[:, 1], target[:, 1])
        enc_x2 = torch.max(pred[:, 2], target[:, 2])
        enc_y2 = torch.max(pred[:, 3], target[:, 3])
        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
        
        giou = iou - (enc_area - union_area) / (enc_area + 1e-8)
        
        return 1 - giou
    
    def _get_head_indices(self, question_types: List[str]) -> Dict[str, List[int]]:
        """Map question types to answer heads."""
        try:
            from data.mimic_cxr_dataset import QUESTION_TYPE_MAP
        except ImportError:
            QUESTION_TYPE_MAP = {}
        
        head_indices = {'binary': [], 'category': [], 'region': [], 'severity': []}
        
        for idx, q_type in enumerate(question_types):
            head = QUESTION_TYPE_MAP.get(q_type)
            
            if head is None:
                q_lower = q_type.lower()
                if any(x in q_lower for x in ['is_abnormal', 'is_normal', 'has_']):
                    head = 'binary'
                elif any(x in q_lower for x in ['where_is', 'describe_region']):
                    head = 'region'
                elif 'severe' in q_lower:
                    head = 'severity'
                else:
                    head = 'category'
            
            if head in head_indices:
                head_indices[head].append(idx)
        
        return head_indices


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
