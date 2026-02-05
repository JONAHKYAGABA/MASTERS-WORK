"""
Evaluation Metrics for MIMIC-CXR VQA

Computes metrics for:
- Multi-head VQA classification (accuracy, F1 per head)
- Answer generation (BLEU, ROUGE, exact match)
- CheXpert classification (AUROC, F1)
- Scene graph generation (entity recall, relationship accuracy)
- Visual grounding (IoU, pointing accuracy)
- Attention analysis (entropy, plausibility)

All metrics integrated into single VQAMetrics class.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

# Try to import NLP metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


# =============================================================================
# IoU Utilities
# =============================================================================

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / (union_area + 1e-8)


def compute_batch_iou(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Compute IoU for batch of boxes."""
    B = pred_boxes.shape[0]
    ious = np.zeros(B)
    for i in range(B):
        ious[i] = compute_iou(pred_boxes[i], gt_boxes[i])
    return ious


# =============================================================================
# TEXT GENERATION METRICS
# =============================================================================

def compute_bleu(reference: str, hypothesis: str, n: int = 4) -> float:
    """Compute BLEU score between reference and hypothesis."""
    if not NLTK_AVAILABLE:
        return 0.0
    
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    if len(hyp_tokens) == 0:
        return 0.0
    
    weights = tuple([1.0 / n] * n)
    smoothing = SmoothingFunction().method1
    
    try:
        return sentence_bleu([ref_tokens], hyp_tokens, weights=weights, smoothing_function=smoothing)
    except:
        return 0.0


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 score."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return 0.0
    
    # Compute LCS
    m, n = len(ref_tokens), len(hyp_tokens)
    lcs = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                lcs[i][j] = lcs[i-1][j-1] + 1
            else:
                lcs[i][j] = max(lcs[i-1][j], lcs[i][j-1])
    
    lcs_len = lcs[m][n]
    
    precision = lcs_len / n if n > 0 else 0
    recall = lcs_len / m if m > 0 else 0
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return f1


def compute_exact_match(reference: str, hypothesis: str) -> float:
    """Compute exact match (after normalization)."""
    ref_norm = reference.lower().strip()
    hyp_norm = hypothesis.lower().strip()
    return 1.0 if ref_norm == hyp_norm else 0.0


def compute_word_overlap(reference: str, hypothesis: str) -> float:
    """Compute word overlap F1."""
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())
    
    if len(ref_words) == 0 or len(hyp_words) == 0:
        return 0.0
    
    overlap = ref_words & hyp_words
    precision = len(overlap) / len(hyp_words)
    recall = len(overlap) / len(ref_words)
    
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


# =============================================================================
# UNIFIED VQA METRICS CLASS
# =============================================================================

class VQAMetrics:
    """
    Unified metrics class for VQA evaluation.
    
    Includes:
    - Classification VQA metrics (always active)
    - Generation metrics (active if generated text provided)
    - CheXpert metrics (always active)
    - Scene graph metrics (active if sg data provided)
    - Grounding metrics (active if grounding data provided)
    - Attention metrics (active if attention data provided)
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated values."""
        # Base VQA Classification
        self.predictions = defaultdict(list)
        self.targets = defaultdict(list)
        self.question_types = []
        
        # Answer Generation
        self.generated_answers = []
        self.reference_answers = []
        self.template_answers = []
        
        # CheXpert
        self.chexpert_preds = []
        self.chexpert_labels = []
        self.chexpert_masks = []
        
        # Scene Graph
        self.sg_entity_preds = []
        self.sg_entity_targets = []
        self.sg_region_preds = []
        self.sg_region_targets = []
        self.sg_bbox_preds = []
        self.sg_bbox_targets = []
        
        # Grounding
        self.grounding_bbox_preds = []
        self.grounding_bbox_targets = []
        self.pointing_preds = []
        self.pointing_targets = []
        
        # Attention
        self.attention_entropies = []
        self.attention_plausibilities = []
    
    def update(
        self,
        outputs: Any,
        vqa_targets: Dict[str, torch.Tensor],
        chexpert_labels: torch.Tensor,
        chexpert_mask: torch.Tensor,
        question_types: List[str],
        # Optional: Reference answers for generation
        reference_answers: Optional[List[str]] = None,
        # Optional: Scene graph data
        gt_sg_entities: Optional[List[np.ndarray]] = None,
        gt_sg_regions: Optional[List[np.ndarray]] = None,
        gt_sg_bboxes: Optional[List[np.ndarray]] = None,
        # Optional: Grounding data
        gt_grounding_bboxes: Optional[np.ndarray] = None,
        gt_pointing_valid: Optional[np.ndarray] = None,
        # Optional: Attention ROIs
        gt_attention_rois: Optional[np.ndarray] = None,
    ):
        """Update metrics with batch of predictions."""
        # Get outputs
        vqa_logits = outputs.get('vqa_logits', {}) if isinstance(outputs, dict) else {}
        chexpert_logits = outputs.get('chexpert_logits') if isinstance(outputs, dict) else None
        generated_text = outputs.get('generated_answer_text') if isinstance(outputs, dict) else None
        template_text = outputs.get('template_answer') if isinstance(outputs, dict) else None
        
        # Store question types
        self.question_types.extend(question_types)
        
        # Map question types to heads
        head_indices = self._get_head_indices(question_types)
        
        # =====================================================
        # CLASSIFICATION VQA METRICS
        # =====================================================
        for head_name, indices in head_indices.items():
            if not indices or head_name not in vqa_logits:
                continue
            
            head_logits = vqa_logits[head_name]
            
            if len(indices) < head_logits.shape[0]:
                indices_tensor = torch.tensor(indices, device=head_logits.device)
                head_logits = head_logits[indices_tensor]
            
            preds = head_logits.argmax(dim=-1).cpu().numpy()
            self.predictions[head_name].extend(preds.tolist())
            
            if head_name in vqa_targets:
                head_targets = vqa_targets[head_name]
                if len(indices) < head_targets.shape[0]:
                    indices_tensor = torch.tensor(indices, device=head_targets.device)
                    head_targets = head_targets[indices_tensor]
                targets = head_targets.cpu().numpy()
                self.targets[head_name].extend(targets.tolist())
        
        # =====================================================
        # GENERATION METRICS
        # =====================================================
        if generated_text is not None:
            self.generated_answers.extend(generated_text)
        
        if template_text is not None:
            self.template_answers.extend(template_text)
        
        if reference_answers is not None:
            self.reference_answers.extend(reference_answers)
        
        # =====================================================
        # CHEXPERT METRICS
        # =====================================================
        if chexpert_logits is not None:
            probs = torch.sigmoid(chexpert_logits).cpu().numpy()
            self.chexpert_preds.append(probs)
            self.chexpert_labels.append(chexpert_labels.cpu().numpy())
            self.chexpert_masks.append(chexpert_mask.cpu().numpy())
        
        # =====================================================
        # SCENE GRAPH METRICS
        # =====================================================
        sg_outputs = outputs.get('scene_graph_outputs') if isinstance(outputs, dict) else None
        
        if sg_outputs is not None and gt_sg_entities is not None:
            entity_logits = sg_outputs['entity_logits']
            region_logits = sg_outputs['region_logits']
            bbox_preds = sg_outputs['bbox_preds']
            
            B = entity_logits.shape[0]
            
            for b in range(min(B, len(gt_sg_entities))):
                if gt_sg_entities[b] is None:
                    continue
                
                ent_pred = entity_logits[b].argmax(dim=-1).cpu().numpy()
                ent_target = gt_sg_entities[b]
                N = min(len(ent_pred), len(ent_target))
                
                self.sg_entity_preds.extend(ent_pred[:N].tolist())
                self.sg_entity_targets.extend(ent_target[:N].tolist())
                
                if gt_sg_regions is not None and b < len(gt_sg_regions) and gt_sg_regions[b] is not None:
                    reg_pred = region_logits[b].argmax(dim=-1).cpu().numpy()
                    reg_target = gt_sg_regions[b]
                    N = min(len(reg_pred), len(reg_target))
                    self.sg_region_preds.extend(reg_pred[:N].tolist())
                    self.sg_region_targets.extend(reg_target[:N].tolist())
                
                if gt_sg_bboxes is not None and b < len(gt_sg_bboxes) and gt_sg_bboxes[b] is not None:
                    box_pred = bbox_preds[b].cpu().numpy()
                    box_target = gt_sg_bboxes[b]
                    N = min(len(box_pred), len(box_target))
                    for i in range(N):
                        self.sg_bbox_preds.append(box_pred[i])
                        self.sg_bbox_targets.append(box_target[i])
        
        # =====================================================
        # GROUNDING METRICS
        # =====================================================
        grounding_outputs = outputs.get('grounding_outputs') if isinstance(outputs, dict) else None
        
        if grounding_outputs is not None and gt_grounding_bboxes is not None:
            bbox_pred = grounding_outputs['bbox_pred']
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.cpu().numpy()
            
            pointing_score = grounding_outputs['pointing_score']
            if isinstance(pointing_score, torch.Tensor):
                pointing_score = pointing_score.cpu().numpy()
            
            B = bbox_pred.shape[0]
            for b in range(B):
                self.grounding_bbox_preds.append(bbox_pred[b])
                self.grounding_bbox_targets.append(gt_grounding_bboxes[b])
                
                score = pointing_score[b, 0] if pointing_score.ndim > 1 else pointing_score[b]
                self.pointing_preds.append(score)
                
                if gt_pointing_valid is not None:
                    self.pointing_targets.append(gt_pointing_valid[b])
                else:
                    self.pointing_targets.append(1.0)
        
        # =====================================================
        # ATTENTION METRICS
        # =====================================================
        attention_weights = outputs.get('attention_weights') if isinstance(outputs, dict) else None
        
        if attention_weights is not None:
            attn = attention_weights.get('text_to_visual')
            if attn is not None:
                if isinstance(attn, torch.Tensor):
                    attn = attn.cpu().numpy()
                
                B = attn.shape[0]
                for b in range(B):
                    attn_map = attn[b].mean(axis=0)[0]
                    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                    
                    # Entropy
                    attn_prob = attn_map / (attn_map.sum() + 1e-8)
                    entropy = -np.sum(attn_prob * np.log(attn_prob + 1e-10))
                    self.attention_entropies.append(entropy)
                    
                    # Plausibility
                    if gt_attention_rois is not None and b < len(gt_attention_rois):
                        plaus = self._compute_attention_plausibility(attn_map, gt_attention_rois[b])
                        self.attention_plausibilities.append(plaus)
    
    def _get_head_indices(self, question_types: List[str]) -> Dict[str, List[int]]:
        """Map question types to answer heads."""
        head_indices = {'binary': [], 'category': [], 'region': [], 'severity': []}
        
        binary_keywords = ['abnormal', 'normal', 'has_', 'is_']
        region_keywords = ['where', 'describe_region', 'location']
        severity_keywords = ['severe', 'severity']
        
        for idx, q_type in enumerate(question_types):
            q_lower = q_type.lower()
            
            if any(k in q_lower for k in severity_keywords):
                head_indices['severity'].append(idx)
            elif any(k in q_lower for k in region_keywords):
                head_indices['region'].append(idx)
            elif any(k in q_lower for k in binary_keywords):
                head_indices['binary'].append(idx)
            else:
                head_indices['category'].append(idx)
        
        return head_indices
    
    def _compute_attention_plausibility(self, attn_map: np.ndarray, roi_bbox: np.ndarray, threshold: float = 0.5) -> float:
        """Compute IoU between thresholded attention and ROI."""
        sqrt_len = int(np.sqrt(len(attn_map)))
        if sqrt_len * sqrt_len == len(attn_map):
            attn_2d = attn_map.reshape(sqrt_len, sqrt_len)
        else:
            attn_2d = attn_map[:49].reshape(7, 7)
        
        h, w = attn_2d.shape
        attn_mask = (attn_2d >= threshold).astype(float)
        
        roi_mask = np.zeros((h, w))
        x1 = int(roi_bbox[0] * w)
        y1 = int(roi_bbox[1] * h)
        x2 = int(roi_bbox[2] * w)
        y2 = int(roi_bbox[3] * h)
        
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        roi_mask[y1:y2, x1:x2] = 1.0
        
        intersection = np.logical_and(attn_mask, roi_mask).sum()
        union = np.logical_or(attn_mask, roi_mask).sum()
        
        return intersection / (union + 1e-8)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        metrics = {}
        
        # =====================================================
        # CLASSIFICATION VQA METRICS
        # =====================================================
        overall_correct = 0
        overall_total = 0
        
        for head_name in ['binary', 'category', 'region', 'severity']:
            preds = np.array(self.predictions[head_name])
            targets = np.array(self.targets[head_name])
            
            if len(preds) == 0 or len(targets) == 0:
                continue
            
            valid_mask = targets >= 0
            preds = preds[valid_mask]
            targets = targets[valid_mask]
            
            if len(preds) == 0:
                continue
            
            acc = accuracy_score(targets, preds)
            metrics[f'{head_name}_accuracy'] = acc
            
            try:
                f1 = f1_score(targets, preds, average='macro', zero_division=0)
                metrics[f'{head_name}_f1'] = f1
            except:
                metrics[f'{head_name}_f1'] = 0.0
            
            if head_name == 'binary':
                try:
                    metrics['binary_precision'] = precision_score(targets, preds, average='binary', zero_division=0)
                    metrics['binary_recall'] = recall_score(targets, preds, average='binary', zero_division=0)
                except:
                    pass
            
            overall_correct += (preds == targets).sum()
            overall_total += len(preds)
        
        metrics['classification_accuracy'] = overall_correct / overall_total if overall_total > 0 else 0.0
        
        # =====================================================
        # GENERATION METRICS
        # =====================================================
        if self.generated_answers and self.reference_answers:
            bleu_scores = []
            rouge_scores = []
            exact_matches = []
            word_overlaps = []
            
            for gen, ref in zip(self.generated_answers, self.reference_answers):
                if gen and ref:
                    bleu_scores.append(compute_bleu(ref, gen))
                    rouge_scores.append(compute_rouge_l(ref, gen))
                    exact_matches.append(compute_exact_match(ref, gen))
                    word_overlaps.append(compute_word_overlap(ref, gen))
            
            if bleu_scores:
                metrics['generation_bleu'] = np.mean(bleu_scores)
                metrics['generation_rouge_l'] = np.mean(rouge_scores)
                metrics['generation_exact_match'] = np.mean(exact_matches)
                metrics['generation_word_overlap'] = np.mean(word_overlaps)
        
        # Template answer metrics
        if self.template_answers and self.reference_answers:
            template_bleu = []
            template_overlap = []
            
            for tmpl, ref in zip(self.template_answers, self.reference_answers):
                if tmpl and ref:
                    template_bleu.append(compute_bleu(ref, tmpl))
                    template_overlap.append(compute_word_overlap(ref, tmpl))
            
            if template_bleu:
                metrics['template_bleu'] = np.mean(template_bleu)
                metrics['template_word_overlap'] = np.mean(template_overlap)
        
        # =====================================================
        # CHEXPERT METRICS
        # =====================================================
        if self.chexpert_preds:
            try:
                all_probs = np.concatenate(self.chexpert_preds, axis=0)
                all_labels = np.concatenate(self.chexpert_labels, axis=0)
                all_masks = np.concatenate(self.chexpert_masks, axis=0)
                
                aurocs = []
                for i in range(all_labels.shape[1]):
                    valid = all_masks[:, i] > 0.5
                    if valid.sum() < 10:
                        continue
                    
                    labels_i = all_labels[valid, i]
                    probs_i = all_probs[valid, i]
                    
                    if len(np.unique(labels_i)) < 2:
                        continue
                    
                    try:
                        auroc = roc_auc_score(labels_i, probs_i)
                        aurocs.append(auroc)
                    except ValueError:
                        pass
                
                if aurocs:
                    metrics['chexpert_auroc'] = np.mean(aurocs)
            except:
                pass
        
        # =====================================================
        # SCENE GRAPH METRICS
        # =====================================================
        if self.sg_entity_preds:
            ent_preds = np.array(self.sg_entity_preds)
            ent_targets = np.array(self.sg_entity_targets)
            metrics['sg_entity_accuracy'] = accuracy_score(ent_targets, ent_preds)
            metrics['sg_entity_recall'] = recall_score(ent_targets, ent_preds, average='macro', zero_division=0)
        
        if self.sg_region_preds:
            reg_preds = np.array(self.sg_region_preds)
            reg_targets = np.array(self.sg_region_targets)
            metrics['sg_region_accuracy'] = accuracy_score(reg_targets, reg_preds)
        
        if self.sg_bbox_preds:
            ious = compute_batch_iou(np.array(self.sg_bbox_preds), np.array(self.sg_bbox_targets))
            metrics['sg_mean_iou'] = float(ious.mean())
            metrics['sg_iou_50'] = float((ious >= 0.5).mean())
        
        # =====================================================
        # GROUNDING METRICS
        # =====================================================
        if self.grounding_bbox_preds:
            pred_boxes = np.array(self.grounding_bbox_preds)
            gt_boxes = np.array(self.grounding_bbox_targets)
            ious = compute_batch_iou(pred_boxes, gt_boxes)
            
            metrics['grounding_mean_iou'] = float(ious.mean())
            metrics['grounding_acc_iou25'] = float((ious >= 0.25).mean())
            metrics['grounding_acc_iou50'] = float((ious >= 0.5).mean())
            
            if self.pointing_preds:
                pointing_preds = np.array(self.pointing_preds) > 0.5
                pointing_targets = np.array(self.pointing_targets) > 0.5
                metrics['pointing_accuracy'] = accuracy_score(pointing_targets, pointing_preds)
        
        # =====================================================
        # ATTENTION METRICS
        # =====================================================
        if self.attention_entropies:
            metrics['attention_mean_entropy'] = float(np.mean(self.attention_entropies))
            metrics['attention_focused_ratio'] = float(np.mean([e < 2.0 for e in self.attention_entropies]))
        
        if self.attention_plausibilities:
            metrics['attention_plausibility'] = float(np.mean(self.attention_plausibilities))
        
        return metrics


# =============================================================================
# CONFUSION MATRIX UTILITIES
# =============================================================================

def compute_confusion_matrix(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute confusion matrix."""
    valid_mask = targets >= 0
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    return confusion_matrix(targets, predictions, labels=list(range(num_classes)))


def compute_per_class_metrics(predictions: np.ndarray, targets: np.ndarray, class_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision, recall, F1."""
    valid_mask = targets >= 0
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    
    classes = np.unique(np.concatenate([predictions, targets]))
    
    results = {}
    for c in classes:
        tp = ((predictions == c) & (targets == c)).sum()
        fp = ((predictions == c) & (targets != c)).sum()
        fn = ((predictions != c) & (targets == c)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_name = class_names[c] if class_names and c < len(class_names) else str(c)
        results[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int((targets == c).sum())
        }
    
    return results
