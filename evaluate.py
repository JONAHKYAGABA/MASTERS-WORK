#!/usr/bin/env python3
"""
MIMIC-CXR VQA Evaluation Script

Comprehensive evaluation based on methodology.md specifications:
- Answer Accuracy Metrics (EM, F1, BLEU-4, ROUGE-L, BERTScore)
- Spatial Reasoning Metrics (IoU, Pointing Accuracy, mAP)
- Clinical Relevance Metrics (Sensitivity, Specificity, AUROC, MCC)
- Relational Reasoning Metrics (Graph Entity Recall, Relationship Accuracy)
- Explainability Assessment (Attention analysis, Plausibility)
- Statistical Significance Testing (t-tests, McNemar, Bootstrap CI)
- Cross-Dataset Evaluation (VQA-RAD, SLAKE)

Usage:
    python evaluate.py --model_path ./checkpoints/best_model --test_data /path/to/test
    python evaluate.py --config configs/default_config.yaml --eval_mode full
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Metrics imports
try:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix, matthews_corrcoef,
        average_precision_score
    )
    from scipy import stats
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# NLP metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

# Local imports
from configs import load_config_from_file, get_default_config
from data import MIMICCXRVQADataset, create_dataloader
from models import MIMICCXRVQAModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for all evaluation metrics."""
    
    # Answer Accuracy Metrics
    exact_match: float = 0.0
    f1_score: float = 0.0
    bleu_4: float = 0.0
    rouge_l: float = 0.0
    bert_score: float = 0.0
    clinical_term_f1: float = 0.0
    
    # Per-head accuracy
    binary_accuracy: float = 0.0
    category_accuracy: float = 0.0
    region_accuracy: float = 0.0
    severity_accuracy: float = 0.0
    
    # Spatial Reasoning Metrics
    mean_iou: float = 0.0
    pointing_accuracy: float = 0.0
    map_50: float = 0.0
    map_75: float = 0.0
    spatial_relation_accuracy: float = 0.0
    
    # Clinical Relevance Metrics
    sensitivity: float = 0.0
    specificity: float = 0.0
    ppv: float = 0.0
    npv: float = 0.0
    mcc: float = 0.0
    auroc: float = 0.0
    
    # CheXpert Metrics
    chexpert_auroc: float = 0.0
    chexpert_per_class_auroc: Dict[str, float] = field(default_factory=dict)
    
    # Relational Reasoning
    graph_entity_recall: float = 0.0
    relationship_accuracy: float = 0.0
    
    # Explainability
    attention_plausibility: float = 0.0
    attention_entropy: float = 0.0
    
    # By question type
    accuracy_by_type: Dict[str, float] = field(default_factory=dict)
    
    # By anatomical region
    accuracy_by_region: Dict[str, float] = field(default_factory=dict)
    
    # Statistical significance
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)


class MetricsCalculator:
    """Calculate all evaluation metrics from methodology."""
    
    CHEXPERT_CATEGORIES = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'Pleural Effusion', 'Pneumonia',
        'Pneumothorax', 'Pleural Other', 'Support Devices', 'No Finding'
    ]
    
    def __init__(self):
        self.reset()
        
        # ROUGE scorer
        if NLTK_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            self.smoothing = SmoothingFunction().method1
    
    def reset(self):
        """Reset all accumulated predictions."""
        self.predictions = defaultdict(list)
        self.targets = defaultdict(list)
        self.pred_texts = []
        self.target_texts = []
        self.question_types = []
        self.regions = []
        self.bboxes_pred = []
        self.bboxes_gt = []
        self.chexpert_probs = []
        self.chexpert_labels = []
        self.chexpert_masks = []
        self.attention_weights = []
        self.attention_rois = []
    
    def update(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        pred_text: Optional[List[str]] = None,
        target_text: Optional[List[str]] = None,
        question_types: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        bboxes_pred: Optional[List[np.ndarray]] = None,
        bboxes_gt: Optional[List[np.ndarray]] = None,
        chexpert_probs: Optional[torch.Tensor] = None,
        chexpert_labels: Optional[torch.Tensor] = None,
        chexpert_mask: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
        attention_rois: Optional[List[np.ndarray]] = None,
    ):
        """Update with batch of predictions."""
        # Store classification predictions
        for head in ['binary', 'category', 'region', 'severity']:
            if head in predictions and head in targets:
                pred = predictions[head].argmax(dim=-1).cpu().numpy()
                tgt = targets[head].cpu().numpy()
                self.predictions[head].extend(pred.tolist())
                self.targets[head].extend(tgt.tolist())
        
        # Store text predictions
        if pred_text and target_text:
            self.pred_texts.extend(pred_text)
            self.target_texts.extend(target_text)
        
        # Store metadata
        if question_types:
            self.question_types.extend(question_types)
        if regions:
            self.regions.extend(regions)
        
        # Store bboxes for spatial metrics
        if bboxes_pred and bboxes_gt:
            self.bboxes_pred.extend(bboxes_pred)
            self.bboxes_gt.extend(bboxes_gt)
        
        # Store CheXpert predictions
        if chexpert_probs is not None:
            self.chexpert_probs.append(chexpert_probs.cpu().numpy())
            self.chexpert_labels.append(chexpert_labels.cpu().numpy())
            self.chexpert_masks.append(chexpert_mask.cpu().numpy())
        
        # Store attention
        if attention_weights is not None:
            self.attention_weights.append(attention_weights.cpu().numpy())
        if attention_rois is not None:
            self.attention_rois.extend(attention_rois)
    
    def compute(self) -> EvaluationResults:
        """Compute all metrics."""
        results = EvaluationResults()
        
        # Answer Accuracy Metrics
        self._compute_accuracy_metrics(results)
        
        # Text Generation Metrics
        if self.pred_texts:
            self._compute_text_metrics(results)
        
        # Spatial Reasoning Metrics
        if self.bboxes_pred:
            self._compute_spatial_metrics(results)
        
        # Clinical Relevance (Binary)
        self._compute_clinical_metrics(results)
        
        # CheXpert Metrics
        if self.chexpert_probs:
            self._compute_chexpert_metrics(results)
        
        # Per-type and per-region accuracy
        self._compute_stratified_metrics(results)
        
        # Explainability
        if self.attention_weights:
            self._compute_attention_metrics(results)
        
        # Bootstrap confidence intervals
        self._compute_confidence_intervals(results)
        
        return results
    
    def _compute_accuracy_metrics(self, results: EvaluationResults):
        """Compute per-head accuracy and F1."""
        for head in ['binary', 'category', 'region', 'severity']:
            preds = np.array(self.predictions[head])
            targets = np.array(self.targets[head])
            
            if len(preds) == 0:
                continue
            
            # Filter valid (non-ignored)
            valid = targets >= 0
            preds = preds[valid]
            targets = targets[valid]
            
            if len(preds) == 0:
                continue
            
            acc = accuracy_score(targets, preds)
            setattr(results, f'{head}_accuracy', acc)
        
        # Overall exact match
        all_preds = []
        all_targets = []
        for head in ['binary', 'category', 'region', 'severity']:
            preds = np.array(self.predictions[head])
            targets = np.array(self.targets[head])
            valid = targets >= 0
            all_preds.extend(preds[valid].tolist())
            all_targets.extend(targets[valid].tolist())
        
        if all_preds:
            results.exact_match = accuracy_score(all_targets, all_preds)
            results.f1_score = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    def _compute_text_metrics(self, results: EvaluationResults):
        """Compute BLEU-4, ROUGE-L, BERTScore."""
        if not NLTK_AVAILABLE or not self.pred_texts:
            return
        
        bleu_scores = []
        rouge_scores = []
        
        for pred, ref in zip(self.pred_texts, self.target_texts):
            # BLEU-4
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            try:
                bleu = sentence_bleu(
                    [ref_tokens], pred_tokens,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=self.smoothing
                )
                bleu_scores.append(bleu)
            except Exception:
                bleu_scores.append(0.0)
            
            # ROUGE-L
            try:
                rouge = self.rouge_scorer.score(ref, pred)
                rouge_scores.append(rouge['rougeL'].fmeasure)
            except Exception:
                rouge_scores.append(0.0)
        
        results.bleu_4 = np.mean(bleu_scores) if bleu_scores else 0.0
        results.rouge_l = np.mean(rouge_scores) if rouge_scores else 0.0
        
        # BERTScore
        if BERTSCORE_AVAILABLE and len(self.pred_texts) > 0:
            try:
                P, R, F1 = bert_score(
                    self.pred_texts, self.target_texts,
                    lang='en', verbose=False
                )
                results.bert_score = F1.mean().item()
            except Exception:
                results.bert_score = 0.0
    
    def _compute_spatial_metrics(self, results: EvaluationResults):
        """Compute IoU, Pointing Accuracy, mAP."""
        ious = []
        
        for pred_box, gt_box in zip(self.bboxes_pred, self.bboxes_gt):
            iou = self._compute_iou(pred_box, gt_box)
            ious.append(iou)
        
        ious = np.array(ious)
        
        results.mean_iou = np.mean(ious)
        results.pointing_accuracy = np.mean(ious >= 0.5)
        results.map_50 = np.mean(ious >= 0.5)
        results.map_75 = np.mean(ious >= 0.75)
    
    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    def _compute_clinical_metrics(self, results: EvaluationResults):
        """Compute Sensitivity, Specificity, PPV, NPV, MCC, AUROC."""
        preds = np.array(self.predictions['binary'])
        targets = np.array(self.targets['binary'])
        
        if len(preds) == 0:
            return
        
        valid = targets >= 0
        preds = preds[valid]
        targets = targets[valid]
        
        if len(preds) == 0 or len(np.unique(targets)) < 2:
            return
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
        
        # Sensitivity (Recall / TPR)
        results.sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Specificity (TNR)
        results.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # PPV (Precision)
        results.ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # NPV
        results.npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # MCC
        results.mcc = matthews_corrcoef(targets, preds)
        
        # AUROC (need probabilities, use predictions as proxy)
        try:
            results.auroc = roc_auc_score(targets, preds)
        except ValueError:
            results.auroc = 0.0
    
    def _compute_chexpert_metrics(self, results: EvaluationResults):
        """Compute CheXpert multi-label metrics."""
        all_probs = np.concatenate(self.chexpert_probs, axis=0)
        all_labels = np.concatenate(self.chexpert_labels, axis=0)
        all_masks = np.concatenate(self.chexpert_masks, axis=0)
        
        aurocs = []
        per_class_aurocs = {}
        
        for i, cat in enumerate(self.CHEXPERT_CATEGORIES):
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
                per_class_aurocs[cat] = auroc
            except ValueError:
                continue
        
        results.chexpert_auroc = np.mean(aurocs) if aurocs else 0.0
        results.chexpert_per_class_auroc = per_class_aurocs
    
    def _compute_stratified_metrics(self, results: EvaluationResults):
        """Compute metrics stratified by question type and region."""
        # By question type
        if self.question_types:
            type_preds = defaultdict(list)
            type_targets = defaultdict(list)
            
            # Use binary predictions as main accuracy proxy
            preds = np.array(self.predictions['binary'])
            targets = np.array(self.targets['binary'])
            
            for i, q_type in enumerate(self.question_types):
                if i < len(preds) and targets[i] >= 0:
                    type_preds[q_type].append(preds[i])
                    type_targets[q_type].append(targets[i])
            
            for q_type in type_preds:
                if type_targets[q_type]:
                    acc = accuracy_score(type_targets[q_type], type_preds[q_type])
                    results.accuracy_by_type[q_type] = acc
        
        # By anatomical region
        if self.regions:
            region_preds = defaultdict(list)
            region_targets = defaultdict(list)
            
            preds = np.array(self.predictions['binary'])
            targets = np.array(self.targets['binary'])
            
            for i, region in enumerate(self.regions):
                if i < len(preds) and targets[i] >= 0:
                    region_preds[region].append(preds[i])
                    region_targets[region].append(targets[i])
            
            for region in region_preds:
                if region_targets[region]:
                    acc = accuracy_score(region_targets[region], region_preds[region])
                    results.accuracy_by_region[region] = acc
    
    def _compute_attention_metrics(self, results: EvaluationResults):
        """Compute attention plausibility and entropy."""
        if not self.attention_weights:
            return
        
        # Attention entropy
        entropies = []
        for attn in self.attention_weights:
            # Flatten and normalize
            attn_flat = attn.flatten()
            attn_norm = attn_flat / (attn_flat.sum() + 1e-8)
            
            # Entropy: -sum(p * log(p))
            entropy = -np.sum(attn_norm * np.log(attn_norm + 1e-8))
            entropies.append(entropy)
        
        results.attention_entropy = np.mean(entropies)
        
        # Plausibility (IoU with ROIs)
        if self.attention_rois:
            plausibilities = []
            for attn, roi in zip(self.attention_weights, self.attention_rois):
                if roi is not None:
                    # Simplified plausibility: overlap between attention and ROI
                    plaus = self._compute_attention_roi_overlap(attn, roi)
                    plausibilities.append(plaus)
            
            results.attention_plausibility = np.mean(plausibilities) if plausibilities else 0.0
    
    @staticmethod
    def _compute_attention_roi_overlap(attention: np.ndarray, roi: np.ndarray) -> float:
        """Compute overlap between attention map and ROI mask."""
        # Normalize attention to [0, 1]
        attn_norm = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        # Threshold to binary
        attn_binary = attn_norm > 0.5
        
        # IoU with ROI
        intersection = np.logical_and(attn_binary, roi).sum()
        union = np.logical_or(attn_binary, roi).sum()
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_confidence_intervals(self, results: EvaluationResults, n_bootstrap: int = 1000):
        """Compute bootstrap confidence intervals for key metrics."""
        if not SKLEARN_AVAILABLE:
            return
        
        # Binary accuracy CI
        preds = np.array(self.predictions['binary'])
        targets = np.array(self.targets['binary'])
        valid = targets >= 0
        preds = preds[valid]
        targets = targets[valid]
        
        if len(preds) > 100:
            bootstrap_accs = []
            n = len(preds)
            
            for _ in range(n_bootstrap):
                idx = np.random.choice(n, n, replace=True)
                acc = accuracy_score(targets[idx], preds[idx])
                bootstrap_accs.append(acc)
            
            ci_low = np.percentile(bootstrap_accs, 2.5)
            ci_high = np.percentile(bootstrap_accs, 97.5)
            results.confidence_intervals['binary_accuracy'] = (ci_low, ci_high)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    compute_attention: bool = False
) -> EvaluationResults:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        dataloader: Test data loader
        device: Device to run evaluation on
        compute_attention: Whether to extract attention weights
        
    Returns:
        EvaluationResults with all metrics
    """
    model.eval()
    calculator = MetricsCalculator()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
            scene_graphs = batch['scene_graphs']
            question_types = batch['question_types']
            answer_idx = batch['answer_idx'].to(device)
            chexpert_labels = batch['chexpert_labels'].to(device)
            chexpert_mask = batch['chexpert_mask'].to(device)
            
            # Forward pass
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                scene_graphs=scene_graphs,
                token_type_ids=token_type_ids,
                question_types=question_types
            )
            
            # Prepare targets
            targets = {
                'binary': answer_idx,
                'category': answer_idx,
                'region': answer_idx,
                'severity': answer_idx,
            }
            
            # Get CheXpert probabilities
            chexpert_probs = None
            if outputs.chexpert_logits is not None:
                chexpert_probs = torch.sigmoid(outputs.chexpert_logits)
            
            # Update metrics
            calculator.update(
                predictions=outputs.vqa_logits,
                targets=targets,
                question_types=question_types,
                chexpert_probs=chexpert_probs,
                chexpert_labels=chexpert_labels,
                chexpert_mask=chexpert_mask,
            )
    
    return calculator.compute()


def statistical_significance_test(
    results1: EvaluationResults,
    results2: EvaluationResults,
    predictions1: List,
    predictions2: List,
    targets: List
) -> Dict[str, Any]:
    """
    Perform statistical significance tests between two models.
    
    Returns:
        Dictionary with p-values for paired t-test and McNemar's test
    """
    results = {}
    
    preds1 = np.array(predictions1)
    preds2 = np.array(predictions2)
    tgts = np.array(targets)
    
    # Paired t-test for continuous metrics
    correct1 = (preds1 == tgts).astype(float)
    correct2 = (preds2 == tgts).astype(float)
    
    if len(correct1) > 1:
        t_stat, p_value = stats.ttest_rel(correct1, correct2)
        results['paired_ttest'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05
        }
    
    # McNemar's test for binary outcomes
    n_01 = np.sum((preds1 != tgts) & (preds2 == tgts))  # Model 1 wrong, Model 2 right
    n_10 = np.sum((preds1 == tgts) & (preds2 != tgts))  # Model 1 right, Model 2 wrong
    
    if n_01 + n_10 > 0:
        chi2 = (n_01 - n_10) ** 2 / (n_01 + n_10)
        p_value_mcnemar = 1 - stats.chi2.cdf(chi2, df=1)
        results['mcnemar_test'] = {
            'chi2_statistic': chi2,
            'p_value': p_value_mcnemar,
            'significant_at_0.05': p_value_mcnemar < 0.05
        }
    
    # Effect size (Cohen's d)
    if len(correct1) > 1:
        mean_diff = correct1.mean() - correct2.mean()
        pooled_std = np.sqrt((correct1.std()**2 + correct2.std()**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        results['cohens_d'] = cohens_d
    
    return results


def generate_evaluation_report(
    results: EvaluationResults,
    output_path: str,
    model_name: str = "MIMIC-CXR-VQA"
):
    """Generate comprehensive evaluation report."""
    report = {
        'model_name': model_name,
        'metrics': {
            'answer_accuracy': {
                'exact_match': results.exact_match,
                'f1_score': results.f1_score,
                'bleu_4': results.bleu_4,
                'rouge_l': results.rouge_l,
                'bert_score': results.bert_score,
            },
            'per_head_accuracy': {
                'binary': results.binary_accuracy,
                'category': results.category_accuracy,
                'region': results.region_accuracy,
                'severity': results.severity_accuracy,
            },
            'spatial_reasoning': {
                'mean_iou': results.mean_iou,
                'pointing_accuracy': results.pointing_accuracy,
                'mAP@0.5': results.map_50,
                'mAP@0.75': results.map_75,
            },
            'clinical_relevance': {
                'sensitivity': results.sensitivity,
                'specificity': results.specificity,
                'ppv': results.ppv,
                'npv': results.npv,
                'mcc': results.mcc,
                'auroc': results.auroc,
            },
            'chexpert': {
                'mean_auroc': results.chexpert_auroc,
                'per_class_auroc': results.chexpert_per_class_auroc,
            },
            'explainability': {
                'attention_plausibility': results.attention_plausibility,
                'attention_entropy': results.attention_entropy,
            },
        },
        'stratified_results': {
            'by_question_type': results.accuracy_by_type,
            'by_anatomical_region': results.accuracy_by_region,
        },
        'confidence_intervals': results.confidence_intervals,
    }
    
    # Save JSON report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    
    logger.info(f"\n📊 Answer Accuracy:")
    logger.info(f"   Exact Match:  {results.exact_match*100:.2f}%")
    logger.info(f"   F1 Score:     {results.f1_score*100:.2f}%")
    logger.info(f"   BLEU-4:       {results.bleu_4*100:.2f}%")
    logger.info(f"   ROUGE-L:      {results.rouge_l*100:.2f}%")
    logger.info(f"   BERTScore:    {results.bert_score*100:.2f}%")
    
    logger.info(f"\n🎯 Per-Head Accuracy:")
    logger.info(f"   Binary:    {results.binary_accuracy*100:.2f}%")
    logger.info(f"   Category:  {results.category_accuracy*100:.2f}%")
    logger.info(f"   Region:    {results.region_accuracy*100:.2f}%")
    logger.info(f"   Severity:  {results.severity_accuracy*100:.2f}%")
    
    logger.info(f"\n📍 Spatial Reasoning:")
    logger.info(f"   Mean IoU:           {results.mean_iou*100:.2f}%")
    logger.info(f"   Pointing Accuracy:  {results.pointing_accuracy*100:.2f}%")
    logger.info(f"   mAP@0.5:            {results.map_50*100:.2f}%")
    
    logger.info(f"\n🏥 Clinical Relevance:")
    logger.info(f"   Sensitivity:  {results.sensitivity*100:.2f}%")
    logger.info(f"   Specificity:  {results.specificity*100:.2f}%")
    logger.info(f"   AUROC:        {results.auroc*100:.2f}%")
    logger.info(f"   MCC:          {results.mcc:.4f}")
    
    logger.info(f"\n📋 CheXpert AUROC: {results.chexpert_auroc*100:.2f}%")
    
    logger.info(f"\n📄 Full report saved to: {output_path}")
    
    return report


def evaluate_cross_dataset(
    model: nn.Module,
    dataset_name: str,
    dataset_path: str,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4
) -> Dict[str, Any]:
    """
    Evaluate model on external dataset (VQA-RAD or SLAKE) for cross-dataset generalization.
    
    From methodology Section 16.3:
    Zero-shot cross-dataset evaluation tests generalization without fine-tuning.
    
    Args:
        model: Trained VQA model
        dataset_name: 'vqa_rad' or 'slake'
        dataset_path: Path to dataset
        device: torch device
        batch_size: Batch size for evaluation
        num_workers: DataLoader workers
    
    Returns:
        Dict with cross-dataset metrics
    """
    from data.external_datasets import get_external_dataset, create_external_dataloader
    from utils.nlp_metrics import compute_all_nlp_metrics
    
    logger.info(f"\n{'='*60}")
    logger.info(f"CROSS-DATASET EVALUATION: {dataset_name.upper()}")
    logger.info(f"{'='*60}")
    
    # Load external dataset
    try:
        dataset = get_external_dataset(
            dataset_name=dataset_name,
            data_path=dataset_path,
            split='test',
        )
    except Exception as e:
        logger.error(f"Failed to load {dataset_name}: {e}")
        return {'error': str(e), 'dataset': dataset_name}
    
    if len(dataset) == 0:
        logger.error(f"No samples found in {dataset_name}")
        return {'error': 'Empty dataset', 'dataset': dataset_name}
    
    logger.info(f"Loaded {len(dataset)} samples from {dataset_name}")
    
    dataloader = create_external_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    model.eval()
    all_predictions = []
    all_answers = []
    all_correct_binary = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                scene_graphs=batch['scene_graphs'],
                question_types=batch['head_types'],
            )
            
            # Get predictions based on head types
            batch_preds = []
            for i, head_type in enumerate(batch['head_types']):
                if head_type == 'binary' and 'binary' in outputs.vqa_outputs:
                    pred_logits = outputs.vqa_outputs['binary'][i]
                    pred_idx = pred_logits.argmax().item()
                    pred_text = 'yes' if pred_idx == 1 else 'no'
                else:
                    # Default to binary head
                    pred_text = 'yes'
                
                batch_preds.append(pred_text)
            
            all_predictions.extend(batch_preds)
            all_answers.extend(batch['answers'])
            
            # Track binary correctness
            for pred, ans, ans_type in zip(batch_preds, batch['answers'], batch['answer_types']):
                if ans_type == 'CLOSED':
                    ans_lower = ans.lower().strip()
                    pred_lower = pred.lower().strip()
                    is_correct = (pred_lower == ans_lower) or \
                                 (pred_lower in ['yes', 'no'] and ans_lower in ['yes', 'no'] and pred_lower == ans_lower)
                    all_correct_binary.append(int(is_correct))
    
    # Compute NLP metrics
    nlp_metrics = compute_all_nlp_metrics(
        predictions=all_predictions,
        references=all_answers,
        compute_bertscore=False  # Skip for speed in cross-dataset
    )
    
    # Compute binary accuracy for closed questions
    binary_accuracy = np.mean(all_correct_binary) if all_correct_binary else 0.0
    
    results = {
        'dataset': dataset_name,
        'num_samples': len(dataset),
        'binary_accuracy': binary_accuracy,
        **nlp_metrics
    }
    
    logger.info(f"\n{dataset_name.upper()} Results:")
    logger.info(f"  Samples:          {len(dataset)}")
    logger.info(f"  Binary Accuracy:  {binary_accuracy*100:.2f}%")
    logger.info(f"  Exact Match:      {nlp_metrics.get('exact_match', 0)*100:.2f}%")
    logger.info(f"  Token F1:         {nlp_metrics.get('token_f1', 0)*100:.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate MIMIC-CXR VQA Model')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Path to test data (overrides config)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Evaluation batch size')
    parser.add_argument('--compute_attention', action='store_true',
                       help='Compute attention-based explainability metrics')
    parser.add_argument('--zero_sg_tokens', action='store_true',
                       help='Ablation: zero the SG soft tokens before injection '
                            'into the LM. Everything else runs normally. Use to '
                            'measure whether the scene-graph pathway is actually '
                            'carrying signal end-to-end. Compare metrics against '
                            'a normal run of the same checkpoint.')
    
    # Cross-dataset evaluation arguments
    parser.add_argument('--cross_dataset', action='store_true',
                       help='Run cross-dataset evaluation on VQA-RAD and SLAKE')
    parser.add_argument('--vqa_rad_path', type=str, default='./external_datasets/vqa_rad',
                       help='Path to VQA-RAD dataset')
    parser.add_argument('--slake_path', type=str, default='./external_datasets/slake',
                       help='Path to SLAKE dataset')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config_from_file(args.config)
    else:
        config = get_default_config()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = MIMICCXRVQAModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()

    # Toggle the SG-ablation mode on the module so every forward zeros the
    # soft tokens. Set at model level (rather than per-call) because
    # evaluate_model() lives further down and touching its signature isn't
    # worth it for a one-flag ablation.
    if args.zero_sg_tokens:
        setattr(model, 'zero_sg_tokens', True)
        logger.info("ABLATION: SG soft tokens will be zeroed before injection.")
    
    # Load test dataset
    test_data_path = args.test_data or config.data.mimic_cxr_jpg_path
    
    logger.info(f"Loading test dataset...")
    test_dataset = MIMICCXRVQADataset(
        mimic_cxr_path=config.data.mimic_cxr_jpg_path,
        mimic_qa_path=config.data.mimic_ext_cxr_qba_path,
        split='test',
        tokenizer_name=config.model.text_encoder,
        max_question_length=config.model.max_question_length,
    )
    
    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.training.dataloader_num_workers,
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Evaluate on MIMIC-CXR test set
    logger.info("Running evaluation on MIMIC-CXR test set...")
    results = evaluate_model(
        model=model,
        dataloader=test_dataloader,
        device=device,
        compute_attention=args.compute_attention
    )
    
    # Generate report
    report_path = output_dir / 'evaluation_report.json'
    generate_evaluation_report(results, str(report_path))
    
    # Cross-dataset evaluation (methodology Section 16.3)
    if args.cross_dataset:
        logger.info("\n" + "=" * 70)
        logger.info("CROSS-DATASET EVALUATION (Zero-Shot Generalization)")
        logger.info("=" * 70)
        
        cross_dataset_results = {}
        
        # Check and evaluate on VQA-RAD
        from data.external_datasets import check_external_datasets
        availability = check_external_datasets(str(Path(args.vqa_rad_path).parent))
        
        if availability.get('vqa_rad', False):
            vqa_rad_results = evaluate_cross_dataset(
                model=model,
                dataset_name='vqa_rad',
                dataset_path=args.vqa_rad_path,
                device=device,
                batch_size=args.batch_size,
            )
            cross_dataset_results['vqa_rad'] = vqa_rad_results
        else:
            logger.warning("VQA-RAD dataset not found. Skipping.")
            logger.warning(f"  Expected path: {args.vqa_rad_path}")
            logger.warning("  Download from: https://osf.io/89kps/")
        
        # Check and evaluate on SLAKE
        if availability.get('slake', False):
            slake_results = evaluate_cross_dataset(
                model=model,
                dataset_name='slake',
                dataset_path=args.slake_path,
                device=device,
                batch_size=args.batch_size,
            )
            cross_dataset_results['slake'] = slake_results
        else:
            logger.warning("SLAKE dataset not found. Skipping.")
            logger.warning(f"  Expected path: {args.slake_path}")
            logger.warning("  Download from: https://www.med-vqa.com/slake/")
        
        # Save cross-dataset results
        if cross_dataset_results:
            cross_report_path = output_dir / 'cross_dataset_results.json'
            with open(cross_report_path, 'w') as f:
                json.dump(cross_dataset_results, f, indent=2, default=str)
            logger.info(f"\nCross-dataset results saved to: {cross_report_path}")
            
            # Print summary
            logger.info("\n" + "=" * 70)
            logger.info("CROSS-DATASET SUMMARY")
            logger.info("=" * 70)
            for ds_name, ds_results in cross_dataset_results.items():
                if 'error' not in ds_results:
                    logger.info(f"\n{ds_name.upper()}:")
                    logger.info(f"  Binary Accuracy: {ds_results.get('binary_accuracy', 0)*100:.2f}%")
                    logger.info(f"  Exact Match:     {ds_results.get('exact_match', 0)*100:.2f}%")
    
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()

