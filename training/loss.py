"""
Multi-Task Loss Functions for MIMIC-CXR VQA (v2)

Single unified MultiTaskLoss class that handles all tasks:
- VQA multi-head classification loss (auxiliary in v2)
- Answer generation loss (consumes Qwen's lm_loss directly when present)
- CheXpert auxiliary classification loss
- Scene graph generation loss (only active in 'sg_only' mode for v2)
- Visual grounding loss (priority #1 in v2)

V2 priorities (in order): grounding, generation, classification, chexpert.
Therefore grounding gets the largest weight and classification/chexpert are
demoted to small auxiliary signals.

Training Mode Weights (v2):
============================
'sg_only':    sg=1.0, others=0          # Stage 1
'alignment':  generation=1.0, others=0  # Stage 2 (LLM frozen, SG-text contrastive done elsewhere)
'pretrain':   generation=1.0, grounding=2.0, vqa=0.05, chexpert=0.05, sg=0   # Stage 3a
'finetune':   generation=1.0, grounding=3.0, vqa=0.05, chexpert=0.05, sg=0   # Stage 3b
'rl':         generation=0, grounding=2.0, vqa=0, chexpert=0, sg=0           # Stage 4 (rewards drive)
'standard':   alias for 'pretrain' (back-compat)
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
    
    # Preset weights for v2 training stages.
    # generation absorbs Qwen's lm_loss when outputs['lm_loss'] is present,
    # otherwise falls back to CE on outputs['generated_answer_logits'].
    MODE_WEIGHTS = {
        # Stage 1: SG generator solo on Chest ImaGenome
        'sg_only':   {'vqa': 0.0,  'generation': 0.0, 'chexpert': 0.0,  'scene_graph': 1.0, 'grounding': 0.0},
        # Stage 2: SG-LLM alignment (LLM frozen; contrastive often run separately)
        'alignment': {'vqa': 0.0,  'generation': 1.0, 'chexpert': 0.05, 'scene_graph': 0.0, 'grounding': 0.0},
        # Stage 3a: SFT on B-grade QA
        'pretrain':  {'vqa': 0.05, 'generation': 1.0, 'chexpert': 0.05, 'scene_graph': 0.0, 'grounding': 2.0},
        # Stage 3b: SFT on A-grade QA (boost grounding)
        'finetune':  {'vqa': 0.05, 'generation': 1.0, 'chexpert': 0.05, 'scene_graph': 0.0, 'grounding': 3.0},
        # Stage 4: GRPO outer loop drives the gradient; SFT signals stay light
        'rl':        {'vqa': 0.0,  'generation': 0.0, 'chexpert': 0.0,  'scene_graph': 0.0, 'grounding': 2.0},
        # Back-compat alias
        'standard':  {'vqa': 0.05, 'generation': 1.0, 'chexpert': 0.05, 'scene_graph': 0.0, 'grounding': 2.0},
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
        # ---- Class-weight files (produced by scripts/compute_class_weights.py).
        # When supplied, the SG entity / region / polarity CE losses use
        # per-class weights drawn from MIMIC-Ext-CXR-QBA's published
        # frequency stats (study_observations.csv, study_regions.csv).
        # That's the principled fix for the "predict the most common
        # class for everything" mode collapse the model exhibited.
        entity_weights_json: Optional[str] = None,
        region_weights_json: Optional[str] = None,
        polarity_weights_json: Optional[str] = None,
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

        # ---- Per-class CE losses for scene-graph heads (fixes mode collapse).
        # Held as buffers (not Parameters) so they move with .to(device) but
        # don't show up in optimizer state.
        self.entity_ce_loss   = self._build_weighted_ce(
            entity_weights_json, ignore_index, label_smoothing, "entity"
        )
        self.region_ce_loss   = self._build_weighted_ce(
            region_weights_json, ignore_index, label_smoothing, "region"
        )
        self.polarity_ce_loss = self._build_weighted_ce(
            polarity_weights_json, ignore_index, label_smoothing, "polarity"
        )

    def _build_weighted_ce(
        self,
        weights_json: Optional[str],
        ignore_index: int,
        label_smoothing: float,
        name: str,
    ) -> nn.CrossEntropyLoss:
        """Build a CE loss with per-class weights from a JSON file.

        Falls back to unweighted CE (same as self.ce_loss) when no path
        is supplied. The JSON schema is the one produced by
        scripts/compute_class_weights.py — a dict with a 'weights' list.
        """
        if not weights_json:
            return nn.CrossEntropyLoss(
                ignore_index=ignore_index, label_smoothing=label_smoothing,
            )
        import json as _json
        from pathlib import Path as _Path
        p = _Path(weights_json)
        if not p.exists():
            import warnings as _w
            _w.warn(f"{name} weights file not found: {p} — falling back to "
                    "unweighted CE.")
            return nn.CrossEntropyLoss(
                ignore_index=ignore_index, label_smoothing=label_smoothing,
            )
        payload = _json.loads(p.read_text())
        w = torch.tensor(payload["weights"], dtype=torch.float32)
        print(f"[loss] {name} CE: loaded {len(w)} per-class weights "
              f"from {p} (min={w.min():.3f} max={w.max():.3f})")
        return nn.CrossEntropyLoss(
            weight=w, ignore_index=ignore_index, label_smoothing=label_smoothing,
        )
    
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
        # V2: Qwen returns lm_loss directly. Prefer it over recomputing CE on
        # generated_answer_logits, which avoids a redundant softmax pass and
        # respects Qwen's own label masking (assistant-turn-only).
        precomputed_lm_loss = outputs.get('lm_loss') if isinstance(outputs, dict) else None
        
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

                # CRITICAL: aux_heads can produce NaN logits when their input
                # (Qwen's pooled hidden state) is contaminated by NaN from
                # upstream (e.g. SG token injection on fp16/Turing). The old
                # code silently dropped NaN losses → vqa_loss stuck at 0.0
                # for thousands of steps with zero gradient signal. Now we
                # scrub NaN/Inf at the logit level so CE produces a real loss,
                # AND emit a one-time-per-process warning so we know it's
                # happening (instead of staring at vqa=0.0000 for hours).
                head_logits = head_logits.float()
                if not torch.isfinite(head_logits).all():
                    if not getattr(self, '_warned_nan_vqa', False):
                        import warnings as _w
                        _w.warn(
                            f"vqa_{head_name}_logits contains NaN/Inf — "
                            f"scrubbing with nan_to_num. This usually means "
                            f"the SG injection or pooled hidden state is "
                            f"producing NaN upstream. Investigate the model "
                            f"forward path.",
                            stacklevel=2,
                        )
                        self._warned_nan_vqa = True
                    head_logits = torch.nan_to_num(head_logits, nan=0.0, posinf=1e4, neginf=-1e4)

                head_loss = self.ce_loss(head_logits, head_targets)

                if not torch.isnan(head_loss):
                    loss_dict[f'vqa_{head_name}_loss'] = head_loss
                    total_vqa_loss = total_vqa_loss + self.head_weights.get(head_name, 1.0) * head_loss
        
        loss_dict['vqa_loss'] = total_vqa_loss
        
        # =====================================================
        # ANSWER GENERATION LOSS
        # =====================================================
        # V2: prefer Qwen's precomputed lm_loss (already shifts + masks the
        # assistant turn). Fall back to manual CE on logits for the legacy
        # decoder path or for any non-Qwen forward.
        generation_loss = torch.tensor(0.0, device=device)

        if precomputed_lm_loss is not None and torch.is_tensor(precomputed_lm_loss):
            generation_loss = precomputed_lm_loss
            loss_dict['generation_loss'] = generation_loss
            # Surface the report-only loss (LM CE restricted to <think>...</think>
            # tokens — see SSGVQANetV2._compute_report_loss). This is a MONITORED
            # metric only: it is NOT added to total_loss because the underlying
            # tokens are already supervised by generation_loss above. Showing it
            # separately lets the trainer/wandb plot report-generation quality.
            precomputed_report_loss = outputs.get('report_loss') if isinstance(outputs, dict) else None
            if precomputed_report_loss is not None and torch.is_tensor(precomputed_report_loss):
                loss_dict['report_loss'] = precomputed_report_loss
        elif generation_logits is not None and answer_ids is not None:
            # Shift logits and labels for causal LM
            # logits: (B, T, V), answer_ids: (B, T)
            shift_logits = generation_logits[:, :-1, :].contiguous()
            shift_labels = answer_ids[:, 1:].contiguous()

            # Flatten and compute loss
            B, T, V = shift_logits.shape
            generation_loss = self.generation_ce_loss(
                shift_logits.view(B * T, V).float(),
                shift_labels.view(B * T)
            )
            loss_dict['generation_loss'] = generation_loss
        
        # =====================================================
        # CHEXPERT LOSS
        # =====================================================
        chexpert_loss = torch.tensor(0.0, device=device)
        
        if chexpert_logits is not None:
            # Compute BCE in float32 (avoids FP16 overflow in exp/log)
            raw_loss = self.bce_loss(chexpert_logits.float(), chexpert_labels.float())
            masked_loss = raw_loss * chexpert_mask.float()
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
        # TOTAL LOSS (computed in float32 to prevent FP16 overflow)
        # =====================================================
        total_loss = (
            self.vqa_weight * total_vqa_loss.float() +
            self.generation_weight * generation_loss.float() +
            self.chexpert_weight * chexpert_loss.float() +
            self.scene_graph_weight * sg_loss.float() +
            self.grounding_weight * grounding_loss.float()
        )
        
        return total_loss, loss_dict
    
    # -------------- DETR-style helpers (cost matrix + focal BCE) --------------

    @staticmethod
    def _pairwise_l1_cost(pred_np, gt_np):
        """(N, M) L1 distance between (x1,y1,x2,y2) predictions and GTs."""
        import numpy as np
        if pred_np.shape[0] == 0 or gt_np.shape[0] == 0:
            return np.zeros((pred_np.shape[0], gt_np.shape[0]), dtype=np.float32)
        diff = np.abs(pred_np[:, None, :] - gt_np[None, :, :]).sum(axis=-1)  # (N, M)
        return diff.astype(np.float32)

    @staticmethod
    def _pairwise_giou_cost(pred_np, gt_np):
        """(N, M) GIoU loss (= 1 - GIoU). Penalises bad localisation more
        smoothly than IoU when boxes don't overlap (covers the cold-start
        case where IoU is 0 everywhere but L1 still has a gradient)."""
        import numpy as np
        if pred_np.shape[0] == 0 or gt_np.shape[0] == 0:
            return np.zeros((pred_np.shape[0], gt_np.shape[0]), dtype=np.float32)
        a = pred_np[:, None, :].astype(np.float32)
        b = gt_np[None, :, :].astype(np.float32)
        # Intersection box
        ix1 = np.maximum(a[..., 0], b[..., 0])
        iy1 = np.maximum(a[..., 1], b[..., 1])
        ix2 = np.minimum(a[..., 2], b[..., 2])
        iy2 = np.minimum(a[..., 3], b[..., 3])
        iw = np.clip(ix2 - ix1, 0, None)
        ih = np.clip(iy2 - iy1, 0, None)
        inter = iw * ih
        area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
        area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
        union = area_a + area_b - inter
        iou = inter / (union + 1e-8)
        # Smallest enclosing box
        ex1 = np.minimum(a[..., 0], b[..., 0])
        ey1 = np.minimum(a[..., 1], b[..., 1])
        ex2 = np.maximum(a[..., 2], b[..., 2])
        ey2 = np.maximum(a[..., 3], b[..., 3])
        enc = np.clip(ex2 - ex1, 0, None) * np.clip(ey2 - ey1, 0, None)
        giou = iou - (enc - union) / (enc + 1e-8)
        return (1.0 - giou).astype(np.float32)

    @staticmethod
    def _pairwise_class_cost(entity_logits_b, region_logits_b, ent_target, reg_target,
                             num_ent, num_reg, ignore_index=-100):
        """(N, M) cost = -log p(gt_class | pred_logits) summed over entity+region.
        Computed in numpy on detached softmax probs — used only for matching."""
        import numpy as np
        import torch
        N = entity_logits_b.shape[0]
        M = ent_target.shape[0] if ent_target is not None else 0
        if N == 0 or M == 0:
            return np.zeros((N, M), dtype=np.float32)
        # Softmax probs (detached, fp32)
        ent_p = torch.softmax(entity_logits_b.detach().float(), dim=-1).cpu().numpy()  # (N, num_ent)
        reg_p = torch.softmax(region_logits_b.detach().float(), dim=-1).cpu().numpy()  # (N, num_reg)
        ent_t = ent_target.detach().cpu().numpy() if ent_target is not None else None  # (M,)
        reg_t = reg_target.detach().cpu().numpy() if reg_target is not None else None  # (M,)
        cost = np.zeros((N, M), dtype=np.float32)
        eps = 1e-8
        for m in range(M):
            if ent_t is not None:
                e = int(ent_t[m])
                if 0 <= e < num_ent:
                    cost[:, m] += -np.log(ent_p[:, e] + eps)
            if reg_t is not None:
                r = int(reg_t[m])
                if 0 <= r < num_reg:
                    cost[:, m] += -np.log(reg_p[:, r] + eps)
        return cost

    @staticmethod
    def _focal_bce_with_logits(logits, targets, alpha=0.25, gamma=2.0):
        """Focal-BCE for objectness imbalance (N=300 queries, M~10 positives).

        Reference: Lin et al. (2017) "Focal Loss for Dense Object Detection".
        With α=0.25, γ=2.0 (RetinaNet defaults): negatives get heavy down-
        weighting once the model is confident they're negative, so the small
        positive set drives the gradient even when 95% of queries are
        background.
        """
        logits = logits.float()
        targets = targets.float()
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        focal = alpha_t * (1.0 - p_t) ** gamma * ce
        return focal.mean()

    def _compute_scene_graph_loss(
        self,
        sg_outputs: Dict[str, torch.Tensor],
        gt_bboxes: Optional[List[torch.Tensor]],
        gt_entities: Optional[List[torch.Tensor]],
        gt_regions: Optional[List[torch.Tensor]],
        device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """DETR-style scene graph loss with three upgrades:

          1) COMBINED HUNGARIAN COST (was: IoU only).
             cost = w_iou * (1 - IoU) + w_l1 * L1 + w_giou * (1 - GIoU)
                  + w_class * (-log p(gt_class | pred_logits))
             Default weights mirror DETR: w_l1=5, w_giou=2, w_iou=2, w_class=1.
             Even at init when IoU is 0 everywhere, L1 + class probability
             still provide a smooth cost surface, so Hungarian picks
             non-random assignments from step 1.

          2) FOCAL BCE for objectness (was: plain BCE).
             N=300 queries, M~10 positives → 96.6% background; focal loss
             (α=0.25, γ=2) prevents the positive gradient signal from being
             drowned in the negative mean.

          3) BBOX SIGMOID + CLAMP for cold-start stability.
             The SG generator emits raw (x1,y1,x2,y2) which can be outside
             [0,1] at init, making IoU degenerate and L1 unbounded. We
             apply sigmoid (no-op on already-normalised values up to a soft
             squash; aligned with DETR's bbox_embed.sigmoid()) and clamp
             coords to [0,1] before matching and bbox loss. This stabilises
             the first ~50 steps when bbox regression hasn't warmed up.
        """
        from .metrics import pairwise_iou, hungarian_match

        loss_dict = {}

        bbox_preds_raw = sg_outputs['bbox_preds']    # (B, N, 4) — ALREADY sigmoid'd
                                                     # in SceneGraphGenerator.forward (line 2625)
        entity_logits = sg_outputs['entity_logits']  # (B, N, num_entities)
        region_logits = sg_outputs['region_logits']  # (B, N, num_regions)
        objectness = sg_outputs['objectness_scores'] # (B, N)

        # === FIX 3 (corrected): clamp ONLY — DO NOT sigmoid again ===
        # SceneGraphGenerator already applies torch.sigmoid() to its bbox
        # output at L2625. A second sigmoid here would map [0,1] → [0.5,0.73],
        # crushing the box-movement signal and stalling bbox_loss. We just
        # clamp defensively in case of fp16 over/underflow at the boundaries.
        bbox_preds = bbox_preds_raw.clamp(0.0, 1.0)

        B, N = bbox_preds.shape[:2]

        entity_loss = torch.tensor(0.0, device=device)
        region_loss = torch.tensor(0.0, device=device)
        bbox_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        num_valid = 0
        total_matched_pairs = 0

        if gt_bboxes is None:
            return torch.tensor(0.0, device=device), {}

        # === FIX 1: DETR matching cost weights ===
        W_IOU = 2.0
        W_L1 = 5.0
        W_GIOU = 2.0
        W_CLASS = 1.0

        model_dt = bbox_preds.dtype

        for b in range(B):
            if b >= len(gt_bboxes) or gt_bboxes[b] is None:
                continue

            gt_box = gt_bboxes[b].to(device=device, dtype=model_dt).clamp(0.0, 1.0)
            M = gt_box.shape[0]
            if M == 0:
                continue

            pred_np = bbox_preds[b].detach().float().cpu().numpy()  # (N, 4)
            gt_np = gt_box.detach().float().cpu().numpy()           # (M, 4)

            # === FIX 1: combined cost matrix (not IoU-only) ===
            iou_mat = pairwise_iou(pred_np, gt_np)                  # (N, M)
            l1_cost = self._pairwise_l1_cost(pred_np, gt_np)        # (N, M)
            giou_cost = self._pairwise_giou_cost(pred_np, gt_np)    # (N, M)
            # Class cost depends on this batch's entity/region GTs
            ent_t_for_cost = None
            if gt_entities is not None and b < len(gt_entities) and gt_entities[b] is not None:
                ent_t_for_cost = gt_entities[b].to(device).long()
            reg_t_for_cost = None
            if gt_regions is not None and b < len(gt_regions) and gt_regions[b] is not None:
                reg_t_for_cost = gt_regions[b].to(device).long()
            class_cost = self._pairwise_class_cost(
                entity_logits[b], region_logits[b],
                ent_t_for_cost, reg_t_for_cost,
                entity_logits.shape[-1], region_logits.shape[-1],
                self.ignore_index,
            )
            # Hungarian on -IoU (minimise cost). Combine:
            total_cost = (
                W_IOU * (1.0 - iou_mat)
                + W_L1 * l1_cost
                + W_GIOU * giou_cost
                + W_CLASS * class_cost
            )
            # hungarian_match expects an iou-like matrix to MAXIMISE — we
            # pass -total_cost so the lsa picks lowest cost = highest score.
            pairs = hungarian_match(-total_cost)
            if not pairs:
                continue

            pred_idx_list = [p[0] for p in pairs]
            gt_idx_list = [p[1] for p in pairs]
            pred_idx = torch.tensor(pred_idx_list, device=device, dtype=torch.long)
            gt_idx = torch.tensor(gt_idx_list, device=device, dtype=torch.long)
            P = pred_idx.shape[0]
            total_matched_pairs += P

            # === Bbox loss on matched pairs (uses sigmoid+clamped preds) ===
            matched_pred_box = bbox_preds[b, pred_idx]    # (P, 4)
            matched_gt_box = gt_box[gt_idx]                # (P, 4)
            if self.use_giou:
                bbox_loss = bbox_loss + self._giou_loss(matched_pred_box, matched_gt_box).mean()
            else:
                bbox_loss = bbox_loss + F.smooth_l1_loss(
                    matched_pred_box.float(), matched_gt_box.float()
                )

            # === Entity loss on matched pairs (PER-PAIR CLAMP) ===
            # Why: nn.CrossEntropyLoss(reduction='mean') with class weights is
            # bounded by max_weight × log(C), but in practice a single
            # confidently-wrong rare-class prediction gives per-pair CE of
            # 20-45 (log(p≈1e-13) ≈ 30) and the WEIGHTED MEAN inherits that.
            # On batches dominated by such pairs, sg_loss can spike to 80+
            # even with capped weights. Per-pair clamp to max=10.0 bounds each
            # pair's contribution. The clamp does NOT affect gradients on
            # well-predicted pairs (their loss is < 10 anyway), only the
            # extreme outliers that destabilise the optimizer.
            if ent_t_for_cost is not None:
                safe_gt_idx = gt_idx.clamp(max=ent_t_for_cost.shape[0] - 1) if ent_t_for_cost.numel() else gt_idx
                ent_target = ent_t_for_cost[safe_gt_idx] if ent_t_for_cost.numel() else torch.full((P,), self.ignore_index, device=device, dtype=torch.long)
                num_ent = entity_logits.shape[-1]
                ent_target = torch.where(
                    (ent_target >= 0) & (ent_target < num_ent),
                    ent_target,
                    torch.full_like(ent_target, self.ignore_index),
                )
                if (ent_target != -100).any():
                    ent_weight = self.entity_ce_loss.weight
                    if ent_weight is not None:
                        ent_weight = ent_weight.to(device)
                    ent_per_pair = F.cross_entropy(
                        entity_logits[b, pred_idx].float(),
                        ent_target,
                        weight=ent_weight,
                        ignore_index=self.ignore_index,
                        label_smoothing=getattr(self.entity_ce_loss, 'label_smoothing', 0.0),
                        reduction='none',
                    )  # (P,)
                    ent_per_pair_clipped = ent_per_pair.clamp(max=10.0)
                    # MEAN only over non-ignored pairs (matches behaviour of
                    # reduction='mean' for CE with ignore_index)
                    valid = (ent_target != self.ignore_index)
                    denom = valid.float().sum().clamp(min=1.0)
                    entity_loss = entity_loss + (ent_per_pair_clipped * valid.float()).sum() / denom

            # === Region loss on matched pairs (PER-PAIR CLAMP) ===
            if reg_t_for_cost is not None:
                safe_gt_idx_r = gt_idx.clamp(max=reg_t_for_cost.shape[0] - 1) if reg_t_for_cost.numel() else gt_idx
                reg_target = reg_t_for_cost[safe_gt_idx_r] if reg_t_for_cost.numel() else torch.full((P,), self.ignore_index, device=device, dtype=torch.long)
                num_reg = region_logits.shape[-1]
                reg_target = torch.where(
                    (reg_target >= 0) & (reg_target < num_reg),
                    reg_target,
                    torch.full_like(reg_target, self.ignore_index),
                )
                if (reg_target != -100).any():
                    reg_weight = self.region_ce_loss.weight
                    if reg_weight is not None:
                        reg_weight = reg_weight.to(device)
                    reg_per_pair = F.cross_entropy(
                        region_logits[b, pred_idx].float(),
                        reg_target,
                        weight=reg_weight,
                        ignore_index=self.ignore_index,
                        label_smoothing=getattr(self.region_ce_loss, 'label_smoothing', 0.0),
                        reduction='none',
                    )  # (P,)
                    reg_per_pair_clipped = reg_per_pair.clamp(max=10.0)
                    valid = (reg_target != self.ignore_index)
                    denom = valid.float().sum().clamp(min=1.0)
                    region_loss = region_loss + (reg_per_pair_clipped * valid.float()).sum() / denom

            # === FIX 2: focal BCE for objectness (was: plain BCE) ===
            obj_target = torch.zeros(N, dtype=torch.float32, device=device)
            obj_target[pred_idx] = 1.0
            obj_loss = obj_loss + self._focal_bce_with_logits(
                objectness[b], obj_target, alpha=0.25, gamma=2.0,
            )

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
        loss_dict['sg_matched_pairs_per_batch'] = torch.tensor(
            float(total_matched_pairs) / max(1, num_valid), device=device
        )

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
        
        # Bbox loss (float32 for numerical stability)
        if self.use_giou:
            bbox_loss = self._giou_loss(bbox_pred, gt_bboxes).mean()
        else:
            bbox_loss = F.smooth_l1_loss(bbox_pred.float(), gt_bboxes.float())
        
        loss_dict['grounding_bbox_loss'] = bbox_loss
        
        # Pointing loss (compute in float32 — BCE can produce -inf in FP16 for p≈0 or p≈1).
        # CRITICAL: .clamp() PROPAGATES NaN unchanged. Real crash hit at Stage 2
        # step 10344: pointing_score became NaN upstream (mHC/sinkhorn instability),
        # clamp kept it as NaN, BCE asserted `input_val >= 0 && <= 1` and SIGABRTed
        # the rank. nan_to_num replaces NaN with 0.5 (neutral) before clamp so
        # one bad batch can't take down the entire training run.
        score = pointing_score.view(B, 1).float()
        score = torch.nan_to_num(score, nan=0.5, posinf=1.0, neginf=0.0)
        score = score.clamp(1e-6, 1.0 - 1e-6)
        if gt_pointing_valid is not None:
            pointing_target = gt_pointing_valid.float().view(B, 1)
            pointing_target = torch.nan_to_num(pointing_target, nan=0.0)
        else:
            pointing_target = torch.ones(B, 1, dtype=torch.float32, device=device)

        # BCE is in PyTorch's autocast denylist — it refuses to run inside
        # an autocast region even with fp32 inputs ("unsafe to autocast").
        # Wrap in autocast(enabled=False) to compute the BCE in pure fp32.
        # We can't switch to BCE-with-logits here because pointing_score is
        # already post-sigmoid coming out of the grounding head; converting
        # back via log(p/(1-p)) is numerically lossier than just disabling
        # autocast for one cheap loss term.
        with torch.amp.autocast('cuda', enabled=False):
            pointing_loss = F.binary_cross_entropy(
                score.float(), pointing_target.float()
            )
        loss_dict['grounding_pointing_loss'] = pointing_loss
        
        total = bbox_loss + 0.5 * pointing_loss
        return total, loss_dict
    
    def _giou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute GIoU loss in float32 (FP16-safe)."""
        # Upcast to float32 — area/division ops overflow FP16 easily
        pred = pred.float()
        target = target.float()
        
        # Ensure box ordering: x1<=x2, y1<=y2 (prevents negative areas)
        pred_x1 = torch.min(pred[:, 0], pred[:, 2])
        pred_y1 = torch.min(pred[:, 1], pred[:, 3])
        pred_x2 = torch.max(pred[:, 0], pred[:, 2])
        pred_y2 = torch.max(pred[:, 1], pred[:, 3])
        pred = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)
        
        tgt_x1 = torch.min(target[:, 0], target[:, 2])
        tgt_y1 = torch.min(target[:, 1], target[:, 3])
        tgt_x2 = torch.max(target[:, 0], target[:, 2])
        tgt_y2 = torch.max(target[:, 1], target[:, 3])
        target = torch.stack([tgt_x1, tgt_y1, tgt_x2, tgt_y2], dim=-1)
        
        x1 = torch.max(pred[:, 0], target[:, 0])
        y1 = torch.max(pred[:, 1], target[:, 1])
        x2 = torch.min(pred[:, 2], target[:, 2])
        y2 = torch.min(pred[:, 3], target[:, 3])
        
        inter_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union_area = pred_area + target_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        
        enc_x1 = torch.min(pred[:, 0], target[:, 0])
        enc_y1 = torch.min(pred[:, 1], target[:, 1])
        enc_x2 = torch.max(pred[:, 2], target[:, 2])
        enc_y2 = torch.max(pred[:, 3], target[:, 3])
        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
        
        giou = iou - (enc_area - union_area) / (enc_area + 1e-6)
        
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
