"""
MIMIC-CXR VQA Dataset

Handles loading and preprocessing of:
- MIMIC-CXR-JPG images
- MIMIC-Ext-CXR-QBA scene graphs and QA pairs
- CheXpert labels for auxiliary supervision

Based on MIMIC_CXR_VQA_ANALYSIS.md specifications.
"""

import os
import json
import logging
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path('.cache/dataset_samples')


# =============================================================================
# QBA QUALITY CRITERION → LETTER GRADE
# =============================================================================
# Per MIMIC-Ext-CXR-QBA spec (Section 3a, Methods). The dataset stores
# extraction_quality / question_img_localization_quality / etc. as DICTS of
# criterion-level states (e.g. {"region_quality": "RESOLVED_REGIONS_ONLY",
# "entity_quality": "RESOLVED_ENTITIES_ONLY", ...}) — NOT pre-computed
# letter grades. Per the QBA paper, the overall grade for a question is the
# MIN (worst) letter grade across all criteria in all of its quality dicts.
#
# The mapping below is taken verbatim from the QBA paper tables for both
# scene-graph extraction criteria (Section 3a) and QA evaluation criteria
# (LLM-as-judge entailment/relevance/completeness/clarity).
#
# Without this mapping, the old code's `v.get('overall', v.get('grade', 'B'))`
# defaulted to 'B' for every dict (no 'overall' or 'grade' key exists in
# QBA's per-criterion structure) — so every question got rated B and the
# A-grade filter rejected ALL 7M+ questions.
# =============================================================================
QBA_CRITERION_GRADE: Dict[str, str] = {
    # --- region_quality ---
    'NO_REGIONS': 'B',
    'DEFAULT_REGIONS_ONLY': 'B',
    'CONTAINS_DEFAULT_REGIONS': 'A',
    'CONTAINS_NON_RESOLVED_REGIONS': 'A',
    'RESOLVED_REGIONS_ONLY': 'A++',
    # --- entity_quality ---
    'NO_ENTITIES': 'B',
    'CONTAINS_NON_RESOLVED_ENTITIES': 'A',
    'RESOLVED_ENTITIES_ONLY': 'A++',
    # --- sentence_name_quality / change_quality / issue_level (NO_ISSUES shared) ---
    'NO_ISSUES': 'A++',
    'CHANGE_IN_SENTENCE_OR_NAME': 'B',
    'UNDERSCORES_IN_SENTENCE_OR_NAME': 'A',
    # --- change_quality ---
    'CHANGE_SENTENCE_REMOVED': 'B',
    'UNDERSCORES_IN_CHANGE_SENTENCE': 'A',
    'CONTAINS_NON_RESOLVED_CHANGES': 'A',
    # --- issue_level ---
    'DISCARDED': 'D',
    'NON_INTERPRETABLE': 'C',
    'MOSTLY_INTERPRETABLE': 'B',
    'IGNORABLE': 'A',
    'FIXABLE': 'A+',
    # --- localization_quality (scene-graph extraction) ---
    'NO_LOCALIZATION': 'B',
    'FALLBACK_LOCALIZATION': 'B',
    'INCOMPLETE_LOCALIZATION': 'A',
    'BBOX_LOCALIZATION': 'A++',
    'BBOX_AND_MASK_LOCALIZATION': 'A++',
    # === QA evaluation criteria (LLM-as-judge) ===
    # --- Entailment ---
    'ALIGNED_MENTIONED': 'A++',
    'ALIGNED_INFERABLE': 'A++',
    'ALIGNED_NEGATIVE_NOT_MENTIONED': 'A',
    'ALIGNED_GENERAL_STATEMENT': 'A',
    'NON_ALIGNED_NON_INFERABLE': 'B',
    'NON_ALIGNED_MISLEADING': 'C',
    'NON_ALIGNED_CONTRADICTING': 'D',
    # --- Relevance ---
    'RELEVANT_MAIN_ANSWER': 'A++',
    'RELATED_INFO': 'A++',
    'REDUNDANT_INFO': 'A',
    'IRRELEVANT_INFO': 'A',
    # --- Completeness ---
    'FULLY_COMPLETE': 'A++',
    'DETAILS_MISSING': 'A+',
    'NOT_ANSWERED': 'B',
    'INCOMPLETE_NON_MISLEADING': 'B',
    'INCOMPLETE_MISLEADING': 'C',
    # --- Question / Answer clarity ---
    'OPTIMAL': 'A++',
    'UNUSUAL_SENTENCE_STRUCTURE': 'A',
    'GRAMMATICAL_ERRORS': 'A',
    'UNCLEAR_QUESTION': 'B',
    'UNRELATED_TO_CHEST_XRAY': 'B',
    'UNANSWERABLE': 'C',
    'UNCLEAR_ANSWER': 'B',
    'NOT_UNDERSTANDABLE': 'C',
}

QBA_GRADE_ORDER: Dict[str, int] = {
    'A++': 5, 'A+': 4, 'A': 3, 'B': 2, 'C': 1, 'D': 0, 'U': 0,
}


# QBA stores per-criterion quality as INTEGERS (positional index into each
# criterion's ordered enum, worst → best). Map int → letter grade per
# criterion. Key names below MATCH the actual JSON keys in the dump on disk
# (verified by dumping s55433201.qa.json from the QBA scene_data tree).
QBA_CRITERION_INT_GRADES: Dict[str, List[str]] = {
    # 5 levels: NO_REGIONS, DEFAULT_REGIONS_ONLY, CONTAINS_DEFAULT_REGIONS,
    #          CONTAINS_NON_RESOLVED_REGIONS, RESOLVED_REGIONS_ONLY
    'region_extract_quality': ['B', 'B', 'A', 'A', 'A++'],
    # 3 levels: NO_ENTITIES, CONTAINS_NON_RESOLVED_ENTITIES, RESOLVED_ENTITIES_ONLY
    'entity_extract_quality': ['B', 'A', 'A++'],
    # 3 levels: CHANGE_IN_SENTENCE_OR_NAME, UNDERSCORES_IN_SENTENCE_OR_NAME, NO_ISSUES
    'sentence_name_quality': ['B', 'A', 'A++'],
    # 4 levels: CHANGE_SENTENCE_REMOVED, UNDERSCORES_IN_CHANGE_SENTENCE,
    #          CONTAINS_NON_RESOLVED_CHANGES, NO_ISSUES
    'change_quality': ['B', 'A', 'A', 'A++'],
    # 6 levels: DISCARDED, NON_INTERPRETABLE, MOSTLY_INTERPRETABLE,
    #          IGNORABLE, FIXABLE, NO_ISSUES
    'general_issue_level': ['D', 'C', 'B', 'A', 'A+', 'A++'],
    # 5 levels: NO_LOCALIZATION, FALLBACK_LOCALIZATION, INCOMPLETE_LOCALIZATION,
    #          BBOX_LOCALIZATION, BBOX_AND_MASK_LOCALIZATION
    'localization_quality': ['B', 'B', 'A', 'A++', 'A++'],
    # Legacy alias for older dumps that used the paper's exact key names
    'region_quality': ['B', 'B', 'A', 'A', 'A++'],
    'entity_quality': ['B', 'A', 'A++'],
    'issue_level': ['D', 'C', 'B', 'A', 'A+', 'A++'],
}


def _qba_dict_to_grade(d: Any) -> Optional[str]:
    """Compute the worst letter grade across all criteria in a QBA quality dict.

    Handles THREE storage formats observed across QBA dumps:
      1. Per-criterion INTEGERS (current MIMIC-Ext-CXR-QBA v1.0 dump):
         {'region_extract_quality': 4, 'localization_quality': 1, ...}
         Integer is positional index into criterion's ordered enum.
         Mapped via QBA_CRITERION_INT_GRADES[<criterion_key>][<int>].
      2. Per-criterion STRINGS (paper spec / older dumps):
         {'region_quality': 'RESOLVED_REGIONS_ONLY', ...}
         Mapped via QBA_CRITERION_GRADE[<string>].
      3. Pre-computed legacy: {'overall': 'A'} or {'grade': 'A++'}.

    Returns the worst letter grade, or None if nothing parseable.
    """
    if not isinstance(d, dict):
        return None
    # Fast path: pre-computed grade keys (legacy / convenience)
    if isinstance(d.get('overall'), str):
        return d['overall']
    if isinstance(d.get('grade'), str):
        return d['grade']
    # Aggregate path: scan each criterion and map to a letter grade
    grades: List[str] = []
    for k, v in d.items():
        if isinstance(v, int) and not isinstance(v, bool):
            # Per-criterion integer (current QBA format)
            enum = QBA_CRITERION_INT_GRADES.get(k)
            if enum is not None and 0 <= v < len(enum):
                grades.append(enum[v])
        elif isinstance(v, str):
            # Per-criterion string (paper spec / older format)
            g = QBA_CRITERION_GRADE.get(v)
            if g is not None:
                grades.append(g)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    g = QBA_CRITERION_GRADE.get(item)
                    if g is not None:
                        grades.append(g)
        elif isinstance(v, dict):
            sub = _qba_dict_to_grade(v)
            if sub is not None:
                grades.append(sub)
    if not grades:
        return None
    return min(grades, key=lambda g: QBA_GRADE_ORDER.get(g, 0))


# CheXpert categories
CHEXPERT_CATEGORIES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pneumonia',
    'Pneumothorax', 'Pleural Other', 'Support Devices', 'No Finding'
]

# Question types mapping - includes all MIMIC-Ext-CXR-QBA question types
# Maps question_type to answer head: binary, category, region, or severity
QUESTION_TYPE_MAP = {
    # === MIMIC-Ext-CXR-QBA Question Types (from analysis) ===
    
    # Binary Head (Yes/No) - ~1.2M pairs
    'C03_is_abnormal_region': 'binary',      # 124,434 pairs
    'C04_is_normal_region': 'binary',        # 124,478 pairs
    'C08_has_region_device': 'binary',       # 206,215 pairs
    'D02_has_finding': 'binary',             # 112,049 pairs
    'D06_has_device': 'binary',              # 14,320 pairs
    'B10_is_abnormal_subcat': 'binary',      # 104,254 pairs
    'B11_is_normal_subcat': 'binary',        # 73,544 pairs
    'B13_has_devices': 'binary',             # 42,112 pairs
    
    # Region Head (Anatomical Location) - ~350K pairs
    'C01_describe_region': 'region',         # 124,312 pairs
    'C02_describe_abnormal_region': 'region', # 124,498 pairs
    'D03_where_is_finding': 'region',        # 88,990 pairs
    'D07_where_is_device': 'region',         # 14,114 pairs
    
    # Severity Head - ~58K pairs
    'D04_how_severe_is_finding': 'severity', # 58,490 pairs
    
    # Category Head (Finding Type / Entity) - ~580K pairs
    'D01_describe_finding': 'category',      # 112,193 pairs
    'D05_describe_device': 'category',       # 14,355 pairs
    'C07_describe_region_device': 'category', # 204,916 pairs
    'B08_describe_subcat': 'category',       # 104,254 pairs
    'B09_describe_abnormal_subcat': 'category', # 104,254 pairs
    'B12_describe_device': 'category',       # 42,112 pairs
    'A_indication': 'category',              # 9,645 pairs
    
    # === Legacy/Short Form Mappings (backward compatibility) ===
    'is_abnormal': 'binary',
    'is_normal': 'binary',
    'has_finding': 'binary',
    'has_device': 'binary',
    'is_abnormal_region': 'binary',
    'describe_finding': 'category',
    'describe_device': 'category',
    'where_is_finding': 'region',
    'where_is_device': 'region',
    'describe_region': 'region',
    'how_severe': 'severity',
    'compare': 'category',
    'indication': 'category',
}


class CheXpertLabelLoader:
    """
    Loads and preprocesses CheXpert labels with uncertainty handling.
    """
    
    def __init__(
        self, 
        labels_path: Optional[str] = None,
        uncertainty_policy: str = 'ignore'  # ignore, positive, negative, soft
    ):
        self.labels_df = None
        self.uncertainty_policy = uncertainty_policy
        
        if labels_path and os.path.exists(labels_path):
            if labels_path.endswith('.gz'):
                self.labels_df = pd.read_csv(labels_path, compression='gzip')
            else:
                self.labels_df = pd.read_csv(labels_path)
            
            # Create index for fast lookup
            if 'subject_id' in self.labels_df.columns and 'study_id' in self.labels_df.columns:
                self.labels_df = self.labels_df.set_index(['subject_id', 'study_id'])
                logger.info(f"Loaded CheXpert labels: {len(self.labels_df)} studies")
    
    def get_labels(
        self, 
        subject_id: int, 
        study_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get CheXpert labels and mask for a study.
        
        Returns:
            labels: (14,) array of labels (0, 1, or 0.5 for uncertain)
            mask: (14,) array where 1=use label, 0=ignore
        """
        labels = np.zeros(len(CHEXPERT_CATEGORIES), dtype=np.float32)
        mask = np.ones(len(CHEXPERT_CATEGORIES), dtype=np.float32)
        
        if self.labels_df is None:
            mask = np.zeros(len(CHEXPERT_CATEGORIES), dtype=np.float32)
            return labels, mask
        
        try:
            row = self.labels_df.loc[(subject_id, study_id)]
            
            for i, cat in enumerate(CHEXPERT_CATEGORIES):
                val = row.get(cat, np.nan) if hasattr(row, 'get') else np.nan
                
                if pd.isna(val):
                    labels[i] = 0.0
                    mask[i] = 0.0  # Ignore missing
                elif val == 1.0:
                    labels[i] = 1.0
                elif val == 0.0:
                    labels[i] = 0.0
                elif val == -1.0:  # Uncertain
                    if self.uncertainty_policy == 'ignore':
                        labels[i] = 0.5
                        mask[i] = 0.0
                    elif self.uncertainty_policy == 'positive':
                        labels[i] = 1.0
                    elif self.uncertainty_policy == 'negative':
                        labels[i] = 0.0
                    else:  # soft
                        labels[i] = 0.5
                        
        except KeyError:
            mask = np.zeros(len(CHEXPERT_CATEGORIES), dtype=np.float32)
        
        return labels, mask


class SceneGraphProcessor:
    """
    Processes scene graphs from MIMIC-Ext-CXR-QBA format.
    
    Scene graph structure (from scene_data.zip):
    {
        "patient_id": "p1xxxxxxx",
        "study_id": "sxxxxxxxx",
        "sentences": {...},
        "top_level_obs_ids": ["O01", ...],
        "observations": {
            "O01": {
                "obs_id": "O01",
                "name": "...",
                "regions": [{"region": "lungs", ...}],
                "obs_entities": ["consolidation"],
                "positiveness": "neg"/"pos",
                "localization": {
                    "[image_id]": {
                        "bboxes": [[x1, y1, x2, y2], ...],
                        ...
                    }
                },
                ...
            }
        },
        "regions": {...},
        "located_at_relations": [...],
        ...
    }
    """
    
    def __init__(
        self,
        num_regions: int = 310,
        num_entities: int = 237
    ):
        self.num_regions = num_regions
        self.num_entities = num_entities
        
        # Region and entity vocabularies (loaded from dataset)
        self.region_to_idx: Dict[str, int] = {}
        self.entity_to_idx: Dict[str, int] = {}
        self.category_to_idx: Dict[str, int] = {}
        
    def load_vocab(self, dataset_info_path: str):
        """Load region and entity vocabularies from dataset_info.json."""
        if not os.path.exists(dataset_info_path):
            logger.warning(f"Dataset info not found: {dataset_info_path}")
            return
            
        with open(dataset_info_path) as f:
            info = json.load(f)
        
        # Load regions (e.g., "lungs", "left lung", "heart", etc.)
        for idx, region in enumerate(info.get('regions', info.get('region_names', []))):
            self.region_to_idx[region.lower()] = idx
            
        # Load finding entities (e.g., "consolidation", "effusion", etc.)
        for idx, entity in enumerate(info.get('finding_entities', info.get('entity_names', []))):
            self.entity_to_idx[entity.lower()] = idx
        
        # Load categories if available
        for idx, cat in enumerate(info.get('finding_categories', [])):
            self.category_to_idx[cat.lower()] = idx
            
        logger.info(f"Loaded vocab: {len(self.region_to_idx)} regions, {len(self.entity_to_idx)} entities")
    
    def process(
        self,
        scene_graph: Dict[str, Any],
        image_width: int,
        image_height: int,
        image_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a scene_graph.json into per-(region, bbox) training samples.

        DESIGN CHANGE (2026-06-10) — this method previously had seven bugs
        that flattened the rich per-(image, region) localization in
        MIMIC-Ext-CXR-QBA into a single bbox per observation, dropping
        ~half the available supervision and pairing entities with the
        wrong regions. The fixed version:

          1. Emits ONE sample per (region, bbox) pair, not per observation.
             An NG tube with bboxes in [esophagus, stomach] becomes 2
             samples instead of 1.
          2. Picks `localization_reference_ids` and matches each to its
             corresponding bbox by INDEX (the dataset publishes them in
             parallel arrays — taking [0] of each loses everything past
             the first).
          3. Skips `is_fallback=True` entries (those are generic anatomy
             templates, not real localized findings — including them is
             what taught the model to predict "prosthesis @ hemiazygos
             vein" with degenerate centre bboxes).
          4. Skips observations without any localization for the current
             image instead of defaulting to a whole-image bbox.
          5. Picks the right per-image localization when `image_id` is
             provided, instead of "first available".
          6. Skips child observations that are sub-parts of a parent
             that's already in the list (avoids duplicate "NG tube" +
             "NG tube tip" rows pointing at the same anatomy). Set
             ``include_children=True`` on the instance to keep them.
          7. Skips degenerate bboxes (< 0.5% of image area) — these are
             usually annotation artefacts that don't help the detector.

        Net effect: same-shape dict ({bboxes, region_ids, entity_ids,
        positiveness, num_objects}) but with honest, per-anatomy
        supervision instead of templated noise.
        """
        observations = scene_graph.get('observations', {})
        if not observations:
            return self._empty_result()

        # Optional: include child observations (e.g. "NG tube tip" as a
        # child of "NG tube"). Defaults to False — children duplicate
        # the parent's anatomy with slightly more specific labels and
        # tend to confuse the detector.
        include_children = getattr(self, "include_children", False)

        # Optional escape hatch — set on the instance for A/B testing
        # against the old behaviour. Off by default (the old behaviour
        # is buggy enough that we don't want it unless explicitly opted in).
        if getattr(self, "use_legacy_processor", False):
            return self._process_legacy(scene_graph, image_width, image_height, image_id)

        bboxes: List[List[float]] = []
        region_ids: List[int] = []
        entity_ids: List[int] = []
        positiveness_list: List[int] = []

        n_skipped_no_loc = 0
        n_skipped_fallback = 0
        n_skipped_degenerate = 0
        n_skipped_child = 0

        for obs_id, obs in observations.items():
            # Bug-fix #6: skip child observations by default (they duplicate
            # the parent's localization with a slightly more specific label).
            if not include_children and obs.get("child_level", 0) > 0:
                n_skipped_child += 1
                continue

            # Bug-fix #5: pick per-image localization correctly.
            loc = obs.get("localization") or {}
            if not isinstance(loc, dict) or not loc:
                n_skipped_no_loc += 1
                continue
            if image_id and image_id in loc:
                img_loc = loc[image_id]
            else:
                # Multi-view study: pick the first available view rather
                # than crashing. The training loop should ideally pass
                # image_id; if it doesn't, this is a sensible fallback.
                img_loc = next(iter(loc.values()))
            if not isinstance(img_loc, dict):
                n_skipped_no_loc += 1
                continue

            # Bug-fix #3: skip fallback (anatomy-template) bboxes entirely.
            # These are the "consolidation in lungs gets the same bbox as
            # pleural effusion in pleura" entries — they teach the
            # detector nothing useful.
            if img_loc.get("is_fallback", False):
                n_skipped_fallback += 1
                continue

            # Parallel arrays — same length, indexed together.
            ref_regions = img_loc.get("localization_reference_ids") or []
            img_bboxes  = img_loc.get("bboxes") or []
            if not ref_regions or not img_bboxes:
                n_skipped_no_loc += 1
                continue

            # Get the observation's entity (one per obs)
            entities = obs.get("obs_entities") or []
            entity_name = (entities[0] if entities and isinstance(entities[0], str)
                           else "unknown").lower()
            entity_id = self.entity_to_idx.get(entity_name, 0)

            # Polarity (one per obs)
            pos = obs.get("positiveness", "neg")
            polarity = 1 if pos == "pos" else 0

            # Bug-fix #1, #2: emit ONE sample per (region, bbox) pair —
            # paired by index, not by taking [0].
            n_pairs = min(len(ref_regions), len(img_bboxes))
            for i in range(n_pairs):
                region_name = ref_regions[i]
                bbox = img_bboxes[i]
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue

                # Normalise pixel → [0, 1] using the actual image size.
                # MIMIC-Ext bboxes are in PIXELS; the dataset's
                # metadata.json provides per-image dimensions but the
                # trainer historically uses image_width / image_height
                # from MIMIC-CXR metadata CSV instead. Either source
                # works as long as it's the correct per-image size.
                try:
                    x1 = max(0.0, min(float(bbox[0]) / image_width,  1.0))
                    y1 = max(0.0, min(float(bbox[1]) / image_height, 1.0))
                    x2 = max(0.0, min(float(bbox[2]) / image_width,  1.0))
                    y2 = max(0.0, min(float(bbox[3]) / image_height, 1.0))
                except (TypeError, ZeroDivisionError):
                    continue

                # Bug-fix #7: skip degenerate bboxes (< 0.5% of image
                # area). These are almost always annotation artefacts
                # — a 2-pixel-wide box conveys no information and
                # destabilises bbox-regression heads.
                if (x2 - x1) < 0.005 or (y2 - y1) < 0.005:
                    n_skipped_degenerate += 1
                    continue

                bboxes.append([x1, y1, x2, y2])
                region_ids.append(self.region_to_idx.get(region_name.lower(), 0))
                entity_ids.append(entity_id)
                positiveness_list.append(polarity)

        # Diagnostic counters (optional — uncomment to surface during
        # the audit run; off in production to keep the loader silent).
        # logger.debug(
        #     f"SG processed: kept={len(bboxes)} "
        #     f"skipped_child={n_skipped_child} "
        #     f"skipped_no_loc={n_skipped_no_loc} "
        #     f"skipped_fallback={n_skipped_fallback} "
        #     f"skipped_degenerate={n_skipped_degenerate}"
        # )

        if not bboxes:
            # Preserve downstream contract (the trainer expects num_objects
            # >= 1 in some code paths) — emit one dummy that's CLEARLY
            # marked (entity_id=0, region_id=0, polarity=0, full-image
            # bbox). The loss should mask these via the ignore_index path
            # we set up in training/loss.py.
            return {
                "bboxes": np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
                "region_ids": np.array([0], dtype=np.int64),
                "entity_ids": np.array([0], dtype=np.int64),
                "positiveness": np.array([0], dtype=np.int64),
                "num_objects": 1,
            }

        return {
            "bboxes": np.array(bboxes, dtype=np.float32),
            "region_ids": np.array(region_ids, dtype=np.int64),
            "entity_ids": np.array(entity_ids, dtype=np.int64),
            "positiveness": np.array(positiveness_list, dtype=np.int64),
            "num_objects": len(bboxes),
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "bboxes": np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
            "region_ids": np.array([0], dtype=np.int64),
            "entity_ids": np.array([0], dtype=np.int64),
            "positiveness": np.array([0], dtype=np.int64),
            "num_objects": 1,
        }

    def _process_legacy(
        self,
        scene_graph: Dict[str, Any],
        image_width: int,
        image_height: int,
        image_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Original (buggy) processor preserved as an A/B baseline.

        Enable by setting ``processor.use_legacy_processor = True``. Useful
        when reproducing earlier results or debugging whether a regression
        comes from the new processor.
        """
        observations = scene_graph.get('observations', {})
        if not observations:
            return self._empty_result()

        bboxes, region_ids, entity_ids, positiveness_list = [], [], [], []
        for obs_id, obs in observations.items():
            bbox = [0, 0, image_width, image_height]
            if 'localization' in obs and obs['localization']:
                loc = obs['localization']
                if isinstance(loc, dict):
                    if image_id and image_id in loc:
                        img_loc = loc[image_id]
                    else:
                        img_loc = next(iter(loc.values()), {})
                    if isinstance(img_loc, dict) and 'bboxes' in img_loc:
                        if img_loc['bboxes'] and len(img_loc['bboxes']) > 0:
                            bbox = img_loc['bboxes'][0]
            x1 = max(0, min(bbox[0] / image_width, 1.0))
            y1 = max(0, min(bbox[1] / image_height, 1.0))
            x2 = max(0, min(bbox[2] / image_width, 1.0))
            y2 = max(0, min(bbox[3] / image_height, 1.0))
            bboxes.append([x1, y1, x2, y2])

            regions = obs.get('regions', [])
            if regions:
                region_name = regions[0].get('region', 'unknown') if isinstance(regions[0], dict) else str(regions[0])
                region_id = self.region_to_idx.get(region_name.lower(), 0)
            else:
                region_id = 0
            region_ids.append(region_id)

            entities = obs.get('obs_entities', [])
            entity_name = entities[0] if entities and isinstance(entities[0], str) else 'unknown'
            entity_id = self.entity_to_idx.get(entity_name.lower(), 0)
            entity_ids.append(entity_id)

            pos = obs.get('positiveness', 'neg')
            positiveness_list.append(1 if pos == 'pos' else 0)

        return {
            'bboxes': np.array(bboxes, dtype=np.float32),
            'region_ids': np.array(region_ids, dtype=np.int64),
            'entity_ids': np.array(entity_ids, dtype=np.int64),
            'positiveness': np.array(positiveness_list, dtype=np.int64),
            'num_objects': len(bboxes)
        }


class MIMICCXRVQADataset(Dataset):
    """
    PyTorch Dataset for MIMIC-CXR VQA.
    
    Loads:
    - Chest X-ray images from MIMIC-CXR-JPG
    - Scene graphs and QA pairs from MIMIC-Ext-CXR-QBA
    - CheXpert labels for auxiliary supervision
    
    MIMIC-CXR-JPG Structure:
        files/p{XX}/p{subject_id}/s{study_id}/{dicom_id}.jpg
        Example: files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
    
    MIMIC-Ext-CXR-QBA Structure (after extraction):
        qa/p{XX}/p{subject_id}/s{study_id}.qa.json
        scene_data/p{XX}/p{subject_id}/s{study_id}.scene_graph.json
    """
    
    # Frontal view positions in MIMIC-CXR
    FRONTAL_VIEWS = {'PA', 'AP', 'AP AXIAL', 'PA LLD'}
    LATERAL_VIEWS = {'LATERAL', 'LL', 'LAO', 'RAO'}
    
    def __init__(
        self,
        mimic_cxr_path: str,
        mimic_qa_path: str,
        split: str = 'train',
        tokenizer_name: str = 'emilyalsentzer/Bio_ClinicalBERT',
        max_question_length: int = 128,
        quality_grade: str = 'A',
        view_filter: str = 'frontal_only',
        question_types: Optional[List[str]] = None,
        skip_question_types: Optional[List[str]] = None,  # NEW: blacklist (Stage 4 drops A_*)
        chexpert_labels_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        transform: Optional[Any] = None,
        use_exports: bool = False,  # Use pre-filtered exports folder
        cache_dir: Optional[str] = None,  # Cache directory for samples
        use_cache: bool = True,  # Whether to use caching
        force_rebuild_cache: bool = False,  # Force rebuild even if cache exists
        prebuilt_cache_path: Optional[str] = None,  # Optional: load samples from external prebuilt cache (.pkl)
        sg_cache_root: Optional[str] = None,  # Stage-3+ pre-generated SG cache (precompute_sg_cache.py)
        one_question_per_image: bool = False,  # Dedupe to one sample per image (max image diversity per max_samples)
        use_reports: bool = False,  # Inject radiologist INDICATION as input + FINDINGS/IMPRESSION as <think> target
        min_localization_quality: int = 0,  # NEW: Stage 1 only; drop obs with localization_quality < N
    ):
        self.mimic_cxr_path = Path(mimic_cxr_path)
        self.mimic_qa_path = Path(mimic_qa_path)
        self.split = split
        self.max_question_length = max_question_length
        self.quality_grade = quality_grade
        self.view_filter = view_filter
        self.question_types = question_types
        # Normalise blacklist to a set of lowercase strings for cheap membership tests
        self.skip_question_types = (
            {str(q).lower() for q in skip_question_types}
            if skip_question_types else set()
        )
        self.max_samples = max_samples
        self.use_exports = use_exports
        self.use_cache = use_cache
        self.force_rebuild_cache = force_rebuild_cache
        self.prebuilt_cache_path = prebuilt_cache_path
        self.one_question_per_image = bool(one_question_per_image)
        self.use_reports = bool(use_reports)
        # min_localization_quality: only used by SG-target extraction in
        # _extract_gt_targets. Default 0 = off (Stages 2-4 see all observations
        # for VQA). Set to 3 in Stage 1 config to drop NO_LOCALIZATION /
        # FALLBACK_LOCALIZATION obs that gave Stage 1 garbage bbox targets
        # (sg_loss plateaued at 4.71 across runs).
        self.min_localization_quality = int(min_localization_quality)
        # When use_reports is on, we prepend clinical-context ("Clinical context: ...")
        # to the question text before tokenization. Indication is short (~30-80 tokens)
        # so bump max_question_length so the question itself doesn't get truncated.
        if self.use_reports and max_question_length < 256:
            max_question_length = 256

        # Setup cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Tokenizer (IMPORTANT):
        # HuggingFace fast tokenizers can deadlock when a Dataset is constructed
        # in the main process and then DataLoader forks worker processes.
        # To avoid fork-safety issues, initialize lazily inside the worker the
        # first time __getitem__ is called.
        self.tokenizer_name = tokenizer_name
        self.tokenizer = None
        
        # Initialize transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        # Load metadata for view filtering
        self.metadata_df = self._load_metadata()
        
        # Initialize CheXpert loader.
        # Falls back to mimic_cxr_path/mimic-cxr-2.0.0-chexpert.csv.gz if the
        # configured path is empty OR missing on this machine. Configs often
        # carry stale absolute paths from another developer's box ("/home/X/...")
        # and would otherwise silently leave chex_loss=0 for the whole run.
        default_chexpert = self.mimic_cxr_path / 'mimic-cxr-2.0.0-chexpert.csv.gz'
        chosen_path: Optional[str] = None
        if chexpert_labels_path and os.path.exists(chexpert_labels_path):
            chosen_path = chexpert_labels_path
        elif default_chexpert.exists():
            if chexpert_labels_path:
                logger.warning(
                    f"chexpert_labels_path={chexpert_labels_path} does not exist; "
                    f"falling back to {default_chexpert}"
                )
            chosen_path = str(default_chexpert)
        else:
            logger.warning(
                f"No CheXpert CSV found (tried {chexpert_labels_path or '(empty)'} "
                f"and {default_chexpert}). chex_loss will stay at 0."
            )
        self.chexpert_loader = CheXpertLabelLoader(chosen_path)
        
        # Initialize scene graph processor
        self.sg_processor = SceneGraphProcessor()
        
        # Load dataset info - try multiple locations
        dataset_info_paths = [
            self.mimic_qa_path / 'metadata' / 'dataset_info.json',
            self.mimic_qa_path / 'dataset_info.json',
        ]
        for dataset_info_path in dataset_info_paths:
            if dataset_info_path.exists():
                self.sg_processor.load_vocab(str(dataset_info_path))
                break
        
        # Load samples (with caching support)
        self.samples = self._load_samples_with_cache()

        # Pre-generated SG cache (Stage 3+ only — frozen + deterministic generator).
        # Manifest verification is the trainer's responsibility (it needs the live
        # model to compute the expected signature). Here we just record the root
        # and let __getitem__ attempt loads.
        self.sg_cache_root: Optional[Path] = Path(sg_cache_root) if sg_cache_root else None
        if self.sg_cache_root is not None:
            mpath = self.sg_cache_root / "manifest.json"
            if not mpath.exists():
                raise FileNotFoundError(
                    f"sg_cache_root={self.sg_cache_root} has no manifest.json. "
                    f"Run scripts/precompute_sg_cache.py first, or pass "
                    f"sg_cache_root=None to fall back to on-the-fly."
                )
        # Process-local hit-rate counters (workers each maintain their own;
        # the trainer aggregates after the first epoch via assert_sg_cache_hit_rate).
        self._sg_cache_hits = 0
        self._sg_cache_misses = 0
    def _get_tokenizer(self):
        """Lazily initialize tokenizer (fork-safe for DataLoader workers)."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self.tokenizer

    def _load_generated_sg_for(self, subject_id: int, study_id: int) -> Optional[Dict[str, Any]]:
        """Try to load a pre-generated SG dict from disk. Returns None on miss.

        Trap-aware: uses weights_only=True (the file contains only tensors +
        plain python primitives saved by scripts/precompute_sg_cache.py). If
        the file is malformed, count it as a miss and let the model fall back
        to on-the-fly generation rather than crash the whole run.
        """
        if self.sg_cache_root is None:
            return None
        try:
            p_short = f"p{str(subject_id)[:2]}"
            p_full = f"p{subject_id}"
            cache_file = self.sg_cache_root / p_short / p_full / f"s{study_id}.sg.pt"
            if not cache_file.exists():
                self._sg_cache_misses += 1
                return None
            d = torch.load(cache_file, map_location="cpu", weights_only=True)
            self._sg_cache_hits += 1
            # Cast id tensors back to long for downstream indexing. Bboxes /
            # relations stay fp16 on disk; the encoder upcasts them.
            return {
                "bboxes":       d["bboxes"].float() if d["bboxes"].numel() else d["bboxes"],
                "entity_ids":   d["entity_ids"].long(),
                "region_ids":   d["region_ids"].long(),
                "positiveness": d["positiveness"].long(),
                "relations":    d["relations"].float() if d["relations"].numel() else d["relations"],
                "num_objects":  int(d["num_objects"]),
            }
        except Exception as e:
            logger.debug(f"SG cache load failed for s={study_id}: {e}")
            self._sg_cache_misses += 1
            return None

    @staticmethod
    def _extract_report_sections(scene_graph: Dict[str, Any]) -> Dict[str, str]:
        """Pull original radiologist text out of an already-loaded scene_graph JSON.

        QBA's `s*.scene_graph.json` embeds the full report text in two places:
          1) `sentences`: a dict of per-sentence entries with `section_type` field
             (FINDINGS, IMPRESSION, INDICATION, HISTORY, EXAM_TECHNIQUE, IGNORE)
          2) `indication`: a structured block with `indication_summary`,
             `indication`, `evaluation`, `patient_info`

        Returns:
            {
              'indication_text': clinical context (model INPUT — what radiologist
                                 knows BEFORE looking at the image),
              'findings_impression': real radiologist FINDINGS + IMPRESSION
                                     (model OUTPUT target — what to generate),
            }
        Both strings may be '' if the source report didn't have those sections.
        """
        out = {'indication_text': '', 'findings_impression': ''}
        if not isinstance(scene_graph, dict):
            return out

        # Bucket sentences by section_type
        by_section: Dict[str, List[str]] = {}
        for s in (scene_graph.get('sentences') or {}).values():
            if not isinstance(s, dict):
                continue
            sect = (s.get('section_type') or s.get('section') or '').upper()
            text = (s.get('sentence') or '').strip()
            if text:
                by_section.setdefault(sect, []).append(text)

        # === INPUT context: prefer structured indication_summary, fall back to raw sentences ===
        bits: List[str] = []
        ind_obj = scene_graph.get('indication')
        if isinstance(ind_obj, dict):
            summary = (ind_obj.get('indication_summary') or '').strip()
            if summary:
                bits.append(summary)
            else:
                patient = (ind_obj.get('patient_info') or '').strip()
                indtxt = (ind_obj.get('indication') or '').strip()
                evltxt = (ind_obj.get('evaluation') or '').strip()
                if patient:
                    bits.append(patient)
                if indtxt:
                    bits.append(f"Indication: {indtxt}")
                if evltxt:
                    bits.append(f"Evaluation: {evltxt}")
        if not bits:
            bits.extend(by_section.get('INDICATION', []))
        # HISTORY and COMPARISON are also valid INPUT context — they're info
        # the radiologist had BEFORE looking at the image (prior studies, clinical
        # history). EXAM_TECHNIQUE is included if non-trivial.
        bits.extend(by_section.get('HISTORY', []))
        bits.extend(by_section.get('COMPARISON', []))
        # EXAM_TECHNIQUE is usually just "FINAL REPORT" boilerplate; include only
        # if it has substantive content (>30 chars after stripping the header).
        for et in by_section.get('EXAM_TECHNIQUE', []):
            if len(et) > 30 and 'final report' not in et.lower():
                bits.append(et)
        out['indication_text'] = ' '.join(bits).strip()

        # === OUTPUT target: FINDINGS + IMPRESSION (what the radiologist wrote) ===
        fi_parts: List[str] = []
        findings = ' '.join(by_section.get('FINDINGS', [])).strip()
        impression = ' '.join(by_section.get('IMPRESSION', [])).strip()
        if findings:
            fi_parts.append(f"FINDINGS: {findings}")
        if impression:
            fi_parts.append(f"IMPRESSION: {impression}")
        out['findings_impression'] = '\n'.join(fi_parts).strip()
        return out

    def get_sg_cache_stats(self) -> Dict[str, int]:
        """Snapshot of this dataset's hit/miss counters. Workers each maintain
        their own; in DDP the trainer must reduce across ranks if it wants a
        global rate."""
        return {"hits": self._sg_cache_hits, "misses": self._sg_cache_misses}
    
    def _load_metadata(self) -> Optional[pd.DataFrame]:
        """
        Load MIMIC-CXR-JPG metadata for comprehensive image information.
        
        MIMIC-CXR-JPG metadata columns (from mimic-cxr-2.0.0-metadata.csv.gz):
        - dicom_id: Image identifier (JPG filename stem)
        - PerformedProcedureStepDescription: Type of study ("CHEST (PA AND LAT)", etc.)
        - ViewPosition: Orientation ("AP", "PA", "LATERAL", etc.)
        - Rows: Image height in pixels
        - Columns: Image width in pixels
        - StudyDate: Anonymized but chronologically consistent date
        - StudyTime: Time of study (hours:minutes:seconds.fraction)
        - ProcedureCodeSequence_CodeMeaning: Human-readable procedure description
        - ViewCodeSequence_CodeMeaning: Human-readable view orientation
        - PatientOrientationCodeSequence_CodeMeaning: "Erect", "Recumbent", or null
        """
        metadata_file = self.mimic_cxr_path / 'mimic-cxr-2.0.0-metadata.csv.gz'
        
        if not metadata_file.exists():
            metadata_file = self.mimic_cxr_path / 'mimic-cxr-2.0.0-metadata.csv'
        
        if metadata_file.exists():
            try:
                if str(metadata_file).endswith('.gz'):
                    df = pd.read_csv(metadata_file, compression='gzip')
                else:
                    df = pd.read_csv(metadata_file)
                
                # Create indexed lookup dicts for O(1) access
                self._dicom_to_view = {}
                self._dicom_to_metadata = {}  # Full metadata for each image
                
                for _, row in df.iterrows():
                    dicom_id = row.get('dicom_id')
                    if dicom_id:
                        dicom_str = str(dicom_id)
                        self._dicom_to_view[dicom_str] = row.get('ViewPosition')
                        
                        # Store full metadata for enhanced features
                        self._dicom_to_metadata[dicom_str] = {
                            'view_position': row.get('ViewPosition'),
                            'view_code': row.get('ViewCodeSequence_CodeMeaning'),
                            'procedure': row.get('PerformedProcedureStepDescription'),
                            'procedure_code': row.get('ProcedureCodeSequence_CodeMeaning'),
                            'patient_orientation': row.get('PatientOrientationCodeSequence_CodeMeaning'),
                            'original_rows': int(row.get('Rows', 0)) if pd.notna(row.get('Rows')) else None,
                            'original_cols': int(row.get('Columns', 0)) if pd.notna(row.get('Columns')) else None,
                            'study_date': row.get('StudyDate'),
                            'study_time': row.get('StudyTime'),
                        }
                
                logger.info(f"Loaded metadata: {len(df)} images (indexed for fast lookup)")
                return df
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        
        self._dicom_to_view = {}
        self._dicom_to_metadata = {}
        return None
    
    def _get_view_position(self, dicom_id: str) -> Optional[str]:
        """Get ViewPosition for a DICOM ID from metadata (O(1) lookup)."""
        return self._dicom_to_view.get(str(dicom_id))
    
    def _get_image_metadata(self, dicom_id: str) -> Dict[str, Any]:
        """
        Get full MIMIC-CXR-JPG metadata for an image.
        
        Returns dict with:
        - view_position: "AP", "PA", "LATERAL", etc.
        - view_code: Human-readable view ("postero-anterior", "antero-posterior", etc.)
        - procedure: Type of study ("CHEST (PA AND LAT)", "CHEST (PORTABLE AP)", etc.)
        - patient_orientation: "Erect", "Recumbent", or None
        - original_rows: Original image height before resize
        - original_cols: Original image width before resize
        - study_date: Anonymized date (chronologically consistent)
        - study_time: Time of study
        """
        default = {
            'view_position': None,
            'view_code': None,
            'procedure': None,
            'procedure_code': None,
            'patient_orientation': None,
            'original_rows': None,
            'original_cols': None,
            'study_date': None,
            'study_time': None,
        }
        return self._dicom_to_metadata.get(str(dicom_id), default)
    
    def _get_cache_key(self) -> str:
        """Generate a unique cache key based on dataset configuration.

        MUST include every field that affects which samples land in the
        cache. skip_question_types + min_localization_quality were added
        for the v2 retrain fixes; forgetting them here means a stale
        pickle from a prior run would be silently reused with the new
        filters ignored (Stage 4 would still see A_* questions, Stage 1
        would still get garbage bboxes).
        """
        config_str = (
            f"cxr:{self.mimic_cxr_path}|"
            f"qa:{self.mimic_qa_path}|"
            f"split:{self.split}|"
            f"quality:{self.quality_grade}|"
            f"view:{self.view_filter}|"
            f"qtypes:{sorted(self.question_types) if self.question_types else 'all'}|"
            f"skip_qtypes:{sorted(self.skip_question_types) if self.skip_question_types else 'none'}|"
            f"min_loc_q:{self.min_localization_quality}|"
            f"max:{self.max_samples or 'none'}|"
            f"exports:{self.use_exports}|"
            f"dedupe_img:{int(self.one_question_per_image)}|"
            f"scan_seed:{42 if self.one_question_per_image else 'noshuffle'}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    @staticmethod
    def _dedupe_by_image(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep at most one sample per unique STUDY (= unique scene graph + unique image).

        In MIMIC-Ext-CXR-QBA, scene graphs are per-STUDY (one .scene_graph.json per
        study), so two samples that share study_id share the same SG even if they
        use different frontal images. To guarantee "100K unique images AND 100K
        unique scene graphs", we dedupe on study_id (with image_path fallback for
        legacy caches that lack study_id).

        Preserves order: first occurrence per study wins.
        """
        seen: set = set()
        out: List[Dict[str, Any]] = []
        for s in samples:
            key = s.get('study_id') or s.get('scene_graph_path') or s.get('image_path') or s.get('dicom_id')
            if key is None or key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out
    
    def _get_cache_path(self) -> Path:
        """Get the cache file path for current configuration."""
        cache_key = self._get_cache_key()
        return self.cache_dir / f"samples_{self.split}_{cache_key}.pkl"
    
    def _load_samples_with_cache(self) -> List[Dict[str, Any]]:
        """Load samples with caching support for faster distributed training."""
        import gc
        import time
        
        # Detect distributed environment (DeepSpeed/DDP)
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))
        is_distributed = world_size > 1
        
        # If an explicit external cache is provided (e.g., produced by scripts/prebuild_cache.py),
        # load it directly. This is especially useful for subset caches for pretraining.
        if self.prebuilt_cache_path:
            p = Path(self.prebuilt_cache_path)
            if p.exists():
                try:
                    # ============================================================
                    # DISTRIBUTED: Check for pre-sharded cache files first
                    # Format: cache.pkl -> cache.shard0of4.pkl, cache.shard1of4.pkl, etc.
                    # ============================================================
                    if is_distributed:
                        shard_path = p.parent / f"{p.stem}.shard{rank}of{world_size}.pkl"
                        if shard_path.exists():
                            logger.info(f"[Rank {rank}] Loading PRE-SHARDED cache: {shard_path}")
                            with open(shard_path, 'rb') as f:
                                samples = pickle.load(f)
                            logger.info(f"[Rank {rank}] Loaded {len(samples)} samples from shard")

                            # ── Dedupe to one question per image (max image diversity) ──
                            if self.one_question_per_image:
                                pre = len(samples)
                                samples = self._dedupe_by_image(samples)
                                logger.info(
                                    f"[Rank {rank}] one_question_per_image: "
                                    f"{pre:,} → {len(samples):,} samples "
                                    f"({len(samples)/max(1,pre):.1%} kept)"
                                )

                            # ── Apply max_samples to shard ──────────────
                            if self.max_samples and len(samples) > self.max_samples:
                                orig_len = len(samples)
                                samples = samples[:self.max_samples]
                                # The old list (2M items) lost its reference;
                                # 1.5M tail items are now unreachable → GC them
                                gc.collect()
                                try:
                                    import ctypes
                                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                                except (OSError, AttributeError):
                                    pass
                                logger.info(
                                    f"[Rank {rank}] Truncated shard from {orig_len:,} → "
                                    f"{self.max_samples:,} samples (freed ~"
                                    f"{(orig_len - self.max_samples) * 15 / 1024:.0f} MB)"
                                )
                            
                            return samples
                        
                        # No pre-sharded cache exists - use staggered loading
                        # Each rank waits (rank * 30 seconds) before loading to avoid simultaneous OOM
                        wait_time = rank * 30  # 0s, 30s, 60s, 90s for ranks 0,1,2,3
                        if wait_time > 0:
                            logger.info(f"[Rank {rank}] Waiting {wait_time}s for staggered loading (avoids OOM)...")
                            time.sleep(wait_time)
                    
                    logger.info(f"[Rank {rank}/{world_size}] Loading samples from PREBUILT cache: {p}")
                    with open(p, 'rb') as f:
                        samples = pickle.load(f)

                    # Best-effort filtering to align with dataset config
                    if self.view_filter and self.view_filter != 'all':
                        filtered = []
                        for s in samples:
                            view_pos = s.get('view_position')
                            if view_pos is None and s.get('dicom_id'):
                                view_pos = self._get_view_position(s.get('dicom_id'))
                            if self._is_valid_view(view_pos):
                                filtered.append(s)
                        samples = filtered

                    if self.question_types:
                        samples = [s for s in samples if s.get('question_type') in set(self.question_types)]

                    # Quality filtering is not possible if the prebuilt cache doesn't carry quality metadata
                    if self.quality_grade and self.quality_grade.lower() not in ('', 'all', 'none'):
                        logger.warning(
                            "prebuilt_cache_path is set but quality_grade filtering requires per-question quality metadata "
                            "(not present in prebuilt cache). Proceeding without quality filtering."
                        )

                    # Dedupe to one question per image (max image diversity for SG generator)
                    if self.one_question_per_image:
                        pre = len(samples)
                        samples = self._dedupe_by_image(samples)
                        logger.info(
                            f"[Rank {rank}] one_question_per_image: "
                            f"{pre:,} → {len(samples):,} samples "
                            f"({len(samples)/max(1,pre):.1%} kept)"
                        )

                    # Apply max_samples limit if needed
                    if self.max_samples and len(samples) > self.max_samples:
                        samples = samples[:self.max_samples]
                    
                    # ============================================================
                    # DISTRIBUTED SHARDING: Each rank keeps only its portion
                    # ============================================================
                    if is_distributed:
                        total_samples = len(samples)
                        # Shard samples: rank i gets samples[i::world_size]
                        my_samples = samples[rank::world_size]
                        
                        # Save shard to disk for future fast loading
                        shard_path = p.parent / f"{p.stem}.shard{rank}of{world_size}.pkl"
                        try:
                            logger.info(f"[Rank {rank}] Saving shard to {shard_path} for future runs...")
                            with open(shard_path, 'wb') as f:
                                pickle.dump(my_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
                        except Exception as e:
                            logger.warning(f"[Rank {rank}] Could not save shard: {e}")
                        
                        # Clear full samples list and keep only our shard
                        del samples
                        samples = my_samples
                        gc.collect()
                        
                        logger.info(
                            f"[Rank {rank}/{world_size}] Sharded: keeping {len(samples)}/{total_samples} samples "
                            f"(~{100/world_size:.0f}% per rank)"
                        )

                    logger.info(f"[Rank {rank}] Loaded {len(samples)} samples from PREBUILT cache")
                    return samples
                except Exception as e:
                    logger.warning(f"Failed to load prebuilt cache '{p}', falling back to normal loading: {e}")
            else:
                logger.warning(f"prebuilt_cache_path does not exist: {p} (falling back to normal loading)")

        cache_path = self._get_cache_path()
        
        # Try to load from cache
        if self.use_cache and not self.force_rebuild_cache and cache_path.exists():
            try:
                logger.info(f"[Rank {rank}/{world_size}] Loading samples from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    samples = pickle.load(f)

                # Dedupe to one question per image (max image diversity for SG generator).
                # Cache key already encodes self.one_question_per_image, so a cache file
                # loaded here was built with the same flag — but apply defensively in case
                # an old cache predates the flag.
                if self.one_question_per_image:
                    pre = len(samples)
                    samples = self._dedupe_by_image(samples)
                    if len(samples) != pre:
                        logger.info(
                            f"[Rank {rank}] one_question_per_image: "
                            f"{pre:,} → {len(samples):,} samples "
                            f"({len(samples)/max(1,pre):.1%} kept)"
                        )

                # Apply max_samples limit if needed
                if self.max_samples and len(samples) > self.max_samples:
                    samples = samples[:self.max_samples]
                
                # Distributed sharding for regular cache too
                if is_distributed:
                    total_samples = len(samples)
                    samples = samples[rank::world_size]
                    logger.info(
                        f"[Rank {rank}/{world_size}] Sharded: keeping {len(samples)}/{total_samples} samples"
                    )
                    import gc
                    gc.collect()
                
                logger.info(f"[Rank {rank}] Loaded {len(samples)} samples from cache")
                return samples
            except Exception as e:
                logger.warning(f"Failed to load cache, rebuilding: {e}")
        
        # Build samples from scratch
        logger.info("Building samples from dataset (this may take 10-15 minutes)...")
        samples = self._load_samples()
        
        # Save to cache
        if self.use_cache and len(samples) > 0:
            try:
                logger.info(f"Saving {len(samples)} samples to cache: {cache_path}")
                with open(cache_path, 'wb') as f:
                    pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info("Cache saved successfully!")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
        
        return samples
    
    @classmethod
    def prebuild_cache(
        cls,
        mimic_cxr_path: str,
        mimic_qa_path: str,
        splits: List[str] = ['train', 'val', 'test'],
        quality_grade: str = 'all',
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Pre-build cache for all splits (run before distributed training).
        
        Usage:
            python -c "from data.mimic_cxr_dataset import MIMICCXRVQADataset; \\
                MIMICCXRVQADataset.prebuild_cache('/path/to/cxr', '/path/to/qa')"
        """
        for split in splits:
            logger.info(f"\n{'='*60}")
            logger.info(f"Pre-building cache for split: {split}")
            logger.info(f"{'='*60}")
            
            dataset = cls(
                mimic_cxr_path=mimic_cxr_path,
                mimic_qa_path=mimic_qa_path,
                split=split,
                quality_grade=quality_grade,
                cache_dir=cache_dir,
                use_cache=True,
                force_rebuild_cache=True,  # Force rebuild
                **kwargs
            )
            logger.info(f"Split {split}: {len(dataset)} samples cached")
        
        logger.info(f"\n{'='*60}")
        logger.info("Cache pre-building complete!")
        logger.info(f"{'='*60}")
    
    def _is_valid_view(self, view_position: Optional[str]) -> bool:
        """Check if view position matches filter criteria."""
        if self.view_filter == 'all' or view_position is None:
            return True
        
        view_upper = view_position.upper() if view_position else ''
        
        if self.view_filter == 'frontal_only':
            return view_upper in self.FRONTAL_VIEWS
        elif self.view_filter == 'lateral_only':
            return view_upper in self.LATERAL_VIEWS
        
        return True
    
    def _meets_quality_grade(self, actual_grade: str, required_grade: str) -> bool:
        """
        Check if actual quality grade meets or exceeds required grade.
        
        Quality hierarchy: A++ > A+ > A > B > C > U
        """
        grade_order = {'A++': 5, 'A+': 4, 'A': 3, 'B': 2, 'C': 1, 'U': 0}
        
        actual_val = grade_order.get(actual_grade, 0)
        required_val = grade_order.get(required_grade, 0)
        
        return actual_val >= required_val
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load all QA samples for this split."""
        samples = []
        
        # Load split information
        split_file = self.mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv.gz'
        if split_file.exists():
            splits_df = pd.read_csv(split_file, compression='gzip')
        else:
            split_file = self.mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv'
            if split_file.exists():
                splits_df = pd.read_csv(split_file)
            else:
                logger.warning("Split file not found, using all data")
                splits_df = None
        
        if splits_df is not None:
            # Map split names (MIMIC uses 'validate' not 'val')
            split_name = 'validate' if self.split == 'val' else self.split
            splits_df = splits_df[splits_df['split'] == split_name]
            valid_studies = set(zip(splits_df['subject_id'].astype(int), splits_df['study_id'].astype(int)))
            logger.info(f"Found {len(valid_studies)} studies in '{split_name}' split")
        else:
            valid_studies = None
        
        # Check for QA directory - must be extracted from qa.zip
        qa_dir = self.mimic_qa_path / 'qa'
        if not qa_dir.exists():
            # Check if zip exists but not extracted
            qa_zip = self.mimic_qa_path / 'qa.zip'
            if qa_zip.exists():
                logger.error("=" * 60)
                logger.error("QA DATA NOT EXTRACTED!")
                logger.error("=" * 60)
                logger.error(f"Found qa.zip but 'qa/' folder missing.")
                logger.error(f"Please extract: {qa_zip}")
                logger.error("")
                logger.error("On Windows PowerShell:")
                logger.error(f"  Expand-Archive -Path '{qa_zip}' -DestinationPath '{self.mimic_qa_path}'")
                logger.error("")
                logger.error("On Linux/Mac:")
                logger.error(f"  unzip '{qa_zip}' -d '{self.mimic_qa_path}'")
                logger.error("=" * 60)
            else:
                logger.warning(f"QA directory not found: {qa_dir}")
            # Create dummy samples for testing
            return self._create_dummy_samples()
        
        # Iterate through patient directories
        # Structure: qa/p{XX}/p{subject_id}/s{study_id}.qa.json
        p_groups = sorted([p for p in qa_dir.iterdir() if p.is_dir() and p.name.startswith('p')])
        logger.info(f"Scanning {len(p_groups)} patient groups for QA files...")

        files_scanned = 0
        skipped_split = 0
        skipped_image = 0
        skipped_quality = 0
        skipped_dup_image = 0

        # When one_question_per_image is set, we dedupe at the STUDY level so
        # each kept sample has a UNIQUE scene graph (SGs are per-study in QBA)
        # AND a unique image. max_samples then counts unique studies/images/SGs.
        # Name kept as seen_image_paths for back-compat with downstream logging.
        seen_image_paths: set = set() if self.one_question_per_image else None

        # Build the iteration order. When dedupe_by_study is active, randomize
        # study order so the first max_samples studies are sampled UNIFORMLY
        # across the whole train split — otherwise the lex-ordered walk
        # (p10 → p11 → ...) means the first 100K studies all come from a
        # few low-numbered patient groups. Deterministic seed → reproducible
        # subset across runs. Seed is folded into the cache key.
        scan_seed = 42  # bump to invalidate caches if you want a different draw
        flat_iter = []  # list of (subject_id, patient_dir, qa_file)
        for p_group in p_groups:
            for patient_dir in p_group.iterdir():
                if not patient_dir.is_dir() or not patient_dir.name.startswith('p'):
                    continue
                try:
                    subject_id_int = int(patient_dir.name[1:])
                except ValueError:
                    continue
                for qa_file in patient_dir.glob('s*.qa.json'):
                    flat_iter.append((subject_id_int, patient_dir, qa_file))

        if self.one_question_per_image:
            import random as _random
            rng = _random.Random(scan_seed)
            rng.shuffle(flat_iter)
            logger.info(
                f"one_question_per_image: shuffled {len(flat_iter):,} qa-files "
                f"with seed={scan_seed} for uniform-across-cohort sampling"
            )

        for subject_id_int, patient_dir, qa_file in flat_iter:
            files_scanned += 1
            if files_scanned % 5000 == 0:
                logger.info(f"  [{files_scanned}] samples={len(samples)}, skip_split={skipped_split}, skip_img={skipped_image}, skip_qual={skipped_quality}, skip_dup_img={skipped_dup_image}")
            try:
                # Parse IDs from path
                # patient_dir.name = "p10000032" -> subject_id = 10000032
                # qa_file.stem = "s50414267.qa" -> study_id = 50414267
                subject_id = subject_id_int
                study_id_str = qa_file.stem.split('.')[0]  # "s50414267"
                study_id = int(study_id_str[1:])  # Remove 's' prefix

                # Check if in valid split
                if valid_studies and (subject_id, study_id) not in valid_studies:
                    skipped_split += 1
                    continue

                # In one_question_per_image mode: skip studies we've already
                # added (each kept study contributes one sample = one image +
                # one unique scene graph). Checked BEFORE loading the JSON so
                # we don't pay parse cost for studies we'll skip.
                if seen_image_paths is not None and (subject_id, study_id) in seen_image_paths:
                    skipped_dup_image += 1
                    continue

                # Load QA data
                with open(qa_file) as f:
                    qa_data = json.load(f)

                # Find corresponding image (best frontal view, PA > AP).
                image_path, dicom_id = self._find_image(subject_id, study_id)
                if image_path is None:
                    skipped_image += 1
                    continue

                # Find scene graph
                sg_path = self._find_scene_graph(subject_id, study_id)

                # Process each question
                questions = qa_data.get('questions', [])

                # Log first question structure once for debugging
                if files_scanned == 1 and len(questions) > 0:
                    first_q_keys = list(questions[0].keys())
                    logger.info(f"  Sample question keys: {first_q_keys[:10]}")

                for q in questions:
                    # Quality filter (skip if quality_grade is empty/None/"all").
                    # QBA stores quality at TWO independent keys:
                    #   - extraction_quality: how reliably the question was
                    #     extracted from the report (used for pretrain/finetune split)
                    #   - question_img_localization_quality: bbox grounding quality
                    #     (only meaningful for grounding-relevant questions)
                    # Take the MIN of the two so 'A' = both are A or better.
                    # Legacy 'question_quality' / 'quality' keys are kept as fallback
                    # for older QBA dumps.
                    if self.quality_grade and self.quality_grade.lower() not in ('', 'all', 'none'):
                        # FIXED: QBA stores quality as per-criterion INT dicts
                        # ({'region_extract_quality': 4, 'localization_quality': 1, ...})
                        # — _qba_dict_to_grade maps each int to a letter grade
                        # via the per-criterion enum (QBA_CRITERION_INT_GRADES)
                        # and returns the worst.
                        #
                        # NOTE on question_img_localization_quality: it's a per-IMAGE
                        # dict ({dicom_id: int}) with a different semantic — the
                        # values were 0 across all observed samples even on
                        # otherwise-A-grade questions, which would always fail
                        # A-grade filter. Drop it from quality aggregation; use
                        # ONLY extraction_quality + legacy_quality (per QBA paper
                        # which defines fine-tuning grade by extraction quality).
                        ex_q = _qba_dict_to_grade(q.get('extraction_quality'))
                        legacy_raw = q.get('question_quality', q.get('quality'))
                        if isinstance(legacy_raw, str):
                            legacy = legacy_raw
                        else:
                            legacy = _qba_dict_to_grade(legacy_raw)
                        grades = [g for g in (ex_q, legacy) if g]
                        if not grades:
                            quality_rating = 'B'  # no quality info → assume B
                        else:
                            quality_rating = min(grades, key=lambda g: QBA_GRADE_ORDER.get(g, 0))

                        if not self._meets_quality_grade(quality_rating, self.quality_grade):
                            skipped_quality += 1
                            continue

                    # Question type filter — whitelist (question_types) and
                    # blacklist (skip_question_types). Whitelist takes
                    # precedence; blacklist runs after so it can prune even
                    # whitelist-passing types (Stage 4: question_types=null
                    # but A_* are blacklisted).
                    q_type = q.get('question_type', 'unknown')
                    if self.question_types and q_type not in self.question_types:
                        continue
                    if self.skip_question_types and q_type.lower() in self.skip_question_types:
                        continue

                    samples.append({
                        'subject_id': subject_id,
                        'study_id': study_id,
                        'dicom_id': dicom_id,
                        'image_path': str(image_path),
                        'scene_graph_path': str(sg_path) if sg_path else None,
                        'question_id': q.get('question_id', ''),
                        'question_type': q_type,
                        'question_strategy': q.get('question_strategy', ''),
                        'question': q.get('question', ''),
                        'answers': q.get('answers', []),
                        'obs_ids': q.get('obs_ids', []),
                    })

                    if self.max_samples and len(samples) >= self.max_samples:
                        return samples

                    # one_question_per_image: mark this study as taken
                    # (= 1 unique image + 1 unique SG) and skip remaining
                    # questions for it (they would reuse the same image+SG).
                    if seen_image_paths is not None:
                        seen_image_paths.add((subject_id, study_id))
                        break

            except Exception as e:
                logger.debug(f"Error loading {qa_file}: {e}")
                continue
        
        logger.info(f"Scan complete: {files_scanned} files, {len(samples)} samples")
        logger.info(f"  Skipped: split={skipped_split}, image={skipped_image}, quality={skipped_quality}")
        
        if len(samples) == 0:
            logger.warning("No samples found, creating dummy samples")
            return self._create_dummy_samples()
        
        return samples
    
    def _create_dummy_samples(self) -> List[Dict[str, Any]]:
        """Create dummy samples for testing when no real data is available."""
        return [{
            'subject_id': 10000032,
            'study_id': 50000001,
            'image_path': None,  # Will use dummy image
            'scene_graph_path': None,
            'question_id': 'dummy_001',
            'question_type': 'is_abnormal',
            'question': 'Is there any abnormality visible in the chest X-ray?',
            'answers': [{'text': 'Yes', 'confidence': 1.0}],
        }] * min(100, self.max_samples or 100)
    
    def _find_image(self, subject_id: int, study_id: int) -> Tuple[Optional[Path], Optional[str]]:
        """
        Find the best image file for a study.
        
        MIMIC-CXR-JPG structure: files/p{XX}/p{subject_id}/s{study_id}/{dicom_id}.jpg
        Example: files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
        
        Returns:
            Tuple of (image_path, dicom_id) or (None, None) if not found
        """
        # Build path: p{XX} is first 2 chars of subject_id
        p_group = f"p{str(subject_id)[:2]}"
        study_dir = self.mimic_cxr_path / 'files' / p_group / f"p{subject_id}" / f"s{study_id}"
        
        if not study_dir.exists():
            return None, None
        
        # Get all images in the study
        images = list(study_dir.glob('*.jpg'))
        if not images:
            return None, None
        
        # Filter by view if metadata available
        valid_images = []
        for img_file in images:
            dicom_id = img_file.stem  # Filename without extension
            view_pos = self._get_view_position(dicom_id)
            
            if self._is_valid_view(view_pos):
                # Prioritize PA over AP (PA is generally higher quality)
                priority = 0
                if view_pos:
                    if view_pos.upper() == 'PA':
                        priority = 2
                    elif view_pos.upper() == 'AP':
                        priority = 1
                valid_images.append((img_file, dicom_id, priority))
        
        if not valid_images:
            # Fall back to first image if no valid views found
            img_file = images[0]
            return img_file, img_file.stem
        
        # Sort by priority (highest first) and return best
        valid_images.sort(key=lambda x: x[2], reverse=True)
        best_img, best_dicom_id, _ = valid_images[0]
        
        return best_img, best_dicom_id
    
    def _find_scene_graph(self, subject_id: int, study_id: int) -> Optional[Path]:
        """Find the scene graph file for a study."""
        p_group = f"p{str(subject_id)[:2]}"
        
        # Try multiple possible paths (extracted from scene_data.zip)
        possible_paths = [
            # After extracting scene_data.zip
            self.mimic_qa_path / 'scene_data' / p_group / f"p{subject_id}" / f"s{study_id}.scene_graph.json",
            self.mimic_qa_path / 'scene_data' / p_group / f"p{subject_id}" / f"s{study_id}.json",
            # Alternative structure
            self.mimic_qa_path / 'scene_graphs' / p_group / f"p{subject_id}" / f"s{study_id}.json",
            self.mimic_qa_path / 'scene_graphs' / p_group / f"p{subject_id}" / f"s{study_id}.scene_graph.json",
        ]
        
        for sg_file in possible_paths:
            if sg_file.exists():
                return sg_file
        
        return None
    
    def _get_answer_idx(self, question_type: str, answers: List[Dict]) -> int:
        """
        Convert answer to index for classification.
        
        Answer format from MIMIC-Ext-CXR-QBA:
        {
            "answer_id": "...",
            "answer_type": "main_answer",
            "text": "There is no focal consolidation.",
            "positiveness": "neg"/"pos"/"neutral",
            "regions": ["lungs", "left lung"],
            "obs_entities": ["consolidation", "opacity"],
            "modifiers": [["severity", "mild"], ...],
            ...
        }
        
        Returns answer index based on question type and head:
        - binary: 0=No, 1=Yes
        - severity: 0=none, 1=mild, 2=moderate, 3=severe
        - region: index into region vocabulary (0-25)
        - category: index into entity vocabulary (0-13 for CheXpert categories)
        """
        if not answers:
            return 0
        
        # Get main answer (first answer or first with answer_type="main_answer")
        main_answer = answers[0]
        for ans in answers:
            if ans.get('answer_type') == 'main_answer':
                main_answer = ans
                break
        
        answer_text = main_answer.get('text', '').lower()
        positiveness = main_answer.get('positiveness', '')
        
        # Determine head type from question type
        head_type = QUESTION_TYPE_MAP.get(question_type, 'binary')
        
        # =========================================================================
        # BINARY HEAD (Yes/No questions)
        # =========================================================================
        if head_type == 'binary':
            # Use positiveness field directly (most reliable)
            if positiveness:
                # Handle 'is_normal' and 'C04_is_normal_region' inversions
                is_normal_question = 'normal' in question_type.lower() and 'abnormal' not in question_type.lower()
                
                if positiveness == 'pos' or positiveness == 'positive':
                    return 0 if is_normal_question else 1
                elif positiveness == 'neg' or positiveness == 'negative':
                    return 1 if is_normal_question else 0
                elif positiveness == 'neutral':
                    return 0  # Treat neutral as negative for binary
            
            # Fall back to text parsing
            if any(w in answer_text for w in ['yes', 'present', 'positive', 'abnormal', 'there is']):
                return 1
            elif any(w in answer_text for w in ['no', 'absent', 'negative', 'normal', 'there is no']):
                return 0
            return 0
        
        # =========================================================================
        # SEVERITY HEAD (none/mild/moderate/severe)
        # =========================================================================
        elif head_type == 'severity':
            # First check modifiers field (most reliable)
            modifiers = main_answer.get('modifiers', [])
            
            # Modifiers format: [["severity", "mild"], ["change", "improved"], ...]
            for mod in modifiers:
                if isinstance(mod, list) and len(mod) >= 2:
                    if mod[0].lower() == 'severity':
                        severity_mod = mod[1].lower()
                        if severity_mod in ['none', 'no', 'absent']:
                            return 0
                        elif severity_mod in ['mild', 'small', 'minimal']:
                            return 1
                        elif severity_mod in ['moderate', 'medium']:
                            return 2
                        elif severity_mod in ['severe', 'large', 'significant']:
                            return 3
            
            # Fall back to text parsing
            if any(w in answer_text for w in ['none', 'no ', 'absent', 'not present']):
                return 0
            elif any(w in answer_text for w in ['mild', 'small', 'minimal', 'trace']):
                return 1
            elif any(w in answer_text for w in ['moderate', 'medium']):
                return 2
            elif any(w in answer_text for w in ['severe', 'large', 'significant', 'massive']):
                return 3
            return 0
        
        # =========================================================================
        # REGION HEAD (Anatomical Location)
        # =========================================================================
        elif head_type == 'region':
            # Use regions field directly (most reliable)
            regions = main_answer.get('regions', [])
            if regions:
                region_name = regions[0].lower() if isinstance(regions[0], str) else str(regions[0]).lower()
                # Map to region index using scene graph processor vocabulary
                region_idx = self.sg_processor.region_to_idx.get(region_name, 0)
                # Clamp to 26 major regions for the head (reduce vocabulary)
                return min(region_idx, 25)
            return 0
        
        # =========================================================================
        # CATEGORY HEAD (Finding Type / Entity)
        # =========================================================================
        elif head_type == 'category':
            # Use obs_entities field directly (most reliable)
            entities = main_answer.get('obs_entities', [])
            if entities:
                entity_name = entities[0].lower() if isinstance(entities[0], str) else str(entities[0]).lower()
                
                # Map to CheXpert category index (14 categories)
                chexpert_mapping = {
                    'atelectasis': 0, 'cardiomegaly': 1, 'consolidation': 2, 'edema': 3,
                    'enlarged cardiomediastinum': 4, 'fracture': 5, 'lung lesion': 6,
                    'lung opacity': 7, 'pleural effusion': 8, 'pneumonia': 9,
                    'pneumothorax': 10, 'pleural other': 11, 'support devices': 12, 
                    'no finding': 13,
                    # Common aliases
                    'opacity': 7, 'effusion': 8, 'lesion': 6, 'mass': 6, 'nodule': 6,
                    'device': 12, 'tube': 12, 'line': 12, 'catheter': 12, 'pacemaker': 12,
                }
                
                # Try exact match first
                if entity_name in chexpert_mapping:
                    return chexpert_mapping[entity_name]
                
                # Try partial match
                for key, idx in chexpert_mapping.items():
                    if key in entity_name or entity_name in key:
                        return idx
                
                # Default to entity vocabulary if no CheXpert match
                entity_idx = self.sg_processor.entity_to_idx.get(entity_name, 0)
                return min(entity_idx, 13)  # Clamp to 14 categories
            return 0
        
        # Default
        return 0
    
    def _get_answer_text(self, answers: List[Dict]) -> str:
        """Get the main answer text for text-based evaluation."""
        if not answers:
            return ""
        
        # Get main answer
        for ans in answers:
            if ans.get('answer_type') == 'main_answer':
                return ans.get('text', '')
        
        return answers[0].get('text', '')
    
    def _get_full_answer_text(self, answers: List[Dict]) -> str:
        """
        Get the complete hierarchical answer text for decoder training.
        
        Concatenates main_answer + details in natural sentence form,
        matching the rich MIMIC-Ext-CXR-QBA answer structure.
        """
        if not answers:
            return "No findings to report."
        
        parts = []
        
        for ans in answers:
            answer_type = ans.get('answer_type', 'main_answer')
            text = ans.get('text', '')
            
            if not text:
                continue
            
            # Main answers come first
            if answer_type == 'main_answer':
                parts.insert(0, text)
            # Details are appended
            elif answer_type == 'details':
                parts.append(text)
            # Related info goes at the end
            elif answer_type == 'related_information':
                parts.append(f"Additionally, {text.lower()}" if text[0].isupper() else text)
            
            # Process sub-answers recursively
            sub_answers = ans.get('sub_answers', [])
            for sub in sub_answers:
                sub_text = sub.get('text', '')
                if sub_text:
                    parts.append(sub_text)
        
        # Join into coherent answer
        if not parts:
            return "No significant findings."
        
        # Clean up the combined text
        full_text = ' '.join(parts)
        # Remove duplicate periods
        full_text = full_text.replace('..', '.').replace('. .', '.')
        
        return full_text
    
    def _get_grounding_data(
        self, 
        answers: List[Dict], 
        image_width: int, 
        image_height: int,
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract visual grounding data from answers.
        
        MIMIC-Ext-CXR-QBA provides bounding boxes per answer via the localization field:
        {
            "localization": {
                "[image_id]": {
                    "bboxes": [[x1, y1, x2, y2], ...],
                    ...
                }
            }
        }
        
        Returns:
            Dict with normalized bbox, validity flag, and regions
        """
        # Default: full image, not a valid grounding target
        default_bbox = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        
        if not answers:
            return {
                'bbox': default_bbox,
                'valid': False,
                'regions': []
            }
        
        # Find main answer with localization
        for ans in answers:
            if ans.get('answer_type') != 'main_answer':
                continue
            
            localization = ans.get('localization', {})
            if not localization:
                continue
            
            # Get localization for specific image or first available
            if image_id and image_id in localization:
                img_loc = localization[image_id]
            elif localization:
                img_loc = next(iter(localization.values()), {})
            else:
                continue
            
            bboxes = img_loc.get('bboxes', [])
            if not bboxes or len(bboxes) == 0:
                continue
            
            # Get first bbox and normalize
            bbox = bboxes[0]
            x1 = max(0, min(bbox[0] / image_width, 1.0))
            y1 = max(0, min(bbox[1] / image_height, 1.0))
            x2 = max(0, min(bbox[2] / image_width, 1.0))
            y2 = max(0, min(bbox[3] / image_height, 1.0))
            
            # Ensure valid bbox
            if x2 <= x1 or y2 <= y1:
                continue
            
            return {
                'bbox': np.array([x1, y1, x2, y2], dtype=np.float32),
                'valid': True,
                'regions': ans.get('regions', [])
            }
        
        return {
            'bbox': default_bbox,
            'valid': False,
            'regions': []
        }
    
    def _get_scene_graph_targets(
        self, 
        scene_graph: Dict[str, Any],
        image_width: int,
        image_height: int,
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract ground truth targets for scene graph generation training.
        
        From MIMIC-Ext-CXR-QBA scene_graph.json:
        - observations: contains entities, regions, bboxes
        - regions: anatomical region nodes with bboxes
        - located_at_relations: observation-region edges
        
        Returns:
            Dict with gt_bboxes, gt_entities, gt_regions, gt_positiveness
        """
        observations = scene_graph.get('observations', {})
        
        if not observations:
            return {
                'gt_bboxes': None,
                'gt_entities': None,
                'gt_regions': None,
                'gt_positiveness': None,
                'gt_relationships': None,
            }
        
        bboxes = []
        entities = []
        regions = []
        positiveness_list = []
        
        for obs_id, obs in observations.items():
            # Stage-1-only filter: drop observations whose localization_quality
            # is below BBOX_LOCALIZATION (level 3). QBA's level-0/1/2 entries
            # (NO_LOCALIZATION, FALLBACK_LOCALIZATION, INCOMPLETE_LOCALIZATION)
            # produce whole-image or whole-region bboxes that taught the
            # Stage-1 SG generator nothing useful (sg_loss plateaued ~4.7).
            # min_localization_quality defaults to 0 = off; only Stage 1
            # turns this on. See QBA_CRITERION_INT_GRADES['localization_quality'].
            if self.min_localization_quality > 0:
                loc_q = obs.get('localization_quality')
                if isinstance(loc_q, dict):
                    # Per-image int dict {dicom_id: int}; pick the worst across
                    # images so a single-bad-localization study isn't kept.
                    vals = [v for v in loc_q.values() if isinstance(v, int)]
                    loc_q_int = min(vals) if vals else None
                elif isinstance(loc_q, int):
                    loc_q_int = loc_q
                else:
                    loc_q_int = None
                if loc_q_int is None or loc_q_int < self.min_localization_quality:
                    continue

            # Extract bbox
            bbox = None
            if 'localization' in obs and obs['localization']:
                loc = obs['localization']
                if isinstance(loc, dict):
                    if image_id and image_id in loc:
                        img_loc = loc[image_id]
                    else:
                        img_loc = next(iter(loc.values()), {})

                    if isinstance(img_loc, dict) and 'bboxes' in img_loc:
                        if img_loc['bboxes'] and len(img_loc['bboxes']) > 0:
                            raw_bbox = img_loc['bboxes'][0]
                            # Normalize
                            x1 = max(0, min(raw_bbox[0] / image_width, 1.0))
                            y1 = max(0, min(raw_bbox[1] / image_height, 1.0))
                            x2 = max(0, min(raw_bbox[2] / image_width, 1.0))
                            y2 = max(0, min(raw_bbox[3] / image_height, 1.0))
                            if x2 > x1 and y2 > y1:
                                bbox = [x1, y1, x2, y2]

            if bbox is None:
                bbox = [0.0, 0.0, 1.0, 1.0]  # Default to full image
            bboxes.append(bbox)
            
            # Entity (finding)
            obs_entities = obs.get('obs_entities', [])
            if obs_entities:
                entity_name = obs_entities[0].lower() if isinstance(obs_entities[0], str) else 'unknown'
                entity_id = self.sg_processor.entity_to_idx.get(entity_name, 0)
            else:
                entity_id = 0
            entities.append(entity_id)
            
            # Region
            obs_regions = obs.get('regions', [])
            if obs_regions:
                if isinstance(obs_regions[0], dict):
                    region_name = obs_regions[0].get('region', 'unknown').lower()
                else:
                    region_name = str(obs_regions[0]).lower()
                region_id = self.sg_processor.region_to_idx.get(region_name, 0)
            else:
                region_id = 0
            regions.append(region_id)
            
            # Positiveness
            pos = obs.get('positiveness', 'neg')
            positiveness_list.append(1 if pos == 'pos' else 0)
        
        return {
            'gt_bboxes': np.array(bboxes, dtype=np.float32) if bboxes else None,
            'gt_entities': np.array(entities, dtype=np.int64) if entities else None,
            'gt_regions': np.array(regions, dtype=np.int64) if regions else None,
            'gt_positiveness': np.array(positiveness_list, dtype=np.int64) if positiveness_list else None,
            'gt_relationships': None,  # TODO: Extract from located_at_relations
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load image
        if sample['image_path'] and os.path.exists(sample['image_path']):
            image = Image.open(sample['image_path']).convert('RGB')
            original_size = image.size
        else:
            # Create dummy image for testing
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            original_size = (224, 224)
        
        image_tensor = self.transform(image)
        image_width, image_height = original_size
        
        # Get dicom_id for image-specific localization
        dicom_id = sample.get('dicom_id', None)
        
        # Load scene graph
        scene_graph = {}
        if sample['scene_graph_path'] and os.path.exists(sample['scene_graph_path']):
            try:
                with open(sample['scene_graph_path']) as f:
                    scene_graph = json.load(f)
            except Exception as e:
                logger.debug(f"Error loading scene graph: {e}")
        
        # Process scene graph for model input
        sg_features = self.sg_processor.process(
            scene_graph, 
            image_width, 
            image_height,
            image_id=dicom_id
        )
        
        # Get tokenizer
        tokenizer = self._get_tokenizer()

        # === REPORT INJECTION (input side) ===
        # When use_reports is on, pull INDICATION + HISTORY from the scene_graph
        # JSON and prepend as clinical context to the question. This mirrors what
        # a radiologist receives before reading the X-ray. No leakage because
        # FINDINGS/IMPRESSION (what the report SAYS about the image) are kept
        # separate and only used as an OUTPUT target — see structured_answer_text
        # construction below.
        report_sections = {'indication_text': '', 'findings_impression': ''}
        if self.use_reports:
            report_sections = self._extract_report_sections(scene_graph)
        if report_sections['indication_text']:
            question_text_for_model = (
                f"Clinical context: {report_sections['indication_text']}\n\n"
                f"Question: {sample['question']}"
            )
        else:
            question_text_for_model = sample['question']

        # Tokenize question (with optional clinical context prepended)
        question_inputs = tokenizer(
            question_text_for_model,
            padding='max_length',
            truncation=True,
            max_length=self.max_question_length,
            return_tensors='pt'
        )
        
        # Get answer index (for classification heads)
        answer_idx = self._get_answer_idx(
            sample['question_type'],
            sample['answers']
        )
        
        # =====================================================
        # ANSWER GENERATION DATA (for decoder training)
        # =====================================================
        # Get full hierarchical answer text from MIMIC-Ext-CXR-QBA
        full_answer_text = self._get_full_answer_text(sample['answers'])
        
        # Tokenize answer for decoder training
        # Format: [CLS] answer text [SEP]
        answer_inputs = tokenizer(
            full_answer_text,
            padding='max_length',
            truncation=True,
            max_length=64,  # Max answer length
            return_tensors='pt'
        )
        answer_ids = answer_inputs['input_ids'].squeeze(0)
        
        # =====================================================
        # VISUAL GROUNDING DATA (for grounding head training)
        # =====================================================
        grounding_data = self._get_grounding_data(
            sample['answers'],
            image_width,
            image_height,
            image_id=dicom_id
        )
        grounding_bbox = torch.tensor(grounding_data['bbox'], dtype=torch.float)
        grounding_valid = torch.tensor(1.0 if grounding_data['valid'] else 0.0, dtype=torch.float)
        
        # =====================================================
        # SCENE GRAPH GENERATION DATA (for SG generator training)
        # =====================================================
        sg_targets = self._get_scene_graph_targets(
            scene_graph,
            image_width,
            image_height,
            image_id=dicom_id
        )

        # =====================================================
        # PRE-GENERATED SG (Stage 3+ only — cache hit or None)
        # =====================================================
        # When sg_cache_root is configured, attempt to load a cached graph
        # that the frozen Stage-1 generator produced offline. On a hit the
        # trainer's pretrain/finetune gate will pass this in as scene_graphs,
        # so the model skips _run_sg_generator entirely (sg_outputs=None,
        # sg_loss=0 — expected under caching, see scripts/precompute_sg_cache.py
        # docstring).
        generated_sg = self._load_generated_sg_for(
            sample['subject_id'], sample['study_id']
        )
        
        # =====================================================
        # CHEXPERT LABELS (for auxiliary supervision)
        # =====================================================
        chexpert_labels, chexpert_mask = self.chexpert_loader.get_labels(
            sample['subject_id'],
            sample['study_id']
        )
        
        # Get main answer metadata for evaluation
        main_answer = sample['answers'][0] if sample['answers'] else {}
        answer_text = main_answer.get('text', '')
        answer_regions = main_answer.get('regions', [])
        answer_entities = main_answer.get('obs_entities', [])
        answer_positiveness = main_answer.get('positiveness', '')
        
        # =====================================================
        # MIMIC-CXR-JPG IMAGE METADATA (from metadata CSV)
        # =====================================================
        # Get comprehensive image metadata for context-aware processing
        img_metadata = self._get_image_metadata(dicom_id) if dicom_id else {}
        
        # View position info (AP, PA, LATERAL) - important for model understanding
        view_position = img_metadata.get('view_position') or sample.get('view_position')
        view_code = img_metadata.get('view_code', '')  # "postero-anterior", "antero-posterior", etc.
        
        # Patient orientation (Erect, Recumbent) - affects image interpretation
        patient_orientation = img_metadata.get('patient_orientation', '')
        
        # Procedure info - "CHEST (PA AND LAT)", "CHEST (PORTABLE AP)", etc.
        procedure = img_metadata.get('procedure', '')
        
        # Original image dimensions (before resize)
        original_rows = img_metadata.get('original_rows') or image_height
        original_cols = img_metadata.get('original_cols') or image_width
        
        # Study timing (for longitudinal analysis if needed)
        study_date = img_metadata.get('study_date')
        study_time = img_metadata.get('study_time')
        
        # Encode view position for model (useful for view-aware processing)
        view_encoding = self._encode_view_position(view_position)
        
        # =====================================================
        # V2 STRUCTURED ANSWER (for Qwen LM supervision)
        # =====================================================
        # Format: <think>{cot}</think><box>{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}</box><answer>{ans}</answer>
        # cot is rule-generated from CheXpert labels + answer regions when no
        # teacher trace is available. Replace the cot string here with a
        # distilled trace (GPT-4o / Claude / Qwen-72B) once one exists.
        if grounding_data['valid']:
            _bx = grounding_data['bbox']
        else:
            _bx = [0.0, 0.0, 1.0, 1.0]  # whole-image fallback
        _box_str = f"{_bx[0]:.3f},{_bx[1]:.3f},{_bx[2]:.3f},{_bx[3]:.3f}"
        # === REPORT INJECTION (output side) ===
        # When use_reports is on and FINDINGS/IMPRESSION are present in the SG
        # JSON, use the radiologist's ACTUAL writing as the <think> CoT target.
        # The model learns to generate report-style reasoning from the image
        # alone (it doesn't SEE these strings at inference). Falls back to the
        # rule-generated CoT when the report sections are missing (~5% of studies).
        if self.use_reports and report_sections['findings_impression']:
            _cot = report_sections['findings_impression']
        else:
            _cot_bits: List[str] = []
            if answer_regions:
                _cot_bits.append(f"Region(s) of interest: {', '.join(map(str, answer_regions))}.")
            if answer_entities:
                _cot_bits.append(f"Observed: {', '.join(map(str, answer_entities))}.")
            if answer_positiveness:
                _cot_bits.append(f"Positiveness: {answer_positiveness}.")
            _cot = " ".join(_cot_bits) if _cot_bits else "Reviewing the chest radiograph."
        structured_answer_text = (
            f"<think>{_cot}</think>"
            f"<box>{_box_str}</box>"
            f"<answer>{full_answer_text}</answer>"
        )

        return {
            # === MODEL INPUTS ===
            'images': image_tensor,
            'input_ids': question_inputs['input_ids'].squeeze(0),
            'attention_mask': question_inputs['attention_mask'].squeeze(0),
            'token_type_ids': question_inputs.get('token_type_ids', torch.zeros_like(question_inputs['input_ids'])).squeeze(0),
            'scene_graphs': sg_features,

            # === V2 INPUTS (raw text + raw PIL image for Qwen processor) ===
            'pil_image': image,                 # PIL.Image.Image (pre-transform RGB)
            'question_text': question_text_for_model,  # question, with clinical context prepended when use_reports=True
            'structured_answer_text': structured_answer_text,  # <think><box><answer> format

            # === REPORT TARGET (for dedicated report_loss in loss.py) ===
            # Raw FINDINGS+IMPRESSION text without any tag wrapping. The loss
            # module can tokenize this and compute a separately-monitored CE
            # against the model's <think>-portion output. Empty string when
            # use_reports=False or the source study had no report sections.
            'report_target_text': report_sections['findings_impression'],
            
            # === ROUTING ===
            'question_types': sample['question_type'],
            
            # === CLASSIFICATION TARGETS ===
            'answer_idx': torch.tensor(answer_idx, dtype=torch.long),
            'chexpert_labels': torch.tensor(chexpert_labels, dtype=torch.float),
            'chexpert_mask': torch.tensor(chexpert_mask, dtype=torch.float),
            
            # === ANSWER GENERATION TARGETS ===
            'answer_ids': answer_ids,                # Tokenized answer for decoder
            'reference_answer': full_answer_text,    # Full text for metrics
            
            # === VISUAL GROUNDING TARGETS ===
            'grounding_bbox': grounding_bbox,        # (4,) normalized [x1, y1, x2, y2]
            'grounding_valid': grounding_valid,      # 1.0 if valid target, 0.0 otherwise
            
            # === SCENE GRAPH GENERATION TARGETS ===
            'gt_sg_bboxes': sg_targets['gt_bboxes'],        # (N, 4) or None
            'gt_sg_entities': sg_targets['gt_entities'],    # (N,) entity indices or None
            'gt_sg_regions': sg_targets['gt_regions'],      # (N,) region indices or None
            'gt_sg_positiveness': sg_targets['gt_positiveness'],  # (N,) 0/1 or None

            # === PRE-GENERATED SG (Stage 3+ cache hit, else None) ===
            'generated_sg': generated_sg,
            
            # === PATIENT/STUDY METADATA ===
            'subject_id': sample['subject_id'],
            'study_id': sample['study_id'],
            'dicom_id': dicom_id,
            'question_id': sample.get('question_id', ''),
            
            # === ANSWER METADATA (for evaluation) ===
            'answer_text': answer_text,
            'answer_regions': answer_regions,
            'answer_entities': answer_entities,
            'answer_positiveness': answer_positiveness,
            
            # === MIMIC-CXR-JPG IMAGE METADATA ===
            'image_width': image_width,
            'image_height': image_height,
            'original_rows': original_rows,           # Original size before transform
            'original_cols': original_cols,
            'view_position': view_position,           # "AP", "PA", "LATERAL", etc.
            'view_code': view_code,                   # "postero-anterior", "antero-posterior"
            'view_encoding': torch.tensor(view_encoding, dtype=torch.float),  # One-hot [4]
            'patient_orientation': patient_orientation,  # "Erect", "Recumbent"
            'procedure': procedure,                   # "CHEST (PA AND LAT)", etc.
            'study_date': study_date,                 # Anonymized but chronologically consistent
            'study_time': study_time,
        }
    
    def _encode_view_position(self, view_position: Optional[str]) -> List[float]:
        """
        Encode view position as a one-hot vector for view-aware model processing.
        
        Categories from MIMIC-CXR-JPG:
        - PA (Posterior-Anterior): Standard upright view
        - AP (Anterior-Posterior): Portable/bedside view
        - LATERAL: Side view
        - Other/Unknown
        
        Returns:
            [4] one-hot encoding: [PA, AP, LATERAL, OTHER]
        """
        encoding = [0.0, 0.0, 0.0, 0.0]
        
        if view_position is None or not isinstance(view_position, str):
            encoding[3] = 1.0  # Unknown / missing / NaN
        elif view_position.upper() in ('PA', 'PA LLD'):
            encoding[0] = 1.0  # PA
        elif view_position.upper() in ('AP', 'AP AXIAL'):
            encoding[1] = 1.0  # AP
        elif view_position.upper() in ('LATERAL', 'LL', 'LAO', 'RAO'):
            encoding[2] = 1.0  # LATERAL
        else:
            encoding[3] = 1.0  # Other
        
        return encoding


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for MIMIC-CXR VQA dataset.
    
    Handles:
    - Fixed-size tensors (images, tokens, labels) -> stacked
    - Variable-length lists (scene_graphs, gt_sg_*, metadata) -> collected as lists
    - Answer generation targets -> stacked
    - Grounding targets -> stacked
    """
    
    # === Stack fixed-size tensors ===
    images = torch.stack([item['images'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    answer_idx = torch.stack([item['answer_idx'] for item in batch])
    chexpert_labels = torch.stack([item['chexpert_labels'] for item in batch])
    chexpert_mask = torch.stack([item['chexpert_mask'] for item in batch])
    
    # === Answer Generation Targets ===
    answer_ids = torch.stack([item['answer_ids'] for item in batch])
    reference_answers = [item.get('reference_answer', '') for item in batch]
    
    # === Visual Grounding Targets ===
    grounding_bboxes = torch.stack([item['grounding_bbox'] for item in batch])
    grounding_valid = torch.stack([item['grounding_valid'].unsqueeze(0) for item in batch])
    
    # === Scene Graph Generation Targets (variable length - keep as lists) ===
    # Convert numpy arrays to torch tensors where not None
    gt_sg_bboxes = []
    gt_sg_entities = []
    gt_sg_regions = []
    
    for item in batch:
        if item['gt_sg_bboxes'] is not None:
            gt_sg_bboxes.append(torch.from_numpy(item['gt_sg_bboxes']))
        else:
            gt_sg_bboxes.append(None)
        
        if item['gt_sg_entities'] is not None:
            gt_sg_entities.append(torch.from_numpy(item['gt_sg_entities']))
        else:
            gt_sg_entities.append(None)
        
        if item['gt_sg_regions'] is not None:
            gt_sg_regions.append(torch.from_numpy(item['gt_sg_regions']))
        else:
            gt_sg_regions.append(None)
    
    # === Collect variable-length scene graphs ===
    scene_graphs = [item['scene_graphs'] for item in batch]

    # === Collect pre-generated SGs (Stage 3+ only; per-item None on miss) ===
    generated_sg = [item.get('generated_sg') for item in batch]
    
    # === Collect routing info ===
    question_types = [item['question_types'] for item in batch]
    
    # === Collect image dimensions (needed for bbox denormalization) ===
    image_widths = torch.tensor([item.get('image_width', 224) for item in batch], dtype=torch.long)
    image_heights = torch.tensor([item.get('image_height', 224) for item in batch], dtype=torch.long)
    
    # === Stack MIMIC-CXR-JPG Image Metadata ===
    view_encodings = torch.stack([item['view_encoding'] for item in batch])
    original_rows = torch.tensor([item.get('original_rows', 224) for item in batch], dtype=torch.long)
    original_cols = torch.tensor([item.get('original_cols', 224) for item in batch], dtype=torch.long)
    
    # === V2 raw inputs for Qwen processor ===
    pil_images = [item.get('pil_image') for item in batch]
    questions = [item.get('question_text', '') for item in batch]
    structured_answer_texts = [
        item.get('structured_answer_text', '') for item in batch
    ]

    result = {
        # === Model Inputs ===
        'images': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'scene_graphs': scene_graphs,

        # === V2 (Qwen) raw inputs ===
        'pil_images': pil_images,                # List[PIL.Image]
        'questions': questions,                  # List[str]
        # 'answer_texts' below holds the structured <think><box><answer> string
        # used for Qwen LM supervision. The plain reference answer remains
        # available via 'reference_answers' / 'answer_texts_raw' for metrics.
        'answer_texts': structured_answer_texts, # List[str]
        
        # === Routing ===
        'question_types': question_types,
        
        # === Classification Targets ===
        'answer_idx': answer_idx,
        'chexpert_labels': chexpert_labels,
        'chexpert_mask': chexpert_mask,
        
        # === Answer Generation Targets ===
        'answer_ids': answer_ids,               # (B, T) tokenized answers for decoder
        'reference_answers': reference_answers,  # List[str] for metrics
        
        # === Visual Grounding Targets ===
        'gt_grounding_bboxes': grounding_bboxes,  # (B, 4) normalized bboxes
        'gt_pointing_valid': grounding_valid,     # (B, 1) validity flags
        
        # === Scene Graph Generation Targets ===
        'gt_sg_bboxes': gt_sg_bboxes,      # List[Tensor(N, 4) | None]
        'gt_sg_entities': gt_sg_entities,  # List[Tensor(N,) | None]
        'gt_sg_regions': gt_sg_regions,    # List[Tensor(N,) | None]

        # === Pre-generated SGs (Stage 3+ cache; None on miss/disabled) ===
        'generated_sg': generated_sg,      # List[Dict | None]
        
        # === MIMIC-CXR-JPG Image Metadata ===
        'image_widths': image_widths,           # Current image width
        'image_heights': image_heights,         # Current image height
        'original_rows': original_rows,         # Original image rows (before resize)
        'original_cols': original_cols,         # Original image cols (before resize)
        'view_encodings': view_encodings,       # (B, 4) one-hot [PA, AP, LATERAL, OTHER]
        'view_positions': [item.get('view_position', '') for item in batch],  # List[str]
        'patient_orientations': [item.get('patient_orientation', '') for item in batch],  # List[str] "Erect"/"Recumbent"
        'procedures': [item.get('procedure', '') for item in batch],  # List[str] "CHEST (PA AND LAT)", etc.
    }
    
    # === Study/Patient Metadata (for evaluation & tracking) ===
    if 'answer_text' in batch[0]:
        # Plain reference answer text (un-structured); 'answer_texts' above
        # holds the v2 structured <think><box><answer> string.
        result['answer_texts_raw'] = [item.get('answer_text', '') for item in batch]
    if 'answer_regions' in batch[0]:
        result['answer_regions'] = [item.get('answer_regions', []) for item in batch]
    if 'answer_entities' in batch[0]:
        result['answer_entities'] = [item.get('answer_entities', []) for item in batch]
    if 'answer_positiveness' in batch[0]:
        result['answer_positiveness'] = [item.get('answer_positiveness', '') for item in batch]
    if 'question_id' in batch[0]:
        result['question_ids'] = [item.get('question_id', '') for item in batch]
    if 'subject_id' in batch[0]:
        result['subject_ids'] = [item.get('subject_id', 0) for item in batch]
    if 'study_id' in batch[0]:
        result['study_ids'] = [item.get('study_id', 0) for item in batch]
    if 'dicom_id' in batch[0]:
        result['dicom_ids'] = [item.get('dicom_id', '') for item in batch]
    if 'study_date' in batch[0]:
        result['study_dates'] = [item.get('study_date') for item in batch]
    if 'study_time' in batch[0]:
        result['study_times'] = [item.get('study_time') for item in batch]
    
    return result


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    sampler: Optional[Any] = None,
    prefetch_factor: int = 2
) -> DataLoader:
    """
    Create a DataLoader with custom collate function.
    
    For distributed training, pass a DistributedSampler and set shuffle=False.
    The sampler handles shuffling in distributed mode.
    
    Args:
        dataset: The dataset to load
        batch_size: Batch size per GPU
        shuffle: Whether to shuffle (ignored if sampler is provided)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        sampler: Optional sampler for distributed training
        prefetch_factor: Number of batches to prefetch per worker
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),  # Don't shuffle if using sampler
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        sampler=sampler,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0  # Keep workers alive between batches for speed; RSS growth is handled by malloc_trim in train loop
    )
