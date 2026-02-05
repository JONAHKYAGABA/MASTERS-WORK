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
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process scene graph into features.
        
        Args:
            scene_graph: Scene graph dict from .scene_graph.json
            image_width: Original image width in pixels
            image_height: Original image height in pixels
            image_id: Specific image ID to get localization for (optional)
        
        Returns:
            dict with:
                - bboxes: (N, 4) normalized bbox coordinates
                - region_ids: (N,) region indices
                - entity_ids: (N,) entity indices
                - positiveness: (N,) 1 for positive, 0 for negative findings
                - num_objects: int
        """
        observations = scene_graph.get('observations', {})
        
        if not observations:
            # Return dummy observation
            return {
                'bboxes': np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
                'region_ids': np.array([0], dtype=np.int64),
                'entity_ids': np.array([0], dtype=np.int64),
                'positiveness': np.array([0], dtype=np.int64),
                'num_objects': 1
            }
        
        bboxes = []
        region_ids = []
        entity_ids = []
        positiveness_list = []
        
        for obs_id, obs in observations.items():
            # Extract bbox from localization
            # Format: localization -> [image_id] -> bboxes -> [[x1, y1, x2, y2], ...]
            bbox = [0, 0, image_width, image_height]  # Default to full image
            
            if 'localization' in obs and obs['localization']:
                loc = obs['localization']
                if isinstance(loc, dict):
                    # If specific image_id provided, use that; otherwise take first
                    if image_id and image_id in loc:
                        img_loc = loc[image_id]
                    else:
                        # Take first available image localization
                        img_loc = next(iter(loc.values()), {})
                    
                    if isinstance(img_loc, dict) and 'bboxes' in img_loc:
                        if img_loc['bboxes'] and len(img_loc['bboxes']) > 0:
                            bbox = img_loc['bboxes'][0]  # Take first bbox
            
            # Normalize bbox to [0, 1]
            x1 = max(0, min(bbox[0] / image_width, 1.0))
            y1 = max(0, min(bbox[1] / image_height, 1.0))
            x2 = max(0, min(bbox[2] / image_width, 1.0))
            y2 = max(0, min(bbox[3] / image_height, 1.0))
            bboxes.append([x1, y1, x2, y2])
            
            # Get region from "regions" field
            # Format: regions -> [{"region": "lungs", "distances": []}, ...]
            regions = obs.get('regions', [])
            if regions:
                if isinstance(regions[0], dict):
                    region_name = regions[0].get('region', 'unknown')
                else:
                    region_name = str(regions[0])
                region_id = self.region_to_idx.get(region_name.lower(), 0)
            else:
                region_id = 0
            region_ids.append(region_id)
            
            # Get entity from "obs_entities" field
            # Format: obs_entities -> ["consolidation", ...]
            entities = obs.get('obs_entities', [])
            if entities:
                entity_name = entities[0] if isinstance(entities[0], str) else 'unknown'
                entity_id = self.entity_to_idx.get(entity_name.lower(), 0)
            else:
                entity_id = 0
            entity_ids.append(entity_id)
            
            # Get positiveness (pos/neg finding)
            # Format: positiveness -> "pos" or "neg"
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
        chexpert_labels_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        transform: Optional[Any] = None,
        use_exports: bool = False,  # Use pre-filtered exports folder
        cache_dir: Optional[str] = None,  # Cache directory for samples
        use_cache: bool = True,  # Whether to use caching
        force_rebuild_cache: bool = False,  # Force rebuild even if cache exists
        prebuilt_cache_path: Optional[str] = None  # Optional: load samples from external prebuilt cache (.pkl)
    ):
        self.mimic_cxr_path = Path(mimic_cxr_path)
        self.mimic_qa_path = Path(mimic_qa_path)
        self.split = split
        self.max_question_length = max_question_length
        self.quality_grade = quality_grade
        self.view_filter = view_filter
        self.question_types = question_types
        self.max_samples = max_samples
        self.use_exports = use_exports
        self.use_cache = use_cache
        self.force_rebuild_cache = force_rebuild_cache
        self.prebuilt_cache_path = prebuilt_cache_path
        
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
        
        # Initialize CheXpert loader
        if chexpert_labels_path:
            self.chexpert_loader = CheXpertLabelLoader(chexpert_labels_path)
        else:
            # Default to MIMIC-CXR-JPG chexpert labels
            default_chexpert = self.mimic_cxr_path / 'mimic-cxr-2.0.0-chexpert.csv.gz'
            self.chexpert_loader = CheXpertLabelLoader(str(default_chexpert) if default_chexpert.exists() else None)
        
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
    def _get_tokenizer(self):
        """Lazily initialize tokenizer (fork-safe for DataLoader workers)."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self.tokenizer
    
    def _load_metadata(self) -> Optional[pd.DataFrame]:
        """Load MIMIC-CXR metadata for view filtering."""
        metadata_file = self.mimic_cxr_path / 'mimic-cxr-2.0.0-metadata.csv.gz'
        
        if not metadata_file.exists():
            metadata_file = self.mimic_cxr_path / 'mimic-cxr-2.0.0-metadata.csv'
        
        if metadata_file.exists():
            try:
                if str(metadata_file).endswith('.gz'):
                    df = pd.read_csv(metadata_file, compression='gzip')
                else:
                    df = pd.read_csv(metadata_file)
                
                # Create indexed lookup dict for O(1) access (MUCH faster than DataFrame scans)
                self._dicom_to_view = {}
                for _, row in df.iterrows():
                    dicom_id = row.get('dicom_id')
                    if dicom_id:
                        self._dicom_to_view[str(dicom_id)] = row.get('ViewPosition')
                
                logger.info(f"Loaded metadata: {len(df)} images (indexed for fast lookup)")
                return df
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        
        self._dicom_to_view = {}
        return None
    
    def _get_view_position(self, dicom_id: str) -> Optional[str]:
        """Get ViewPosition for a DICOM ID from metadata (O(1) lookup)."""
        return self._dicom_to_view.get(str(dicom_id))
    
    def _get_cache_key(self) -> str:
        """Generate a unique cache key based on dataset configuration."""
        config_str = (
            f"cxr:{self.mimic_cxr_path}|"
            f"qa:{self.mimic_qa_path}|"
            f"split:{self.split}|"
            f"quality:{self.quality_grade}|"
            f"view:{self.view_filter}|"
            f"qtypes:{sorted(self.question_types) if self.question_types else 'all'}|"
            f"max:{self.max_samples or 'none'}|"
            f"exports:{self.use_exports}"
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
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
        
        for p_group in p_groups:
            for patient_dir in p_group.iterdir():
                if not patient_dir.is_dir() or not patient_dir.name.startswith('p'):
                    continue
                
                for qa_file in patient_dir.glob('s*.qa.json'):
                    files_scanned += 1
                    if files_scanned % 5000 == 0:
                        logger.info(f"  [{files_scanned}] samples={len(samples)}, skip_split={skipped_split}, skip_img={skipped_image}, skip_qual={skipped_quality}")
                    try:
                        # Parse IDs from path
                        # patient_dir.name = "p10000032" -> subject_id = 10000032
                        # qa_file.stem = "s50414267.qa" -> study_id = 50414267
                        subject_id = int(patient_dir.name[1:])  # Remove 'p' prefix
                        study_id_str = qa_file.stem.split('.')[0]  # "s50414267"
                        study_id = int(study_id_str[1:])  # Remove 's' prefix
                        
                        # Check if in valid split
                        if valid_studies and (subject_id, study_id) not in valid_studies:
                            skipped_split += 1
                            continue
                        
                        # Load QA data
                        with open(qa_file) as f:
                            qa_data = json.load(f)
                        
                        # Find corresponding image
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
                            # Quality filter (skip if quality_grade is empty/None/"all")
                            if self.quality_grade and self.quality_grade.lower() not in ('', 'all', 'none'):
                                q_quality = q.get('question_quality', q.get('quality', {}))
                                if isinstance(q_quality, dict):
                                    quality_rating = q_quality.get('overall', q_quality.get('grade', 'B'))
                                elif isinstance(q_quality, str):
                                    quality_rating = q_quality
                                else:
                                    quality_rating = 'B'  # Default to B if no quality info
                                
                                if not self._meets_quality_grade(quality_rating, self.quality_grade):
                                    skipped_quality += 1
                                    continue
                            
                            # Question type filter
                            q_type = q.get('question_type', 'unknown')
                            if self.question_types and q_type not in self.question_types:
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
                                'obs_ids': q.get('obs_ids', []),  # Scene graph observation IDs
                            })
                            
                            # Check max samples
                            if self.max_samples and len(samples) >= self.max_samples:
                                return samples
                                
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
        
        # Load scene graph
        scene_graph = {}
        if sample['scene_graph_path'] and os.path.exists(sample['scene_graph_path']):
            try:
                with open(sample['scene_graph_path']) as f:
                    scene_graph = json.load(f)
            except Exception as e:
                logger.debug(f"Error loading scene graph: {e}")
        
        # Process scene graph
        sg_features = self.sg_processor.process(
            scene_graph, 
            original_size[0], 
            original_size[1]
        )
        
        # Tokenize question
        tokenizer = self._get_tokenizer()
        question_inputs = tokenizer(
            sample['question'],
            padding='max_length',
            truncation=True,
            max_length=self.max_question_length,
            return_tensors='pt'
        )
        
        # Get answer index
        answer_idx = self._get_answer_idx(
            sample['question_type'],
            sample['answers']
        )
        
        # Get CheXpert labels
        chexpert_labels, chexpert_mask = self.chexpert_loader.get_labels(
            sample['subject_id'],
            sample['study_id']
        )
        
        # Get additional answer metadata for evaluation
        main_answer = sample['answers'][0] if sample['answers'] else {}
        answer_text = main_answer.get('text', '')
        answer_regions = main_answer.get('regions', [])
        answer_entities = main_answer.get('obs_entities', [])
        answer_positiveness = main_answer.get('positiveness', '')
        
        return {
            # === MODEL INPUTS ===
            'images': image_tensor,
            'input_ids': question_inputs['input_ids'].squeeze(0),
            'attention_mask': question_inputs['attention_mask'].squeeze(0),
            'token_type_ids': question_inputs.get('token_type_ids', torch.zeros_like(question_inputs['input_ids'])).squeeze(0),
            'scene_graphs': sg_features,
            
            # === ROUTING ===
            'question_types': sample['question_type'],
            
            # === TARGETS ===
            'answer_idx': torch.tensor(answer_idx, dtype=torch.long),
            'chexpert_labels': torch.tensor(chexpert_labels, dtype=torch.float),
            'chexpert_mask': torch.tensor(chexpert_mask, dtype=torch.float),
            
            # === METADATA (for evaluation/debugging) ===
            'subject_id': sample['subject_id'],
            'study_id': sample['study_id'],
            'question_id': sample.get('question_id', ''),
            
            # === ANSWER METADATA (for enhanced evaluation) ===
            'answer_text': answer_text,              # Raw answer text
            'answer_regions': answer_regions,        # Ground truth regions
            'answer_entities': answer_entities,      # Ground truth entities
            'answer_positiveness': answer_positiveness,  # pos/neg/neutral
            
            # === IMAGE METADATA ===
            'image_width': original_size[0],
            'image_height': original_size[1],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for variable-length scene graphs.
    
    Handles:
    - Fixed-size tensors (images, tokens, labels) -> stacked
    - Variable-length lists (scene_graphs, metadata) -> collected as lists
    """
    
    # === Stack fixed-size tensors ===
    images = torch.stack([item['images'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    answer_idx = torch.stack([item['answer_idx'] for item in batch])
    chexpert_labels = torch.stack([item['chexpert_labels'] for item in batch])
    chexpert_mask = torch.stack([item['chexpert_mask'] for item in batch])
    
    # === Collect variable-length scene graphs ===
    scene_graphs = [item['scene_graphs'] for item in batch]
    
    # === Collect routing info ===
    question_types = [item['question_types'] for item in batch]
    
    # === Collect image dimensions (needed for bbox denormalization) ===
    image_widths = torch.tensor([item.get('image_width', 224) for item in batch], dtype=torch.long)
    image_heights = torch.tensor([item.get('image_height', 224) for item in batch], dtype=torch.long)
    
    result = {
        # Model inputs
        'images': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'scene_graphs': scene_graphs,
        
        # Routing
        'question_types': question_types,
        
        # Targets
        'answer_idx': answer_idx,
        'chexpert_labels': chexpert_labels,
        'chexpert_mask': chexpert_mask,
        
        # Image dimensions
        'image_widths': image_widths,
        'image_heights': image_heights,
    }
    
    # === Optionally collect evaluation metadata (if present) ===
    if 'answer_text' in batch[0]:
        result['answer_texts'] = [item.get('answer_text', '') for item in batch]
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
        persistent_workers=num_workers > 0
    )
