#!/usr/bin/env python3
"""
Pre-build dataset cache using MapReduce for parallel processing.

MapReduce approach:
- MAP: Each worker processes a chunk of QA files independently
- REDUCE: Main process concatenates all sample lists

Includes validation to ensure cache matches model input requirements.

Run (full dataset):
    python scripts/prebuild_cache.py --config configs/pretrain_config.yaml
    python scripts/prebuild_cache.py --config configs/pretrain_config.yaml --num_workers 24
    python scripts/prebuild_cache.py --config configs/pretrain_config.yaml --validate_only

Run (SUBSET for pipeline testing - recommended for initial development):
    # Use 5% of data for quick pipeline verification
    python scripts/prebuild_cache.py --config configs/pretrain_config.yaml --sample_percent 5
    
    # Use exact number of samples (e.g., 10,000 for testing)
    python scripts/prebuild_cache.py --config configs/pretrain_config.yaml --max_samples 10000
    
    # Very small subset for debugging (1%)
    python scripts/prebuild_cache.py --config configs/pretrain_config.yaml --sample_percent 1 --split train
    python scripts/prebuild_cache.py --config configs/pretrain_config.yaml --sample_percent 1 --split val

Resume or checkpoint long runs:
    # Save partial cache every 50 chunks
    python scripts/prebuild_cache.py --config configs/pretrain_config.yaml --checkpoint_every 50
    # Resume from the last checkpoint if it exists
    python scripts/prebuild_cache.py --config configs/pretrain_config.yaml --resume
"""

import os
import sys
import argparse
import json
import pickle
import hashlib
from pathlib import Path
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from functools import partial
import time
import random
import tempfile


# =============================================================================
# CHECKPOINTING (optional)
# =============================================================================

def _load_checkpoint(checkpoint_path: Path) -> dict:
    """Load checkpoint state from disk."""
    if not checkpoint_path.exists():
        return {}
    try:
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return {}


def _save_checkpoint(checkpoint_path: Path, state: dict) -> None:
    """Persist checkpoint state to disk."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.pkl', prefix=checkpoint_path.name + '.tmp.', dir=str(checkpoint_path.parent))
    try:
        with os.fdopen(tmp_fd, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp_path, str(checkpoint_path))
    except Exception:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# EXPECTED DATA STRUCTURES (for validation)
# =============================================================================

# Required keys in each cached sample (matches MIMICCXRVQADataset expectations)
REQUIRED_SAMPLE_KEYS = {
    'subject_id',      # int: Patient ID
    'study_id',        # int: Study ID  
    'dicom_id',        # str: DICOM image ID
    'image_path',      # str: Path to .jpg file
    'question_type',   # str: e.g., "D02_has_finding"
    'question',        # str: Question text
    'answers',         # list: Answer dicts
}

OPTIONAL_SAMPLE_KEYS = {
    'scene_graph_path',    # str or None: Path to scene graph JSON
    'question_id',         # str: Unique question ID
    'question_strategy',   # str: Question strategy type
    'obs_ids',             # list: Observation IDs linked to question
    'view_position',       # str: Image view (PA, AP, LATERAL, etc.)
    'num_study_images',    # int: Number of images in the study
}

# Required keys in scene graph JSON (for SceneGraphEncoder)
REQUIRED_SG_KEYS = {
    'observations',  # dict: Observation ID -> observation data
}

# Required keys in each observation (for scene graph processing)
REQUIRED_OBS_KEYS = {
    'regions',        # list of region dicts with 'region' key
    'obs_entities',   # list of entity strings
    'positiveness',   # str: 'pos', 'neg', or 'unknown'
}

# Required keys in answer dict
REQUIRED_ANSWER_KEYS = {
    'text',  # str: Answer text
}

# View position preferences (frontal views preferred for VQA)
# MIMIC-CXR has ViewPosition in metadata - we use filename heuristics as backup
FRONTAL_VIEWS = {'PA', 'AP', 'AP_AXIAL', 'PA_LLD'}
LATERAL_VIEWS = {'LATERAL', 'LL', 'LAO', 'RAO'}

# Question types that the model supports (from loss.py mapping)
SUPPORTED_QUESTION_TYPES = {
    # Binary questions (D00-D01, D03, D05, D08-D09, D11-D12, D14-D23, D31-D32)
    'D00', 'D01', 'D03', 'D05', 'D08', 'D09', 'D11', 'D12', 
    'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 
    'D22', 'D23', 'D31', 'D32',
    # Category questions (D02, D04, D06, D10, D13, D24, D25)
    'D02', 'D04', 'D06', 'D10', 'D13', 'D24', 'D25',
    # Region questions (D07, D27, D28, D30)
    'D07', 'D27', 'D28', 'D30',
    # Severity questions (D26, D29)
    'D26', 'D29',
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_sample(sample: dict, check_files: bool = False) -> tuple:
    """
    Validate a single cached sample matches model input requirements.
    
    Args:
        sample: Sample dict from cache
        check_files: Whether to verify file paths exist on disk
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Check required keys
    for key in REQUIRED_SAMPLE_KEYS:
        if key not in sample:
            issues.append(f"Missing required key: {key}")
        elif sample[key] is None and key not in {'scene_graph_path'}:
            issues.append(f"Required key is None: {key}")
    
    if issues:
        return False, issues
    
    # Type checks
    if not isinstance(sample['subject_id'], int):
        issues.append(f"subject_id should be int, got {type(sample['subject_id'])}")
    
    if not isinstance(sample['study_id'], int):
        issues.append(f"study_id should be int, got {type(sample['study_id'])}")
    
    if not isinstance(sample['dicom_id'], str):
        issues.append(f"dicom_id should be str, got {type(sample['dicom_id'])}")
    
    if not isinstance(sample['question'], str) or len(sample['question']) < 5:
        issues.append(f"question should be non-empty str")
    
    if not isinstance(sample['answers'], list):
        issues.append(f"answers should be list, got {type(sample['answers'])}")
    elif len(sample['answers']) == 0:
        issues.append("answers list is empty")
    else:
        # Check first answer has required keys
        first_answer = sample['answers'][0]
        if not isinstance(first_answer, dict):
            issues.append(f"answer should be dict, got {type(first_answer)}")
        elif 'text' not in first_answer:
            issues.append("answer missing 'text' key")
    
    # Check question_type format
    q_type = sample.get('question_type', '')
    if not q_type:
        issues.append("question_type is empty")
    else:
        # Extract DXX prefix
        prefix = q_type.split('_')[0] if '_' in q_type else q_type[:3]
        if prefix not in SUPPORTED_QUESTION_TYPES and not q_type.startswith('D'):
            issues.append(f"Unknown question_type prefix: {prefix}")
    
    # File existence checks (optional, slow)
    if check_files:
        img_path = sample.get('image_path')
        if img_path and not Path(img_path).exists():
            issues.append(f"Image file not found: {img_path}")
        
        sg_path = sample.get('scene_graph_path')
        if sg_path and not Path(sg_path).exists():
            issues.append(f"Scene graph file not found: {sg_path}")
    
    return len(issues) == 0, issues


def validate_scene_graph(sg_path: str) -> tuple:
    """
    Validate a scene graph JSON file matches model requirements.
    
    The SceneGraphProcessor expects:
    - observations: dict of observation_id -> observation
    - Each observation should have: regions, obs_entities, positiveness, localization
    
    Returns:
        (is_valid, dict_of_stats_and_issues)
    """
    result = {
        'valid': False,
        'num_observations': 0,
        'num_with_bboxes': 0,
        'num_with_regions': 0,
        'num_with_entities': 0,
        'issues': [],
    }
    
    try:
        with open(sg_path, 'r') as f:
            sg = json.load(f)
    except Exception as e:
        result['issues'].append(f"Failed to load JSON: {e}")
        return False, result
    
    # Check required top-level keys
    if 'observations' not in sg:
        result['issues'].append("Missing 'observations' key")
        return False, result
    
    observations = sg['observations']
    if not isinstance(observations, dict):
        result['issues'].append(f"'observations' should be dict, got {type(observations)}")
        return False, result
    
    result['num_observations'] = len(observations)
    
    if len(observations) == 0:
        result['issues'].append("No observations in scene graph")
        # Empty is technically valid (just means no findings)
        result['valid'] = True
        return True, result
    
    # Check observation structure
    for obs_id, obs in list(observations.items())[:10]:  # Sample first 10
        if not isinstance(obs, dict):
            result['issues'].append(f"Observation {obs_id} is not a dict")
            continue
        
        # Check regions (required for SceneGraphProcessor)
        regions = obs.get('regions', [])
        if regions:
            result['num_with_regions'] += 1
            # Check region format
            if isinstance(regions, list) and len(regions) > 0:
                first_reg = regions[0]
                if isinstance(first_reg, dict) and 'region' not in first_reg:
                    result['issues'].append(f"Obs {obs_id}: region dict missing 'region' key")
        
        # Check entities
        entities = obs.get('obs_entities', [])
        if entities:
            result['num_with_entities'] += 1
        
        # Check localization (bboxes)
        loc = obs.get('localization', {})
        if loc and isinstance(loc, dict):
            for img_id, loc_data in loc.items():
                bboxes = loc_data.get('bboxes', [])
                if bboxes:
                    result['num_with_bboxes'] += 1
                    # Check bbox format
                    if isinstance(bboxes, list) and len(bboxes) > 0:
                        first_bbox = bboxes[0]
                        if not isinstance(first_bbox, list) or len(first_bbox) != 4:
                            result['issues'].append(f"Obs {obs_id}: bbox should be [x1,y1,x2,y2]")
                    break
        
        # Check positiveness
        pos = obs.get('positiveness', '')
        if pos not in {'pos', 'neg', 'unknown', ''}:
            result['issues'].append(f"Obs {obs_id}: invalid positiveness '{pos}'")
    
    result['valid'] = len(result['issues']) == 0
    return result['valid'], result


def validate_cache(cache_path: str, num_samples: int = 100, check_files: bool = True) -> dict:
    """
    Validate an entire cache file.
    
    Args:
        cache_path: Path to .pkl cache file
        num_samples: Number of random samples to validate
        check_files: Whether to check file existence
    
    Returns:
        Validation report dict
    """
    report = {
        'cache_path': cache_path,
        'total_samples': 0,
        'samples_checked': 0,
        'valid_samples': 0,
        'invalid_samples': 0,
        'scene_graphs_checked': 0,
        'valid_scene_graphs': 0,
        'question_type_distribution': {},
        'view_position_distribution': {},  # NEW: Track view positions
        'multi_image_studies': 0,          # NEW: Studies with >1 image
        'single_image_studies': 0,         # NEW: Studies with 1 image
        'sample_issues': [],
        'sg_issues': [],
        'overall_valid': False,
    }
    
    # Load cache
    try:
        with open(cache_path, 'rb') as f:
            samples = pickle.load(f)
    except Exception as e:
        report['sample_issues'].append(f"Failed to load cache: {e}")
        return report
    
    report['total_samples'] = len(samples)
    
    if len(samples) == 0:
        report['sample_issues'].append("Cache is empty")
        return report
    
    # Sample for validation
    check_indices = random.sample(range(len(samples)), min(num_samples, len(samples)))
    
    # Track scene graphs to check (unique)
    sg_paths_to_check = set()
    
    for idx in check_indices:
        sample = samples[idx]
        report['samples_checked'] += 1
        
        # Validate sample
        is_valid, issues = validate_sample(sample, check_files=check_files)
        
        if is_valid:
            report['valid_samples'] += 1
        else:
            report['invalid_samples'] += 1
            if len(report['sample_issues']) < 10:  # Keep first 10 issues
                report['sample_issues'].append({
                    'idx': idx,
                    'issues': issues
                })
        
        # Track question types
        q_type = sample.get('question_type', 'unknown')
        prefix = q_type.split('_')[0] if '_' in q_type else q_type[:3]
        report['question_type_distribution'][prefix] = report['question_type_distribution'].get(prefix, 0) + 1
        
        # Track view positions (NEW)
        view = sample.get('view_position', 'unknown')
        if view:
            report['view_position_distribution'][view] = report['view_position_distribution'].get(view, 0) + 1
        
        # Track multi-image studies (NEW)
        num_imgs = sample.get('num_study_images', 1)
        if num_imgs > 1:
            report['multi_image_studies'] += 1
        else:
            report['single_image_studies'] += 1
        
        # Collect scene graph path
        sg_path = sample.get('scene_graph_path')
        if sg_path:
            sg_paths_to_check.add(sg_path)
    
    # Validate scene graphs (sample up to 20)
    sg_sample = list(sg_paths_to_check)[:20]
    for sg_path in sg_sample:
        report['scene_graphs_checked'] += 1
        is_valid, sg_result = validate_scene_graph(sg_path)
        
        if is_valid:
            report['valid_scene_graphs'] += 1
        else:
            if len(report['sg_issues']) < 5:
                report['sg_issues'].append({
                    'path': sg_path,
                    'issues': sg_result['issues']
                })
    
    # Overall validity
    valid_rate = report['valid_samples'] / max(report['samples_checked'], 1)
    sg_valid_rate = report['valid_scene_graphs'] / max(report['scene_graphs_checked'], 1)
    
    report['overall_valid'] = (valid_rate >= 0.95) and (sg_valid_rate >= 0.90)
    report['sample_valid_rate'] = f"{valid_rate * 100:.1f}%"
    report['sg_valid_rate'] = f"{sg_valid_rate * 100:.1f}%"
    
    return report


def print_validation_report(report: dict):
    """Print a formatted validation report."""
    print("\n" + "=" * 70)
    print("  CACHE VALIDATION REPORT")
    print("=" * 70)
    
    print(f"\n  Cache: {report['cache_path']}")
    print(f"  Total samples: {report['total_samples']:,}")
    
    print(f"\n  SAMPLE VALIDATION ({report['samples_checked']} checked):")
    print(f"    ✓ Valid: {report['valid_samples']}")
    print(f"    ✗ Invalid: {report['invalid_samples']}")
    print(f"    Rate: {report.get('sample_valid_rate', 'N/A')}")
    
    if report['sample_issues']:
        print(f"\n  Sample Issues (showing first {len(report['sample_issues'])}):")
        for issue in report['sample_issues'][:5]:
            print(f"    - Sample {issue['idx']}: {issue['issues'][:2]}")
    
    print(f"\n  SCENE GRAPH VALIDATION ({report['scene_graphs_checked']} checked):")
    print(f"    ✓ Valid: {report['valid_scene_graphs']}")
    print(f"    Rate: {report.get('sg_valid_rate', 'N/A')}")
    
    if report['sg_issues']:
        print(f"\n  Scene Graph Issues:")
        for issue in report['sg_issues'][:3]:
            print(f"    - {Path(issue['path']).name}: {issue['issues'][:2]}")
    
    print(f"\n  QUESTION TYPE DISTRIBUTION:")
    for q_type, count in sorted(report['question_type_distribution'].items()):
        status = "✓" if q_type in SUPPORTED_QUESTION_TYPES else "?"
        print(f"    {status} {q_type}: {count}")
    
    # NEW: View position distribution
    if report.get('view_position_distribution'):
        print(f"\n  VIEW POSITION DISTRIBUTION:")
        for view, count in sorted(report['view_position_distribution'].items(), key=lambda x: -x[1]):
            is_frontal = view in {'PA', 'AP', 'AP AXIAL', 'PA LLD', 'AP_AXIAL', 'PA_LLD'}
            status = "✓" if is_frontal else "⚠"
            print(f"    {status} {view}: {count}")
    
    # NEW: Multi-image study stats
    multi = report.get('multi_image_studies', 0)
    single = report.get('single_image_studies', 0)
    total = multi + single
    if total > 0:
        print(f"\n  MULTI-IMAGE STUDY HANDLING:")
        print(f"    Single image studies: {single} ({single/total*100:.1f}%)")
        print(f"    Multi-image studies:  {multi} ({multi/total*100:.1f}%)")
        print(f"    → Selected PRIMARY FRONTAL image (PA > AP > other)")
    
    print("\n" + "-" * 70)
    if report['overall_valid']:
        print("  ✓ CACHE VALIDATION PASSED - Ready for training!")
    else:
        print("  ✗ CACHE VALIDATION FAILED - Check issues above")
    print("-" * 70)


# =============================================================================
# IMAGE SELECTION (for multi-image studies)
# =============================================================================

# Global metadata cache (loaded once, shared via initializer)
_METADATA_CACHE = None


def _load_metadata_cache(mimic_cxr_path: str) -> dict:
    """
    Load MIMIC-CXR metadata to get ViewPosition for each image.
    
    Returns:
        dict: dicom_id -> {'view': 'PA', 'subject_id': X, 'study_id': Y}
    """
    import pandas as pd
    
    metadata_file = Path(mimic_cxr_path) / 'mimic-cxr-2.0.0-metadata.csv.gz'
    if not metadata_file.exists():
        metadata_file = Path(mimic_cxr_path) / 'mimic-cxr-2.0.0-metadata.csv'
    
    if not metadata_file.exists():
        return {}
    
    try:
        compression = 'gzip' if str(metadata_file).endswith('.gz') else None
        df = pd.read_csv(metadata_file, compression=compression,
                        usecols=['dicom_id', 'subject_id', 'study_id', 'ViewPosition'])
        
        # Build lookup dict
        cache = {}
        for _, row in df.iterrows():
            dicom_id = str(row['dicom_id'])
            cache[dicom_id] = {
                'view': str(row.get('ViewPosition', '')).upper(),
                'subject_id': int(row['subject_id']),
                'study_id': int(row['study_id']),
            }
        return cache
    except Exception:
        return {}


def _select_best_image(jpg_files: list, metadata_cache: dict) -> tuple:
    """
    Select the best image from multiple images in a study.
    
    Priority:
    1. PA (Posterior-Anterior) - best quality frontal
    2. AP (Anterior-Posterior) - common frontal view
    3. AP AXIAL, PA LLD - other frontal variants
    4. Any other image (fallback)
    
    Args:
        jpg_files: List of Path objects to .jpg files
        metadata_cache: dict of dicom_id -> {'view': 'PA', ...}
    
    Returns:
        (selected_path, dicom_id, view_position, num_images_in_study)
    """
    if not jpg_files:
        return None, None, None, 0
    
    num_images = len(jpg_files)
    
    # Categorize images by view
    frontal_pa = []
    frontal_ap = []
    frontal_other = []
    lateral = []
    unknown = []
    
    for img_path in jpg_files:
        dicom_id = img_path.stem
        view = metadata_cache.get(dicom_id, {}).get('view', '')
        
        if view == 'PA':
            frontal_pa.append((img_path, dicom_id, view))
        elif view == 'AP':
            frontal_ap.append((img_path, dicom_id, view))
        elif view in {'AP AXIAL', 'PA LLD', 'AP_AXIAL', 'PA_LLD'}:
            frontal_other.append((img_path, dicom_id, view))
        elif view in {'LATERAL', 'LL', 'LAO', 'RAO'}:
            lateral.append((img_path, dicom_id, view))
        else:
            unknown.append((img_path, dicom_id, view))
    
    # Select best image (priority order)
    if frontal_pa:
        selected = frontal_pa[0]
    elif frontal_ap:
        selected = frontal_ap[0]
    elif frontal_other:
        selected = frontal_other[0]
    elif unknown:
        # Unknown might be frontal - use first
        selected = unknown[0]
    elif lateral:
        # Lateral is last resort
        selected = lateral[0]
    else:
        # Fallback
        selected = (jpg_files[0], jpg_files[0].stem, '')
    
    return str(selected[0]), selected[1], selected[2], num_images


# =============================================================================
# MAPREDUCE FUNCTIONS (Top-level for pickling)
# =============================================================================

# Global metadata cache for worker processes (set via Pool initializer)
METADATA_CACHE = None

def _init_worker(metadata_cache_path):
    """Initializer for Pool workers: load metadata cache from a file path."""
    global METADATA_CACHE
    METADATA_CACHE = {}
    try:
        if metadata_cache_path:
            with open(metadata_cache_path, 'rb') as f:
                METADATA_CACHE = pickle.load(f)
    except Exception:
        METADATA_CACHE = {}


def _map_qa_file(args_tuple):
    """
    MAP function: Process a single QA file and return samples.
    
    Args:
        args_tuple: (qa_file_path, valid_studies_set, mimic_cxr_path, sg_dir_path, metadata_cache)
    
    Returns:
        List of sample dicts, or empty list on failure
    """
    qa_file_str, valid_studies_frozen, mimic_cxr_str, sg_dir_str, metadata_cache = args_tuple
    # If metadata_cache not provided in args (to avoid pickling large dicts),
    # fall back to the global METADATA_CACHE set by the Pool initializer.
    if metadata_cache is None:
        metadata_cache = METADATA_CACHE or {}
    
    try:
        qa_file = Path(qa_file_str)
        mimic_cxr_path = Path(mimic_cxr_str)
        sg_dir = Path(sg_dir_str) if sg_dir_str else None
        
        # Parse IDs from path: .../p10/p10000032/s50414267.qa.json
        subject_id = int(qa_file.parent.name[1:])  # p10000032 -> 10000032
        study_id = int(qa_file.stem.split('.')[0][1:])  # s50414267.qa -> 50414267
        
        # Check if in valid split
        if (subject_id, study_id) not in valid_studies_frozen:
            return []
        
        # Find study directory
        p_prefix = f"p{str(subject_id)[:2]}"
        study_dir = mimic_cxr_path / 'files' / p_prefix / f"p{subject_id}" / f"s{study_id}"
        
        if not study_dir.exists():
            return []
        
        jpg_files = list(study_dir.glob('*.jpg'))
        if not jpg_files:
            return []
        
        # Select best image (frontal preferred)
        img_path, dicom_id, view_position, num_study_images = _select_best_image(
            jpg_files, metadata_cache or {}
        )
        
        # Find scene graph
        sg_path = None
        if sg_dir:
            sg_file = sg_dir / p_prefix / f"p{subject_id}" / f"s{study_id}.scene_graph.json"
            if sg_file.exists():
                sg_path = str(sg_file)
        
        # Load QA data
        with open(qa_file, 'r') as f:
            qa_data = json.load(f)
        
        # Build samples for each question
        samples = []
        for q in qa_data.get('questions', []):
            samples.append({
                # Core identifiers
                'subject_id': subject_id,
                'study_id': study_id,
                'dicom_id': dicom_id,
                
                # File paths
                'image_path': img_path,
                'scene_graph_path': sg_path,
                
                # Question data
                'question_id': q.get('question_id', ''),
                'question_type': q.get('question_type', 'unknown'),
                'question_strategy': q.get('question_strategy', ''),
                'question': q.get('question', ''),
                'answers': q.get('answers', []),
                'obs_ids': q.get('obs_ids', []),
                
                # Multi-image study metadata
                'view_position': view_position,           # e.g., 'PA', 'AP', 'LATERAL'
                'num_study_images': num_study_images,     # How many images in this study
            })
        
        return samples
    
    except Exception:
        return []


def _map_qa_chunk(chunk_args):
    """
    MAP function for a CHUNK of QA files.
    Processes multiple files and returns combined samples.
    This reduces IPC overhead significantly.
    
    Args:
        chunk_args: (chunk_idx, list_of_qa_file_paths, valid_studies_frozen, mimic_cxr_path, sg_dir_path, metadata_cache)
    
    Returns:
        dict with 'samples' list and stats
    """
    chunk_idx, file_paths, valid_studies_frozen, mimic_cxr_str, sg_dir_str, metadata_cache = chunk_args
    
    all_samples = []
    files_processed = 0
    files_skipped_split = 0
    files_skipped_img = 0
    
    for qa_file_str in file_paths:
        result = _map_qa_file((qa_file_str, valid_studies_frozen, mimic_cxr_str, sg_dir_str, metadata_cache))
        
        if result:
            all_samples.extend(result)
            files_processed += 1
        else:
            # Determine skip reason (simplified - count as split skip if file was readable)
            try:
                qa_file = Path(qa_file_str)
                subject_id = int(qa_file.parent.name[1:])
                study_id = int(qa_file.stem.split('.')[0][1:])
                if (subject_id, study_id) not in valid_studies_frozen:
                    files_skipped_split += 1
                else:
                    files_skipped_img += 1
            except:
                files_skipped_img += 1
    
    # To avoid sending very large objects through the multiprocessing
    # pipe (which can cause BrokenPipeError), write the chunk results to
    # a temporary file and return only the file path and small stats.
    try:
        fd, tmp_path = tempfile.mkstemp(suffix='.pkl', prefix='prebuild_chunk_')
        with os.fdopen(fd, 'wb') as tf:
            pickle.dump(
                {
                    'samples': all_samples,
                    'files_processed': files_processed,
                    'files_skipped_split': files_skipped_split,
                    'files_skipped_img': files_skipped_img,
                },
                tf,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        # Clear large list to free memory in worker (best-effort)
        all_samples = None

        return {
            'chunk_idx': chunk_idx,
            'temp_file': tmp_path,
            'files_processed': files_processed,
            'files_skipped_split': files_skipped_split,
            'files_skipped_img': files_skipped_img,
        }
    except Exception:
        # Fallback to in-memory return (rare)
        return {
            'chunk_idx': chunk_idx,
            'samples': all_samples,
            'files_processed': files_processed,
            'files_skipped_split': files_skipped_split,
            'files_skipped_img': files_skipped_img,
        }


def _chunk_list(lst, chunk_size):
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Pre-build dataset cache with MapReduce')
    parser.add_argument('--config', default='configs/pretrain_config.yaml')
    parser.add_argument('--mimic_cxr_path', type=str)
    parser.add_argument('--mimic_qa_path', type=str)
    parser.add_argument('--split', default='train')
    parser.add_argument('--cache_dir', default='.cache/dataset_samples')
    parser.add_argument('--num_workers', type=int, default=None, 
                        help='Number of worker processes (default: CPU count - 2)')
    parser.add_argument('--chunk_size', type=int, default=500,
                        help='Files per chunk (default: 500)')
    parser.add_argument('--force', action='store_true', help='Rebuild even if cache exists')
    parser.add_argument('--validate_only', action='store_true',
                        help='Only validate existing cache, do not rebuild')
    parser.add_argument('--validate_samples', type=int, default=200,
                        help='Number of samples to validate (default: 200)')
    parser.add_argument('--skip_file_checks', action='store_true',
                        help='Skip checking if image/scene_graph files exist')
    # ===== NEW: Subset sampling for pipeline testing =====
    parser.add_argument('--sample_percent', type=float, default=100.0,
                        help='Percentage of data to use (1-100). Use 5 for 5%% subset to test pipeline. Default: 100')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to cache. Overrides --sample_percent if set.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for subset sampling (default: 42)')
    parser.add_argument('--maxtasksperchild', type=int, default=100,
                        help='Recycle worker after this many tasks (default: 100)')
    parser.add_argument('--checkpoint_every', type=int, default=0,
                        help='Save partial cache every N chunks (0 disables).')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint/partial cache if available.')
    args = parser.parse_args()
    
    # Validate sample_percent
    if args.sample_percent <= 0 or args.sample_percent > 100:
        print(f"ERROR: --sample_percent must be between 0 and 100, got {args.sample_percent}")
        sys.exit(1)
    
    print("=" * 70)
    print("  MAPREDUCE CACHE BUILDER")
    print("=" * 70)
    
    # Subset mode detection
    is_subset = args.sample_percent < 100.0 or args.max_samples is not None
    if is_subset:
        print(f"\n  ⚡ SUBSET MODE: Testing pipeline with limited data")
        if args.max_samples:
            print(f"     Max samples: {args.max_samples:,}")
        else:
            print(f"     Sample: {args.sample_percent}% of data")
        print(f"     Seed: {args.seed}")
    
    # Load config
    config = None
    if args.config and os.path.exists(args.config):
        try:
            from configs.mimic_cxr_config import load_config_from_file
            config = load_config_from_file(args.config)
            print(f"  Config: {args.config}")
        except Exception as e:
            print(f"  Warning: Could not load config: {e}")
    
    # Get paths
    mimic_cxr_path = Path(args.mimic_cxr_path or (config.data.mimic_cxr_jpg_path if config else None))
    mimic_qa_path = Path(args.mimic_qa_path or (config.data.mimic_ext_cxr_qba_path if config else None))
    
    if not mimic_cxr_path or not mimic_qa_path:
        print("ERROR: Must provide --mimic_cxr_path and --mimic_qa_path or valid config")
        sys.exit(1)
    
    print(f"  CXR: {mimic_cxr_path}")
    print(f"  QA:  {mimic_qa_path}")
    
    # Cache path - include subset info in filename if using subset
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if is_subset:
        # Include subset percentage/max in cache key for unique naming
        subset_str = f"_max{args.max_samples}" if args.max_samples else f"_{args.sample_percent}pct"
        cache_key = hashlib.md5(f"{mimic_cxr_path}|{mimic_qa_path}|{args.split}|subset{subset_str}|seed{args.seed}".encode()).hexdigest()[:12]
        cache_path = cache_dir / f"samples_{args.split}{subset_str}_{cache_key}.pkl"
    else:
        cache_key = hashlib.md5(f"{mimic_cxr_path}|{mimic_qa_path}|{args.split}".encode()).hexdigest()[:12]
        cache_path = cache_dir / f"samples_{args.split}_{cache_key}.pkl"
    
    print(f"  Cache: {cache_path}")

    checkpoint_path = cache_path.with_suffix('.checkpoint.pkl')
    partial_cache_path = cache_path.with_suffix('.partial.pkl')
    
    # Handle --validate_only
    if args.validate_only:
        if not cache_path.exists():
            print(f"\n✗ Cache does not exist: {cache_path}")
            print("  Run without --validate_only to build cache first")
            sys.exit(1)
        
        print("\n" + "-" * 70)
        print("VALIDATE ONLY MODE")
        print("-" * 70)
        
        report = validate_cache(
            str(cache_path),
            num_samples=args.validate_samples,
            check_files=not args.skip_file_checks
        )
        print_validation_report(report)
        
        sys.exit(0 if report['overall_valid'] else 1)
    
    if cache_path.exists() and not args.force:
        with open(cache_path, 'rb') as f:
            samples = pickle.load(f)
        print(f"\n✓ Cache exists: {len(samples):,} samples")
        print(f"  Use --force to rebuild")
        print(f"  Use --validate_only to check cache validity")
        return
    
    # Determine workers
    num_workers = args.num_workers or max(1, cpu_count() - 2)
    print(f"  Workers: {num_workers}")
    print(f"  Chunk size: {args.chunk_size}")

    # Prefer 'forkserver' start method (safer for large multiprocessing jobs on Unix)
    # Fall back to 'spawn' if 'forkserver' isn't available.
    try:
        available = mp.get_all_start_methods()
        if 'forkserver' in available:
            mp.set_start_method('forkserver', force=True)
        else:
            mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # start method already set; ignore
        pass
    
    # =========================================================================
    # STEP 1: Load split info
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 1: Loading split info...")
    print("-" * 70)
    
    import pandas as pd
    split_file = mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv.gz'
    if not split_file.exists():
        split_file = mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv'
    
    if not split_file.exists():
        print(f"ERROR: Split file not found: {split_file}")
        sys.exit(1)
    
    df = pd.read_csv(split_file, compression='gzip' if str(split_file).endswith('.gz') else None)
    split_name = 'validate' if args.split == 'val' else args.split
    df = df[df['split'] == split_name]
    
    # Create frozenset for efficient lookup (hashable for multiprocessing)
    valid_studies = frozenset(zip(df['subject_id'].astype(int), df['study_id'].astype(int)))
    print(f"  {len(valid_studies):,} studies in '{split_name}' split")
    
    # =========================================================================
    # STEP 2: Collect QA file paths
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 2: Collecting QA file paths...")
    print("-" * 70)
    
    qa_dir = mimic_qa_path / 'qa'
    sg_dir = mimic_qa_path / 'scene_data'
    
    if not qa_dir.exists():
        print(f"ERROR: QA directory not found: {qa_dir}")
        sys.exit(1)
    
    # Collect all QA file paths as strings (for pickling)
    qa_files = []
    for p_group in sorted(qa_dir.iterdir()):
        if p_group.is_dir() and p_group.name.startswith('p'):
            for patient_dir in p_group.iterdir():
                if patient_dir.is_dir() and patient_dir.name.startswith('p'):
                    qa_files.extend([str(f) for f in patient_dir.glob('s*.qa.json')])
    
    total_qa_files = len(qa_files)
    print(f"  {total_qa_files:,} QA files found")
    
    if not qa_files:
        print("ERROR: No QA files found")
        sys.exit(1)
    
    # ===== SUBSET SAMPLING =====
    if is_subset:
        print(f"\n  Applying subset sampling...")
        random.seed(args.seed)
        
        if args.max_samples:
            # Estimate samples per file (~5-10 questions per study on average)
            # Sample more files than needed to account for filtered studies
            est_samples_per_file = 5
            files_needed = min(len(qa_files), int((args.max_samples / est_samples_per_file) * 1.5))
            qa_files = random.sample(qa_files, files_needed)
            print(f"  Sampled {len(qa_files):,} files (targeting ~{args.max_samples:,} samples)")
        else:
            # Sample by percentage
            sample_count = max(1, int(len(qa_files) * (args.sample_percent / 100.0)))
            qa_files = random.sample(qa_files, sample_count)
            print(f"  Sampled {len(qa_files):,} files ({args.sample_percent}% of {total_qa_files:,})")

    # ===== RESUME FROM CHECKPOINT (optional) =====
    resume_state = {}
    if args.resume and checkpoint_path.exists():
        print("\n  ♻ Resuming from checkpoint...")
        resume_state = _load_checkpoint(checkpoint_path)
        if resume_state:
            # Trust saved QA list to keep sampling consistent
            saved_qa_files = resume_state.get('qa_files')
            if saved_qa_files:
                qa_files = saved_qa_files
                print(f"  Loaded saved QA file list: {len(qa_files):,} files")
            else:
                print("  ⚠ Checkpoint missing QA file list; resume may be inconsistent")
        else:
            print("  ⚠ Failed to load checkpoint; starting fresh")
    
    # =========================================================================
    # STEP 3: MapReduce processing
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 3: MapReduce processing...")
    print("-" * 70)
    
    # Create chunks
    chunks = _chunk_list(qa_files, args.chunk_size)
    print(f"  Created {len(chunks)} chunks (~{args.chunk_size} files each)")
    
    # Load image metadata for view position selection
    print(f"\n  Loading image metadata for view selection...")
    metadata_cache = _load_metadata_cache(str(mimic_cxr_path))
    print(f"  Loaded metadata for {len(metadata_cache):,} images")
    
    if len(metadata_cache) == 0:
        print("  ⚠ No metadata loaded - will use first image from each study")
    
    # Prepare chunk arguments (each chunk gets the same shared data)
    mimic_cxr_str = str(mimic_cxr_path)
    sg_dir_str = str(sg_dir) if sg_dir.exists() else None
    
    # Persist metadata_cache to a temp file so workers can load it from disk
    # (avoids pickling the large dict into IPC for each worker).
    metadata_tmp_path = None
    if metadata_cache:
        try:
            md_fd, md_tmp = tempfile.mkstemp(suffix='.pkl', prefix='prebuild_metadata_')
            with os.fdopen(md_fd, 'wb') as mf:
                pickle.dump(metadata_cache, mf, protocol=pickle.HIGHEST_PROTOCOL)
            metadata_tmp_path = md_tmp
        except Exception:
            metadata_tmp_path = None

    # Pass None for metadata_cache in chunk args; workers will load via initializer
    chunk_args_list = [
        (idx, chunk, valid_studies, mimic_cxr_str, sg_dir_str, None)
        for idx, chunk in enumerate(chunks)
    ]
    
    # ============ MAP PHASE ============
    print(f"\n  MAP phase: Processing {len(chunks)} chunks with {num_workers} workers...")
    
    all_samples = []
    total_processed = 0
    total_skipped_split = 0
    total_skipped_img = 0
    processed_chunks = set()

    # Resume partial cache (if requested)
    if args.resume and resume_state:
        processed_chunks = set(resume_state.get('processed_chunks', []))
        total_processed = resume_state.get('files_processed', 0)
        total_skipped_split = resume_state.get('files_skipped_split', 0)
        total_skipped_img = resume_state.get('files_skipped_img', 0)
        if partial_cache_path.exists():
            try:
                with open(partial_cache_path, 'rb') as f:
                    all_samples = pickle.load(f)
                print(f"  Loaded partial cache: {len(all_samples):,} samples")
            except Exception:
                print("  ⚠ Failed to load partial cache; continuing without it")
                all_samples = []

        if processed_chunks:
            print(f"  Resuming from {len(processed_chunks):,} completed chunks")

    # Filter chunks to resume only remaining work
    if processed_chunks:
        chunk_args_list = [c for c in chunk_args_list if c[0] not in processed_chunks]
        print(f"  Remaining chunks to process: {len(chunk_args_list)}")
    
    start_time = time.time()
    
    def _save_partial_cache(path: Path, samples: list) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix='.pkl', prefix=path.name + '.tmp.', dir=str(path.parent)
        )
        try:
            with os.fdopen(tmp_fd, 'wb') as f:
                pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            os.replace(tmp_path, str(path))
        except Exception:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise

    def _checkpoint_state(processed_idx_set: set) -> dict:
        return {
            'version': 1,
            'cache_path': str(cache_path),
            'cache_key': cache_key,
            'split': args.split,
            'sample_percent': args.sample_percent,
            'max_samples': args.max_samples,
            'seed': args.seed,
            'chunk_size': args.chunk_size,
            'qa_files': qa_files,
            'processed_chunks': sorted(list(processed_idx_set)),
            'files_processed': total_processed,
            'files_skipped_split': total_skipped_split,
            'files_skipped_img': total_skipped_img,
        }

    try:
        # Prepare initializer argument (path to metadata temp file or None)
        init_arg = metadata_tmp_path if metadata_tmp_path else None

        # Use initializer to populate METADATA_CACHE in each worker process.
        with Pool(processes=num_workers, initializer=_init_worker,
                  initargs=(init_arg,), maxtasksperchild=args.maxtasksperchild) as pool:
            # Use imap_unordered for better performance and progress tracking
            for i, result in enumerate(pool.imap_unordered(_map_qa_chunk, chunk_args_list)):
                # Result may contain a temp_file where the worker wrote the
                # full samples to avoid large IPC payloads. Load and cleanup.
                samples = []
                if result is None:
                    continue
                if 'temp_file' in result:
                    tmp = result.get('temp_file')
                    try:
                        with open(tmp, 'rb') as tf:
                            chunk_data = pickle.load(tf)
                        samples = chunk_data.get('samples', [])
                    except Exception:
                        samples = []
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass
                else:
                    samples = result.get('samples', [])

                all_samples.extend(samples)
                total_processed += result.get('files_processed', 0)
                total_skipped_split += result.get('files_skipped_split', 0)
                total_skipped_img += result.get('files_skipped_img', 0)
                if result.get('chunk_idx') is not None:
                    processed_chunks.add(result.get('chunk_idx'))
                
                # Progress update every 10 chunks or at end
                if (i + 1) % 10 == 0 or (i + 1) == len(chunk_args_list):
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (len(chunk_args_list) - i - 1) / rate if rate > 0 else 0
                    
                    print(f"    [{i+1}/{len(chunk_args_list)}] "
                          f"Samples: {len(all_samples):,} | "
                          f"Rate: {rate:.1f} chunks/s | "
                          f"ETA: {eta:.0f}s")

                # Checkpointing (optional)
                if args.checkpoint_every and args.checkpoint_every > 0:
                    if (i + 1) % args.checkpoint_every == 0 or (i + 1) == len(chunk_args_list):
                        print(f"    ⏳ Saving checkpoint at chunk {i+1}...")
                        try:
                            _save_partial_cache(partial_cache_path, all_samples)
                            _save_checkpoint(checkpoint_path, _checkpoint_state(processed_chunks))
                            print(f"    ✓ Checkpoint saved: {partial_cache_path.name}")
                        except Exception as e:
                            print(f"    ⚠ Checkpoint save failed: {e}")
    
    except Exception as e:
        print(f"\n  ⚠ Multiprocessing failed: {e}")
        print("  Falling back to sequential processing...")
        
        # Sequential fallback with progress
        from tqdm import tqdm
        
        all_samples = []
        total_processed = 0
        total_skipped_split = 0
        total_skipped_img = 0
        
        for i, chunk_args in enumerate(tqdm(chunk_args_list, desc="Processing", unit="chunk")):
            result = _map_qa_chunk(chunk_args)
            if result is None:
                continue
            if 'temp_file' in result:
                tmp = result.get('temp_file')
                try:
                    with open(tmp, 'rb') as tf:
                        chunk_data = pickle.load(tf)
                    samples = chunk_data.get('samples', [])
                except Exception:
                    samples = []
                try:
                    os.remove(tmp)
                except Exception:
                    pass
            else:
                samples = result.get('samples', [])

            all_samples.extend(samples)
            total_processed += result.get('files_processed', 0)
            total_skipped_split += result.get('files_skipped_split', 0)
            total_skipped_img += result.get('files_skipped_img', 0)
            if result.get('chunk_idx') is not None:
                processed_chunks.add(result.get('chunk_idx'))

            if args.checkpoint_every and args.checkpoint_every > 0:
                if (i + 1) % args.checkpoint_every == 0 or (i + 1) == len(chunk_args_list):
                    print(f"    ⏳ Saving checkpoint at chunk {i+1}...")
                    try:
                        _save_partial_cache(partial_cache_path, all_samples)
                        _save_checkpoint(checkpoint_path, _checkpoint_state(processed_chunks))
                        print(f"    ✓ Checkpoint saved: {partial_cache_path.name}")
                    except Exception as e:
                        print(f"    ⚠ Checkpoint save failed: {e}")
    
    elapsed = time.time() - start_time

    # Cleanup metadata temp file (workers have already loaded it)
    if metadata_tmp_path:
        try:
            os.remove(metadata_tmp_path)
        except Exception:
            pass
    
    # ============ REDUCE PHASE (already done - just concatenation) ============
    print(f"\n  REDUCE phase: Combined {len(all_samples):,} samples")
    
    # ===== APPLY MAX_SAMPLES LIMIT (if specified) =====
    if args.max_samples and len(all_samples) > args.max_samples:
        print(f"\n  Truncating to --max_samples={args.max_samples:,}...")
        random.seed(args.seed)  # Ensure reproducibility
        all_samples = random.sample(all_samples, args.max_samples)
        print(f"  Final sample count: {len(all_samples):,}")
    
    # =========================================================================
    # STEP 4: Summary and save
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 4: Saving cache...")
    print("-" * 70)
    
    print(f"  Total samples: {len(all_samples):,}")
    if is_subset:
        print(f"  ⚡ SUBSET MODE: This is a reduced dataset for pipeline testing")
    print(f"  Files processed: {total_processed:,}")
    print(f"  Skipped (wrong split): {total_skipped_split:,}")
    print(f"  Skipped (no image): {total_skipped_img:,}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    
    if all_samples:
        # Atomically write cache to avoid leaving a partially-written file
        # if the process is interrupted. Write to a temp file in the same
        # directory and then replace the final path.
        tmp_fd = None
        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix='.pkl', prefix=cache_path.name + '.tmp.', dir=str(cache_path.parent))
            with os.fdopen(tmp_fd, 'wb') as f:
                pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    # On some platforms fsync may fail for special filesystems; ignore
                    pass
            # Atomic replace
            os.replace(tmp_path, str(cache_path))
        except Exception as e:
            # Cleanup temp file if present
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            print(f"  ✗ Failed to save cache atomically: {e}")
            raise
        
        mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"\n  ✓ Saved: {cache_path}")
        print(f"    Size: {mb:.1f} MB")

        # Cleanup partial cache/checkpoint if present
        try:
            if partial_cache_path.exists():
                partial_cache_path.unlink()
            if checkpoint_path.exists():
                checkpoint_path.unlink()
        except Exception:
            pass
    else:
        print("\n  ⚠ No samples to save!")
        sys.exit(1)
    
    # =========================================================================
    # STEP 5: Validate cache
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 5: Validating cache...")
    print("-" * 70)
    
    report = validate_cache(
        str(cache_path),
        num_samples=min(args.validate_samples, len(all_samples)),
        check_files=not args.skip_file_checks
    )
    print_validation_report(report)
    
    # =========================================================================
    # Done
    # =========================================================================
    print("\n" + "=" * 70)
    if report['overall_valid']:
        print("✓ CACHE BUILD COMPLETE AND VALIDATED!")
    else:
        print("⚠ CACHE BUILT BUT VALIDATION FOUND ISSUES")
        print("  Training may still work, but check issues above")
    print("=" * 70)
    
    if is_subset:
        print("\n⚡ SUBSET MODE - Cache is ready for pipeline testing!")
        print(f"   Samples: {len(all_samples):,}")
        print(f"   Cache file: {cache_path.name}")
        print("\nNow test the training pipeline:")
        print("  # Quick single-GPU test:")
        print("  python train_mimic_cxr.py \\")
        print("    --config configs/pretrain_config.yaml \\")
        print("    --max_steps 100 \\")
        print("    --eval_steps 50")
        print("\n  # Multi-GPU test with DeepSpeed:")
        print("  deepspeed --num_gpus=4 train_mimic_cxr.py \\")
        print("    --config configs/pretrain_config.yaml \\")
        print("    --deepspeed_config configs/deepspeed_config.json \\")
        print("    --max_steps 200")
        print("\nOnce pipeline is verified, build FULL cache:")
        print("  python scripts/prebuild_cache.py --config configs/pretrain_config.yaml --force")
    else:
        print("\nNow run training:")
        print("  deepspeed --num_gpus=4 train_mimic_cxr.py \\")
        print("    --config configs/pretrain_config.yaml \\")
        print("    --deepspeed_config configs/deepspeed_config.json")
    print("=" * 70)


if __name__ == '__main__':
    main()
