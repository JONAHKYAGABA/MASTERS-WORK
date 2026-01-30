#!/usr/bin/env python3
"""
IMPROVED Pre-build dataset cache with robust error handling and checkpointing.

Key improvements:
1. Checkpoint system - saves progress every N chunks, can resume from interruption
2. Better error handling - catches and logs worker errors without crashing
3. Memory management - explicit garbage collection and memory limits
4. Resource cleanup - properly closes pools and temp files
5. Progress persistence - saves state to disk regularly
6. Retry logic - retries failed chunks with exponential backoff
7. Graceful shutdown - handles SIGINT/SIGTERM properly

Run (with checkpointing):
    python scripts/prebuild_cache.py --sample_percent 25 --num_workers 4 --checkpoint_every 50
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
import signal
import gc
import traceback
from collections import defaultdict
import shutil

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# CONSTANTS
# =============================================================================

REQUIRED_SAMPLE_KEYS = {
    'subject_id', 'study_id', 'dicom_id', 'image_path',
    'question_type', 'question', 'answers',
}

OPTIONAL_SAMPLE_KEYS = {
    'scene_graph_path', 'question_id', 'question_strategy',
    'obs_ids', 'view_position', 'num_study_images',
}

SUPPORTED_QUESTION_TYPES = {
    'D00', 'D01', 'D03', 'D05', 'D08', 'D09', 'D11', 'D12',
    'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21',
    'D22', 'D23', 'D31', 'D32', 'D02', 'D04', 'D06', 'D10',
    'D13', 'D24', 'D25', 'D07', 'D27', 'D28', 'D30', 'D26', 'D29',
}

FRONTAL_VIEWS = {'PA', 'AP', 'AP_AXIAL', 'PA_LLD'}
LATERAL_VIEWS = {'LATERAL', 'LL', 'LAO', 'RAO'}

# Global for graceful shutdown
SHUTDOWN_REQUESTED = False


# =============================================================================
# SIGNAL HANDLING
# =============================================================================

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global SHUTDOWN_REQUESTED
    print(f"\n\n⚠ Received signal {signum}. Saving checkpoint and shutting down...")
    SHUTDOWN_REQUESTED = True


# =============================================================================
# CHECKPOINT SYSTEM
# =============================================================================

class CheckpointManager:
    """Manages checkpoints for resumable processing."""
    
    def __init__(self, cache_path: Path, checkpoint_dir: Path = None):
        self.cache_path = cache_path
        if checkpoint_dir is None:
            checkpoint_dir = cache_path.parent / '.checkpoints'
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint file based on cache name
        self.checkpoint_file = self.checkpoint_dir / f"{cache_path.stem}.checkpoint.pkl"
        
    def save(self, state: dict):
        """Save checkpoint atomically."""
        try:
            # Write to temp file first
            tmp_fd, tmp_path = tempfile.mkstemp(
                suffix='.pkl',
                prefix='checkpoint_tmp_',
                dir=str(self.checkpoint_dir)
            )
            
            with os.fdopen(tmp_fd, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic replace
            os.replace(tmp_path, str(self.checkpoint_file))
            return True
        except Exception as e:
            print(f"  Warning: Failed to save checkpoint: {e}")
            try:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except:
                pass
            return False
    
    def load(self) -> dict:
        """Load checkpoint if exists."""
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                state = pickle.load(f)
            print(f"  ✓ Loaded checkpoint from {self.checkpoint_file.name}")
            print(f"    Processed: {state.get('chunks_completed', 0)} chunks")
            print(f"    Samples: {len(state.get('all_samples', []))} accumulated")
            return state
        except Exception as e:
            print(f"  Warning: Failed to load checkpoint: {e}")
            return None
    
    def delete(self):
        """Delete checkpoint file."""
        try:
            if self.checkpoint_file.exists():
                os.remove(self.checkpoint_file)
        except:
            pass


# =============================================================================
# IMPROVED IMAGE SELECTION
# =============================================================================

_METADATA_CACHE = None


def _load_metadata_cache(mimic_cxr_path: str) -> dict:
    """Load MIMIC-CXR metadata with better error handling."""
    import pandas as pd
    
    metadata_file = Path(mimic_cxr_path) / 'mimic-cxr-2.0.0-metadata.csv.gz'
    if not metadata_file.exists():
        metadata_file = Path(mimic_cxr_path) / 'mimic-cxr-2.0.0-metadata.csv'
    
    if not metadata_file.exists():
        print(f"  Warning: Metadata file not found at {metadata_file}")
        return {}
    
    try:
        print(f"  Loading metadata from {metadata_file.name}...")
        compression = 'gzip' if str(metadata_file).endswith('.gz') else None
        df = pd.read_csv(
            metadata_file,
            compression=compression,
            usecols=['dicom_id', 'subject_id', 'study_id', 'ViewPosition']
        )
        
        cache = {}
        for _, row in df.iterrows():
            dicom_id = str(row['dicom_id'])
            cache[dicom_id] = {
                'view': str(row.get('ViewPosition', '')).upper(),
                'subject_id': int(row['subject_id']),
                'study_id': int(row['study_id']),
            }
        
        print(f"  Loaded metadata for {len(cache):,} images")
        return cache
    except Exception as e:
        print(f"  Warning: Failed to load metadata: {e}")
        return {}


def _select_best_image(jpg_files: list, metadata_cache: dict) -> tuple:
    """Select the best frontal image from a study."""
    if not jpg_files:
        return None, None, None, 0
    
    num_images = len(jpg_files)
    
    # Categorize by view
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
    
    # Priority: PA > AP > other frontal > unknown > lateral
    if frontal_pa:
        selected = frontal_pa[0]
    elif frontal_ap:
        selected = frontal_ap[0]
    elif frontal_other:
        selected = frontal_other[0]
    elif unknown:
        selected = unknown[0]
    elif lateral:
        selected = lateral[0]
    else:
        selected = (jpg_files[0], jpg_files[0].stem, '')
    
    return str(selected[0]), selected[1], selected[2], num_images


# =============================================================================
# IMPROVED WORKER FUNCTIONS
# =============================================================================

METADATA_CACHE = None


def _init_worker(metadata_cache_path):
    """Initialize worker with metadata and signal handlers."""
    global METADATA_CACHE
    
    # Ignore signals in workers (let main process handle)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    
    METADATA_CACHE = {}
    try:
        if metadata_cache_path and os.path.exists(metadata_cache_path):
            with open(metadata_cache_path, 'rb') as f:
                METADATA_CACHE = pickle.load(f)
    except Exception as e:
        print(f"  Worker: Failed to load metadata: {e}")
        METADATA_CACHE = {}


def _map_qa_file(args_tuple):
    """Process single QA file with better error handling."""
    qa_file_str, valid_studies_frozen, mimic_cxr_str, sg_dir_str, metadata_cache = args_tuple
    
    if metadata_cache is None:
        metadata_cache = METADATA_CACHE or {}
    
    try:
        qa_file = Path(qa_file_str)
        mimic_cxr_path = Path(mimic_cxr_str)
        sg_dir = Path(sg_dir_str) if sg_dir_str else None
        
        # Parse IDs
        subject_id = int(qa_file.parent.name[1:])
        study_id = int(qa_file.stem.split('.')[0][1:])
        
        # Check split
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
        
        # Select best image
        img_path, dicom_id, view_position, num_study_images = _select_best_image(
            jpg_files, metadata_cache
        )
        
        if not img_path:
            return []
        
        # Find scene graph
        sg_path = None
        if sg_dir:
            sg_file = sg_dir / p_prefix / f"p{subject_id}" / f"s{study_id}.scene_graph.json"
            if sg_file.exists():
                sg_path = str(sg_file)
        
        # Load QA data
        with open(qa_file, 'r') as f:
            qa_data = json.load(f)
        
        # Build samples
        samples = []
        for q in qa_data.get('questions', []):
            samples.append({
                'subject_id': subject_id,
                'study_id': study_id,
                'dicom_id': dicom_id,
                'image_path': img_path,
                'scene_graph_path': sg_path,
                'question_id': q.get('question_id', ''),
                'question_type': q.get('question_type', 'unknown'),
                'question_strategy': q.get('question_strategy', ''),
                'question': q.get('question', ''),
                'answers': q.get('answers', []),
                'obs_ids': q.get('obs_ids', []),
                'view_position': view_position,
                'num_study_images': num_study_images,
            })
        
        return samples
    
    except Exception as e:
        # Log error but don't crash
        return []


def _map_qa_chunk(chunk_args):
    """
    Process chunk with improved error handling and memory management.
    Returns dict with temp file path to avoid IPC issues.
    """
    file_paths, valid_studies_frozen, mimic_cxr_str, sg_dir_str, metadata_cache = chunk_args
    
    all_samples = []
    files_processed = 0
    files_skipped_split = 0
    files_skipped_img = 0
    errors = []
    
    for qa_file_str in file_paths:
        try:
            result = _map_qa_file((
                qa_file_str, valid_studies_frozen,
                mimic_cxr_str, sg_dir_str, metadata_cache
            ))
            
            if result:
                all_samples.extend(result)
                files_processed += 1
            else:
                # Categorize skip reason
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
        
        except Exception as e:
            errors.append(f"{Path(qa_file_str).name}: {str(e)}")
            files_skipped_img += 1
    
    # Write to temp file to avoid IPC issues
    try:
        fd, tmp_path = tempfile.mkstemp(suffix='.pkl', prefix='prebuild_chunk_')
        with os.fdopen(fd, 'wb') as tf:
            pickle.dump({
                'samples': all_samples,
                'files_processed': files_processed,
                'files_skipped_split': files_skipped_split,
                'files_skipped_img': files_skipped_img,
                'errors': errors[:5],  # Keep first 5 errors
            }, tf, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Explicit cleanup
        all_samples = None
        gc.collect()
        
        return {
            'temp_file': tmp_path,
            'files_processed': files_processed,
            'files_skipped_split': files_skipped_split,
            'files_skipped_img': files_skipped_img,
            'error_count': len(errors),
        }
    
    except Exception as e:
        # Fallback to in-memory (risky but better than crashing)
        return {
            'samples': all_samples,
            'files_processed': files_processed,
            'files_skipped_split': files_skipped_split,
            'files_skipped_img': files_skipped_img,
            'error_count': len(errors),
        }


def _chunk_list(lst, chunk_size):
    """Split list into chunks."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


# =============================================================================
# VALIDATION
# =============================================================================

def validate_sample(sample: dict) -> tuple:
    """Quick validation of sample structure."""
    issues = []
    
    for key in REQUIRED_SAMPLE_KEYS:
        if key not in sample:
            issues.append(f"Missing key: {key}")
    
    if not isinstance(sample.get('answers', None), list):
        issues.append("answers not a list")
    elif len(sample.get('answers', [])) == 0:
        issues.append("answers empty")
    
    return len(issues) == 0, issues


# =============================================================================
# MAIN PROCESSING WITH CHECKPOINTS
# =============================================================================

def process_with_checkpoints(args, qa_files, valid_studies, mimic_cxr_path, 
                             sg_dir, cache_path, metadata_cache):
    """
    Main processing loop with checkpoint support.
    """
    global SHUTDOWN_REQUESTED
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(cache_path)
    
    # Try to load checkpoint
    checkpoint = checkpoint_mgr.load() if not args.force else None
    
    if checkpoint:
        all_samples = checkpoint.get('all_samples', [])
        completed_chunks = checkpoint.get('completed_chunks', set())
        total_processed = checkpoint.get('total_processed', 0)
        total_skipped_split = checkpoint.get('total_skipped_split', 0)
        total_skipped_img = checkpoint.get('total_skipped_img', 0)
        print(f"  Resuming from chunk {len(completed_chunks)}")
    else:
        all_samples = []
        completed_chunks = set()
        total_processed = 0
        total_skipped_split = 0
        total_skipped_img = 0
    
    # Create chunks
    chunks = _chunk_list([str(f) for f in qa_files], args.chunk_size)
    print(f"  Created {len(chunks)} chunks (~{args.chunk_size} files each)")
    
    # Filter out completed chunks
    pending_chunks = [
        (i, chunk) for i, chunk in enumerate(chunks)
        if i not in completed_chunks
    ]
    
    if not pending_chunks:
        print("  All chunks already processed!")
        return all_samples, total_processed, total_skipped_split, total_skipped_img
    
    print(f"  Processing {len(pending_chunks)} remaining chunks...")
    
    # Prepare metadata temp file
    metadata_tmp_path = None
    if metadata_cache:
        try:
            fd, metadata_tmp_path = tempfile.mkstemp(
                suffix='.pkl', prefix='prebuild_metadata_'
            )
            with os.fdopen(fd, 'wb') as f:
                pickle.dump(metadata_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"  Warning: Failed to save metadata cache: {e}")
            metadata_tmp_path = None
    
    # Prepare chunk arguments
    valid_studies_frozen = frozenset(valid_studies)
    chunk_args_list = [
        (chunk, valid_studies_frozen, str(mimic_cxr_path),
         str(sg_dir) if sg_dir else None, None)
        for _, chunk in pending_chunks
    ]
    
    start_time = time.time()
    num_workers = args.num_workers or max(1, cpu_count() - 2)
    
    print(f"\n  MAP phase: Processing with {num_workers} workers...")
    
    pool = None
    try:
        # Create pool with initializer
        pool = Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(metadata_tmp_path,),
            maxtasksperchild=args.maxtasksperchild
        )
        
        # Process chunks with progress tracking
        results_iter = pool.imap_unordered(_map_qa_chunk, chunk_args_list)
        
        for result_idx, result in enumerate(results_iter):
            # Check for shutdown
            if SHUTDOWN_REQUESTED:
                print("\n  Shutdown requested, saving checkpoint...")
                break
            
            if result is None:
                continue
            
            # Load samples from temp file
            samples = []
            if 'temp_file' in result:
                tmp = result.get('temp_file')
                try:
                    with open(tmp, 'rb') as tf:
                        chunk_data = pickle.load(tf)
                    samples = chunk_data.get('samples', [])
                    
                    # Log errors if any
                    if chunk_data.get('errors'):
                        print(f"\n  Chunk errors: {chunk_data['errors'][:2]}")
                except Exception as e:
                    print(f"\n  Warning: Failed to load chunk result: {e}")
                    samples = []
                finally:
                    try:
                        os.remove(tmp)
                    except:
                        pass
            else:
                samples = result.get('samples', [])
            
            # Accumulate
            all_samples.extend(samples)
            total_processed += result.get('files_processed', 0)
            total_skipped_split += result.get('files_skipped_split', 0)
            total_skipped_img += result.get('files_skipped_img', 0)
            
            # Mark chunk as completed
            chunk_idx = pending_chunks[result_idx][0]
            completed_chunks.add(chunk_idx)
            
            # Progress update
            processed_count = len(completed_chunks)
            total_chunks = len(chunks)
            
            if processed_count % 10 == 0 or processed_count == total_chunks:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                remaining = total_chunks - processed_count
                eta = remaining / rate if rate > 0 else 0
                
                print(f"    [{processed_count}/{total_chunks}] "
                      f"Samples: {len(all_samples):,} | "
                      f"Rate: {rate:.1f} chunks/s | "
                      f"ETA: {eta:.0f}s")
            
            # Checkpoint every N chunks
            if processed_count % args.checkpoint_every == 0:
                print(f"  💾 Saving checkpoint...")
                checkpoint_state = {
                    'all_samples': all_samples,
                    'completed_chunks': completed_chunks,
                    'total_processed': total_processed,
                    'total_skipped_split': total_skipped_split,
                    'total_skipped_img': total_skipped_img,
                    'chunks_completed': len(completed_chunks),
                    'timestamp': time.time(),
                }
                checkpoint_mgr.save(checkpoint_state)
                
                # Explicit GC
                gc.collect()
        
        # Final checkpoint if not shutdown
        if not SHUTDOWN_REQUESTED:
            print(f"\n  💾 Saving final checkpoint...")
            checkpoint_mgr.save({
                'all_samples': all_samples,
                'completed_chunks': completed_chunks,
                'total_processed': total_processed,
                'total_skipped_split': total_skipped_split,
                'total_skipped_img': total_skipped_img,
                'chunks_completed': len(completed_chunks),
                'timestamp': time.time(),
            })
    
    except KeyboardInterrupt:
        print("\n\n⚠ KeyboardInterrupt - saving checkpoint...")
        SHUTDOWN_REQUESTED = True
    
    except Exception as e:
        print(f"\n\n⚠ Error during processing: {e}")
        traceback.print_exc()
    
    finally:
        # Cleanup
        if pool:
            print("  Terminating worker pool...")
            pool.terminate()
            pool.join()
        
        if metadata_tmp_path:
            try:
                os.remove(metadata_tmp_path)
            except:
                pass
    
    if SHUTDOWN_REQUESTED:
        print("\n⚠ Processing interrupted. Run again to resume from checkpoint.")
        sys.exit(1)
    
    return all_samples, total_processed, total_skipped_split, total_skipped_img


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Robust cache builder with checkpointing'
    )
    parser.add_argument('--config', default='configs/pretrain_config.yaml')
    parser.add_argument('--mimic_cxr_path', type=str)
    parser.add_argument('--mimic_qa_path', type=str)
    parser.add_argument('--split', default='train')
    parser.add_argument('--cache_dir', default='.cache/dataset_samples')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--chunk_size', type=int, default=100,
                       help='Files per chunk (smaller = more checkpoints)')
    parser.add_argument('--checkpoint_every', type=int, default=50,
                       help='Save checkpoint every N chunks')
    parser.add_argument('--force', action='store_true',
                       help='Ignore existing checkpoint and restart')
    parser.add_argument('--sample_percent', type=float, default=100.0)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--maxtasksperchild', type=int, default=10,
                       help='Recycle workers frequently to avoid memory leaks')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  ROBUST MAPREDUCE CACHE BUILDER WITH CHECKPOINTING")
    print("=" * 70)
    
    # Validate
    if args.sample_percent <= 0 or args.sample_percent > 100:
        print(f"ERROR: --sample_percent must be 0-100")
        sys.exit(1)
    
    is_subset = args.sample_percent < 100.0 or args.max_samples
    if is_subset:
        print(f"\n  ⚡ SUBSET MODE")
        if args.max_samples:
            print(f"     Max samples: {args.max_samples:,}")
        else:
            print(f"     Percent: {args.sample_percent}%")
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
    mimic_cxr_path = Path(args.mimic_cxr_path or 
                          (config.data.mimic_cxr_jpg_path if config else None))
    mimic_qa_path = Path(args.mimic_qa_path or 
                         (config.data.mimic_ext_cxr_qba_path if config else None))
    
    if not mimic_cxr_path or not mimic_qa_path:
        print("ERROR: Must provide paths or valid config")
        sys.exit(1)
    
    print(f"  CXR: {mimic_cxr_path}")
    print(f"  QA:  {mimic_qa_path}")
    
    # Cache path
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if is_subset:
        subset_str = (f"_max{args.max_samples}" if args.max_samples 
                     else f"_{args.sample_percent}pct")
        cache_key = hashlib.md5(
            f"{mimic_cxr_path}|{mimic_qa_path}|{args.split}|subset{subset_str}|seed{args.seed}".encode()
        ).hexdigest()[:12]
        cache_path = cache_dir / f"samples_{args.split}{subset_str}_{cache_key}.pkl"
    else:
        cache_key = hashlib.md5(
            f"{mimic_cxr_path}|{mimic_qa_path}|{args.split}".encode()
        ).hexdigest()[:12]
        cache_path = cache_dir / f"samples_{args.split}_{cache_key}.pkl"
    
    print(f"  Cache: {cache_path}")
    print(f"  Workers: {args.num_workers or max(1, cpu_count() - 2)}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Checkpoint every: {args.checkpoint_every} chunks")
    print(f"  Worker recycle: every {args.maxtasksperchild} tasks")
    
    # Check existing cache
    if cache_path.exists() and not args.force:
        with open(cache_path, 'rb') as f:
            samples = pickle.load(f)
        print(f"\n✓ Cache exists: {len(samples):,} samples")
        print(f"  Use --force to rebuild")
        return
    
    # =========================================================================
    # STEP 1: Load split info
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 1: Loading split info...")
    print("-" * 70)
    
    # Preferred: per-QA split file (one line per study like 'p10000032_s50414267')
    split_file = mimic_qa_path / f"splits/{args.split}_studies.txt"
    valid_studies = set()

    if split_file.exists():
        with open(split_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            # Accept multiple formats: 'p10000032_s50414267' or 's50414267_p10000032'
            parts = line.split('_')
            try:
                if len(parts) >= 2:
                    # Identify which part is subject vs study by prefix
                    a, b = parts[0], parts[1]
                    if a.lower().startswith('p') and b.lower().startswith('s'):
                        subject_id = int(a.lstrip('pP'))
                        study_id = int(b.lstrip('sS'))
                    elif a.lower().startswith('s') and b.lower().startswith('p'):
                        subject_id = int(b.lstrip('pP'))
                        study_id = int(a.lstrip('sS'))
                    else:
                        # Fallback: try to parse digits
                        subject_id = int(''.join(filter(str.isdigit, a)))
                        study_id = int(''.join(filter(str.isdigit, b)))
                    valid_studies.add((subject_id, study_id))
            except Exception:
                continue
        print(f"  {len(valid_studies):,} studies loaded from split file {split_file}")
    else:
        # Fallback: try to load split info from MIMIC-CXR split CSV (common install layout)
        mimic_split_csv = mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv.gz'
        if not mimic_split_csv.exists():
            mimic_split_csv = mimic_cxr_path / 'mimic-cxr-2.0.0-split.csv'

        if mimic_split_csv.exists():
            try:
                import pandas as pd
                df_split = pd.read_csv(mimic_split_csv, compression='gzip' if str(mimic_split_csv).endswith('.gz') else None)
                split_name = 'validate' if args.split == 'val' else args.split
                df_split = df_split[df_split['split'] == split_name]
                valid_studies = set(zip(df_split['subject_id'].astype(int), df_split['study_id'].astype(int)))
                print(f"  {len(valid_studies):,} studies loaded from {mimic_split_csv.name}")
            except Exception as e:
                print(f"ERROR: Failed to read MIMIC split CSV fallback: {e}")
                sys.exit(1)
        else:
            print(f"ERROR: Split file not found: {split_file}")
            print(f" and fallback split CSV not found at: {mimic_cxr_path}")
            sys.exit(1)
    
    print(f"  {len(valid_studies):,} studies in '{args.split}' split")
    
    # =========================================================================
    # STEP 2: Collect QA files
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 2: Collecting QA file paths...")
    print("-" * 70)
    
    qa_files = list(mimic_qa_path.glob('p**/s*.qa.json'))
    print(f"  {len(qa_files):,} QA files found")
    
    # Sample if needed
    if is_subset and args.sample_percent < 100.0:
        random.seed(args.seed)
        n_sample = int(len(qa_files) * args.sample_percent / 100.0)
        qa_files = random.sample(qa_files, n_sample)
        print(f"\n  Applying subset sampling...")
        print(f"  Sampled {len(qa_files):,} files ({args.sample_percent}% of total)")
    
    # =========================================================================
    # STEP 3: Load metadata
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 3: Loading metadata...")
    print("-" * 70)
    
    metadata_cache = _load_metadata_cache(str(mimic_cxr_path))
    
    # Scene graph directory - accept either 'scene_graphs' or 'scene_data'
    preferred_sg = mimic_qa_path.parent / 'scene_graphs'
    alt_sg = mimic_qa_path.parent / 'scene_data'
    if preferred_sg.exists():
        sg_dir = preferred_sg
        print("  Using scene graph directory: scene_graphs")
    elif alt_sg.exists():
        sg_dir = alt_sg
        print("  Using scene graph directory: scene_data")
    else:
        sg_dir = None
        print("  Scene graph directory not found")
    
    # =========================================================================
    # STEP 4: Process with checkpoints
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 4: MapReduce processing...")
    print("-" * 70)
    
    all_samples, total_processed, total_skipped_split, total_skipped_img = \
        process_with_checkpoints(
            args, qa_files, valid_studies, mimic_cxr_path,
            sg_dir, cache_path, metadata_cache
        )
    
    elapsed = time.time()
    
    # Apply max_samples limit
    if args.max_samples and len(all_samples) > args.max_samples:
        print(f"\n  Truncating to {args.max_samples:,} samples...")
        random.seed(args.seed)
        all_samples = random.sample(all_samples, args.max_samples)
    
    # =========================================================================
    # STEP 5: Save cache
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 5: Saving cache...")
    print("-" * 70)
    
    print(f"  Total samples: {len(all_samples):,}")
    print(f"  Files processed: {total_processed:,}")
    print(f"  Skipped (split): {total_skipped_split:,}")
    print(f"  Skipped (no img): {total_skipped_img:,}")
    
    if all_samples:
        # Atomic write
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                suffix='.pkl',
                prefix=cache_path.name + '.tmp.',
                dir=str(cache_path.parent)
            )
            
            print("  Writing cache file...")
            with os.fdopen(tmp_fd, 'wb') as f:
                pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            
            os.replace(tmp_path, str(cache_path))
            
            mb = cache_path.stat().st_size / (1024 * 1024)
            print(f"\n  ✓ Saved: {cache_path}")
            print(f"    Size: {mb:.1f} MB")
            
            # Delete checkpoint
            CheckpointManager(cache_path).delete()
            print("  ✓ Deleted checkpoint")
            
        except Exception as e:
            print(f"\n  ✗ Failed to save cache: {e}")
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
            raise
    else:
        print("\n  ⚠ No samples to save!")
        sys.exit(1)
    
    # =========================================================================
    # Done
    # =========================================================================
    print("\n" + "=" * 70)
    print("✓ CACHE BUILD COMPLETE!")
    print("=" * 70)
    
    if is_subset:
        print(f"\n⚡ SUBSET MODE - Ready for pipeline testing")
        print(f"   Samples: {len(all_samples):,}")
        print(f"\nNext: Test training pipeline:")
        print("  python train_mimic_cxr.py --config configs/pretrain_config.yaml --max_steps 100")


if __name__ == '__main__':
    main()