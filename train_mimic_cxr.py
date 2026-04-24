#!/usr/bin/env python3
"""
MIMIC-CXR VQA Training Script

Complete training pipeline with:
- Weights & Biases logging
- Hugging Face Hub checkpointing
- DeepSpeed distributed training support
- Mixed precision training
- Gradient checkpointing

Based on methodology.md specifications.

IMPORTANT: Run analyze_data.py FIRST to verify data is ready before training!

Usage:
    # Step 1: Analyze data (required)
    python analyze_data.py --mimic_cxr_path /path/to/MIMIC-CXR-JPG --mimic_qa_path /path/to/QA
    
    # Step 2: Train model (only if analysis passes)
    python train_mimic_cxr.py --config configs/default_config.yaml
    
    # Step 3: Evaluate model
    python evaluate.py --model_path ./checkpoints/best_model --config configs/default_config.yaml

Environment Variables (can be set in ~/.env):
    HF_TOKEN        - HuggingFace API token for model upload
    WANDB_API_KEY   - Weights & Biases API key for experiment tracking
    WANDB_ENTITY    - Wandb username or team name
    WANDB_PROJECT   - Wandb project name
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

import gc
import ctypes

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CPU memory reclamation helper (Linux only)
# On Linux glibc, freed memory is NOT returned to the OS by default. Over
# thousands of training steps the RSS grows monotonically even though the
# Python heap has plenty of free space.  Calling malloc_trim(0) after
# gc.collect() forces glibc to release free pages back to the OS.
# ---------------------------------------------------------------------------
_LIBC = None
try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:
    pass  # Not Linux — skip


def _reclaim_cpu_memory():
    """Force Python GC + glibc malloc_trim to return memory to the OS."""
    gc.collect()
    if _LIBC is not None:
        _LIBC.malloc_trim(0)

# DeepSpeed import (optional)
try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("DeepSpeed not available. Install with: pip install deepspeed")

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from ~/.env file."""
    env_paths = [
        Path.home() / '.env',
        Path('.env'),
        Path('~/.env').expanduser(),
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
            print(f"Loaded environment from: {env_path}")
            return True
    return False

# Load .env file before other imports
load_env_file()

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available. Install with: pip install wandb")

try:
    from huggingface_hub import HfApi, create_repo, upload_folder, login as hf_login
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("huggingface_hub not available. Install with: pip install huggingface-hub")

# Local imports
from configs.mimic_cxr_config import (
    MIMICCXRVQAConfig,
    get_default_config,
    load_config_from_file
)
from data.mimic_cxr_dataset import MIMICCXRVQADataset, create_dataloader
# V2 model: Qwen2.5-VL backbone + SG soft tokens + grounding refinement head.
# The legacy MIMICCXRVQAModel was retired; SSGVQANetV2 lives in
# models/ssg_vqa_net_v2.py and exposes the same forward output keys consumed
# by training.loss.MultiTaskLoss and training.metrics.VQAMetrics.
from models import SSGVQANetV2
from training.loss import MultiTaskLoss
from training.metrics import VQAMetrics
from utils.hardware_utils import (
    detect_hardware,
    print_hardware_info,
    optimize_for_hardware,
    set_optimal_environment,
    get_deepspeed_config_for_hardware,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # PERFORMANCE: benchmark=True auto-tunes convolution algorithms for fixed input sizes (224×224)
    # This alone can give 1.5-2× speedup for conv-heavy models like ConvNeXt.
    # deterministic=False allows faster non-deterministic algorithms.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def setup_distributed(args) -> tuple[int, int, bool]:
    """
    Setup distributed training environment.
    
    Returns:
        local_rank: GPU index on this node
        world_size: Total number of GPUs across all nodes
        is_distributed: Whether distributed training is enabled
    """
    # Check for DeepSpeed environment
    if args.use_deepspeed and DEEPSPEED_AVAILABLE:
        deepspeed.init_distributed()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        is_distributed = world_size > 1
        logger.info(f"DeepSpeed distributed initialized: rank {local_rank}/{world_size}")
        return local_rank, world_size, is_distributed
    
    # Check for torchrun/DDP environment
    if args.use_ddp or 'RANK' in os.environ:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        if world_size > 1:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
            is_distributed = True
            logger.info(f"DDP distributed initialized: rank {local_rank}/{world_size}")
        else:
            is_distributed = False
        
        return local_rank, world_size, is_distributed
    
    # Single GPU / CPU fallback
    return 0, 1, False


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(local_rank: int) -> bool:
    """Check if this is the main process (for logging, saving, etc.)."""
    return local_rank == 0


def get_effective_batch_size(config, world_size: int) -> int:
    """Calculate effective batch size with gradient accumulation."""
    return (
        config.training.batch_size_per_gpu 
        * world_size 
        * config.training.gradient_accumulation_steps
    )


def print_training_info(config, world_size: int, model, device):
    """Print training configuration summary."""
    effective_batch = get_effective_batch_size(config, world_size)
    
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION (per methodology Section 11)")
    logger.info("=" * 60)
    logger.info(f"  Device:                     {device}")
    logger.info(f"  World size (GPUs):          {world_size}")
    logger.info(f"  Batch size per GPU:         {config.training.batch_size_per_gpu}")
    logger.info(f"  Gradient accumulation:      {config.training.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size:       {effective_batch}")
    logger.info(f"  Mixed precision (FP16):     {config.training.fp16}")
    logger.info(f"  Gradient checkpointing:     {config.training.gradient_checkpointing}")
    logger.info(f"  DeepSpeed ZeRO:             {getattr(config.deepspeed, 'enabled', False)}")
    logger.info(f"  Learning rate:              {config.training.learning_rate}")
    logger.info(f"  Warmup ratio:               {config.training.warmup_ratio}")
    logger.info(f"  Weight decay:               {config.training.weight_decay}")
    logger.info(f"  Dataloader workers:         {config.training.dataloader_num_workers}")
    logger.info("=" * 60)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters:           {total_params:,}")
    logger.info(f"  Trainable parameters:       {trainable_params:,}")
    logger.info("=" * 60)


def setup_huggingface(config: MIMICCXRVQAConfig):
    """Setup HuggingFace authentication."""
    if not HF_HUB_AVAILABLE:
        return
    
    hf_token = os.environ.get('HF_TOKEN') or config.training.hub_token
    
    if hf_token:
        try:
            hf_login(token=hf_token, add_to_git_credential=True)
            logger.info("HuggingFace authentication successful")
        except Exception as e:
            logger.warning(f"HuggingFace login failed: {e}")
    else:
        logger.warning("HF_TOKEN not found. Set it in ~/.env or pass --hub_token")


def init_wandb(config: MIMICCXRVQAConfig) -> Optional[Any]:
    """Initialize Weights & Biases tracking."""
    if not WANDB_AVAILABLE or not config.wandb.enabled:
        return None
    
    # Get wandb settings from environment or config
    wandb_api_key = os.environ.get('WANDB_API_KEY')
    wandb_entity = os.environ.get('WANDB_ENTITY') or config.wandb.entity
    wandb_project = os.environ.get('WANDB_PROJECT') or config.wandb.project
    
    # Login if API key available
    if wandb_api_key:
        try:
            wandb.login(key=wandb_api_key, relogin=True)
            logger.info("Wandb authentication successful")
        except Exception as e:
            logger.warning(f"Wandb login failed: {e}")
    
    run_name = config.wandb.name or f"ssg-vqa-{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity or None,
        name=run_name,
        group=config.wandb.group,
        tags=config.wandb.tags,
        notes=config.wandb.notes,
        config=config.to_dict(),
        resume="allow",
        save_code=True,
    )
    
    logger.info(f"Wandb run started: {wandb_project}/{run_name}")
    
    # Define custom metrics
    wandb.define_metric("train/loss", summary="min")
    wandb.define_metric("train/vqa_loss", summary="min")
    wandb.define_metric("train/chexpert_loss", summary="min")
    wandb.define_metric("val/accuracy", summary="max")
    wandb.define_metric("val/binary_accuracy", summary="max")
    wandb.define_metric("val/category_f1", summary="max")
    
    return run


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    global_step: int,
    metrics: Dict[str, float],
    config: MIMICCXRVQAConfig,
    is_best: bool = False
):
    """Save model checkpoint."""
    checkpoint_dir = Path(config.training.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config.to_dict(),
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint-{global_step}"
    checkpoint_path.mkdir(exist_ok=True)
    torch.save(checkpoint, checkpoint_path / "pytorch_model.bin")
    
    # Save config
    with open(checkpoint_path / "config.json", 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Save training metadata
    metadata = {
        'epoch': epoch,
        'global_step': global_step,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
    }
    with open(checkpoint_path / "training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / "best_model"
        best_path.mkdir(exist_ok=True)
        torch.save(checkpoint, best_path / "pytorch_model.bin")
        with open(best_path / "config.json", 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Saved best model to {best_path}")
    
    # Clean up old checkpoints
    _cleanup_old_checkpoints(checkpoint_dir, config.training.save_total_limit)


def _cleanup_old_checkpoints(checkpoint_dir: Path, keep_last: int = 5):
    """Remove old checkpoints, keeping only the most recent ones."""
    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1])
    )
    
    # Remove old checkpoints
    for checkpoint in checkpoints[:-keep_last]:
        import shutil
        shutil.rmtree(checkpoint)
        logger.info(f"Removed old checkpoint: {checkpoint}")


def push_to_hub(
    model: nn.Module,
    config: MIMICCXRVQAConfig,
    metrics: Dict[str, float],
    commit_message: str = "Training checkpoint"
):
    """Push model to Hugging Face Hub."""
    if not HF_HUB_AVAILABLE:
        logger.warning("huggingface_hub not available. Skipping push to hub.")
        return
    
    if not config.training.hub_model_id:
        logger.warning("hub_model_id not set. Skipping push to hub.")
        return
    
    try:
        api = HfApi()
        
        # Create repo if it doesn't exist
        try:
            create_repo(
                config.training.hub_model_id,
                private=config.training.hub_private_repo,
                exist_ok=True
            )
        except Exception as e:
            logger.warning(f"Could not create repo: {e}")
        
        # Save model locally first
        save_dir = Path(config.training.output_dir) / "hub_upload"
        save_dir.mkdir(exist_ok=True)
        
        torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
        
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Generate model card
        model_card = _generate_model_card(config, metrics)
        with open(save_dir / "README.md", 'w') as f:
            f.write(model_card)
        
        # Upload
        upload_folder(
            folder_path=str(save_dir),
            repo_id=config.training.hub_model_id,
            commit_message=commit_message,
        )
        
        logger.info(f"Pushed model to {config.training.hub_model_id}")
        
    except Exception as e:
        logger.error(f"Failed to push to hub: {e}")


def _generate_model_card(config: MIMICCXRVQAConfig, metrics: Dict[str, float]) -> str:
    """Generate Hugging Face model card."""
    # Precompute metric strings to avoid invalid f-string format specifiers
    acc = metrics.get('accuracy')
    binary_acc = metrics.get('binary_accuracy')
    cat_f1 = metrics.get('category_f1')
    chex_auroc = metrics.get('chexpert_auroc')

    acc_str = f"{acc:.3f}" if isinstance(acc, float) else "N/A"
    binary_acc_str = f"{binary_acc:.3f}" if isinstance(binary_acc, float) else "N/A"
    cat_f1_str = f"{cat_f1:.3f}" if isinstance(cat_f1, float) else "N/A"
    chex_auroc_str = f"{chex_auroc:.3f}" if isinstance(chex_auroc, float) else "N/A"

    mixed_precision = 'Enabled' if config.training.fp16 else 'Disabled'

    card = f"""---
language: en
license: mit
library_name: pytorch
tags:
    - medical-vqa
    - chest-x-ray
    - scene-graph
    - visual-question-answering
    - mimic-cxr
datasets:
    - mimic-cxr-jpg
    - mimic-ext-cxr-qba
---

# SSG-VQA-Net for MIMIC-CXR Visual Question Answering

## Model Description

This model adapts the SSG-VQA-Net architecture for chest X-ray visual question answering
using the MIMIC-CXR-JPG images and MIMIC-Ext-CXR-QBA question-answer pairs.

### Architecture

- **Visual Backbone**: ConvNeXt-Base (pre-trained on ImageNet-22k)
- **Text Encoder**: Bio+ClinicalBERT (medical domain)
- **Scene Graph**: Expanded 134-dim embeddings ({config.model.num_regions} regions, {config.model.num_entities} entities)
- **Fusion**: Scene-Embedded Interaction Module (SIM)
- **Answer Heads**: Multi-head (Binary, Category, Region, Severity)

## Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | {acc_str} |
| Binary Accuracy | {binary_acc_str} |
| Category F1 | {cat_f1_str} |
| CheXpert AUROC | {chex_auroc_str} |

## Training Details

- **Batch Size**: {config.training.batch_size_per_gpu} per GPU
- **Learning Rate**: {config.training.learning_rate}
- **Epochs**: {config.training.num_epochs}
- **Mixed Precision**: {mixed_precision}

## Usage

```python
from models import SSGVQANetV2

model = SSGVQANetV2(qwen_model_id="{config.training.hub_model_id}")
```

## Citation

```bibtex
@article{{ssg-vqa-mimic,
    title={{Scene Graph-Enhanced VQA for Chest X-Ray Analysis}},
    year={{2026}}
}}
```
"""

    return card


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    criterion: MultiTaskLoss,
    device: torch.device,
    epoch: int,
    config: MIMICCXRVQAConfig,
    scaler: Optional[GradScaler] = None,
    global_step: int = 0,
    local_rank: int = 0,
    use_deepspeed: bool = False
) -> tuple[float, int]:
    """
    Train for one epoch with gradient accumulation support.
    
    Implements methodology Section 11 optimizations:
    - Gradient accumulation (default: 4 steps)
    - Mixed precision (FP16)
    - Gradient clipping
    """
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    accumulated_loss = 0.0
    
    grad_accum_steps = config.training.gradient_accumulation_steps
    
    # Only show progress bar on main process
    if is_main_process(local_rank):
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            leave=False
        )
    else:
        progress_bar = dataloader
    
    # Zero gradients at start
    if not use_deepspeed:
        optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        images = batch['images'].to(device) if 'images' in batch else None
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
        scene_graphs = batch['scene_graphs']
        question_types = batch['question_types']

        # === V2 raw inputs for Qwen processor (added by collate_fn) ===
        pil_images = batch.get('pil_images')          # List[PIL.Image]
        questions = batch.get('questions')            # List[str]
        answer_texts = batch.get('answer_texts')      # List[str] structured
        # Pass scene_graphs=None to let v2 generate fresh from Qwen's ViT.
        # When ground-truth scene graphs from the dataset are preferred (Stage
        # 1 / debugging), keep batch['scene_graphs']. Default: trust dataset SG
        # during training, regenerate at inference (handled inside SSGVQANetV2).
        answer_idx = batch['answer_idx'].to(device)
        chexpert_labels = batch['chexpert_labels'].to(device)
        chexpert_mask = batch['chexpert_mask'].to(device)
        
        # === Image Metadata (from MIMIC-CXR-JPG) ===
        image_widths = batch.get('image_widths', torch.full((images.shape[0],), 224, dtype=torch.long)).to(device) if images is not None else None
        image_heights = batch.get('image_heights', torch.full((images.shape[0],), 224, dtype=torch.long)).to(device) if images is not None else None
        view_encodings = batch.get('view_encodings', None)
        if view_encodings is not None:
            view_encodings = view_encodings.to(device)
        
        # === Answer Generation Targets (from MIMIC-Ext-CXR-QBA) ===
        answer_ids = batch.get('answer_ids', None)
        if answer_ids is not None:
            answer_ids = answer_ids.to(device)
        reference_answers = batch.get('reference_answers', None)
        
        # === Visual Grounding Targets (from answer localization) ===
        gt_grounding_bboxes = batch.get('gt_grounding_bboxes', None)
        if gt_grounding_bboxes is not None:
            gt_grounding_bboxes = gt_grounding_bboxes.to(device)
        gt_pointing_valid = batch.get('gt_pointing_valid', None)
        if gt_pointing_valid is not None:
            gt_pointing_valid = gt_pointing_valid.to(device)
        
        # === Scene Graph Generation Targets (from scene_graph.json) ===
        # These are lists of tensors (variable length per sample)
        gt_sg_bboxes = batch.get('gt_sg_bboxes', None)
        gt_sg_entities = batch.get('gt_sg_entities', None)
        gt_sg_regions = batch.get('gt_sg_regions', None)
        
        # Move scene graph targets to device (they're lists of tensors or None)
        if gt_sg_bboxes is not None:
            gt_sg_bboxes = [t.to(device) if t is not None else None for t in gt_sg_bboxes]
        if gt_sg_entities is not None:
            gt_sg_entities = [t.to(device) if t is not None else None for t in gt_sg_entities]
        if gt_sg_regions is not None:
            gt_sg_regions = [t.to(device) if t is not None else None for t in gt_sg_regions]
        
        if images is None:
            continue
        
        # Prepare VQA targets - all heads get the same answer_idx
        # The loss function routes to correct head based on question_type
        vqa_targets = {
            'binary': answer_idx,
            'category': answer_idx,
            'region': answer_idx,
            'severity': answer_idx,
        }
        
        # DeepSpeed handles mixed precision and gradient accumulation internally
        if use_deepspeed:
            outputs = model(
                images=images,
                pil_images=pil_images,
                questions=questions,
                answer_texts=answer_texts,
                input_ids=input_ids,
                attention_mask=attention_mask,
                scene_graphs=scene_graphs,
                token_type_ids=token_type_ids,
                question_types=question_types,
                image_widths=image_widths,
                image_heights=image_heights,
                view_encodings=view_encodings,
                gt_bboxes=gt_sg_bboxes,
                gt_entities=gt_sg_entities,
                gt_regions=gt_sg_regions,
                # V2: thread grounding GT into the model so the refinement
                # head trains on a noised-GT init_bbox instead of a fixed
                # anchor — matches the inference distribution where init
                # comes from the LLM's parsed <box>.
                gt_grounding_bboxes=gt_grounding_bboxes,
                gt_pointing_valid=gt_pointing_valid,
                answer_ids=answer_ids,
            )
            
            loss, loss_dict = criterion(
                outputs,
                vqa_targets,
                chexpert_labels,
                chexpert_mask,
                question_types,
                answer_ids=answer_ids,
                gt_sg_bboxes=gt_sg_bboxes,
                gt_sg_entities=gt_sg_entities,
                gt_sg_regions=gt_sg_regions,
                gt_grounding_bboxes=gt_grounding_bboxes,
                gt_pointing_valid=gt_pointing_valid,
            )
            
            # DeepSpeed backward (handles gradient accumulation)
            model.backward(loss)
            model.step()
            
        # Standard PyTorch with gradient accumulation
        else:
            # Mixed precision forward pass
            if scaler is not None and config.training.fp16:
                with autocast('cuda'):
                    outputs = model(
                        images=images,
                        pil_images=pil_images,
                        questions=questions,
                        answer_texts=answer_texts,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        scene_graphs=scene_graphs,
                        token_type_ids=token_type_ids,
                        question_types=question_types,
                        image_widths=image_widths,
                        image_heights=image_heights,
                        view_encodings=view_encodings,
                        gt_bboxes=gt_sg_bboxes,
                        gt_entities=gt_sg_entities,
                        gt_regions=gt_sg_regions,
                        gt_grounding_bboxes=gt_grounding_bboxes,
                        gt_pointing_valid=gt_pointing_valid,
                        answer_ids=answer_ids,
                    )
                    
                    loss, loss_dict = criterion(
                        outputs,
                        vqa_targets,
                        chexpert_labels,
                        chexpert_mask,
                        question_types,
                        answer_ids=answer_ids,
                        gt_sg_bboxes=gt_sg_bboxes,
                        gt_sg_entities=gt_sg_entities,
                        gt_sg_regions=gt_sg_regions,
                        gt_grounding_bboxes=gt_grounding_bboxes,
                        gt_pointing_valid=gt_pointing_valid,
                    )
                    
                    # Scale loss for gradient accumulation
                    loss = loss / grad_accum_steps
                
                # Backward pass with scaled gradients
                scaler.scale(loss).backward()
                
            else:
                outputs = model(
                    images=images,
                    pil_images=pil_images,
                    questions=questions,
                    answer_texts=answer_texts,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    scene_graphs=scene_graphs,
                    token_type_ids=token_type_ids,
                    question_types=question_types,
                    image_widths=image_widths,
                    image_heights=image_heights,
                    view_encodings=view_encodings,
                    gt_bboxes=gt_sg_bboxes,
                    gt_entities=gt_sg_entities,
                    gt_regions=gt_sg_regions,
                    gt_grounding_bboxes=gt_grounding_bboxes,
                    gt_pointing_valid=gt_pointing_valid,
                    answer_ids=answer_ids,
                )
                
                loss, loss_dict = criterion(
                    outputs,
                    vqa_targets,
                    chexpert_labels,
                    chexpert_mask,
                    question_types,
                    answer_ids=answer_ids,
                    gt_sg_bboxes=gt_sg_bboxes,
                    gt_sg_entities=gt_sg_entities,
                    gt_sg_regions=gt_sg_regions,
                    gt_grounding_bboxes=gt_grounding_bboxes,
                    gt_pointing_valid=gt_pointing_valid,
                )
                
                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps
                loss.backward()
            
            # Accumulate loss for logging (unscaled)
            accumulated_loss += loss.item() * grad_accum_steps
            
            # Optimizer step at accumulation boundary
            if (batch_idx + 1) % grad_accum_steps == 0:
                if scaler is not None and config.training.fp16:
                    # Gradient clipping
                    if config.training.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            config.training.max_grad_norm
                        )
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if config.training.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            config.training.max_grad_norm
                        )
                    optimizer.step()
                
                # Update scheduler per optimizer step
                if scheduler is not None:
                    scheduler.step()
                
                # Zero gradients for next accumulation
                optimizer.zero_grad()
                
                # Track loss per actual step
                total_loss += accumulated_loss
                accumulated_loss = 0.0
                num_batches += 1
                global_step += 1
        
        # DeepSpeed tracks its own steps
        if use_deepspeed:
            step_loss = loss.item()  # scalar — no graph reference
            total_loss += step_loss
            num_batches += 1
            global_step += 1
        
        # ------------------------------------------------------------------
        # Snapshot scalar values BEFORE deleting tensors
        # ------------------------------------------------------------------
        _loss_display = loss.item() * (grad_accum_steps if not use_deepspeed else 1)
        _vqa_display = loss_dict.get("vqa_loss", 0)
        _chex_display = loss_dict.get("chexpert_loss", 0)
        if torch.is_tensor(_vqa_display):
            _vqa_display = _vqa_display.item()
        if torch.is_tensor(_chex_display):
            _chex_display = _chex_display.item()
        
        # Update progress bar (main process only)
        if is_main_process(local_rank) and hasattr(progress_bar, 'set_postfix'):
            _gen_display = loss_dict.get("generation_loss", 0)
            _sg_display = loss_dict.get("scene_graph_loss", 0)
            _grd_display = loss_dict.get("grounding_loss", 0)
            if torch.is_tensor(_gen_display):
                _gen_display = _gen_display.item()
            if torch.is_tensor(_sg_display):
                _sg_display = _sg_display.item()
            if torch.is_tensor(_grd_display):
                _grd_display = _grd_display.item()
            progress_bar.set_postfix({
                'loss': f'{_loss_display:.4f}',
                'vqa': f'{_vqa_display:.4f}',
                'chex': f'{_chex_display:.4f}',
                'gen': f'{_gen_display:.4f}',
                'sg': f'{_sg_display:.4f}',
                'grd': f'{_grd_display:.4f}',
                'step': global_step
            })
        
        # Log to wandb (main process only, every logging_steps actual steps)
        if (is_main_process(local_rank) and 
            WANDB_AVAILABLE and 
            config.wandb.enabled and 
            global_step > 0 and 
            global_step % config.training.logging_steps == 0):
            
            # Get learning rate
            if use_deepspeed:
                lr = model.get_lr()[0] if hasattr(model, 'get_lr') else config.training.learning_rate
            else:
                lr = optimizer.param_groups[0]['lr']
            
            # Extract all loss components
            def get_loss_val(key):
                val = loss_dict.get(key, 0)
                return val.item() if torch.is_tensor(val) else val
            
            log_dict = {
                'train/loss': loss.item() * (grad_accum_steps if not use_deepspeed else 1),
                # --- Top-level task losses ---
                'train/vqa_loss': get_loss_val('vqa_loss'),
                'train/chexpert_loss': get_loss_val('chexpert_loss'),
                'train/generation_loss': get_loss_val('generation_loss'),
                'train/scene_graph_loss': get_loss_val('scene_graph_loss'),
                'train/grounding_loss': get_loss_val('grounding_loss'),
                # --- Per-head VQA sub-losses ---
                'train/vqa_binary_loss': get_loss_val('vqa_binary_loss'),
                'train/vqa_category_loss': get_loss_val('vqa_category_loss'),
                'train/vqa_region_loss': get_loss_val('vqa_region_loss'),
                'train/vqa_severity_loss': get_loss_val('vqa_severity_loss'),
                # --- Scene graph sub-losses ---
                'train/sg_entity_loss': get_loss_val('sg_entity_loss'),
                'train/sg_region_loss': get_loss_val('sg_region_loss'),
                'train/sg_bbox_loss': get_loss_val('sg_bbox_loss'),
                'train/sg_objectness_loss': get_loss_val('sg_objectness_loss'),
                # --- Grounding sub-losses ---
                'train/grounding_bbox_loss': get_loss_val('grounding_bbox_loss'),
                'train/grounding_pointing_loss': get_loss_val('grounding_pointing_loss'),
                # --- Training dynamics ---
                'train/learning_rate': lr,
                'train/epoch': epoch,
                'global_step': global_step,
            }
            
            # FP16 loss scale (useful for debugging NaN/overflow)
            if use_deepspeed and hasattr(model, 'optimizer') and hasattr(model.optimizer, 'cur_scale'):
                log_dict['train/fp16_loss_scale'] = model.optimizer.cur_scale
            
            # GPU memory usage
            if torch.cuda.is_available():
                log_dict['train/gpu_mem_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
                log_dict['train/gpu_mem_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
            
            # Log mHC metrics if available
            mhc_metrics = outputs.get('mhc_metrics', {})
            for k, v in mhc_metrics.items():
                log_dict[f'train/mhc_{k}'] = v
            
            wandb.log(log_dict)
        
        # ------------------------------------------------------------------
        # FREE large per-step objects to avoid holding references across
        # iterations.  Without this, Python's refcount collector may keep
        # the entire autograd graph alive until the *next* iteration
        # re-assigns these names, doubling peak RSS.
        # ------------------------------------------------------------------
        del outputs, loss, loss_dict
        del images, input_ids, attention_mask, token_type_ids
        del scene_graphs, vqa_targets
        
        # ------------------------------------------------------------------
        # CRITICAL: Reclaim CPU RSS every 50 steps.
        # Python/glibc never returns freed heap pages to the OS on their
        # own.  gc.collect() + malloc_trim(0) forces this, preventing the
        # monotonic RSS growth that leads to OOM-kill at ~step 468.
        # ------------------------------------------------------------------
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
            _reclaim_cpu_memory()
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    config: MIMICCXRVQAConfig
) -> Dict[str, float]:
    """Validate model with full feature evaluation."""
    model.eval()
    
    metrics_calculator = VQAMetrics()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)
    
    for batch in progress_bar:
        # Move data to device
        images = batch['images'].to(device) if 'images' in batch else None
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
        scene_graphs = batch['scene_graphs']
        question_types = batch['question_types']
        answer_idx = batch['answer_idx'].to(device)
        chexpert_labels = batch['chexpert_labels'].to(device)
        chexpert_mask = batch['chexpert_mask'].to(device)

        # === V2 raw inputs for Qwen processor ===
        # In validation we omit answer_texts so SSGVQANetV2 runs free generation.
        pil_images = batch.get('pil_images')
        questions = batch.get('questions')

        # === Image Metadata (from MIMIC-CXR-JPG) ===
        view_encodings = batch.get('view_encodings', None)
        if view_encodings is not None:
            view_encodings = view_encodings.to(device)
        
        # === Reference answers for generation metrics ===
        reference_answers = batch.get('reference_answers', None)
        
        # === Grounding targets for grounding metrics ===
        gt_grounding_bboxes = batch.get('gt_grounding_bboxes', None)
        if gt_grounding_bboxes is not None:
            gt_grounding_bboxes = gt_grounding_bboxes.cpu().numpy()
        gt_pointing_valid = batch.get('gt_pointing_valid', None)
        if gt_pointing_valid is not None:
            gt_pointing_valid = gt_pointing_valid.cpu().numpy()
        
        # === Scene graph targets for SG metrics ===
        gt_sg_entities = batch.get('gt_sg_entities', None)
        gt_sg_regions = batch.get('gt_sg_regions', None)
        gt_sg_bboxes = batch.get('gt_sg_bboxes', None)
        
        # Convert to numpy for metrics
        if gt_sg_entities is not None:
            gt_sg_entities = [t.cpu().numpy() if t is not None else None for t in gt_sg_entities]
        if gt_sg_regions is not None:
            gt_sg_regions = [t.cpu().numpy() if t is not None else None for t in gt_sg_regions]
        if gt_sg_bboxes is not None:
            gt_sg_bboxes = [t.cpu().numpy() if t is not None else None for t in gt_sg_bboxes]
        
        vqa_targets = {
            'binary': answer_idx,
            'category': answer_idx,
            'region': answer_idx,
            'severity': answer_idx,
        }
        
        if images is None:
            continue
        
        # Forward pass (inference mode - no answer_texts/answer_ids → free generation)
        outputs = model(
            images=images,
            pil_images=pil_images,
            questions=questions,
            input_ids=input_ids,
            attention_mask=attention_mask,
            scene_graphs=scene_graphs,
            token_type_ids=token_type_ids,
            question_types=question_types,
            view_encodings=view_encodings,
        )
        
        loss, _ = criterion(
            outputs,
            vqa_targets,
            chexpert_labels,
            chexpert_mask,
            question_types
        )
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update metrics with all available data
        metrics_calculator.update(
            outputs,
            vqa_targets,
            chexpert_labels,
            chexpert_mask,
            question_types,
            reference_answers=reference_answers,
            gt_sg_entities=gt_sg_entities,
            gt_sg_regions=gt_sg_regions,
            gt_sg_bboxes=gt_sg_bboxes,
            gt_grounding_bboxes=gt_grounding_bboxes,
            gt_pointing_valid=gt_pointing_valid,
        )
    
    # Compute final metrics
    metrics = metrics_calculator.compute()
    metrics['loss'] = total_loss / max(num_batches, 1)
    
    return metrics


def check_data_readiness(config) -> bool:
    """
    Check if data analysis has been run and data is ready.
    
    Returns:
        True if data is ready, False otherwise
    """
    analysis_report_path = Path('./analysis_output/analysis_report.json')
    
    if not analysis_report_path.exists():
        logger.error("=" * 60)
        logger.error("DATA ANALYSIS NOT FOUND!")
        logger.error("=" * 60)
        logger.error("\nPlease run data analysis first:")
        logger.error(f"  python analyze_data.py \\")
        logger.error(f"    --mimic_cxr_path {config.data.mimic_cxr_jpg_path} \\")
        logger.error(f"    --mimic_qa_path {config.data.mimic_ext_cxr_qba_path}")
        logger.error("\nThen run training again.")
        logger.error("=" * 60)
        return False
    
    try:
        with open(analysis_report_path) as f:
            report = json.load(f)
        
        is_ready = report.get('summary', {}).get('is_ready', False)
        
        if not is_ready:
            logger.error("=" * 60)
            logger.error("DATA NOT READY FOR TRAINING!")
            logger.error("=" * 60)
            
            issues = report.get('issues', [])
            if issues:
                logger.error("\nCritical issues found:")
                for issue in issues:
                    logger.error(f"  • {issue}")
            
            warnings = report.get('warnings', [])
            if warnings:
                logger.warning("\nWarnings:")
                for warning in warnings:
                    logger.warning(f"  • {warning}")
            
            logger.error("\nPlease resolve issues and re-run analyze_data.py")
            logger.error("=" * 60)
            return False
        
        # Data is ready - show summary
        summary = report.get('summary', {})
        logger.info("=" * 60)
        logger.info("DATA READINESS CHECK: PASSED ✓")
        logger.info("=" * 60)
        logger.info(f"  Images:       {summary.get('total_images', 0):,}")
        logger.info(f"  QA Pairs:     {summary.get('total_qa_pairs', 0):,}")
        logger.info(f"  Scene Graphs: {summary.get('total_scene_graphs', 0):,}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error reading analysis report: {e}")
        logger.error("Please re-run analyze_data.py")
        return False


def main(args):
    """Main training function with distributed training support."""
    # Set optimal environment variables
    set_optimal_environment()
    
    # Load config
    if args.config and os.path.exists(args.config):
        config = load_config_from_file(args.config)
    else:
        config = get_default_config()
    
    # ========================================
    # HARDWARE AUTO-DETECTION AND OPTIMIZATION
    # ========================================
    if args.auto_optimize:
        hardware_info = detect_hardware()
        
        # Print hardware info (before distributed init, so only once)
        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            print_hardware_info(hardware_info)
        
        # Apply optimal settings to config
        config = optimize_for_hardware(config, auto_detect=True)
        
        # Auto-enable DeepSpeed for multi-GPU if not explicitly set
        if not args.use_deepspeed and not args.use_ddp:
            if hardware_info.num_gpus > 1 and DEEPSPEED_AVAILABLE:
                args.use_deepspeed = True
                logger.info(f"Auto-enabled DeepSpeed ZeRO-{hardware_info.deepspeed_stage} for {hardware_info.num_gpus} GPUs")
            elif hardware_info.num_gpus > 1:
                # IMPORTANT:
                # DDP requires launching with torchrun (WORLD_SIZE>1) to initialize
                # the default process group. If we are running via plain `python`,
                # auto-enabling DDP will crash. Prefer DataParallel in that case.
                logger.info(
                    f"Multiple GPUs detected ({hardware_info.num_gpus}) but DeepSpeed is unavailable. "
                    "Using DataParallel fallback (no distributed init). "
                    "For true DDP, launch with: torchrun --nproc_per_node=<num_gpus> train_mimic_cxr.py --use_ddp ..."
                )
    else:
        hardware_info = None
    
    # Override config with command line args (takes precedence over auto-detect)
    if args.mimic_cxr_path:
        config.data.mimic_cxr_jpg_path = args.mimic_cxr_path
    if args.mimic_qa_path:
        config.data.mimic_ext_cxr_qba_path = args.mimic_qa_path
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.batch_size:
        config.training.batch_size_per_gpu = args.batch_size
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.hub_model_id:
        config.training.hub_model_id = args.hub_model_id
    if args.wandb_project:
        config.wandb.project = args.wandb_project
    if args.disable_wandb:
        config.wandb.enabled = False
    
    # Force disable gradient checkpointing if flag is set
    if args.no_gradient_checkpointing:
        config.training.gradient_checkpointing = False
        logger.info("Gradient checkpointing FORCE DISABLED via --no_gradient_checkpointing")

    # Force disable FP16 if flag is set (required when CUDA toolkit can't compile DeepSpeed FP16 ops)
    if args.no_fp16:
        config.training.fp16 = False
        logger.info("FP16 FORCE DISABLED via --no_fp16")

    # Override phase from CLI
    if args.phase:
        config.training.phase = args.phase
        logger.info(f"Training phase set to '{args.phase}' via --phase")

    # ----- Phase detection and dataset quality alignment -----
    phase = getattr(config.training, 'phase', 'finetune')
    phase = phase.lower() if isinstance(phase, str) else 'finetune'
    if phase == 'pretrain':
        # Pretraining should prefer B-grade (noisy + broad) and a higher LR
        config.data.quality_grade = getattr(config.data, 'quality_grade', 'B') or 'B'
        # If the config uses the default LR, bump for pretraining
        if not args.learning_rate and (not config.training.learning_rate or config.training.learning_rate == 5e-5):
            config.training.learning_rate = 1e-4
        # Default output dir suffix for clarity
        if not args.output_dir or args.output_dir == './checkpoints/mimic-cxr-vqa':
            config.training.output_dir = os.path.join(config.training.output_dir, 'pretrain')
    elif phase == 'finetune':
        # Finetuning should prefer A-grade (clean) and a lower LR
        config.data.quality_grade = getattr(config.data, 'quality_grade', 'A') or 'A'
        if not args.learning_rate and (not config.training.learning_rate or config.training.learning_rate == 5e-5):
            config.training.learning_rate = 5e-5
        if not args.output_dir or args.output_dir == './checkpoints/mimic-cxr-vqa':
            config.training.output_dir = os.path.join(config.training.output_dir, 'finetune')

    
    # Enable DeepSpeed if requested and available
    use_deepspeed = args.use_deepspeed and DEEPSPEED_AVAILABLE
    use_ddp = args.use_ddp and not use_deepspeed
    
    # Setup distributed training
    local_rank, world_size, is_distributed = setup_distributed(args)
    
    # Only main process should check data and log
    if is_main_process(local_rank):
        # Check data readiness (unless skipped)
        if not args.skip_data_check:
            if not check_data_readiness(config):
                logger.info("\nTo skip this check (not recommended), use --skip_data_check")
                sys.exit(1)
        else:
            logger.warning("Skipping data readiness check (--skip_data_check)")
    
    # Sync all processes after data check
    if is_distributed:
        dist.barrier() if dist.is_initialized() else None
    
    # Setup
    seed_everything(config.training.seed + local_rank)  # Different seed per rank for data augmentation
    
    # Device setup
    if torch.cuda.is_available():
        if is_distributed:
            device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(device)
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    if is_main_process(local_rank):
        logger.info(f"Using device: {device}")
        logger.info(f"World size (total GPUs): {world_size}")
        logger.info(f"Local rank: {local_rank}")
        logger.info(f"Distributed training: {is_distributed}")
        logger.info(f"DeepSpeed enabled: {use_deepspeed}")
        logger.info(f"DDP enabled: {use_ddp}")
    
    # Initialize wandb (main process only)
    wandb_run = None
    if is_main_process(local_rank):
        wandb_run = init_wandb(config)
        
        # Create output directory
        os.makedirs(config.training.output_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(config.training.output_dir, 'config.json'), 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    
    # Load datasets with caching support
    # Cache directory for instant loading on distributed training
    cache_dir = getattr(config.data, 'cache_dir', '.cache/dataset_samples')
    
    # ------------------------------------------------------------------
    # AUTO-COMPUTE safe max_samples if not provided by user.
    # Each cached sample is ~15 KB (Python dict overhead).  With 4 ranks
    # each loading 2M samples, that's ~128 GB — leaving almost no room
    # for training.  We cap samples so data uses ≤35% of total RAM.
    # ------------------------------------------------------------------
    if args.max_samples is None:
        try:
            import psutil
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            data_budget_gb = total_ram_gb * 0.35
            system_overhead_gb = 15  # OS + Python + libraries
            per_rank_budget_gb = max(2, (data_budget_gb - system_overhead_gb) / max(1, world_size))
            BYTES_PER_SAMPLE = 15 * 1024  # empirical for MIMIC-Ext-CXR-QBA
            auto_max = int(per_rank_budget_gb * 1024**3 / BYTES_PER_SAMPLE)
            auto_max = max(50_000, auto_max)
            args.max_samples = auto_max
            if is_main_process(local_rank):
                logger.info(
                    f"Auto-computed max_samples={auto_max:,} per rank "
                    f"(RAM={total_ram_gb:.0f}GB, {world_size} ranks, "
                    f"~{per_rank_budget_gb:.1f}GB/rank for data)"
                )
        except ImportError:
            if is_main_process(local_rank):
                logger.warning(
                    "psutil not installed — cannot auto-limit samples. "
                    "If OOM, pass --max_samples 500000"
                )
    elif is_main_process(local_rank):
        logger.info(f"Using user-specified max_samples={args.max_samples:,}")
    
    if is_main_process(local_rank):
        logger.info("Loading datasets (using cache if available)...")
    
    train_dataset = MIMICCXRVQADataset(
        mimic_cxr_path=config.data.mimic_cxr_jpg_path,
        mimic_qa_path=config.data.mimic_ext_cxr_qba_path,
        split='train',
        tokenizer_name=config.model.text_encoder,
        max_question_length=config.model.max_question_length,
        quality_grade=config.data.quality_grade,
        view_filter=config.data.view_filter,
        question_types=config.data.question_types if config.data.question_types else None,
        chexpert_labels_path=config.data.chexpert_labels_path if config.data.chexpert_labels_path else None,
        max_samples=args.max_samples,
        cache_dir=cache_dir,
        use_cache=True,
        prebuilt_cache_path=args.prebuilt_cache_train,
    )
    
    # Barrier to ensure all processes have loaded/cached train data
    if is_distributed:
        dist.barrier()
    
    val_dataset = MIMICCXRVQADataset(
        mimic_cxr_path=config.data.mimic_cxr_jpg_path,
        mimic_qa_path=config.data.mimic_ext_cxr_qba_path,
        split='validate',
        tokenizer_name=config.model.text_encoder,
        max_question_length=config.model.max_question_length,
        quality_grade=config.data.quality_grade,
        view_filter=config.data.view_filter,
        question_types=config.data.question_types if config.data.question_types else None,
        chexpert_labels_path=config.data.chexpert_labels_path if config.data.chexpert_labels_path else None,
        max_samples=args.max_samples // 10 if args.max_samples else None,
        cache_dir=cache_dir,
        use_cache=True,
        prebuilt_cache_path=args.prebuilt_cache_val,
    )
    
    # Barrier to ensure all processes have loaded/cached val data
    if is_distributed:
        dist.barrier()
    
    # ------------------------------------------------------------------
    # Reclaim memory after dataset loading.  The pickle.load() creates
    # all 2M dicts, then we truncate to max_samples.  The freed dicts
    # are unreachable but glibc hasn't returned the pages to the OS.
    # ------------------------------------------------------------------
    _reclaim_cpu_memory()
    
    if is_main_process(local_rank):
        try:
            import psutil
            mem = psutil.virtual_memory()
            logger.info(
                f"Train samples: {len(train_dataset)} | "
                f"Val samples: {len(val_dataset)} | "
                f"RAM after load: {mem.used/1024**3:.1f}GB / {mem.total/1024**3:.1f}GB "
                f"({mem.percent}%)"
            )
        except ImportError:
            logger.info(f"Train samples: {len(train_dataset)}")
            logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create distributed samplers for multi-GPU training
    train_sampler = None
    val_sampler = None
    
    if is_distributed and not use_deepspeed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
            seed=config.training.seed
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=False
        )
    logger.info(f"Using sampler {train_sampler} {val_sampler}")
    # Create dataloaders with distributed samplers
    # Optimized for high-CPU machines (48 vCPU = 10 workers per GPU * 4 GPUs)
    prefetch_factor = getattr(config.training, 'dataloader_prefetch_factor', 4)
    
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size_per_gpu,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        num_workers=config.training.dataloader_num_workers,
        pin_memory=config.training.dataloader_pin_memory,
        sampler=train_sampler,
        prefetch_factor=prefetch_factor,
        drop_last=True  # Drop incomplete batches for stable training
    )
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=config.training.batch_size_per_gpu * 2,
        shuffle=False,
        num_workers=config.training.dataloader_num_workers,
        pin_memory=config.training.dataloader_pin_memory,
        sampler=val_sampler,
        prefetch_factor=prefetch_factor
    )
    
    # Initialize model
    if is_main_process(local_rank):
        logger.info("Initializing model...")
    
    # ------------------------------------------------------------------
    # V2 model construction — Qwen2.5-VL + LoRA + SG soft tokens
    # ------------------------------------------------------------------
    # Compute-capability guard: Turing (cc<8.0, e.g. RTX 8000 / V100) lacks
    # bf16 and FlashAttention-2 — force fp16 compute and 4-bit NF4 weights.
    _phase = getattr(config.training, 'phase', 'pretrain').lower()
    _gpu0_cc = (0, 0)
    if torch.cuda.is_available():
        _props = torch.cuda.get_device_properties(0)
        _gpu0_cc = (_props.major, _props.minor)
    _force_qlora = _gpu0_cc < (8, 0)
    _qwen_dtype = torch.float16 if _force_qlora else torch.bfloat16

    _qwen_id = getattr(config.model, 'qwen_model_id', 'Qwen/Qwen2.5-VL-7B-Instruct')
    _use_quant = bool(getattr(config.model, 'use_quantization', _force_qlora))
    _lora_rank = int(getattr(config.model, 'lora_rank', 16))
    _num_sg_tokens = int(getattr(config.model, 'num_sg_tokens', 8))

    if is_main_process(local_rank):
        logger.info(
            f"V2 model: {_qwen_id} | cc={_gpu0_cc} | quantization={_use_quant} "
            f"| lora_rank={_lora_rank} | num_sg_tokens={_num_sg_tokens} "
            f"| dtype={_qwen_dtype} | training_mode={_phase}"
        )
        if _force_qlora and not _use_quant:
            logger.warning(
                "Compute capability < 8.0 detected but use_quantization=False. "
                "Forcing QLoRA (4-bit NF4) for memory safety."
            )

    model = SSGVQANetV2(
        qwen_model_id=_qwen_id,
        use_quantization=_use_quant or _force_qlora,
        lora_rank=_lora_rank,
        num_sg_tokens=_num_sg_tokens,
        num_regions=config.model.num_regions,
        num_entities=config.model.num_entities,
        num_binary=config.model.num_binary_classes,
        num_category=config.model.num_category_classes,
        num_region_classes=config.model.num_region_classes,
        num_severity=config.model.num_severity_classes,
        training_mode=_phase if _phase in {'sg_only', 'alignment', 'pretrain', 'finetune', 'rl'} else 'pretrain',
        torch_dtype=_qwen_dtype,
    )

    # Dtype consistency: Qwen runs in fp16 (or bf16 on Ampere+) but the v2
    # custom modules (SG encoder/projector, grounding head, aux heads, mHC)
    # initialize in fp32 by default. Without an explicit cast every cross-
    # module hop pays an implicit upcast/downcast — ~10-15% throughput tax
    # on Turing cards. The forward path's _ensure_dtype helpers keep things
    # numerically correct, but we'd rather pay the cast once at startup.
    # NOTE: do NOT cast the SG generator — its BatchNorm layers misbehave
    # in fp16 on Turing. Leave it fp32; _extract_qwen_vit_feature_maps
    # handles the dtype boundary.
    for _mod_name in ("sg_encoder", "sg_projector", "grounding_head", "aux_heads"):
        _mod = getattr(model, _mod_name, None)
        if _mod is not None:
            _mod.to(dtype=_qwen_dtype)
    if is_main_process(local_rank):
        logger.info(f"Cast SG/grounding/aux modules to {_qwen_dtype} to match Qwen.")

    # Mode-specific LR override (ADR-026 / migration notes):
    #   pretrain → 2e-4 (LoRA warmup), finetune → 5e-5 (refinement)
    _MODE_LR = {'sg_only': 1e-4, 'alignment': 5e-5, 'pretrain': 2e-4, 'finetune': 5e-5, 'rl': 1e-5}
    if _phase in _MODE_LR:
        _new_lr = _MODE_LR[_phase]
        if is_main_process(local_rank):
            logger.info(
                f"Mode-specific LR override for '{_phase}': "
                f"{config.training.learning_rate} -> {_new_lr}"
            )
        config.training.learning_rate = _new_lr
    
    # ==================================================================
    # LOAD PRETRAINED CHECKPOINT (for finetuning or resuming)
    # ==================================================================
    resume_step = 0
    resume_epoch = 0
    if args.resume_from_checkpoint:
        ckpt_path = Path(args.resume_from_checkpoint)
        ckpt_file = None
        
        # Find the actual .bin file
        if ckpt_path.is_dir():
            for candidate in ['pytorch_model.bin', 'model.bin', 'checkpoint.bin']:
                if (ckpt_path / candidate).exists():
                    ckpt_file = ckpt_path / candidate
                    break
            # Also check for DeepSpeed-style checkpoint
            if ckpt_file is None:
                ds_ckpt = ckpt_path / 'mp_rank_00_model_states.pt'
                if ds_ckpt.exists():
                    ckpt_file = ds_ckpt
        elif ckpt_path.is_file():
            ckpt_file = ckpt_path
        
        if ckpt_file is not None:
            if is_main_process(local_rank):
                logger.info(f"Loading checkpoint from: {ckpt_file}")
            
            checkpoint = torch.load(ckpt_file, map_location='cpu')
            
            # Extract model state dict (handle different checkpoint formats)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'module' in checkpoint:
                state_dict = checkpoint['module']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Assume the checkpoint IS the state dict
                state_dict = checkpoint
            
            # Load with strict=False to handle architecture changes between
            # pretrain and finetune (e.g., new heads, different projections)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            
            if is_main_process(local_rank):
                if missing:
                    logger.warning(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
                if unexpected:
                    logger.warning(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
                logger.info(f"  Loaded {len(state_dict) - len(missing) - len(unexpected)}/{len(state_dict)} parameters")
            
            # Optionally resume training step/epoch (only if NOT finetuning)
            phase = getattr(config.training, 'phase', 'pretrain').lower()
            if phase != 'finetune' and 'global_step' in checkpoint:
                resume_step = checkpoint.get('global_step', 0)
                resume_epoch = checkpoint.get('epoch', 0)
                if is_main_process(local_rank):
                    logger.info(f"  Resuming from step {resume_step}, epoch {resume_epoch}")
            elif phase == 'finetune':
                if is_main_process(local_rank):
                    logger.info("  Finetuning mode: starting fresh optimizer/scheduler (not resuming step)")
            
            del checkpoint, state_dict  # Free memory
            _reclaim_cpu_memory()
        else:
            logger.error(f"Could not find checkpoint file in {ckpt_path}")
            logger.error("Expected: pytorch_model.bin, model.bin, or mp_rank_00_model_states.pt")
    
    # Enable/disable gradient checkpointing based on config
    if config.training.gradient_checkpointing:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            if is_main_process(local_rank):
                logger.info("Gradient checkpointing ENABLED")
    else:
        # Explicitly disable to ensure BERT encoder doesn't use it
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
        if is_main_process(local_rank):
            logger.info("Gradient checkpointing DISABLED")
    
    # ==================================================================
    # LOSS FUNCTION — uses training_mode from config.training.phase
    # ==================================================================
    # 'pretrain' → vqa=1.0, generation=0.5, chexpert=0.3, SG=0.2, grounding=0.15
    # 'finetune' → vqa=1.0, generation=0.8, chexpert=0.1, SG=0.05, grounding=0.2
    # 'standard' → vqa=1.0, generation=0.5, chexpert=0.3, SG=0.1, grounding=0.1
    training_mode = getattr(config.training, 'phase', 'standard').lower()
    if training_mode not in ('pretrain', 'finetune', 'standard'):
        training_mode = 'standard'
    
    criterion = MultiTaskLoss(
        training_mode=training_mode,
        vqa_weight=config.training.vqa_loss_weight,
        generation_weight=getattr(config.training, 'generation_loss_weight', None),
        chexpert_weight=config.training.chexpert_loss_weight,
        scene_graph_weight=getattr(config.training, 'scene_graph_loss_weight', None),
        grounding_weight=getattr(config.training, 'grounding_loss_weight', None),
        binary_weight=config.training.binary_head_weight,
        category_weight=config.training.category_head_weight,
        region_weight=config.training.region_head_weight,
        severity_weight=config.training.severity_head_weight,
    )
    
    if is_main_process(local_rank):
        logger.info(f"Loss mode: {training_mode} | "
                     f"vqa={criterion.vqa_weight}, gen={criterion.generation_weight}, "
                     f"chex={criterion.chexpert_weight}, sg={criterion.scene_graph_weight}, "
                     f"grd={criterion.grounding_weight}")
    
    # Calculate scheduler steps (accounting for gradient accumulation)
    steps_per_epoch = len(train_dataloader) // config.training.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.training.num_epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    
    # DeepSpeed initialization
    if use_deepspeed:
        if is_main_process(local_rank):
            logger.info("Initializing DeepSpeed...")
        
        # Load DeepSpeed config (optimized for hardware if auto_optimize)
        ds_config_path = args.deepspeed_config or config.deepspeed.config_path
        
        if args.auto_optimize and hardware_info is not None:
            # Generate optimized DeepSpeed config based on hardware
            ds_config = get_deepspeed_config_for_hardware(hardware_info, ds_config_path)
            if is_main_process(local_rank):
                logger.info(f"Using hardware-optimized DeepSpeed config (ZeRO stage {hardware_info.deepspeed_stage})")
        else:
            with open(ds_config_path) as f:
                ds_config = json.load(f)
        
        # Update DeepSpeed config with training parameters
        ds_config['train_micro_batch_size_per_gpu'] = config.training.batch_size_per_gpu
        ds_config['gradient_accumulation_steps'] = config.training.gradient_accumulation_steps
        ds_config['optimizer']['params']['lr'] = config.training.learning_rate
        ds_config['optimizer']['params']['weight_decay'] = config.training.weight_decay
        ds_config['scheduler']['params']['warmup_num_steps'] = warmup_steps
        ds_config['scheduler']['params']['total_num_steps'] = total_steps
        ds_config['scheduler']['params']['warmup_max_lr'] = config.training.learning_rate
        ds_config['fp16']['enabled'] = config.training.fp16
        
        # PyTorch 2.6+ requires contiguous tensors for distributed broadcast
        # Make all parameters contiguous before DeepSpeed initialization
        for param in model.parameters():
            if not param.data.is_contiguous():
                param.data = param.data.contiguous()
        
        # Initialize DeepSpeed
        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )
        
        scaler = None  # DeepSpeed handles FP16 internally
        
    # DDP or DataParallel setup
    elif use_ddp:
        # Guardrail: DDP requires an initialized process group (usually via torchrun).
        # If we're not actually in a distributed run, fall back safely.
        if not dist.is_available() or not dist.is_initialized():
            if is_main_process(local_rank):
                logger.warning(
                    "DDP was enabled but no default process group is initialized. "
                    "Falling back to DataParallel/single-process mode. "
                    "If you intended to use DDP, launch with torchrun --nproc_per_node=<n> and pass --use_ddp."
                )
            use_ddp = False
        if use_ddp:
            if is_main_process(local_rank):
                logger.info("Initializing DistributedDataParallel...")
            
            model = model.to(device)
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True
            )
            
            # Standard optimizer and scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay
            )
            
            from torch.optim.lr_scheduler import OneCycleLR
            scheduler = OneCycleLR(
                optimizer,
                max_lr=config.training.learning_rate,
                total_steps=total_steps,
                pct_start=config.training.warmup_ratio,
                anneal_strategy='cos'
            )
            
            scaler = GradScaler('cuda') if config.training.fp16 else None
        
    # Single GPU / DataParallel fallback
    if not use_deepspeed and not use_ddp:
        model = model.to(device)
        
        # Multi-GPU with DataParallel (less efficient than DDP)
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1 and not is_distributed:
            if is_main_process(local_rank):
                logger.warning(f"Using DataParallel with {n_gpus} GPUs. Consider using --use_deepspeed or --use_ddp for better performance.")
            model = nn.DataParallel(model)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.training.learning_rate,
            total_steps=total_steps,
            pct_start=config.training.warmup_ratio,
            anneal_strategy='cos'
        )
        
        scaler = GradScaler('cuda') if config.training.fp16 else None
    
    # Start epoch and global step (may be set by --resume_from_checkpoint above)
    # For finetuning: always start from epoch 1, step 0 (fresh optimizer/scheduler)
    # For resuming pretrain: pick up where we left off
    start_epoch = resume_epoch + 1 if resume_epoch > 0 else 1
    global_step = resume_step

    # Print training info (main process only)
    if is_main_process(local_rank):
        print_training_info(config, world_size, model, device)
        
        if WANDB_AVAILABLE and config.wandb.enabled and config.wandb.watch_model:
            wandb.watch(model, log="all", log_freq=config.wandb.watch_log_freq)
    
    # Training loop
    best_metric = 0.0
    # global_step is already set from resume_step above (0 for fresh / finetune)
    epochs_without_improvement = 0
    
    if is_main_process(local_rank):
        logger.info("Starting training...")
        logger.info(f"Phase: {getattr(config.training, 'phase', 'standard')}")
        logger.info(f"Start epoch: {start_epoch}, Start step: {global_step}")
        logger.info(f"Effective batch size: {get_effective_batch_size(config, world_size)}")
        logger.info(f"Steps per epoch: {len(train_dataloader) // config.training.gradient_accumulation_steps}")
        logger.info(f"Total training steps: {total_steps}")
        if args.resume_from_checkpoint:
            logger.info(f"Resumed from: {args.resume_from_checkpoint}")
    
    for epoch in range(start_epoch, config.training.num_epochs + 1):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if is_main_process(local_rank):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{config.training.num_epochs}")
            logger.info(f"{'='*50}")
        
        # Train
        train_loss, global_step = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epoch=epoch,
            config=config,
            scaler=scaler,
            global_step=global_step,
            local_rank=local_rank,
            use_deepspeed=use_deepspeed
        )
        
        if is_main_process(local_rank):
            logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate (all processes participate, but only main process logs)
        val_metrics = validate(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
            config=config
        )
        
        # Only main process handles logging and checkpointing
        if is_main_process(local_rank):
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val Accuracy: {val_metrics.get('classification_accuracy', 0):.4f}")
            logger.info(f"Val Binary Acc: {val_metrics.get('binary_accuracy', 0):.4f}")
            logger.info(f"Val Category F1: {val_metrics.get('category_f1', 0):.4f}")
            logger.info(f"Val CheXpert AUROC: {val_metrics.get('chexpert_auroc', 0):.4f}")
            logger.info(f"Val SG Entity Acc: {val_metrics.get('sg_entity_accuracy', 0):.4f}")
            logger.info(f"Val Grounding IoU: {val_metrics.get('grounding_mean_iou', 0):.4f}")
            if val_metrics.get('generation_bleu', 0) > 0:
                logger.info(f"Val Gen BLEU: {val_metrics.get('generation_bleu', 0):.4f}")
                logger.info(f"Val Gen ROUGE-L: {val_metrics.get('generation_rouge_l', 0):.4f}")
            
            # Log to wandb — ALL available metrics
            if WANDB_AVAILABLE and config.wandb.enabled:
                val_log = {
                    'epoch': epoch,
                    'train/epoch_loss': train_loss,
                    'val/loss': val_metrics['loss'],
                    # --- VQA Classification ---
                    'val/classification_accuracy': val_metrics.get('classification_accuracy', 0),
                    'val/binary_accuracy': val_metrics.get('binary_accuracy', 0),
                    'val/binary_f1': val_metrics.get('binary_f1', 0),
                    'val/binary_precision': val_metrics.get('binary_precision', 0),
                    'val/binary_recall': val_metrics.get('binary_recall', 0),
                    'val/category_accuracy': val_metrics.get('category_accuracy', 0),
                    'val/category_f1': val_metrics.get('category_f1', 0),
                    'val/region_accuracy': val_metrics.get('region_accuracy', 0),
                    'val/region_f1': val_metrics.get('region_f1', 0),
                    'val/severity_accuracy': val_metrics.get('severity_accuracy', 0),
                    'val/severity_f1': val_metrics.get('severity_f1', 0),
                    # --- CheXpert ---
                    'val/chexpert_auroc': val_metrics.get('chexpert_auroc', 0),
                    # --- Answer Generation ---
                    'val/generation_bleu': val_metrics.get('generation_bleu', 0),
                    'val/generation_rouge_l': val_metrics.get('generation_rouge_l', 0),
                    'val/generation_exact_match': val_metrics.get('generation_exact_match', 0),
                    'val/generation_word_overlap': val_metrics.get('generation_word_overlap', 0),
                    # --- Scene Graph ---
                    'val/sg_entity_accuracy': val_metrics.get('sg_entity_accuracy', 0),
                    'val/sg_entity_recall': val_metrics.get('sg_entity_recall', 0),
                    'val/sg_region_accuracy': val_metrics.get('sg_region_accuracy', 0),
                    'val/sg_mean_iou': val_metrics.get('sg_mean_iou', 0),
                    'val/sg_iou_50': val_metrics.get('sg_iou_50', 0),
                    # --- Visual Grounding ---
                    'val/grounding_mean_iou': val_metrics.get('grounding_mean_iou', 0),
                    'val/grounding_acc_iou25': val_metrics.get('grounding_acc_iou25', 0),
                    'val/grounding_acc_iou50': val_metrics.get('grounding_acc_iou50', 0),
                    'val/pointing_accuracy': val_metrics.get('pointing_accuracy', 0),
                    # --- Attention ---
                    'val/attention_mean_entropy': val_metrics.get('attention_mean_entropy', 0),
                    'val/attention_focused_ratio': val_metrics.get('attention_focused_ratio', 0),
                }
                wandb.log(val_log)
            
            # Check for best model
            current_metric = val_metrics.get(config.training.metric_for_best_model, 0)
            is_best = current_metric > best_metric
            
            if is_best:
                best_metric = current_metric
                epochs_without_improvement = 0
                logger.info(f"New best {config.training.metric_for_best_model}: {best_metric:.4f}")
            else:
                epochs_without_improvement += 1
            
            # Save checkpoint: always at epoch end, plus best model
            # (cleanup will keep only the most recent save_total_limit)
            model_to_save = model
            if use_deepspeed:
                model_to_save = model.module
            elif hasattr(model, 'module'):
                model_to_save = model.module
            
            save_checkpoint(
                model=model_to_save,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                metrics=val_metrics,
                config=config,
                is_best=is_best
            )
            
            # Push to hub
            if is_best and config.training.hub_model_id:
                model_to_save = model.module if hasattr(model, 'module') else model
                push_to_hub(
                    model=model_to_save,
                    config=config,
                    metrics=val_metrics,
                    commit_message=f"Epoch {epoch} - Acc: {val_metrics.get('accuracy', 0):.4f}"
                )
            
            # Early stopping check
            if epochs_without_improvement >= config.training.early_stopping_patience:
                logger.info(f"Early stopping after {epochs_without_improvement} epochs without improvement")
                break
        
        # Sync all processes at end of epoch
        if is_distributed:
            dist.barrier() if dist.is_initialized() else None
    
    # Final save (main process only) — ALWAYS saves final epoch checkpoint
    if is_main_process(local_rank):
        logger.info("\nTraining complete!")
        logger.info(f"Best {config.training.metric_for_best_model}: {best_metric:.4f}")
        
        # Get the underlying model for saving
        model_to_save = model
        if use_deepspeed:
            model_to_save = model.module
        elif hasattr(model, 'module'):
            model_to_save = model.module
        
        # Always save a numbered checkpoint at the end
        save_checkpoint(
            model=model_to_save,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            metrics=val_metrics,
            config=config,
            is_best=False
        )
        
        # Also save as 'final_model' (never cleaned up) so finetuning always has a fixed path
        final_dir = Path(config.training.output_dir) / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_ckpt = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model_to_save.state_dict(),
            'metrics': val_metrics,
            'config': config.to_dict(),
            'phase': getattr(config.training, 'phase', 'unknown'),
        }
        torch.save(final_ckpt, final_dir / "pytorch_model.bin")
        with open(final_dir / "config.json", 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        with open(final_dir / "training_metadata.json", 'w') as f:
            json.dump({
                'epoch': epoch,
                'global_step': global_step,
                'metrics': val_metrics,
                'phase': getattr(config.training, 'phase', 'unknown'),
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
        logger.info(f"Saved FINAL MODEL to {final_dir}")
        
        # Final push to hub
        if config.training.hub_model_id:
            push_to_hub(
                model=model_to_save,
                config=config,
                metrics={'best_accuracy': best_metric, **val_metrics},
                commit_message=f"Final model - Best Acc: {best_metric:.4f}"
            )
        
        # Close wandb
        if WANDB_AVAILABLE and config.wandb.enabled:
            wandb.finish()
        
        logger.info("Done!")
    
    # Cleanup distributed training
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MIMIC-CXR VQA Model")
    
    # Config
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
    
    # Data paths
    parser.add_argument('--mimic_cxr_path', type=str, help='Path to MIMIC-CXR-JPG dataset')
    parser.add_argument('--mimic_qa_path', type=str, help='Path to MIMIC-Ext-CXR-QBA dataset')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/mimic-cxr-vqa', help='Output directory')
    
    # Training params
    parser.add_argument('--batch_size', type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples (for debugging)')
    parser.add_argument('--prebuilt_cache_train', type=str, default=None,
                       help='Path to prebuilt TRAIN samples .pkl (e.g., from scripts/prebuild_cache.py)')
    parser.add_argument('--prebuilt_cache_val', type=str, default=None,
                       help='Path to prebuilt VAL/VALIDATE samples .pkl (e.g., from scripts/prebuild_cache.py)')
    
    # Distributed training (per methodology Section 11)
    parser.add_argument('--use_deepspeed', action='store_true',
                       help='Enable DeepSpeed ZeRO-2 (recommended for 4+ GPUs)')
    parser.add_argument('--deepspeed_config', type=str, default='configs/deepspeed_config.json',
                       help='Path to DeepSpeed config JSON')
    parser.add_argument('--use_ddp', action='store_true',
                       help='Enable DistributedDataParallel (alternative to DeepSpeed)')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training (set by launcher)')
    
    # Hardware optimization
    parser.add_argument('--auto_optimize', action='store_true', default=True,
                       help='Auto-detect hardware and optimize settings (default: enabled)')
    parser.add_argument('--no_auto_optimize', action='store_false', dest='auto_optimize',
                       help='Disable hardware auto-optimization')
    
    # Training phase
    parser.add_argument('--phase', type=str, default=None, choices=['pretrain', 'finetune'],
                       help='Training phase: pretrain or finetune (overrides config)')
    
    # Hub
    parser.add_argument('--hub_model_id', type=str, help='Hugging Face Hub model ID')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to pretrained checkpoint dir/file to load weights from')
    
    # Wandb
    parser.add_argument('--wandb_project', type=str, default='mimic-cxr-vqa', help='W&B project name')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable W&B logging')
    
    # Data check
    parser.add_argument('--skip_data_check', action='store_true', 
                       help='Skip data readiness check (not recommended)')
    
    # Gradient checkpointing override
    parser.add_argument('--no_gradient_checkpointing', action='store_true',
                       help='Force disable gradient checkpointing (fixes DataParallel deadlock)')
    
    # FP16 override (required when CUDA toolkit is too old to compile DeepSpeed FP16 ops)
    parser.add_argument('--no_fp16', action='store_true',
                       help='Force disable FP16 mixed precision (use when CUDA toolkit cannot compile DeepSpeed ops)')
    
    args = parser.parse_args()
    
    # DeepSpeed adds local_rank argument automatically
    if args.local_rank == -1:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    main(args)

