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

    # ------------------------------------------------------------------
    # Run-ID persistence for power-loss continuity.
    # If a previous run wrote wandb_run_id.txt under output_dir, reuse that
    # run id with resume="allow" so charts continue on the same run instead
    # of forking a new one. New run otherwise.
    # ------------------------------------------------------------------
    prior_run_id = None
    try:
        run_id_file = Path(config.training.output_dir) / "wandb_run_id.txt"
        if run_id_file.exists():
            prior_run_id = run_id_file.read_text().strip() or None
            if prior_run_id:
                logger.info(f"Resuming wandb run id={prior_run_id}")
    except Exception as e:
        logger.warning(f"Could not read wandb_run_id.txt: {e}")

    # Honour WANDB_DISABLE_ALERTS / WANDB_SILENT escape hatches so the
    # user can suppress server-side run-state alert emails ("Run failed",
    # "Run crashed") without editing this file.
    _wandb_settings = None
    try:
        _wandb_settings = wandb.Settings(
            # Don't emit server-side "Run failed" alert email when this
            # process exits with non-zero. The OLD-run alert spam was
            # because every prior crash (NCCL timeout, host reboot,
            # scheduler off-by-one) left the run in `crashed` state and
            # W&B kept re-alerting. Combined with the SIGTERM/atexit
            # handlers below, the run is always wandb.finish()-ed
            # cleanly so the server never sees a crashed state.
            disable_job_creation=True,
            quiet=True,
        )
    except Exception:
        # Older wandb versions may not accept these kwargs — fall back
        # to None and rely solely on the signal/atexit clean-exit path.
        _wandb_settings = None

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity or None,
        name=run_name,
        group=config.wandb.group,
        tags=config.wandb.tags,
        notes=config.wandb.notes,
        config=config.to_dict(),
        id=prior_run_id,
        resume="allow",
        save_code=True,
        settings=_wandb_settings,
    )

    # ------------------------------------------------------------------
    # CLEAN-EXIT HANDLERS for wandb.
    # W&B sends "Run failed" emails when a run's heartbeat dies without a
    # clean wandb.finish(). Register handlers on SIGTERM/SIGINT (the
    # signals torchrun sends on worker death) AND on normal Python exit
    # (atexit) so the run is ALWAYS marked as terminated cleanly,
    # regardless of how this process dies.
    #
    # The remaining failure modes that bypass this:
    #   - Host reboot / kernel panic (host is gone, no signal delivered)
    #   - Manual `kill -9` (SIGKILL cannot be caught)
    # For those, manually mark the old runs as finished in the W&B UI
    # (one-time) and they'll stop alerting.
    # ------------------------------------------------------------------
    import signal as _signal
    import atexit as _atexit

    _wandb_finished_flag = {"done": False}

    def _safe_wandb_finish(exit_code: int = 0):
        if _wandb_finished_flag["done"]:
            return
        _wandb_finished_flag["done"] = True
        try:
            wandb.finish(exit_code=exit_code, quiet=True)
        except Exception:
            pass

    def _signal_handler(sig, _frame):
        # On SIGTERM/SIGINT, mark wandb finished with exit_code=0 (clean
        # termination) so the server doesn't fire the failure alert.
        _safe_wandb_finish(exit_code=0)
        # Re-raise the signal with default handler so process still dies
        _signal.signal(sig, _signal.SIG_DFL)
        os.kill(os.getpid(), sig)

    try:
        _signal.signal(_signal.SIGTERM, _signal_handler)
        _signal.signal(_signal.SIGINT, _signal_handler)
    except Exception as _e:
        logger.warning(f"Could not register wandb signal handlers: {_e}")

    _atexit.register(_safe_wandb_finish, 0)

    # Persist the (possibly newly-assigned) run id immediately so the next
    # crash-recovery launch can pick it up before the first checkpoint save.
    try:
        Path(config.training.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(config.training.output_dir) / "wandb_run_id.txt").write_text(
            run.id + "\n"
        )
    except Exception as e:
        logger.warning(f"Could not write wandb_run_id.txt: {e}")
    
    logger.info(f"Wandb run started: {wandb_project}/{run_name}")
    
    # Define custom metrics
    wandb.define_metric("train/loss", summary="min")
    wandb.define_metric("train/vqa_loss", summary="min")
    wandb.define_metric("train/chexpert_loss", summary="min")
    wandb.define_metric("val/accuracy", summary="max")
    wandb.define_metric("val/binary_accuracy", summary="max")
    wandb.define_metric("val/category_f1", summary="max")
    
    return run


class EMAModel:
    """Tiny exponential-moving-average wrapper around a model's parameters.

    Why a custom class instead of timm.utils.ModelEma: the trainer has zero
    EMA infra today, and pulling in timm just for this is overkill. We hold
    a CPU-side shadow copy of every parameter (avoids ~+8 GB VRAM with a
    Qwen3-VL-8B base), update after each optimiser step, and provide
    swap_in / swap_out for validation + save_checkpoint.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        # State dict only (no autograd, no buffers we don't need). Stored on
        # CPU because the EMA copy is the same size as the trainable params
        # and Turing GPUs are tight enough already.
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().cpu().clone()
            for k, v in self._iter_params(model)
        }
        self._backup: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _iter_params(model: nn.Module):
        """Iterate trainable parameters only. Skips buffers (running stats,
        positional ids, etc.) which the optimiser doesn't touch."""
        target = model.module if hasattr(model, 'module') else model
        for k, v in target.state_dict().items():
            if torch.is_floating_point(v):
                yield k, v

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in self._iter_params(model):
            if k not in self.shadow:
                self.shadow[k] = v.detach().cpu().clone()
                continue
            # shadow = d * shadow + (1-d) * v  (on CPU)
            self.shadow[k].mul_(d).add_(v.detach().cpu(), alpha=1.0 - d)

    @torch.no_grad()
    def swap_in(self, model: nn.Module) -> None:
        """Replace model's live parameters with EMA shadow; remember the
        live values in ``_backup`` so swap_out can restore them."""
        target = model.module if hasattr(model, 'module') else model
        live = target.state_dict()
        self._backup = {}
        for k in self.shadow:
            if k in live:
                self._backup[k] = live[k].detach().clone()
                live[k].copy_(self.shadow[k].to(live[k].device, dtype=live[k].dtype))

    @torch.no_grad()
    def swap_out(self, model: nn.Module) -> None:
        if not self._backup:
            return
        target = model.module if hasattr(model, 'module') else model
        live = target.state_dict()
        for k, v in self._backup.items():
            if k in live:
                live[k].copy_(v)
        self._backup = {}

    def shadow_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return a CPU-side copy of the EMA weights for saving."""
        return {k: v.clone() for k, v in self.shadow.items()}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    global_step: int,
    metrics: Dict[str, float],
    config: MIMICCXRVQAConfig,
    is_best: bool = False,
    ema_model: Optional["EMAModel"] = None,
):
    """Save model checkpoint."""
    checkpoint_dir = Path(config.training.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Capture RNG state so resumed runs reproduce the exact data order and
    # dropout pattern that the original run would have produced from this point.
    # Without this, a power-loss recovery shuffles data differently and
    # invalidates loss-curve comparisons against pre-loss training.
    import random as _random
    try:
        import numpy as _np
        numpy_rng = _np.random.get_state()
    except ImportError:
        numpy_rng = None

    rng_state = {
        'python': _random.getstate(),
        'numpy': numpy_rng,
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }

    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config.to_dict(),
        'rng_state': rng_state,
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint-{global_step}"
    checkpoint_path.mkdir(exist_ok=True)
    torch.save(checkpoint, checkpoint_path / "pytorch_model.bin")

    # Save EMA shadow weights alongside (small, ~same size as trainable params).
    # serve_app can load this in place of pytorch_model.bin for the smoothed
    # version of the model.
    if ema_model is not None:
        try:
            torch.save(
                {'model_state_dict': ema_model.shadow_state_dict()},
                checkpoint_path / "pytorch_model_ema.bin",
            )
        except Exception as _e:
            logger.warning(f"Could not save EMA weights: {_e}")
    
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

    # ------------------------------------------------------------------
    # Power-loss recovery: write 'latest_checkpoint.txt' pointing at this
    # save. --auto_resume reads this on next launch and skips manual
    # --resume_from_checkpoint. We write the *step number* not a symlink
    # (symlinks unreliable on some FUSE-mounted GCS/NFS volumes).
    # ------------------------------------------------------------------
    try:
        latest_pointer = checkpoint_dir / "latest_checkpoint.txt"
        latest_pointer.write_text(checkpoint_path.name + "\n")
    except OSError as e:
        logger.warning(f"Could not update latest_checkpoint.txt: {e}")

    # ------------------------------------------------------------------
    # Persist wandb run_id alongside the checkpoint so a resumed run can
    # call wandb.init(id=..., resume="must") and continue plotting on the
    # same chart rather than starting a fresh run.
    # ------------------------------------------------------------------
    try:
        import wandb as _wandb  # noqa: WPS433
        if _wandb.run is not None:
            run_id_file = checkpoint_dir / "wandb_run_id.txt"
            run_id_file.write_text(_wandb.run.id + "\n")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Could not persist wandb run_id: {e}")

    # ------------------------------------------------------------------
    # Best-model pointer (replaces the old in-place byte copy).
    #
    # Old behaviour: torch.save(...) into best_model/pytorch_model.bin,
    #                overwriting on every is_best=True. Created an ambiguity
    #                where a later eval that happened to score above the
    #                running max could clobber the actual peak checkpoint
    #                bytes (e.g. Stage 4 E2 first-eval overwrote E1's 0.464
    #                peak with a 0.384 model).
    #
    # New behaviour: write best_model/checkpoint_step.txt containing just the
    #                global_step number. The numbered checkpoint
    #                (checkpoint-{global_step}/) stays intact, and
    #                serve_app._resolve_best_pointer() reads the pointer to
    #                load the correct weights. _cleanup_old_checkpoints()
    #                must NOT delete the checkpoint the pointer references.
    # ------------------------------------------------------------------
    if is_best:
        best_path = checkpoint_dir / "best_model"
        best_path.mkdir(exist_ok=True)
        (best_path / "checkpoint_step.txt").write_text(f"{global_step}\n")
        with open(best_path / "config.json", 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        # Record which metrics this is the best for so the dir is
        # self-describing without grepping training logs.
        with open(best_path / "training_metadata.json", 'w') as f:
            json.dump({
                'epoch': epoch,
                'global_step': global_step,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat(),
                'note': "best_model is a pointer; load checkpoint-{global_step}/ "
                        "instead, or use scripts/serve_app.py which resolves "
                        "checkpoint_step.txt automatically.",
            }, f, indent=2)
        logger.info(
            f"Saved best-model pointer at {best_path}/checkpoint_step.txt "
            f"-> checkpoint-{global_step}"
        )

    # Clean up old checkpoints
    _cleanup_old_checkpoints(checkpoint_dir, config.training.save_total_limit)


def _cleanup_old_checkpoints(checkpoint_dir: Path, keep_last: int = 5):
    """Remove old checkpoints, keeping only the most recent ones AND the
    checkpoint that ``best_model/checkpoint_step.txt`` points at (so the
    best-model pointer never becomes dangling).
    """
    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1])
    )

    # Resolve the best-pointer (if any) so we don't delete its referent.
    protected_step: Optional[int] = None
    pointer = checkpoint_dir / "best_model" / "checkpoint_step.txt"
    if pointer.exists():
        try:
            protected_step = int(pointer.read_text().strip())
        except (ValueError, OSError):
            protected_step = None

    # Remove old checkpoints (skip the best-pointer target)
    for checkpoint in checkpoints[:-keep_last]:
        try:
            step = int(checkpoint.name.split("-")[1])
        except ValueError:
            step = None
        if protected_step is not None and step == protected_step:
            logger.info(
                f"Skip cleanup of {checkpoint} (referenced by "
                f"best_model/checkpoint_step.txt)"
            )
            continue
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
        # Token resolution order: explicit env HF_TOKEN > HUGGING_FACE_HUB_TOKEN
        # > token already cached by `huggingface-cli login`. Passing token
        # explicitly avoids surprises in multi-user containers where the
        # cached login may belong to a different user.
        hub_token = (
            os.environ.get('HF_TOKEN')
            or os.environ.get('HUGGING_FACE_HUB_TOKEN')
            or None
        )
        api = HfApi(token=hub_token) if hub_token else HfApi()

        # Create repo if it doesn't exist
        try:
            create_repo(
                config.training.hub_model_id,
                private=config.training.hub_private_repo,
                exist_ok=True,
                token=hub_token,
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
            token=hub_token,
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


def assert_sg_is_generated_not_gt_histogram(
    batch: Dict[str, Any],
    num_regions: int,
    leak_tvd_threshold: float = 0.05,
    flip_tvd_threshold: float = 0.95,
    min_objects: int = 32,
) -> Dict[str, float]:
    """Histogram-based tripwire confirming Stage-3+ region_ids come from the
    generator (cached or live), NOT from GT.

    Why histograms and not per-object compare:
        Cached graphs are in centerness-rank order (n_keep <= max_objects=20).
        GT graphs are in their own annotation order with arbitrary count. Pairing
        by index is meaningless, so a per-object equality check would either
        false-alarm or false-pass depending on how the orderings happen to line
        up. The histogram of region_id COUNTS across the batch is alignment-free
        and still catches the failure mode we care about (the IDs being literally
        GT-equal).

    The check:
        - Build region_id count histograms for cached vs GT across all objects
          in the batch, normalize to distributions p_cached, p_gt.
        - TVD = 0.5 * sum(|p_cached - p_gt|), bounded in [0, 1].
        - TVD ~ 0 → distributions identical → LEAK SUSPECTED (raise)
        - TVD ~ 1 → distributions disjoint → id-space / vocab bug (raise)
        - Middle band → expected (generator argmax accuracy ~0.75; the marginal
          distribution should differ measurably from GT's marginal).

    Two honest limitations baked in:
        1. With very few objects in the batch (< min_objects), the histograms
           are too noisy to be informative — we return {'skipped': 1.0} rather
           than fire a false alarm.
        2. Per-object identity is NOT what this tests. A subtle leak that swaps
           IDs while preserving marginals (e.g. permuted vocab) would slip
           through. The tripwire catches the dumb leaks (literal GT pass-through,
           wrong-field assignment, off-by-vocab); it does NOT certify "correct
           graphs."

    Args:
        batch: collated batch dict with 'generated_sg' and 'gt_sg_regions'
        num_regions: vocab size (model.sg_generator.region_classifier output dim)
        leak_tvd_threshold: TVD below this → raise leak suspicion
        flip_tvd_threshold: TVD above this → raise id-space mismatch
        min_objects: skip the check if fewer than this many comparable objects

    Returns:
        Dict with 'tvd', 'n_cached', 'n_gt', and 'skipped' (1.0 if skipped).
    """
    cached_list = batch.get("generated_sg")
    gt_list = batch.get("gt_sg_regions")
    if cached_list is None or gt_list is None:
        return {"skipped": 1.0, "reason": "missing field"}

    cached_counts = torch.zeros(num_regions, dtype=torch.long)
    gt_counts = torch.zeros(num_regions, dtype=torch.long)
    n_cached = n_gt = 0

    for c, g in zip(cached_list, gt_list):
        if c is not None:
            ids = c.get("region_ids")
            if ids is not None and ids.numel() > 0:
                ids = ids.long().clamp_(0, num_regions - 1)
                cached_counts.index_add_(0, ids, torch.ones_like(ids))
                n_cached += int(ids.numel())
        if g is not None:
            ids = g.long().clamp_(0, num_regions - 1) if hasattr(g, "long") else None
            if ids is not None and ids.numel() > 0:
                gt_counts.index_add_(0, ids, torch.ones_like(ids))
                n_gt += int(ids.numel())

    if n_cached < min_objects or n_gt < min_objects:
        logger.info(
            f"[Stage3 SG check] skipped — too few objects "
            f"(cached={n_cached}, gt={n_gt}, min={min_objects})"
        )
        return {"skipped": 1.0, "n_cached": n_cached, "n_gt": n_gt}

    p_c = cached_counts.float() / max(n_cached, 1)
    p_g = gt_counts.float() / max(n_gt, 1)
    tvd = 0.5 * (p_c - p_g).abs().sum().item()

    if tvd < leak_tvd_threshold:
        raise RuntimeError(
            f"Stage 3 LEAK SUSPECTED: cached region_id distribution matches GT "
            f"with TVD={tvd:.4f} (threshold {leak_tvd_threshold}). "
            f"The trainer is likely passing GT instead of generated graphs. "
            f"n_cached={n_cached}, n_gt={n_gt}, num_regions={num_regions}."
        )
    if tvd > flip_tvd_threshold:
        raise RuntimeError(
            f"Stage 3 ID-SPACE MISMATCH: cached vs GT region distributions are "
            f"effectively disjoint (TVD={tvd:.4f}, threshold {flip_tvd_threshold}). "
            f"Likely an off-by-vocab or wrong-field bug. "
            f"n_cached={n_cached}, n_gt={n_gt}, num_regions={num_regions}."
        )

    logger.info(
        f"[Stage3 SG check] OK — region histogram TVD vs GT = {tvd:.3f} "
        f"(n_cached={n_cached}, n_gt={n_gt}) — generated, not GT."
    )
    return {"tvd": tvd, "n_cached": n_cached, "n_gt": n_gt, "skipped": 0.0}


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
    use_deepspeed: bool = False,
    ema_model: Optional["EMAModel"] = None,
    val_dataloader: Optional[DataLoader] = None,
    best_metric: float = 0.0,
) -> tuple[float, int, float]:
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
        # CURRICULUM-CORRECT SCENE GRAPH ROUTING (per PDF spec §4-5):
        #   sg_only / alignment → use GT scene graphs from dataset
        #                         (clean signal, encoder learns the manifold)
        #   pretrain / finetune / rl → use GENERATED scene graphs from the
        #                         frozen Stage-1 SG generator (matches inference;
        #                         avoids GT→noisy distribution mismatch)
        # Passing GT graphs in Stage 3+ creates a train/test leak: the encoder
        # sees clean entity/region IDs during training but argmax(noisy generator
        # logits) at inference, inflating val metrics and collapsing on deployment.
        _phase_for_sg = getattr(config.training, 'phase', 'pretrain').lower()
        if _phase_for_sg in {'sg_only', 'alignment'}:
            scene_graphs = batch['scene_graphs']      # GT — clean signal
        else:
            # Stage 3+: prefer cached pre-generated SGs (frozen + deterministic
            # generator → cache hit is bit-identical to on-the-fly). Per-item
            # cache miss → None for that item, which the model resolves by
            # falling back to _run_sg_generator at inference semantics.
            # NOTE: when ANY item supplies a cached graph, the model skips the
            #   generator forward → outputs['scene_graph_outputs'] is None →
            #   sg_loss = 0. This is expected, not a regression — the generator
            #   is frozen at Stage 3+, so its detection loss is unoptimizable.
            #   The Stage-3 leak corroborator is now histogram-divergence (see
            #   assert_sg_is_generated_not_gt_histogram), not sg_loss != 0.
            _gen_sg = batch.get('generated_sg')
            if _gen_sg is not None and any(g is not None for g in _gen_sg):
                scene_graphs = _gen_sg
            else:
                scene_graphs = None                   # cache disabled → on-the-fly

            # First-batch tripwire (Stage 3+ only): confirm the SGs we're about
            # to train on are NOT GT. Runs once per training launch on the main
            # process. Raises on leak; skips quietly if too few objects to tell.
            if (
                batch_idx == 0
                and is_main_process(local_rank)
                and not getattr(train_epoch, "_sg_check_done", False)
            ):
                try:
                    _num_regions = int(
                        (model.module if hasattr(model, "module") else model)
                        .sg_generator.region_classifier[-1].out_features
                    )
                    assert_sg_is_generated_not_gt_histogram(batch, _num_regions)
                except RuntimeError:
                    raise  # leak / id-space mismatch — fail loud
                except Exception as e:
                    logger.debug(f"SG histogram check could not run: {e}")
                train_epoch._sg_check_done = True  # one-shot per process
        question_types = batch['question_types']

        # === V2 raw inputs for Qwen processor (added by collate_fn) ===
        pil_images = batch.get('pil_images')          # List[PIL.Image]
        questions = batch.get('questions')            # List[str]
        answer_texts = batch.get('answer_texts')      # List[str] structured
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

                # EMA update — once per actual optimiser step (after grad-accum
                # boundary). Cheap (CPU copy) but adds nothing if disabled.
                if ema_model is not None:
                    ema_model.update(model)

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
        # MID-EPOCH SAVE — power-loss recovery point.
        # Without this, a crash mid-epoch loses everything since the previous
        # epoch boundary. Honored only on the main process and only when
        # save_steps > 0. Saves to checkpoint-{global_step}/ which the
        # --auto_resume flag picks up on next launch.
        # ------------------------------------------------------------------
        if (
            is_main_process(local_rank)
            and global_step > 0
            and getattr(config.training, 'save_steps', 0) > 0
            and global_step % config.training.save_steps == 0
        ):
            model_to_save = model
            if use_deepspeed:
                model_to_save = model.module
            elif hasattr(model, 'module'):
                model_to_save = model.module
            try:
                save_checkpoint(
                    model=model_to_save,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    metrics={'train_loss_running': total_loss / max(num_batches, 1)},
                    config=config,
                    is_best=False,
                    ema_model=ema_model,
                )
                logger.info(
                    f"[mid-epoch] saved checkpoint at step {global_step} "
                    f"(epoch {epoch}, save_steps={config.training.save_steps})"
                )
                # Mid-epoch HF push (optional). Push only when the user
                # explicitly opted in — pushes are 30s-2min each and would
                # slow training to a crawl on small save_steps intervals.
                if getattr(config.training, 'push_every_save', False):
                    push_to_hub(
                        model=model_to_save,
                        config=config,
                        metrics={'global_step': global_step, 'epoch': epoch},
                        commit_message=f"Mid-epoch checkpoint @ step {global_step}",
                    )
            except Exception as e:
                logger.error(f"Mid-epoch save failed (continuing training): {e}")

        # ------------------------------------------------------------------
        # MID-EPOCH VALIDATION
        # Triggered on every config.training.eval_steps boundary. Without
        # this, val only runs at end-of-epoch -- with num_epochs=1 (Stage 4
        # retrain recipe) that means a single val read total, no way to
        # catch the peak before degradation. validate() must be called by
        # ALL ranks (DDP barrier inside), but best-tracking + save happen
        # only on rank 0.
        # ------------------------------------------------------------------
        _eval_steps_cfg = int(getattr(config.training, 'eval_steps', 0) or 0)
        if (
            val_dataloader is not None
            and _eval_steps_cfg > 0
            and global_step > 0
            and global_step % _eval_steps_cfg == 0
        ):
            # Swap EMA in for validation (smoothed weights), restore after.
            if ema_model is not None:
                ema_model.swap_in(model)
            try:
                _val_metrics = validate(
                    model=model,
                    dataloader=val_dataloader,
                    criterion=criterion,
                    device=device,
                    config=config,
                )
            finally:
                if ema_model is not None:
                    ema_model.swap_out(model)

            if is_main_process(local_rank):
                _metric_key = getattr(
                    config.training, 'metric_for_best_model',
                    'classification_accuracy'
                )
                _current = float(_val_metrics.get(_metric_key, 0.0) or 0.0)
                _is_best = _current > best_metric
                logger.info(
                    f"[mid-epoch eval @ step {global_step}] "
                    f"{_metric_key}={_current:.4f} "
                    f"(prev best {best_metric:.4f}) "
                    f"{'NEW BEST' if _is_best else ''}"
                )
                # Also surface key panel metrics so the log shows the trajectory
                logger.info(
                    f"  Val Acc={_val_metrics.get('classification_accuracy', 0):.4f} "
                    f"Bin={_val_metrics.get('binary_accuracy', 0):.4f} "
                    f"Grd IoU={_val_metrics.get('grounding_mean_iou', 0):.4f} "
                    f"Chex AUROC={_val_metrics.get('chexpert_auroc', 0):.4f}"
                )
                if _is_best:
                    best_metric = _current
                    _model_to_save = (
                        model.module if hasattr(model, 'module') else model
                    )
                    try:
                        save_checkpoint(
                            model=_model_to_save,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            global_step=global_step,
                            metrics=_val_metrics,
                            config=config,
                            is_best=True,
                            ema_model=ema_model,
                        )
                    except Exception as e:
                        logger.error(
                            f"Mid-epoch best-checkpoint save failed: {e}"
                        )
            # validate() flipped the model to eval mode; restore train.
            model.train()

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
    return avg_loss, global_step, best_metric


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
        # Validation MUST always use generated scene graphs — at deployment
        # we'll never have GT. Using GT here would inflate val metrics and
        # give a misleading signal to early-stopping / best-model selection.
        # Same routing rule as training (see train_epoch above).
        _phase_for_sg = getattr(config.training, 'phase', 'pretrain').lower()
        if _phase_for_sg in {'sg_only', 'alignment'}:
            scene_graphs = batch['scene_graphs']      # GT — only stages that train ON them
        else:
            # Validation MUST match training's SG source, otherwise val/train
            # distributions diverge and metrics lie. Same cache-first rule.
            _gen_sg = batch.get('generated_sg')
            if _gen_sg is not None and any(g is not None for g in _gen_sg):
                scene_graphs = _gen_sg
            else:
                scene_graphs = None                   # all eval paths see generated SGs
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
    if args.gradient_accumulation_steps:
        config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.hub_model_id:
        config.training.hub_model_id = args.hub_model_id
    if getattr(args, 'no_push', False):
        # Disable all hub push paths for this run. Reason: HF upload on rank 0
        # blocks the rank for many minutes (an 11 GB checkpoint @ 5 MB/s is
        # >30 minutes), while rank 1 keeps running into the next NCCL
        # all-reduce and trips the 10-min watchdog timeout → SIGABRT.
        # If you need a checkpoint on the Hub after training finishes, push
        # ./checkpoints/.../best_model manually with `huggingface-cli upload`.
        config.training.hub_model_id = ""
        config.training.push_every_save = False
        logger.info("--no_push set; disabling all HF Hub uploads for this run.")
    if args.hub_token:
        # Set in env so push_to_hub() picks it up via its env-fallback chain.
        os.environ['HF_TOKEN'] = args.hub_token
    if args.push_every_save:
        config.training.push_every_save = True
    if args.save_steps is not None:
        config.training.save_steps = args.save_steps
        logger.info(f"save_steps overridden to {args.save_steps} via CLI")
    if args.wandb_project:
        config.wandb.project = args.wandb_project
    if args.disable_wandb:
        config.wandb.enabled = False

    # ------------------------------------------------------------------
    # --auto_resume: if a latest_checkpoint.txt exists under output_dir,
    # point the resume flag at the latest checkpoint dir. Takes precedence
    # over --resume_from_checkpoint when both are set AND a latest exists —
    # this is the right behavior for the curriculum: --resume_from_checkpoint
    # supplies the previous stage's weights as the starting point, but if
    # the current stage has already produced its own in-progress checkpoint
    # (mid-stage crash + relaunch), we want to continue THAT, not restart
    # from the previous stage's snapshot.
    # ------------------------------------------------------------------
    if args.auto_resume:
        out_dir = Path(
            args.output_dir
            or config.training.output_dir
            or './checkpoints/mimic-cxr-vqa'
        )
        pointer = out_dir / 'latest_checkpoint.txt'
        if pointer.exists():
            ckpt_name = pointer.read_text().strip()
            ckpt_dir = out_dir / ckpt_name
            if ckpt_dir.is_dir():
                if args.resume_from_checkpoint and str(ckpt_dir) != args.resume_from_checkpoint:
                    logger.info(
                        f"--auto_resume: in-progress checkpoint exists at "
                        f"{ckpt_dir}, taking precedence over "
                        f"--resume_from_checkpoint={args.resume_from_checkpoint}"
                    )
                    # In-progress recovery: discard --load_weights_only because
                    # we want FULL state (optimizer/scheduler/step) restored
                    # to truly continue mid-stage.
                    args.load_weights_only = False
                args.resume_from_checkpoint = str(ckpt_dir)
                logger.info(
                    f"--auto_resume: resuming from latest checkpoint {ckpt_dir}"
                )
            else:
                logger.warning(
                    f"--auto_resume: latest_checkpoint.txt points at "
                    f"{ckpt_dir} which doesn't exist. Falling back to "
                    f"--resume_from_checkpoint={args.resume_from_checkpoint or 'None'}."
                )
        else:
            logger.info(
                f"--auto_resume: no latest_checkpoint.txt at {pointer}. "
                f"Falling back to --resume_from_checkpoint={args.resume_from_checkpoint or 'None'}."
            )
    
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

    # CLI override wins — useful when your QBA dataset doesn't have the
    # phase-default grade (e.g. user has only B_frontal, finetune wants A).
    if args.quality_grade:
        old = config.data.quality_grade
        config.data.quality_grade = args.quality_grade
        logger.info(f"quality_grade overridden via CLI: {old} → {args.quality_grade}")

    
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
    
    # New optional config flags (default-safe so older configs still load):
    #   skip_question_types: list[str] - blacklist applied after the
    #     question_types whitelist (Stage 4 drops A_* report-text qs).
    #   min_localization_quality: int - drop obs with QBA loc_q < N from
    #     SG-target extraction (Stage 1 only; default 0 = off).
    #   val_samples: int - cap on val dataset size when not using max_samples.
    #   val_quality_grade: str - override quality_grade for val dataset only.
    #     Enables "train on B (includes A), validate on A" pattern for
    #     Stage 4 to prevent overfitting on the ~76K A-only pool.
    _skip_qtypes = getattr(config.data, 'skip_question_types', None) or None
    _min_loc_q = int(getattr(config.data, 'min_localization_quality', 0))
    _val_samples = getattr(config.data, 'val_samples', None)
    if _val_samples is not None:
        _val_samples = int(_val_samples)
    _val_quality_grade = getattr(config.data, 'val_quality_grade', None) or config.data.quality_grade

    train_dataset = MIMICCXRVQADataset(
        mimic_cxr_path=config.data.mimic_cxr_jpg_path,
        mimic_qa_path=config.data.mimic_ext_cxr_qba_path,
        split='train',
        tokenizer_name=config.model.text_encoder,
        max_question_length=config.model.max_question_length,
        quality_grade=config.data.quality_grade,
        view_filter=config.data.view_filter,
        question_types=config.data.question_types if config.data.question_types else None,
        skip_question_types=_skip_qtypes,
        chexpert_labels_path=config.data.chexpert_labels_path if config.data.chexpert_labels_path else None,
        max_samples=args.max_samples,
        cache_dir=cache_dir,
        use_cache=True,
        prebuilt_cache_path=args.prebuilt_cache_train,
        one_question_per_image=args.one_question_per_image,
        use_reports=args.use_reports,
        min_localization_quality=_min_loc_q,
    )

    # Barrier to ensure all processes have loaded/cached train data
    if is_distributed:
        dist.barrier()

    # Val sample cap: CLI --max_samples//10 wins; otherwise use config.val_samples.
    _val_cap = (args.max_samples // 10) if args.max_samples else _val_samples
    if is_main_process(local_rank) and _val_quality_grade != config.data.quality_grade:
        logger.info(
            f"Val uses stricter quality_grade='{_val_quality_grade}' "
            f"(train uses '{config.data.quality_grade}'). Enables "
            f"train-on-B-validate-on-A pattern."
        )
    val_dataset = MIMICCXRVQADataset(
        mimic_cxr_path=config.data.mimic_cxr_jpg_path,
        mimic_qa_path=config.data.mimic_ext_cxr_qba_path,
        split='validate',
        tokenizer_name=config.model.text_encoder,
        max_question_length=config.model.max_question_length,
        quality_grade=_val_quality_grade,
        view_filter=config.data.view_filter,
        question_types=config.data.question_types if config.data.question_types else None,
        skip_question_types=_skip_qtypes,
        chexpert_labels_path=config.data.chexpert_labels_path if config.data.chexpert_labels_path else None,
        max_samples=_val_cap,
        cache_dir=cache_dir,
        use_cache=True,
        prebuilt_cache_path=args.prebuilt_cache_val,
        one_question_per_image=args.one_question_per_image,
        use_reports=args.use_reports,
        min_localization_quality=_min_loc_q,
    )
    
    # Barrier to ensure all processes have loaded/cached val data
    if is_distributed:
        dist.barrier()

    # ------------------------------------------------------------------
    # OPTIONAL: carve N samples off the end of train as an INTERNAL VAL.
    #
    # MIMIC-Ext-CXR-QBA's official val split has only 1,805 studies (vs
    # 222K train) — comparable to other published baselines but too small
    # for reliable per-epoch val_loss tracking on the noisy B-grade pool.
    # When --val_from_train N is set, we slice the LAST N samples from the
    # already-shuffled-and-deduped train cache and replace val_dataset's
    # samples with them. Train sees the FIRST (len-N) samples.
    #
    # Determinism: seed-42 shuffle in the cache means the slice is stable
    # across runs. No leakage because the slice is disjoint from train.
    # Official val (1,805 studies) is preserved on-disk for FINAL test-set
    # comparisons in your paper; we just don't use it during training.
    # ------------------------------------------------------------------
    if getattr(args, 'val_from_train', 0) and args.val_from_train > 0:
        per_rank_n = args.val_from_train // max(1, int(os.environ.get('WORLD_SIZE', 1)))
        if per_rank_n <= 0:
            if is_main_process(local_rank):
                logger.warning(
                    f"--val_from_train={args.val_from_train} too small for "
                    f"world_size={os.environ.get('WORLD_SIZE','1')}; ignoring"
                )
        elif per_rank_n >= len(train_dataset.samples):
            if is_main_process(local_rank):
                logger.warning(
                    f"--val_from_train={args.val_from_train} >= train size "
                    f"({len(train_dataset.samples) * max(1, int(os.environ.get('WORLD_SIZE','1')))}); ignoring"
                )
        else:
            train_size_before = len(train_dataset.samples)
            val_size_before = len(val_dataset.samples)
            # Carve last per_rank_n samples off train; they become the new val.
            val_dataset.samples = train_dataset.samples[-per_rank_n:]
            train_dataset.samples = train_dataset.samples[:-per_rank_n]
            if is_main_process(local_rank):
                logger.info(
                    f"--val_from_train={args.val_from_train}: carved last "
                    f"{per_rank_n:,}/rank from train. "
                    f"Train: {train_size_before:,} -> {len(train_dataset.samples):,} per rank. "
                    f"Val: {val_size_before:,} -> {len(val_dataset.samples):,} per rank "
                    f"(official PhysioNet val preserved on disk for final reporting)."
                )

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

    # CLI flag --qwen_model_id wins over config; config wins over default.
    _qwen_id = (
        getattr(args, 'qwen_model_id', None)
        or getattr(config.model, 'qwen_model_id', None)
        or 'Qwen/Qwen3-VL-8B-Instruct'
    )
    _use_quant = bool(getattr(config.model, 'use_quantization', _force_qlora))
    _lora_rank = int(getattr(config.model, 'lora_rank', 16))
    _lora_alpha = int(getattr(config.model, 'lora_alpha', 2 * _lora_rank))
    _lora_targets = getattr(config.model, 'lora_target_modules', None) or None
    _num_sg_tokens = int(getattr(config.model, 'num_sg_tokens', 8))

    if is_main_process(local_rank):
        logger.info(
            f"V2 model: {_qwen_id} | cc={_gpu0_cc} | quantization={_use_quant} "
            f"| lora_rank={_lora_rank} | lora_alpha={_lora_alpha} "
            f"| lora_targets={_lora_targets or '(default: attn only)'} "
            f"| num_sg_tokens={_num_sg_tokens} "
            f"| dtype={_qwen_dtype} | training_mode={_phase}"
        )
        if _force_qlora and not _use_quant:
            logger.warning(
                "Compute capability < 8.0 detected but use_quantization=False. "
                "Forcing QLoRA (4-bit NF4) for memory safety."
            )

    # freeze_sg_generator controls whether the ViT and SG generator forward
    # under torch.no_grad(). Default is True (correct for Stages 2-4 which
    # CONSUME frozen pre-computed SGs). In Stage 1 (sg_only) the whole
    # POINT is to train the SG generator — leaving freeze=True means
    # sg_loss gradients flow into a no_grad() block and the generator
    # never updates (sg_loss plateaus at random-init entropy ~9.5).
    _freeze_sg = (_phase != 'sg_only')
    if is_main_process(local_rank):
        logger.info(
            f"SG generator: {'FROZEN' if _freeze_sg else 'TRAINABLE'} "
            f"(phase={_phase})"
        )

    model = SSGVQANetV2(
        qwen_model_id=_qwen_id,
        use_quantization=_use_quant or _force_qlora,
        lora_rank=_lora_rank,
        lora_alpha=_lora_alpha,
        lora_target_modules=_lora_targets,
        num_sg_tokens=_num_sg_tokens,
        num_regions=config.model.num_regions,
        num_entities=config.model.num_entities,
        num_binary=config.model.num_binary_classes,
        num_category=config.model.num_category_classes,
        num_region_classes=config.model.num_region_classes,
        num_severity=config.model.num_severity_classes,
        training_mode=_phase if _phase in {'sg_only', 'alignment', 'pretrain', 'finetune', 'rl'} else 'pretrain',
        torch_dtype=_qwen_dtype,
        freeze_sg_generator=_freeze_sg,
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

    # Re-cast mHC back to fp32 (Option C — mHC fp32 compute path).
    # On Turing GPUs (RTX 8000, cc 7.5) without bf16, the Sinkhorn-Knopp
    # projection in mHC produces gradient overflow under fp16 within
    # ~4-10K training steps once grounding_head unfreezes. Keeping mHC's
    # weights in fp32 (small block, ~256 KB) lets the manifold math run
    # numerically stable. mHCBlock.forward casts input fp16→fp32 on entry
    # and fp32→fp16 on exit, so the rest of grounding_head still runs in
    # the dominant dtype.
    if hasattr(model, "grounding_head") and getattr(model.grounding_head, "mhc", None) is not None:
        model.grounding_head.mhc.to(dtype=torch.float32)
        if is_main_process(local_rank):
            logger.info(f"Re-cast mHC sub-block to fp32 (Option C — stability fix).")

    # CRITICAL: trainable params must stay fp32, even when the base model is fp16.
    # PyTorch's GradScaler.unscale_() refuses to operate on fp16 gradients
    # ("Attempting to unscale FP16 gradients") because the optimizer needs
    # numerically-safe fp32 grads. Stage 1 (sg_only) didn't hit this because
    # its trainable module (SG generator) is already fp32 above; Stage 2+
    # (alignment / pretrain / finetune) unfreeze sg_encoder / sg_projector /
    # aux_heads / LoRA, which got swept into the bulk fp16 cast.
    # Standard QLoRA pattern: forward in fp16 (via autocast), grads stored in
    # fp32 on the trainable params themselves so the scaler can unscale them.
    _n_recast_fp32 = 0
    for _p in model.parameters():
        if _p.requires_grad and _p.dtype == torch.float16:
            _p.data = _p.data.float()
            _n_recast_fp32 += 1
    if is_main_process(local_rank) and _n_recast_fp32 > 0:
        logger.info(
            f"Re-cast {_n_recast_fp32} trainable params to fp32 "
            "(GradScaler requires fp32 grads — fp16 trainables crash unscale_)."
        )

    # Mode-specific LR fallback (ADR-026 / migration notes).
    # Priority order (highest wins):
    #   1. CLI --learning_rate (args.learning_rate)
    #   2. YAML config.training.learning_rate, if it's a "real" value (not the
    #      legacy sentinel 5e-5 that the codebase historically meant "I don't
    #      care, pick the mode default for me")
    #   3. Mode default from _MODE_LR
    # Earlier behaviour clobbered the YAML even when it carried a deliberate
    # value (e.g. Stage 4's 5.0e-6 -> mode-default 5e-5 = 10x too high), which
    # is the exact footgun that caused the Stage 4 E2 overfitting cliff.
    _MODE_LR = {'sg_only': 1e-4, 'alignment': 5e-5, 'pretrain': 2e-4, 'finetune': 5e-5, 'rl': 1e-5}
    _SENTINEL_LRS = {None, 0.0, 5e-5}  # 5e-5 was the historical "default" value
    if args.learning_rate:
        if is_main_process(local_rank):
            logger.info(
                f"LR source = CLI --learning_rate: {args.learning_rate} "
                f"(YAML had {config.training.learning_rate}; mode default for "
                f"'{_phase}' is {_MODE_LR.get(_phase)})"
            )
        config.training.learning_rate = args.learning_rate
    elif config.training.learning_rate not in _SENTINEL_LRS:
        if is_main_process(local_rank):
            logger.info(
                f"LR source = YAML: {config.training.learning_rate} "
                f"(mode default for '{_phase}' would be "
                f"{_MODE_LR.get(_phase)}; pass --learning_rate to override)"
            )
        # leave config.training.learning_rate alone
    elif _phase in _MODE_LR:
        _new_lr = _MODE_LR[_phase]
        if is_main_process(local_rank):
            logger.info(
                f"LR source = mode default for '{_phase}': "
                f"{config.training.learning_rate} -> {_new_lr} "
                f"(YAML had the sentinel value; pass --learning_rate or "
                f"set a non-sentinel value in the YAML to override)"
            )
        config.training.learning_rate = _new_lr
    
    # ==================================================================
    # LOAD PRETRAINED CHECKPOINT (for finetuning or resuming)
    # ==================================================================
    resume_step = 0
    resume_epoch = 0
    # Stashed state from a resumed checkpoint, restored AFTER optimizer +
    # scheduler are constructed further below. Initialized here at function
    # scope so they're always defined regardless of which branch fires.
    resume_optimizer_state = None
    resume_scheduler_state = None
    resume_rng_state = None
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
            
            # weights_only=False required because our checkpoint dict embeds
            # DeepSpeed objects (DynamicLossScaler) in the optimizer state.
            # PyTorch 2.6+ defaults to weights_only=True for security but
            # rejects these classes by default. Our checkpoints are written
            # by this same trainer (trusted source), so opting out is safe.
            checkpoint = torch.load(ckpt_file, map_location='cpu', weights_only=False)
            
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
            
            # Optionally resume training step/epoch (only if NOT finetuning
            # and NOT --load_weights_only).
            # Stash optimizer/scheduler/RNG state so we can restore them AFTER
            # those objects are constructed below. For finetuning OR
            # explicit --load_weights_only, we discard them on purpose — both
            # cross-stage transfer (sg_only→alignment, etc.) and finetune
            # want a fresh optimizer because the trainable parameter set is
            # different from what produced the saved optimizer state.
            phase = getattr(config.training, 'phase', 'pretrain').lower()
            weights_only = getattr(args, 'load_weights_only', False)
            if (
                not weights_only
                and phase != 'finetune'
                and 'global_step' in checkpoint
            ):
                resume_step = checkpoint.get('global_step', 0)
                resume_epoch = checkpoint.get('epoch', 0)
                resume_optimizer_state = checkpoint.get('optimizer_state_dict')
                resume_scheduler_state = checkpoint.get('scheduler_state_dict')
                resume_rng_state = checkpoint.get('rng_state')
                if is_main_process(local_rank):
                    logger.info(f"  Resuming from step {resume_step}, epoch {resume_epoch}")
                    logger.info(
                        f"  Stashed: optimizer={resume_optimizer_state is not None}, "
                        f"scheduler={resume_scheduler_state is not None}, "
                        f"rng={resume_rng_state is not None}"
                    )
            elif weights_only:
                if is_main_process(local_rank):
                    logger.info("  --load_weights_only: model weights restored, fresh optimizer/scheduler/step/RNG")
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
    # MultiTaskLoss.MODE_WEIGHTS supports all 5 curriculum phases (per PDF
    # spec). The earlier downgrade to 'standard' silently broke sg_only
    # (sg_weight became 0.1 instead of 1.0) — RPN/entity/region/bbox losses
    # got 10x weaker signal than the curriculum design intended.
    if training_mode not in ('sg_only', 'alignment', 'pretrain', 'finetune', 'rl', 'standard'):
        training_mode = 'standard'
    
    # Resolve class-weight files for SG heads. Pass --class_weights_dir
    # pointing at the output of scripts/compute_class_weights.py to
    # enable inverse-frequency weighting (fixes mode collapse to common
    # entities/regions). When the dir is missing, loss falls back to
    # unweighted CE (same behaviour as before).
    cw_dir = getattr(args, 'class_weights_dir', None)
    if cw_dir:
        cw_dir_path = Path(cw_dir)
        ent_w = str(cw_dir_path / "entity_weights.json")   if (cw_dir_path / "entity_weights.json").exists()   else None
        reg_w = str(cw_dir_path / "region_weights.json")   if (cw_dir_path / "region_weights.json").exists()   else None
        pol_w = str(cw_dir_path / "polarity_weights.json") if (cw_dir_path / "polarity_weights.json").exists() else None
        if is_main_process(local_rank):
            logger.info(f"Class weights: dir={cw_dir} entity={'Y' if ent_w else 'N'} "
                        f"region={'Y' if reg_w else 'N'} polarity={'Y' if pol_w else 'N'}")
    else:
        ent_w = reg_w = pol_w = None

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
        entity_weights_json=ent_w,
        region_weights_json=reg_w,
        polarity_weights_json=pol_w,
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
            # DDP config for stages with dynamic per-sample routing:
            # - find_unused_parameters=True: required because grounding/pointing/
            #   aux heads only receive gradients when their target is valid for
            #   the given sample (some MIMIC questions have no bbox, no pointing
            #   target, etc.). Different ranks see different unused param sets,
            #   so static_graph=True is NOT viable.
            # - static_graph=False: see above.
            # The conflicting "mark a variable ready only once" error that
            # static_graph was meant to fix comes from gradient checkpointing's
            # reentrant mode wrapping mHC twice. Fix that at its source:
            # pass --no_gradient_checkpointing for Stage 3+, OR ensure any
            # remaining checkpoint() call uses use_reentrant=False.
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
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
    
    # ------------------------------------------------------------------
    # Restore stashed optimizer/scheduler/RNG state from resumed checkpoint.
    # Done HERE (not at model-load time) because optimizer/scheduler are
    # built above this point but BELOW the model load. For DeepSpeed this
    # is currently a no-op — deepspeed.initialize manages its own optimizer
    # state and the user should pass the DS checkpoint via DS APIs. For
    # plain DDP / single-GPU AdamW + OneCycleLR the restore is meaningful.
    # ------------------------------------------------------------------
    if not use_deepspeed:
        if resume_optimizer_state is not None:
            try:
                optimizer.load_state_dict(resume_optimizer_state)
                if is_main_process(local_rank):
                    logger.info("Restored optimizer state from checkpoint")
            except Exception as e:
                logger.warning(f"Could not restore optimizer state: {e}")
        if resume_scheduler_state is not None and scheduler is not None:
            if getattr(args, 'reset_scheduler', False):
                # User requested a fresh scheduler — common when crossing
                # curriculum-stage boundaries (e.g. Stage 1 → Stage 2 → Stage 3)
                # where each stage has its own num_epochs / total_steps target.
                # OneCycleLR caps at exactly its configured total_steps and will
                # raise "Tried to step N times. The specified number of total
                # steps is M" if we inherit Stage K's schedule into Stage K+1.
                if is_main_process(local_rank):
                    logger.info(
                        "--reset_scheduler set: SKIPPING scheduler restore. "
                        "LR ramp will start fresh for this stage's total_steps."
                    )
            else:
                try:
                    scheduler.load_state_dict(resume_scheduler_state)
                    if is_main_process(local_rank):
                        logger.info("Restored scheduler state from checkpoint")
                except Exception as e:
                    logger.warning(f"Could not restore scheduler state: {e}")
        if resume_rng_state is not None:
            try:
                import random as _random
                _random.setstate(resume_rng_state.get('python'))
                if resume_rng_state.get('numpy') is not None:
                    import numpy as _np
                    _np.random.set_state(resume_rng_state['numpy'])
                if resume_rng_state.get('torch_cpu') is not None:
                    torch.set_rng_state(resume_rng_state['torch_cpu'])
                if (
                    resume_rng_state.get('torch_cuda') is not None
                    and torch.cuda.is_available()
                ):
                    torch.cuda.set_rng_state_all(resume_rng_state['torch_cuda'])
                if is_main_process(local_rank):
                    logger.info("Restored RNG state (python/numpy/torch/cuda)")
            except Exception as e:
                logger.warning(f"Could not restore RNG state: {e}")

    # Start epoch and global step (may be set by --resume_from_checkpoint above).
    # For finetuning: always start from epoch 1, step 0 (fresh optimizer/scheduler).
    # For resuming pretrain: pick up where we left off.
    #
    # Mid-epoch resume detection: the saved `epoch` field is the epoch that
    # was IN PROGRESS when the checkpoint was written. If the saved global_step
    # is not a clean multiple of steps_per_epoch, we were mid-epoch — continue
    # from the SAME epoch. Otherwise the previous epoch finished cleanly and we
    # advance to epoch+1.
    #
    # Without this, mid-epoch saves silently skip the rest of the epoch on
    # resume (you lose up to (1 - save_step%steps_per_epoch)/steps_per_epoch
    # of training for that epoch). Real bug discovered in Stage 2 of the
    # budget curriculum — lost ~96% of epoch 1 after auto_resume from
    # checkpoint-500 (12500 steps_per_epoch → 500/12500 = 4% complete).
    if resume_epoch > 0:
        steps_per_epoch = len(train_dataloader) // config.training.gradient_accumulation_steps
        was_mid_epoch = steps_per_epoch > 0 and (resume_step % steps_per_epoch) != 0
        if was_mid_epoch:
            start_epoch = resume_epoch  # continue same epoch
            if is_main_process(local_rank):
                logger.info(
                    f"Resume is MID-EPOCH (step {resume_step} % {steps_per_epoch} "
                    f"steps_per_epoch != 0). Continuing epoch {resume_epoch} "
                    f"rather than advancing to epoch {resume_epoch + 1}."
                )
        else:
            start_epoch = resume_epoch + 1  # previous epoch finished cleanly
    else:
        start_epoch = 1
    global_step = resume_step

    # Print training info (main process only)
    if is_main_process(local_rank):
        print_training_info(config, world_size, model, device)
        
        if WANDB_AVAILABLE and config.wandb.enabled and config.wandb.watch_model:
            # log="parameters" (NOT "all" / "gradients") — wandb's gradient
            # logging registers a per-param hook that calls .data on the grad
            # tensor. When the model has dynamic per-sample routing (some
            # heads only fire when their target is valid — finetune mode hits
            # this), some params get grad=None on some iterations and the
            # hook crashes with AttributeError: 'NoneType' object has no
            # attribute 'data'. Parameter logging alone is enough for the
            # weight distributions in the W&B UI.
            wandb.watch(model, log="parameters", log_freq=config.wandb.watch_log_freq)
    
    # ------------------------------------------------------------------
    # EMA (exponential moving average of weights).
    # Enabled iff config.training.ema_decay > 0. CPU-side shadow copy so it
    # doesn't eat VRAM on a Qwen3-VL-8B base. Updated after each optimiser
    # step; swapped in for validation and saved alongside each checkpoint.
    # ------------------------------------------------------------------
    _ema_decay = float(getattr(config.training, 'ema_decay', 0.0))
    ema_model: Optional[EMAModel] = None
    if _ema_decay > 0.0 and not use_deepspeed:
        ema_model = EMAModel(model, decay=_ema_decay)
        if is_main_process(local_rank):
            logger.info(
                f"EMA enabled: decay={_ema_decay}, "
                f"shadow params={len(ema_model.shadow)}, "
                f"shadow lives on CPU."
            )
    elif _ema_decay > 0.0 and use_deepspeed:
        if is_main_process(local_rank):
            logger.warning("ema_decay set but DeepSpeed enabled: EMA disabled "
                           "(DS manages its own optimizer state).")

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
        
        # Train (in-epoch validation triggered inside on eval_steps boundary;
        # train_epoch returns the running best_metric so we can keep tracking
        # across epoch boundaries).
        train_loss, global_step, best_metric = train_epoch(
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
            use_deepspeed=use_deepspeed,
            ema_model=ema_model,
            val_dataloader=val_dataloader,
            best_metric=best_metric,
        )

        if is_main_process(local_rank):
            logger.info(f"Train Loss: {train_loss:.4f}")

        # Validate using EMA weights when available (smoothed, less noisy).
        # swap_in copies CPU shadow into the live model; swap_out restores
        # the live weights so the next training epoch isn't perturbed.
        if ema_model is not None:
            ema_model.swap_in(model)
        try:
            val_metrics = validate(
                model=model,
                dataloader=val_dataloader,
                criterion=criterion,
                device=device,
                config=config
            )
        finally:
            if ema_model is not None:
                ema_model.swap_out(model)
        
        # Only main process handles logging and checkpointing
        if is_main_process(local_rank):
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val Accuracy: {val_metrics.get('classification_accuracy', 0):.4f}")
            logger.info(f"Val Binary Acc: {val_metrics.get('binary_accuracy', 0):.4f}")
            logger.info(f"Val Category F1: {val_metrics.get('category_f1', 0):.4f}")
            logger.info(f"Val CheXpert AUROC: {val_metrics.get('chexpert_auroc', 0):.4f}")
            # Hungarian-matched (see training/metrics.py). The IoU-conditioned
            # variants are the ones to put in tables — sg_entity_accuracy
            # alone counts any-IoU>0 matches and inflates with small overlaps.
            logger.info(
                f"Val SG Entity Acc: {val_metrics.get('sg_entity_accuracy', 0):.4f} "
                f"(IoU>=0.5: {val_metrics.get('sg_entity_acc_iou50', 0):.4f}, "
                f"matches: {val_metrics.get('sg_match_count', 0)})"
            )
            logger.info(f"Val SG Mean IoU: {val_metrics.get('sg_mean_iou', 0):.4f} "
                        f"(IoU>=0.5 rate: {val_metrics.get('sg_iou_50', 0):.4f})")
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
                    # --- Scene Graph (Hungarian-matched) ---
                    'val/sg_entity_accuracy': val_metrics.get('sg_entity_accuracy', 0),
                    'val/sg_entity_acc_iou25': val_metrics.get('sg_entity_acc_iou25', 0),
                    'val/sg_entity_acc_iou50': val_metrics.get('sg_entity_acc_iou50', 0),
                    'val/sg_entity_recall': val_metrics.get('sg_entity_recall', 0),
                    'val/sg_region_accuracy': val_metrics.get('sg_region_accuracy', 0),
                    'val/sg_region_acc_iou50': val_metrics.get('sg_region_acc_iou50', 0),
                    'val/sg_mean_iou': val_metrics.get('sg_mean_iou', 0),
                    'val/sg_iou_50': val_metrics.get('sg_iou_50', 0),
                    'val/sg_match_count': val_metrics.get('sg_match_count', 0),
                    'val/sg_match_count_iou50': val_metrics.get('sg_match_count_iou50', 0),
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
                is_best=is_best,
                ema_model=ema_model,
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
            is_best=False,
            ema_model=ema_model,
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
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None,
                        help='Gradient accumulation steps. Effective batch = '
                             'batch_size * num_gpus * grad_accum. Overrides YAML config.')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples (for debugging)')
    parser.add_argument('--prebuilt_cache_train', type=str, default=None,
                       help='Path to prebuilt TRAIN samples .pkl (e.g., from scripts/prebuild_cache.py)')
    parser.add_argument('--prebuilt_cache_val', type=str, default=None,
                       help='Path to prebuilt VAL/VALIDATE samples .pkl (e.g., from scripts/prebuild_cache.py)')
    parser.add_argument('--one_question_per_image', action='store_true',
                       help='Dedupe samples to AT MOST one per STUDY. With this flag, --max_samples N '
                            'gives N unique images AND N unique scene graphs (scene graphs are per-study '
                            'in QBA — different studies = different SGs). Without it, ~82 questions share '
                            'the same image+SG. Use for SG-generator training where image/SG diversity '
                            'matters more than question text. Requires a fresh cache rebuild (cache key '
                            'changes with the flag).')
    parser.add_argument('--val_from_train', type=int, default=0,
                       help='Carve this many TOTAL samples off the end of the (shuffled, deduped) '
                            'train cache and use them as the validation set during training. '
                            'Replaces the small official PhysioNet val (1,805 studies) which is too '
                            'tiny for reliable per-epoch tracking. Recommended: 10000-20000. '
                            'Determinism: the seed-42 shuffle in the train cache means the carve-out '
                            'is reproducible across runs. The official val/test stay on disk for '
                            'final paper-grade reporting; this flag only changes what the trainer '
                            'uses for per-epoch val_loss/val_metrics during the run.')
    parser.add_argument('--reset_scheduler', action='store_true',
                       help='Skip restoring the scheduler state when resuming from a '
                            'checkpoint. Required when crossing curriculum-stage '
                            'boundaries because OneCycleLR is one-shot (it caps at the '
                            "stage's configured total_steps and refuses to continue). "
                            'Without this flag, Stage K+1 inherits Stage K\'s LR ramp '
                            "and aborts with 'Tried to step N+1 times' after consuming "
                            "K's remaining steps. Use whenever --resume_from_checkpoint "
                            'points at a checkpoint from a DIFFERENT phase/config.')
    parser.add_argument('--use_reports', action='store_true',
                       help='Inject the radiologist report into training: INDICATION+HISTORY is '
                            'prepended to the question as clinical context (model INPUT), and '
                            'FINDINGS+IMPRESSION replaces the rule-generated <think> CoT (model '
                            'OUTPUT target). The report text is sourced from the scene_graph.json '
                            'files (no external download required). Bumps max_question_length to '
                            '256 to fit the context. Does NOT invalidate the sample cache (report '
                            'is loaded per-item from SG JSON inside __getitem__).')
    
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
    parser.add_argument('--phase', type=str, default=None,
                       choices=['sg_only', 'alignment', 'pretrain', 'finetune', 'rl'],
                       help='Training phase: sg_only (Stage 1) | alignment (Stage 2) | '
                            'pretrain (Stage 3) | finetune (Stage 4) | rl (Stage 5). '
                            'Overrides config.')
    
    # Hub
    parser.add_argument('--hub_model_id', type=str, help='Hugging Face Hub model ID')
    parser.add_argument('--hub_token', type=str, default=None,
                       help='HF Hub token. If unset, falls back to env HF_TOKEN / HUGGING_FACE_HUB_TOKEN / cached login')
    parser.add_argument('--push_every_save', action='store_true',
                       help='Push to HF Hub on EVERY checkpoint save (not just best). Slower but safer for power loss')
    parser.add_argument('--no_push', action='store_true',
                       help='Disable ALL HF Hub uploads (overrides hub_model_id in YAML). '
                            'Use for short test runs where the 10+ minute checkpoint upload '
                            'would trip the NCCL collective-timeout watchdog (default 600s).')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to pretrained checkpoint dir/file to load weights from')
    parser.add_argument('--auto_resume', action='store_true',
                       help='Auto-resume from latest checkpoint in --output_dir (reads latest_checkpoint.txt). '
                            'Takes precedence over --resume_from_checkpoint when both are set and a latest checkpoint exists.')
    parser.add_argument('--load_weights_only', action='store_true',
                       help='When loading --resume_from_checkpoint, restore model weights only — discard '
                            'optimizer / scheduler / global_step / RNG state. Used for cross-stage curriculum '
                            'transitions (e.g. Stage 1 weights → fresh Stage 2 optimizer).')
    parser.add_argument('--save_steps', type=int, default=None,
                       help='Save mid-epoch checkpoint every N steps (overrides config). 0 disables mid-epoch save')
    parser.add_argument('--quality_grade', type=str, default=None,
                       choices=['A', 'B', 'all'],
                       help='Override QBA quality grade filter (A=clean, B=noisy/broad, all=no filter). '
                            'Default per phase: pretrain→B, finetune→A. Use "all" if your QBA dataset '
                            'lacks the auto-selected grade.')
    parser.add_argument('--qwen_model_id', type=str, default=None,
                       help='Override Qwen model id (e.g. Qwen/Qwen3-VL-4B-Instruct for smoke). '
                            'Default: Qwen/Qwen3-VL-8B-Instruct')

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

    # Class-weight directory (output of scripts/compute_class_weights.py).
    # When supplied, SG entity / region / polarity CE losses use inverse-frequency
    # per-class weights → fixes mode collapse to over-represented classes.
    parser.add_argument('--class_weights_dir', type=str, default=None,
                       help='Directory with {entity,region,polarity}_weights.json '
                            'produced by scripts/compute_class_weights.py. '
                            'Enables inverse-frequency weighting for SG losses.')

    args = parser.parse_args()
    
    # DeepSpeed adds local_rank argument automatically
    if args.local_rank == -1:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    main(args)

