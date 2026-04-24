#!/usr/bin/env python3
"""
Hardware Detection and Optimization Utilities

Automatically detects machine specifications and calculates optimal
training parameters to maximize GPU/CPU utilization.

Based on methodology Section 11: Training Optimizations
Target hardware: 4x NVIDIA L4 GPUs (but auto-adapts to any configuration)
"""

import os
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int
    name: str
    total_memory_mb: int
    free_memory_mb: int
    compute_capability: Tuple[int, int]
    
    @property
    def total_memory_gb(self) -> float:
        return self.total_memory_mb / 1024
    
    @property
    def free_memory_gb(self) -> float:
        return self.free_memory_mb / 1024


@dataclass
class HardwareInfo:
    """Complete hardware information for a machine."""
    # GPU info
    num_gpus: int = 0
    gpus: List[GPUInfo] = field(default_factory=list)
    total_gpu_memory_gb: float = 0.0
    min_gpu_memory_gb: float = 0.0
    
    # CPU info
    num_cpus: int = 1
    cpu_name: str = "Unknown"
    
    # Memory info
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    
    # Computed optimal settings
    optimal_batch_size: int = 16
    optimal_grad_accum: int = 4
    optimal_num_workers: int = 4
    optimal_prefetch_factor: int = 2
    use_fp16: bool = True
    use_bf16: bool = False
    use_gradient_checkpointing: bool = True
    deepspeed_stage: int = 2
    # V2: when compute capability is below sm_80 (Ampere), bf16 isn't
    # supported in hardware. Force QLoRA (4-bit NF4 weights, fp16 compute).
    force_qlora: bool = False
    min_compute_capability: Tuple[int, int] = (0, 0)


def get_gpu_info() -> List[GPUInfo]:
    """Get detailed information about all available GPUs."""
    gpus = []
    
    if not torch.cuda.is_available():
        return gpus
    
    num_gpus = torch.cuda.device_count()
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        
        # Get memory info
        total_mem = props.total_memory // (1024 * 1024)  # MB
        
        # Try to get free memory
        try:
            torch.cuda.set_device(i)
            free_mem = torch.cuda.mem_get_info(i)[0] // (1024 * 1024)
        except:
            free_mem = total_mem  # Assume all available if can't query
        
        gpus.append(GPUInfo(
            index=i,
            name=props.name,
            total_memory_mb=total_mem,
            free_memory_mb=free_mem,
            compute_capability=(props.major, props.minor)
        ))
    
    return gpus


def get_cpu_info() -> Tuple[int, str]:
    """Get CPU information."""
    num_cpus = os.cpu_count() or 1
    
    cpu_name = "Unknown"
    try:
        if os.path.exists('/proc/cpuinfo'):
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if 'model name' in line:
                        cpu_name = line.split(':')[1].strip()
                        break
        else:
            # Windows
            import platform
            cpu_name = platform.processor()
    except:
        pass
    
    return num_cpus, cpu_name


def get_ram_info() -> Tuple[float, float]:
    """Get RAM information in GB."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.total / (1024**3), mem.available / (1024**3)
    except ImportError:
        # Fallback for systems without psutil
        try:
            with open('/proc/meminfo') as f:
                lines = f.readlines()
                total = int(lines[0].split()[1]) / (1024**2)  # KB to GB
                available = int(lines[2].split()[1]) / (1024**2)
                return total, available
        except:
            return 0.0, 0.0


def detect_hardware() -> HardwareInfo:
    """
    Detect all hardware specifications.
    
    Returns:
        HardwareInfo with detected specs and optimal settings
    """
    info = HardwareInfo()
    
    # GPU detection
    info.gpus = get_gpu_info()
    info.num_gpus = len(info.gpus)
    
    if info.gpus:
        info.total_gpu_memory_gb = sum(g.total_memory_gb for g in info.gpus)
        info.min_gpu_memory_gb = min(g.total_memory_gb for g in info.gpus)
    
    # CPU detection
    info.num_cpus, info.cpu_name = get_cpu_info()
    
    # RAM detection
    info.total_ram_gb, info.available_ram_gb = get_ram_info()
    
    # Calculate optimal settings
    _calculate_optimal_settings(info)
    
    return info


def _calculate_optimal_settings(info: HardwareInfo):
    """
    Calculate optimal training settings based on hardware.
    
    Optimization strategy:
    - Maximize batch size while fitting in GPU memory
    - Use gradient accumulation to reach effective batch ~256
    - Workers = 3 per GPU (capped at CPU count)
    - Enable all memory optimizations for smaller GPUs
    """
    
    if info.num_gpus == 0:
        # CPU-only fallback
        info.optimal_batch_size = 4
        info.optimal_grad_accum = 8
        info.optimal_num_workers = min(4, info.num_cpus)
        info.optimal_prefetch_factor = 2
        info.use_fp16 = False
        info.use_gradient_checkpointing = True
        info.deepspeed_stage = 0
        return
    
    # Per-GPU memory determines batch size
    # Rough estimates based on model size (~150M params) and image size (224x224)
    # With FP16 + gradient checkpointing:
    #   - 24GB GPU (L4, RTX 3090): batch 16-24
    #   - 16GB GPU (T4, RTX 4080): batch 8-12
    #   - 12GB GPU (RTX 3060): batch 4-8
    #   - 8GB GPU (RTX 3070): batch 2-4
    
    min_gpu_mem = info.min_gpu_memory_gb
    
    if min_gpu_mem >= 20:  # L4, A10, RTX 3090, RTX 4090
        info.optimal_batch_size = 24  # L4 (22GB) only uses ~1-2 GB; plenty of headroom
        info.use_gradient_checkpointing = True
        info.deepspeed_stage = 2
    elif min_gpu_mem >= 14:  # T4, RTX 4080
        info.optimal_batch_size = 16
        info.use_gradient_checkpointing = True
        info.deepspeed_stage = 2
    elif min_gpu_mem >= 10:  # RTX 3060
        info.optimal_batch_size = 8
        info.use_gradient_checkpointing = True
        info.deepspeed_stage = 2
    elif min_gpu_mem >= 6:  # RTX 3060 Ti
        info.optimal_batch_size = 4
        info.use_gradient_checkpointing = True
        info.deepspeed_stage = 2
    else:
        info.optimal_batch_size = 2
        info.use_gradient_checkpointing = True
        info.deepspeed_stage = 3  # ZeRO-3 for very limited memory
    
    # FP16 supported on all modern GPUs (compute capability >= 7.0)
    # V2: bf16 only on sm_80+ (Ampere/Hopper/Blackwell). Below that we must
    # force QLoRA (4-bit NF4 weights with fp16 compute) for the Qwen backbone.
    if info.gpus:
        min_cc = min(g.compute_capability for g in info.gpus)
        info.min_compute_capability = min_cc
        info.use_fp16 = min_cc[0] >= 7
        info.use_bf16 = min_cc >= (8, 0)
        # Below Ampere: no bf16 → force QLoRA. Above: still allow opt-in QLoRA
        # for memory savings, but don't force.
        info.force_qlora = min_cc < (8, 0)
        if info.force_qlora:
            logger.warning(
                f"Detected GPU with compute capability {min_cc[0]}.{min_cc[1]} (<8.0). "
                f"Forcing QLoRA (4-bit NF4 weights, fp16 compute). bf16 unavailable."
            )
    
    # Calculate gradient accumulation to target effective batch ~192
    # Lower accumulation = fewer wasted cycles = faster wall-clock time
    target_effective_batch = 192
    current_effective = info.optimal_batch_size * info.num_gpus
    info.optimal_grad_accum = max(1, target_effective_batch // current_effective)
    
    # Cap grad accum at reasonable levels
    info.optimal_grad_accum = min(info.optimal_grad_accum, 8)
    
    # Workers PER RANK (each DeepSpeed/DDP rank spawns its own workers).
    # With 4 ranks × N workers = 4N total forked processes, each inheriting
    # the parent's multi-GB sample cache via copy-on-write → RAM grows over time.
    # Safe: 2 workers/rank for multi-GPU, 4 for single GPU.
    if info.num_gpus >= 4:
        info.optimal_num_workers = 2   # 4 ranks × 2 = 8 total
    elif info.num_gpus >= 2:
        info.optimal_num_workers = 4   # 2 ranks × 4 = 8 total
    else:
        info.optimal_num_workers = min(8, max(1, info.num_cpus - 2))
    
    # Prefetch factor based on available RAM
    if info.available_ram_gb > 64:
        info.optimal_prefetch_factor = 4
    elif info.available_ram_gb > 32:
        info.optimal_prefetch_factor = 3
    else:
        info.optimal_prefetch_factor = 2


def print_hardware_info(info: HardwareInfo):
    """Print detected hardware information."""
    print("=" * 70)
    print("HARDWARE DETECTION RESULTS")
    print("=" * 70)
    
    # GPU info
    print(f"\n{'GPU CONFIGURATION':^70}")
    print("-" * 70)
    if info.num_gpus > 0:
        print(f"  Number of GPUs:        {info.num_gpus}")
        print(f"  Total GPU Memory:      {info.total_gpu_memory_gb:.1f} GB")
        print(f"  Min GPU Memory:        {info.min_gpu_memory_gb:.1f} GB")
        print()
        for gpu in info.gpus:
            print(f"  GPU {gpu.index}: {gpu.name}")
            print(f"       Memory: {gpu.total_memory_gb:.1f} GB (Free: {gpu.free_memory_gb:.1f} GB)")
            print(f"       Compute: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
    else:
        print("  No GPUs detected - will use CPU (not recommended)")
    
    # CPU info
    print(f"\n{'CPU CONFIGURATION':^70}")
    print("-" * 70)
    print(f"  CPU:                   {info.cpu_name}")
    print(f"  CPU Cores:             {info.num_cpus}")
    
    # RAM info
    print(f"\n{'MEMORY CONFIGURATION':^70}")
    print("-" * 70)
    print(f"  Total RAM:             {info.total_ram_gb:.1f} GB")
    print(f"  Available RAM:         {info.available_ram_gb:.1f} GB")
    
    # Optimal settings
    print(f"\n{'OPTIMAL TRAINING SETTINGS':^70}")
    print("-" * 70)
    effective_batch = info.optimal_batch_size * max(1, info.num_gpus) * info.optimal_grad_accum
    print(f"  Batch size per GPU:    {info.optimal_batch_size}")
    print(f"  Gradient accumulation: {info.optimal_grad_accum}")
    print(f"  Effective batch size:  {effective_batch}")
    print(f"  DataLoader workers:    {info.optimal_num_workers}")
    print(f"  Prefetch factor:       {info.optimal_prefetch_factor}")
    print(f"  Mixed precision (FP16):{' Enabled' if info.use_fp16 else ' Disabled'}")
    print(f"  Gradient checkpointing:{' Enabled' if info.use_gradient_checkpointing else ' Disabled'}")
    print(f"  DeepSpeed ZeRO stage:  {info.deepspeed_stage}")
    print("=" * 70)


def get_optimal_config_overrides(info: HardwareInfo) -> Dict[str, Any]:
    """
    Get configuration overrides based on detected hardware.
    
    Returns dict that can be used to override config values.
    """
    return {
        'training': {
            'batch_size_per_gpu': info.optimal_batch_size,
            'gradient_accumulation_steps': info.optimal_grad_accum,
            'dataloader_num_workers': info.optimal_num_workers,
            'dataloader_prefetch_factor': info.optimal_prefetch_factor,
            'fp16': info.use_fp16 and not info.use_bf16,
            'bf16': info.use_bf16,
            'gradient_checkpointing': info.use_gradient_checkpointing,
        },
        'model': {
            # V2: force 4-bit QLoRA on Turing-and-below GPUs.
            'use_quantization': info.force_qlora,
        },
        'deepspeed': {
            'enabled': info.num_gpus > 1 and info.deepspeed_stage > 0,
            'stage': info.deepspeed_stage,
        }
    }


def optimize_for_hardware(config: Any, auto_detect: bool = True) -> Any:
    """
    Optimize configuration for detected hardware.
    
    Args:
        config: The MIMICCXRVQAConfig object
        auto_detect: Whether to auto-detect and apply optimal settings
        
    Returns:
        Modified config with optimal settings
    """
    if not auto_detect:
        return config
    
    info = detect_hardware()
    
    # Log hardware info
    logger.info(f"Detected {info.num_gpus} GPUs with {info.total_gpu_memory_gb:.1f} GB total memory")
    logger.info(f"Detected {info.num_cpus} CPU cores with {info.available_ram_gb:.1f} GB available RAM")
    
    # Apply optimal settings
    config.training.batch_size_per_gpu = info.optimal_batch_size
    config.training.gradient_accumulation_steps = info.optimal_grad_accum
    config.training.dataloader_num_workers = info.optimal_num_workers
    config.training.dataloader_prefetch_factor = info.optimal_prefetch_factor
    config.training.fp16 = info.use_fp16
    config.training.gradient_checkpointing = info.use_gradient_checkpointing
    
    if hasattr(config, 'deepspeed'):
        config.deepspeed.enabled = info.num_gpus > 1 and info.deepspeed_stage > 0
        config.deepspeed.stage = info.deepspeed_stage
    
    effective_batch = info.optimal_batch_size * max(1, info.num_gpus) * info.optimal_grad_accum
    logger.info(f"Optimal settings: batch={info.optimal_batch_size}, accum={info.optimal_grad_accum}, effective={effective_batch}")
    
    return config


def set_optimal_environment():
    """Set optimal environment variables for training."""
    
    # CUDA optimizations
    os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')
    
    # Disable tokenizer parallelism (conflicts with DataLoader workers)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    
    # OMP threads (prevent oversubscription)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 0:
        omp_threads = max(1, (os.cpu_count() or 4) // (num_gpus * 4))
        os.environ.setdefault('OMP_NUM_THREADS', str(omp_threads))
    
    # NCCL optimizations for multi-GPU
    if num_gpus > 1:
        os.environ.setdefault('NCCL_DEBUG', 'WARN')
        os.environ.setdefault('NCCL_IB_DISABLE', '1')  # Disable InfiniBand if not available


def get_deepspeed_config_for_hardware(info: HardwareInfo, base_config_path: str) -> Dict[str, Any]:
    """
    Generate optimal DeepSpeed config based on hardware.
    
    Args:
        info: Detected hardware information
        base_config_path: Path to base DeepSpeed config JSON
        
    Returns:
        Modified DeepSpeed config dict
    """
    import json
    
    with open(base_config_path) as f:
        ds_config = json.load(f)
    
    # Update based on hardware
    ds_config['train_micro_batch_size_per_gpu'] = info.optimal_batch_size
    ds_config['gradient_accumulation_steps'] = info.optimal_grad_accum
    
    # ZeRO stage based on memory
    ds_config['zero_optimization']['stage'] = info.deepspeed_stage
    
    # Offload to CPU if memory is tight
    if info.min_gpu_memory_gb < 16:
        ds_config['zero_optimization']['offload_optimizer'] = {
            'device': 'cpu',
            'pin_memory': True
        }
    
    # FP16 settings
    ds_config['fp16']['enabled'] = info.use_fp16
    
    # Activation checkpointing
    ds_config['activation_checkpointing'] = {
        'partition_activations': info.use_gradient_checkpointing,
        'contiguous_memory_optimization': True,
        'cpu_checkpointing': info.min_gpu_memory_gb < 12
    }
    
    return ds_config


if __name__ == "__main__":
    """Run hardware detection and print results."""
    info = detect_hardware()
    print_hardware_info(info)
    
    print("\nConfiguration overrides:")
    overrides = get_optimal_config_overrides(info)
    import json
    print(json.dumps(overrides, indent=2))

