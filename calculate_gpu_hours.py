#!/usr/bin/env python3
"""
Calculate GPU hours for full pretraining and fine-tuning based on:
1. Your actual 11% run time, OR
2. Samples/second throughput from logs
"""

import argparse
from datetime import datetime, timedelta

# Dataset sizes (from MIMIC-Ext-CXR-QBA documentation)
PRETRAIN_TRAIN_SAMPLES = 30_542_190
PRETRAIN_VAL_SAMPLES = 246_233
FINETUNE_TRAIN_SAMPLES = 7_378_344
FINETUNE_VAL_SAMPLES = 58_486

# Training configs
PRETRAIN_EPOCHS = 3
FINETUNE_EPOCHS = 15
NUM_GPUS = 4

# A100 speedup factor (conservative: 2.5x faster than L4)
A100_SPEEDUP = 2.5


def calculate_from_time_percent(time_11_percent_hours: float, dataset_percent: float = 11.0):
    """
    Calculate full training time based on time taken for X% of data.
    
    Args:
        time_11_percent_hours: Time taken for 11% (or dataset_percent) of data
        dataset_percent: Percentage of data used (default 11%)
    
    Returns:
        Time for 100% of data (on same hardware)
    """
    time_full = time_11_percent_hours * (100.0 / dataset_percent)
    return time_full


def calculate_from_throughput(samples_per_second: float, num_samples: int, epochs: int):
    """
    Calculate training time from throughput.
    
    Args:
        samples_per_second: Throughput (samples/sec across all GPUs)
        num_samples: Total samples per epoch
        epochs: Number of epochs
    
    Returns:
        Total time in hours
    """
    time_per_epoch_hours = num_samples / samples_per_second / 3600
    total_hours = time_per_epoch_hours * epochs
    return total_hours


def estimate_gpu_hours():
    """Main calculation function."""
    parser = argparse.ArgumentParser(
        description="Calculate GPU hours for full pretraining and fine-tuning"
    )
    
    # Method 1: From 11% run time
    parser.add_argument(
        '--time_11_percent',
        type=float,
        help='Time taken for 11%% of pretraining data (in hours)'
    )
    
    # Method 2: From throughput
    parser.add_argument(
        '--samples_per_sec',
        type=float,
        help='Throughput in samples/second (across all GPUs)'
    )
    
    # Method 3: From step time
    parser.add_argument(
        '--step_time_sec',
        type=float,
        help='Time per training step in seconds'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Effective batch size (required with --step_time_sec)'
    )
    
    # Buffer percentage
    parser.add_argument(
        '--buffer_percent',
        type=float,
        default=20.0,
        help='Buffer percentage for overhead (default: 20%%)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GPU HOURS CALCULATION FOR MIMIC-CXR VQA TRAINING")
    print("=" * 70)
    print()
    
    # Calculate pretraining time
    if args.time_11_percent:
        print(f"Method: Extrapolating from {args.time_11_percent:.2f} hours for 11% of data")
        print()
        
        # Time on L4 (your current hardware)
        pretrain_l4_hours = calculate_from_time_percent(args.time_11_percent, 11.0)
        print(f"Pretraining (100%) on L4: {pretrain_l4_hours:.2f} hours")
        
        # Time on A100 (requested hardware)
        pretrain_a100_hours = pretrain_l4_hours / A100_SPEEDUP
        print(f"Pretraining (100%) on A100: {pretrain_a100_hours:.2f} hours ({A100_SPEEDUP}x speedup)")
        
        # Fine-tuning estimate (smaller dataset, but more epochs)
        # Assume similar throughput, but account for different dataset size and epochs
        finetune_samples_ratio = FINETUNE_TRAIN_SAMPLES / PRETRAIN_TRAIN_SAMPLES
        finetune_epochs_ratio = FINETUNE_EPOCHS / PRETRAIN_EPOCHS
        finetune_a100_hours = pretrain_a100_hours * finetune_samples_ratio * finetune_epochs_ratio
        print(f"Fine-tuning (100%) on A100: {finetune_a100_hours:.2f} hours")
        print(f"  (Ratio: {finetune_samples_ratio:.2f}x samples, {finetune_epochs_ratio:.1f}x epochs)")
        
    elif args.samples_per_sec:
        print(f"Method: Using throughput of {args.samples_per_sec:.2f} samples/sec")
        print()
        
        pretrain_hours = calculate_from_throughput(
            args.samples_per_sec, PRETRAIN_TRAIN_SAMPLES, PRETRAIN_EPOCHS
        )
        finetune_hours = calculate_from_throughput(
            args.samples_per_sec, FINETUNE_TRAIN_SAMPLES, FINETUNE_EPOCHS
        )
        
        print(f"Pretraining: {pretrain_hours:.2f} hours")
        print(f"Fine-tuning: {finetune_hours:.2f} hours")
        pretrain_a100_hours = pretrain_hours
        finetune_a100_hours = finetune_hours
        
    elif args.step_time_sec and args.batch_size:
        print(f"Method: Using step time of {args.step_time_sec:.2f} sec with batch size {args.batch_size}")
        print()
        
        samples_per_sec = args.batch_size / args.step_time_sec
        print(f"Calculated throughput: {samples_per_sec:.2f} samples/sec")
        print()
        
        pretrain_hours = calculate_from_throughput(
            samples_per_sec, PRETRAIN_TRAIN_SAMPLES, PRETRAIN_EPOCHS
        )
        finetune_hours = calculate_from_throughput(
            samples_per_sec, FINETUNE_TRAIN_SAMPLES, FINETUNE_EPOCHS
        )
        
        print(f"Pretraining: {pretrain_hours:.2f} hours")
        print(f"Fine-tuning: {finetune_hours:.2f} hours")
        pretrain_a100_hours = pretrain_hours
        finetune_a100_hours = finetune_hours
        
    else:
        print("ERROR: Must provide one of:")
        print("  --time_11_percent HOURS")
        print("  --samples_per_sec RATE")
        print("  --step_time_sec SEC --batch_size SIZE")
        return
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Add buffer
    buffer_mult = 1.0 + (args.buffer_percent / 100.0)
    pretrain_total = pretrain_a100_hours * buffer_mult
    finetune_total = finetune_a100_hours * buffer_mult
    
    print(f"Pretraining GPU-hours (with {args.buffer_percent}% buffer):")
    print(f"  Per GPU: {pretrain_a100_hours:.2f} hours")
    print(f"  Total (4 GPUs): {pretrain_a100_hours * NUM_GPUS:.2f} GPU-hours")
    print(f"  With buffer: {pretrain_total * NUM_GPUS:.2f} GPU-hours")
    print()
    
    print(f"Fine-tuning GPU-hours (with {args.buffer_percent}% buffer):")
    print(f"  Per GPU: {finetune_a100_hours:.2f} hours")
    print(f"  Total (4 GPUs): {finetune_a100_hours * NUM_GPUS:.2f} GPU-hours")
    print(f"  With buffer: {finetune_total * NUM_GPUS:.2f} GPU-hours")
    print()
    
    total_gpu_hours = (pretrain_total + finetune_total) * NUM_GPUS
    print(f"TOTAL GPU-HOURS: {total_gpu_hours:.0f} GPU-hours")
    print()
    
    # Estimate calendar time
    total_wall_hours = pretrain_a100_hours + finetune_a100_hours
    print(f"Estimated wall-clock time: {total_wall_hours:.1f} hours ({total_wall_hours/24:.1f} days)")
    print()
    
    # Generate date range suggestion
    print("=" * 70)
    print("SUGGESTED DATE RANGE")
    print("=" * 70)
    print("Note: Andrew mentioned no capacity next week.")
    print("Suggested start date: [DATE AFTER NEXT WEEK]")
    print(f"Suggested end date: [START DATE] + {int(total_wall_hours/24) + 2} days (buffer)")
    print()


if __name__ == "__main__":
    estimate_gpu_hours()
