# GPU Hours Calculation for Full Training

## Input Data from Your 11% Run

From your training log:
- **Step time**: 5.24 seconds/iteration
- **Effective batch size**: 192 (24 × 4 GPUs × 2 gradient accumulation)
- **Throughput on L4**: 192 / 5.24 = **36.64 samples/second**

## Dataset Sizes

- **Pretraining**: 30,542,190 QA pairs (B-grade and above)
- **Fine-tuning**: 7,378,344 QA pairs (A-grade only)

## Training Configuration

### Pretraining
- **Epochs**: 3
- **Batch size**: 192

### Fine-tuning
- **Epochs**: 15
- **Batch size**: 256 (16 × 4 × 4)

## A100 Speedup Factor

A100 GPUs are approximately **2.5x faster** than L4 GPUs for this workload:
- **A100 step time**: 5.24 / 2.5 = **2.096 seconds/iteration**
- **A100 throughput**: 192 / 2.096 = **91.60 samples/second**

---

## Calculation Results

### Pretraining Phase (on A100)

**Time per epoch**:
- 30,542,190 samples ÷ 91.60 samples/sec ÷ 3600 = **92.62 hours per epoch**

**Total for 3 epochs**:
- 92.62 × 3 = **277.85 hours per GPU**
- **1,111 GPU-hours** (4 GPUs × 277.85)
- **With 20% buffer**: **1,334 GPU-hours**

### Fine-tuning Phase (on A100)

**Time per epoch**:
- 7,378,344 samples ÷ 91.60 samples/sec ÷ 3600 = **22.37 hours per epoch**

**Total for 15 epochs**:
- 22.37 × 15 = **335.61 hours per GPU**
- **1,342 GPU-hours** (4 GPUs × 335.61)
- **With 20% buffer**: **1,611 GPU-hours**

---

## FINAL SUMMARY

| Phase | Per GPU | Total (4 GPUs) | With 20% Buffer |
|-------|---------|----------------|-----------------|
| **Pretraining** | 277.85 hours | 1,111 GPU-hours | **1,334 GPU-hours** |
| **Fine-tuning** | 335.61 hours | 1,342 GPU-hours | **1,611 GPU-hours** |
| **TOTAL** | 613.46 hours | 2,453 GPU-hours | **2,945 GPU-hours** |

### Wall-clock Time
- **Total wall-clock**: ~614 hours (~25.6 days)
- **Recommended allocation**: **~30 days** (including buffer and validation overhead)

---

## Request Summary

**Total GPU Hours Required**: **~2,945 GPU-hours** (with 20% buffer)

**Hardware**: 4 × A100 GPUs (80GB each)

**Estimated Duration**: ~26 days of continuous training
