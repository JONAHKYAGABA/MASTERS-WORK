# GPU Hours Estimation for Full Training

## Dataset Sizes (from MIMIC-Ext-CXR-QBA documentation)

### Pretraining Phase
- **Total QA pairs**: ~30,542,190 (B-grade and above, train split)
- **Validation pairs**: ~246,233
- **Quality grade**: All (B-grade and above)
- **View filter**: Frontal only

### Fine-tuning Phase
- **Total QA pairs**: ~7,378,344 (A-grade only, train split)
- **Validation pairs**: ~58,486
- **Quality grade**: A (highest quality)
- **View filter**: Frontal only

---

## Training Configuration

### Pretraining (Phase 1)
- **GPUs**: 4 × A100 (80GB each) - **upgrade from L4**
- **Batch size per GPU**: 24
- **Gradient accumulation**: 2
- **Effective batch size**: 24 × 4 × 2 = **192**
- **Epochs**: 3
- **DeepSpeed ZeRO Stage 2**: Enabled
- **FP16**: Enabled
- **Gradient checkpointing**: Enabled

### Fine-tuning (Phase 2)
- **GPUs**: 4 × A100 (80GB each)
- **Batch size per GPU**: 16
- **Gradient accumulation**: 4
- **Effective batch size**: 16 × 4 × 4 = **256**
- **Epochs**: 15
- **DeepSpeed ZeRO Stage 2**: Enabled
- **FP16**: Enabled
- **Gradient checkpointing**: Enabled

---

## Time Estimation

### Assumptions
1. **A100 vs L4 speedup**: A100 has ~2.5-3x more compute than L4 (312 TFLOPS vs 120 TFLOPS)
   - Conservative estimate: **2.5x faster** per sample
2. **Your 11% run**: Need actual time from your logs
   - Example: If 11% took X hours on L4, full 100% will take ~9.1X hours on L4
   - On A100: ~9.1X / 2.5 = **~3.6X hours**

### Calculation Method

#### Step 1: Estimate from your 11% run
If your 11% run took **T hours** on 4×L4:
- Full dataset (100%) on L4: **T × (100/11) = ~9.1T hours**
- Full dataset on A100: **~9.1T / 2.5 = ~3.6T hours**

#### Step 2: Alternative calculation (if you have samples/sec)
If you know your throughput (samples/second):
- **Pretraining**: 30,542,190 samples ÷ samples/sec ÷ 3600 = hours per epoch
- **Fine-tuning**: 7,378,344 samples ÷ samples/sec ÷ 3600 = hours per epoch

#### Step 3: Conservative estimates (if no logs available)

**Pretraining (conservative estimate)**:
- Assuming ~2-3 samples/sec per GPU on A100 (with gradient checkpointing)
- Total throughput: 4 GPUs × 2.5 samples/sec = **10 samples/sec**
- Samples per epoch: 30,542,190
- Time per epoch: 30,542,190 ÷ 10 ÷ 3600 = **~848 hours per epoch**
- **Total for 3 epochs: ~2,544 GPU-hours** (4 GPUs × 636 hours)

**Fine-tuning (conservative estimate)**:
- Same throughput: **10 samples/sec**
- Samples per epoch: 7,378,344
- Time per epoch: 7,378,344 ÷ 10 ÷ 3600 = **~205 hours per epoch**
- **Total for 15 epochs: ~3,075 GPU-hours** (4 GPUs × 768 hours)

**TOTAL ESTIMATE (conservative)**: **~5,619 GPU-hours**

---

## Recommended Approach

### Option 1: Use your actual training logs
If you have logs showing:
- Time to complete 11% of pretraining
- Or samples/second throughput

I can calculate exact GPU hours.

### Option 2: Conservative buffer
Request **6,000-7,000 GPU-hours** to account for:
- Validation runs
- Checkpoint saving overhead
- Potential restarts
- Hyperparameter tuning

---

## Email Template

**Subject**: GPU Hours Request - MIMIC-CXR VQA Pretraining + Fine-tuning

Hi Andrew,

Thank you for the update. Below are my GPU hours requirements:

**Hardware Request**:
- **4 × A100 GPUs** (80GB each)
- RAM and storage as previously specified

**Training Phases**:

1. **Pretraining Phase**:
   - Dataset: ~30.5M QA pairs (B-grade and above, frontal only)
   - Configuration: 3 epochs, batch size 192 (24×4×2)
   - Estimated GPU hours: **[CALCULATE FROM YOUR LOGS]**

2. **Fine-tuning Phase**:
   - Dataset: ~7.4M QA pairs (A-grade only, frontal only)
   - Configuration: 15 epochs, batch size 256 (16×4×4)
   - Estimated GPU hours: **[CALCULATE FROM YOUR LOGS]**

**Total GPU Hours**: **[PRETRAIN + FINETUNE + 20% buffer]**

**Requested Dates**:
- Start: **[DATE AFTER NEXT WEEK]**
- End: **[DATE + DURATION]**

I have completed preliminary experiments on 11% of the pretraining data using 4×L4 GPUs, which validates the pipeline. The full training will use the same configuration scaled to the complete dataset on A100 hardware.

Please let me know if you need any additional details.

Best regards,
Jonah
