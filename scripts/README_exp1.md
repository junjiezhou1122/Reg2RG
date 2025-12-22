# Experiment 1: Decoder Depth Ablation - Quick Start Guide

## 📋 Overview

This script runs 4 variants of the LIT probe experiment with different decoder depths to validate whether the pretrained Perceiver adapter preserves sufficient anatomical information.

## 🎯 Experiments

| Variant | decoder_layers | Expected Performance |
|---------|---------------|---------------------|
| exp1a | 1 | Low (limited capacity) |
| exp1b | 2 | Medium (current default) |
| exp1c | 4 | High |
| exp1d | 6 | Highest (may overfit) |

## ⚙️ Configuration

- **Epochs**: 5 (reduced from 15 for faster iteration)
- **Batch size**: 1
- **Gradient accumulation**: 8 (effective batch size = 8)
- **Learning rate**: 1e-4
- **Validation frequency**: Every **10,000 steps** (true step-level validation)
- **Early stopping**: Patience = **1 validation** (stop if no improvement after 1 validation check)
- **CUDA device**: Configurable (default: GPU 0)

## 🔄 Validation Frequency

**Step-level validation is now implemented!**

With 24,126 training samples:
- Each epoch = 24,126 / 8 = **~3,015 effective steps**
- Validation runs **every 10,000 steps** = approximately every **3.3 epochs**

**Validation schedule example**:
- Step 0: Epoch-end validation (epoch 1)
- Step 3,015: Epoch-end validation (epoch 2)
- Step 6,030: Epoch-end validation (epoch 3)
- Step 9,045: Epoch-end validation (epoch 4)
- Step 10,000: **Mid-epoch validation** (during epoch 4)
- Step 12,060: Epoch-end validation (epoch 5)

**Benefits**:
- ✅ Validation every ~3-4 epochs provides frequent feedback
- ✅ Early stopping can trigger mid-epoch for faster iteration
- ✅ Best checkpoints saved whenever improvement occurs

## 🚀 Usage

### Step 1: Ensure cache is ready

```bash
# Check cache status
ls -lh /mnt2/ct/RadGenome-ChestCT/cache_lit/*.pt | wc -l
# Should show ~25,000 files (train + val)

# If not ready, run warmup:
python src/lit_recon_probe.py --precache_splits both --precache_only True
```

### Step 2: Run experiments

```bash
# Make script executable (if not already)
chmod +x scripts/run_exp1_decoder_ablation.sh

# Run all 4 variants
bash scripts/run_exp1_decoder_ablation.sh
```

### Step 3: Monitor progress

The script will:
1. Run exp1a (decoder=1 layer)
2. Run exp1b (decoder=2 layers)
3. Run exp1c (decoder=4 layers)
4. Run exp1d (decoder=6 layers)

Each experiment logs to:
- **Console output** (real-time)
- **Log file**: `/mnt/home/zhoujunjie/outputs/LIT_exp1/exp1a/training.log`
- **W&B dashboard**: https://wandb.ai (if enabled)
- **Metrics CSV**: `/mnt/home/zhoujunjie/outputs/LIT_exp1/exp1a/lit_metrics.csv`

## ⏱️ Expected Runtime

With caching enabled:
- **Per epoch**: ~2-3 hours
- **Per experiment**: ~10-15 hours (5 epochs)
- **All 4 experiments**: ~40-60 hours total

## 📊 Results

After completion, check:

```bash
# View summary
cat /mnt/home/zhoujunjie/outputs/LIT_exp1/exp1_summary.txt

# Compare metrics
for exp in exp1a exp1b exp1c exp1d; do
    echo "=== $exp ==="
    tail -5 /mnt/home/zhoujunjie/outputs/LIT_exp1/$exp/lit_metrics.csv
done
```

## 🛑 Early Stopping

If validation `reg_cos` doesn't improve for **1 validation check**, training stops automatically. This is more aggressive than the previous patience=3, allowing faster iteration when a variant isn't learning effectively.

**Example scenario**:
- Step 10,000: reg_cos = 0.65 (new best ✅)
- Step 20,000: reg_cos = 0.63 (no improvement, count=1)
- **Training stops automatically** - saves time!

Note: Early stopping counts validation checks (every 10K steps OR epoch-end), not just epochs.

## ⚠️ Troubleshooting

### Script fails with "cache not found"
```bash
# Re-run cache warmup
python src/lit_recon_probe.py --precache_splits both --precache_only True
```

### Out of memory
```bash
# Reduce num_workers in the script
# Edit line: NUM_WORKERS=4 → NUM_WORKERS=2
```

### Want to resume a failed experiment
```bash
# Find the checkpoint
ls /mnt/home/zhoujunjie/outputs/LIT_exp1/exp1a/checkpoints/

# Resume training
python src/lit_recon_probe.py \
  --resume_from_checkpoint /path/to/checkpoint.pt \
  --decoder_layers 1 \
  --num_train_epochs 5 \
  --output_dir /mnt/home/zhoujunjie/outputs/LIT_exp1/exp1a
```

## 📝 Customization

Edit `scripts/run_exp1_decoder_ablation.sh`:

```bash
# Change epochs
NUM_EPOCHS=10  # Line 34

# Change validation interval
VAL_CHECK_INTERVAL=5000  # Line 41 (every 5K steps instead of 10K)

# Change early stopping patience
EARLY_STOPPING_PATIENCE=2  # Line 42 (wait for 2 validation checks)

# Change CUDA device
CUDA_DEVICE="1"  # Line 45 (use GPU 1 instead of GPU 0)
# Or use multiple GPUs:
CUDA_DEVICE="0,1"  # Use GPU 0 and 1

# Add more decoder variants
DECODER_LAYERS=(1 2 3 4 6 8)  # Line 51
EXP_NAMES=("exp1a" "exp1b" "exp1c" "exp1d" "exp1e" "exp1f")  # Line 52
```

**Parameter explanations**:
- `VAL_CHECK_INTERVAL=0`: Disable step-level validation (only validate at epoch-end)
- `EARLY_STOPPING_PATIENCE=0`: Disable early stopping (train for full epochs)
- `CUDA_DEVICE=""`: Leave empty to use all available GPUs

## 🎓 Interpreting Results

### Hypothesis 1: Adapter is effective
```
If: exp1b (2 layers) ≈ exp1d (6 layers)
Then: Adapter preserves info well, decoder just does linear decoding
Action: Current architecture is good
```

### Hypothesis 2: Adapter loses information
```
If: exp1d (6 layers) >> exp1b (2 layers)
Then: Adapter compresses too much, deep decoder compensates
Action: Need to improve adapter (run Exp2)
```

## 📚 Next Steps

After Exp1 completes:
1. **Analyze results**: Compare reg_cos across variants
2. **Document findings**: Update docs/LIT_Experiment_Design_v2.md
3. **Decide next experiment**: Run Exp2 (joint training) if needed

---

**Questions?** Check the full experiment design: `docs/LIT_Experiment_Design_v2.md`
