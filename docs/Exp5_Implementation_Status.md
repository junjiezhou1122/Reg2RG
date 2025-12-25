# Experiment 5 Implementation Complete ✅

**Date**: 2025-12-25
**Status**: Code infrastructure ready for training
**Experiments**: Exp5a (Extracted 1st Layer) + Exp5b (Fresh 1-Layer)

---

## 📁 Files Created/Modified

### ✅ New Files Created

1. **`Model/one_layer_adapter.py`** (291 lines)
   - `OneLayerAdapter` class: 1-layer Perceiver-style cross-attention adapter
   - Single attention block + FFN (no iterative refinement)
   - Forced to preserve all information in one compression step
   - ~600K parameters (vs 3.5M for 6-layer Perceiver)

2. **`Model/adapter_utils.py`** (260 lines)
   - `load_first_layer_from_6layer_checkpoint()`: Extract layer 0 weights
   - `print_adapter_summary()`: Parameter statistics
   - `compare_adapter_outputs()`: Validation utility
   - Weight mapping: `layers.0.0.xxx → xxx`, `layers.0.1.xxx → ff.xxx`

### ✅ Modified Files

3. **`src/lit_recon_probe.py`** (3 locations modified)

   **a) Imports** (lines 37-48):
   ```python
   from Model.one_layer_adapter import OneLayerAdapter
   from Model.adapter_utils import load_first_layer_from_6layer_checkpoint
   ```

   **b) ModelArguments** (lines 137-161):
   ```python
   adapter_depth: int = 6  # 1 for Exp5, 6 for baseline
   load_first_layer_from_pretrained: bool = False  # Exp5a flag
   random_init_adapter: bool = False  # Exp5b flag
   ```

   **c) LITProbeModel.__init__()** (lines 735-749):
   ```python
   if adapter_depth == 1:
       self.adapter = OneLayerAdapter(...)  # Minimal-capacity
   else:
       self.adapter = PerceiverResampler(...)  # Baseline
   ```

   **d) Model instantiation** (line 1407):
   ```python
   model = LITProbeModel(..., adapter_depth=model_args.adapter_depth)
   ```

   **e) Weight loading** (lines 1420-1463):
   ```python
   if random_init_adapter:
       # Exp5b: Skip loading, keep random weights
   elif adapter_depth == 1 and load_first_layer_from_pretrained:
       # Exp5a: Extract first layer from 6-layer checkpoint
       load_first_layer_from_6layer_checkpoint(...)
   else:
       # Normal: Load full checkpoint
   ```

---

## 🚀 Training Commands

### Experiment 5a: Extracted First-Layer Joint Training

**Hypothesis**: Test if Layer 1 from 6-layer adapter can work standalone when fine-tuned.

```bash
CUDA_VISIBLE_DEVICES=1 python3 src/lit_recon_probe.py \
    --tokenizer_path /mnt/home/zhoujunjie/models/Llama-2-7b-chat-hf \
    --pretrained_visual_encoder /mnt/home/zhoujunjie/models/Reg2RG/RadFM_vit3d.pth \
    --pretrained_adapter /mnt/home/zhoujunjie/models/Reg2RG/RadFM_perceiver_fc.pth \
    --adapter_depth 1 \
    --load_first_layer_from_pretrained True \
    --train_adapter True \
    --decoder_layers 4 \
    --decode_mode pre_proj \
    --data_folder /mnt2/ct/RadGenome-ChestCT/dataset/train_preprocessed \
    --mask_folder /mnt2/ct/RadGenome-ChestCT/dataset/train_region_mask \
    --report_file /mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/train_region_report.csv \
    --val_data_folder /mnt2/ct/RadGenome-ChestCT/dataset/valid_preprocessed \
    --val_mask_folder /mnt2/ct/RadGenome-ChestCT/dataset/valid_region_mask \
    --val_report_file /mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv \
    --monai_cache_dir /mnt2/ct/RadGenome-ChestCT/cache_lit \
    --output_dir /mnt/home/zhoujunjie/outputs/LIT_exp5a \
    --num_train_epochs 20 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --val_check_interval 0 \
    --save_top_k 3 \
    --monitor_metric reg_cos \
    --monitor_mode max \
    --use_wandb True \
    --wandb_project Reg2RG-LIT-Minimal-Capacity \
    --wandb_run_name exp5a_extracted_1st_layer_joint \
    --seed 42
```

**Expected console output:**
```
[INFO] 🔬 Using 1-layer adapter (minimal-capacity probe)
[INFO] 🧪 Exp5a: Loading 1st layer from 6-layer checkpoint
[INFO] Loading 1st layer from 6-layer checkpoint: /mnt/home/.../RadFM_perceiver_fc.pth
   ✓ Copied latents: torch.Size([32, 768])
   ✓ layers.0.0.norm_media.weight → norm_media.weight
   ...
[INFO] Loaded 15 parameters
[INFO] 🔥 Exp2 Mode: Training BOTH adapter + decoder (joint training)
[INFO] Training 589,824 adapter params + 2,359,296 decoder params
```

---

### Experiment 5b: Fresh 1-Layer Joint Training

**Hypothesis**: Fresh 1-layer adapter learns more "honest" compression than extracted layer.

```bash
CUDA_VISIBLE_DEVICES=1 python3 src/lit_recon_probe.py \
    --tokenizer_path /mnt/home/zhoujunjie/models/Llama-2-7b-chat-hf \
    --pretrained_visual_encoder /mnt/home/zhoujunjie/models/Reg2RG/RadFM_vit3d.pth \
    --adapter_depth 1 \
    --random_init_adapter True \
    --train_adapter True \
    --decoder_layers 4 \
    --decode_mode pre_proj \
    --data_folder /mnt2/ct/RadGenome-ChestCT/dataset/train_preprocessed \
    --mask_folder /mnt2/ct/RadGenome-ChestCT/dataset/train_region_mask \
    --report_file /mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/train_region_report.csv \
    --val_data_folder /mnt2/ct/RadGenome-ChestCT/dataset/valid_preprocessed \
    --val_mask_folder /mnt2/ct/RadGenome-ChestCT/dataset/valid_region_mask \
    --val_report_file /mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv \
    --monai_cache_dir /mnt2/ct/RadGenome-ChestCT/cache_lit \
    --output_dir /mnt/home/zhoujunjie/outputs/LIT_exp5b \
    --num_train_epochs 20 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --val_check_interval 0 \
    --save_top_k 3 \
    --monitor_metric reg_cos \
    --monitor_mode max \
    --use_wandb True \
    --wandb_project Reg2RG-LIT-Minimal-Capacity \
    --wandb_run_name exp5b_true_1layer_joint \
    --seed 42
```

**Expected console output:**
```
[INFO] 🔬 Using 1-layer adapter (minimal-capacity probe)
[INFO] ❄️  Adapter randomly initialized (no pretrained weights)
[INFO] 🧪 Exp5b: Training fresh 1-layer adapter from scratch
[INFO] 🔥 Exp2 Mode: Training BOTH adapter + decoder (joint training)
[INFO] Training 589,824 adapter params + 2,359,296 decoder params
```

---

## 📊 Expected Metrics to Monitor

### Critical Success Indicators

**ROI/Global Ratio** (most important):
- Exp2 (6-layer joint): ~0.867
- **Exp5b target**: > 0.92 ✨ (more balanced = "honest" compression)
- Exp5a target: > 0.84 (should be between Exp2 and Exp5b)

**Small Lesion Cosine** (clinical importance):
- Exp2 (6-layer joint): ~0.55
- **Exp5b target**: > 0.65 ✨ (better preservation of critical features)
- Exp5a target: > 0.50

**Global Cosine** (acceptable trade-off):
- Exp2 (6-layer joint): ~0.90
- **Exp5b target**: > 0.85 (slight degradation acceptable)
- Exp5a target: > 0.83

### W&B Dashboard Metrics

Monitor these during training:
- `val/reg_cos`: Region reconstruction quality
- `val/cos`: Global reconstruction quality
- `val/reg_cos / val/cos`: ROI/Global ratio (calculate manually)
- `train/loss_running`: Should converge smoothly

---

## 🔍 Quick Validation Checklist

Before starting full training, run a quick sanity check:

```bash
# Test Exp5a weight loading (5 epochs)
CUDA_VISIBLE_DEVICES=1 python3 src/lit_recon_probe.py \
    --adapter_depth 1 \
    --load_first_layer_from_pretrained True \
    --train_adapter True \
    --num_train_epochs 5 \
    --output_dir outputs/test_exp5a \
    [... other args ...]

# Test Exp5b fresh training (5 epochs)
CUDA_VISIBLE_DEVICES=1 python3 src/lit_recon_probe.py \
    --adapter_depth 1 \
    --random_init_adapter True \
    --train_adapter True \
    --num_train_epochs 5 \
    --output_dir outputs/test_exp5b \
    [... other args ...]
```

**Expected behavior:**
- ✅ No import errors
- ✅ Weight loading messages appear correctly
- ✅ Training loop starts without crashes
- ✅ Metrics logged to W&B (if enabled)
- ✅ First epoch completes in reasonable time (~20-30 min)

---

## 🧪 Debugging Tips

### If weight loading fails (Exp5a):

```python
# Check what's in the checkpoint
import torch
ckpt = torch.load("/path/to/RadFM_perceiver_fc.pth")
print("Keys:", ckpt.keys())
print("Perceiver keys:", list(ckpt["perceiver"].keys())[:10])
```

### If adapter shapes mismatch:

```python
# Compare adapter architectures
from Model.helpers import PerceiverResampler
from Model.one_layer_adapter import OneLayerAdapter

perceiver = PerceiverResampler(dim=768, num_latents=32)
one_layer = OneLayerAdapter(dim=768, num_latents=32)

print("Perceiver params:", sum(p.numel() for p in perceiver.parameters()))
print("OneLayer params:", sum(p.numel() for p in one_layer.parameters()))
```

### If training is unstable:

1. Reduce learning rate: `--learning_rate 5e-5` (instead of 1e-4)
2. Increase warmup: Add learning rate scheduler (future enhancement)
3. Check gradients: Add gradient clipping (future enhancement)

---

## ✅ Implementation Status

- [x] Create `Model/one_layer_adapter.py`
- [x] Create `Model/adapter_utils.py`
- [x] Modify `src/lit_recon_probe.py`:
  - [x] Add imports
  - [x] Add ModelArguments fields
  - [x] Modify LITProbeModel.__init__()
  - [x] Modify adapter initialization logic
  - [x] Modify weight loading logic
  - [x] Update model instantiation
- [x] Document training commands
- [ ] Run quick validation (5 epochs) ← **NEXT STEP**
- [ ] Run full Exp5a training (20 epochs)
- [ ] Run full Exp5b training (20 epochs)
- [ ] Analyze results and compare metrics

---

## 🎯 Decision Tree (After Results)

```python
if exp5b_roi_cos > exp2_roi_cos and exp5b_roi_global_ratio > 0.92:
    conclusion = "✅ Minimal-capacity principle confirmed!"
    next_steps = [
        "Write APR (Adapter Probing via Reconstruction) paper",
        "Use Exp5b adapter as evaluation standard",
        "Proceed to Stage 2 VLM training with 1-layer adapter"
    ]

elif exp5b_roi_cos ≈ exp2_roi_cos:
    conclusion = "✅ 1-layer sufficient, more efficient"
    next_steps = [
        "Recommend 1-layer for prototyping",
        "Optional: Try saliency-guided 1-layer next"
    ]

else:  # exp5b << exp2
    conclusion = "⚠️ Compression ratio too aggressive for 1-layer"
    next_steps = [
        "Test with more latents (64 instead of 32)",
        "Or test 2-layer as minimal capacity",
        "Still valuable: proves depth is needed"
    ]
```

---

**Last Updated**: 2025-12-25
**Ready for**: Quick validation testing (Phase 1 complete ✅)
**Next Phase**: Days 3-4 - Quick validation with 5 epochs
