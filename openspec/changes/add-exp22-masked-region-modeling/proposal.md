# Change: Add Masked Region Modeling (Exp 22)

## Why
Self-supervised pre-training (like MAE) helps models learn robust representations. Apply masked region modeling to learn better compression-robust features - mask some regions, encode remaining, predict masked regions from encoded.

## What Changes
- Add `MaskedRegionModeling` pre-training objective
- Randomly mask 25-50% of regions during pre-training
- Train adapter to predict masked region features from visible ones
- Fine-tune on downstream report generation

## Impact
- Affected specs: self-supervised-pretraining (new)
- Affected code:
  - `src/Model/masked_region_modeling.py` (new)
  - `src/pretrain_mrm.py` (new training script)
- Priority: Medium
- Paper potential: "Masked Region Modeling for Medical Vision Encoders"
