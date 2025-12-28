# Change: Add Hardness-Aware Loss Weighting (Exp 35)

## Why
Training data shows that some regions are much harder to learn than others:
- Thyroid: cos = 0.33 (hardest), but only 2K voxels - gets little gradient signal
- Trachea: cos = 0.70 (easiest), 3K voxels
- Large regions (lung, pleura) dominate the loss due to volume

The problem: Hard regions with small volume get "ignored" because their contribution to total loss is minimal. This leads to poor performance on clinically important small organs like thyroid.

**Key Insight from Data**:
- Thyroid: cos=0.33, F1=0.061 (worst both)
- Trachea: cos=0.70, F1=0.000 (easy to reconstruct but no clinical value!)

## What Changes
- Add `HardnessAwareLoss` module that dynamically weights region losses
- Track running average cos per region using EMA
- Higher weight for regions with lower cos (harder to learn)
- Weight range: [1.0, 2.0] based on hardness = 1 - cos

## Impact
- Affected specs: hardness-loss (new)
- Affected code:
  - `src/Model/hardness_aware_loss.py` (new)
  - `src/lit_recon_probe.py` (integrate loss weighting)
- Priority: **Very High** (simplest to implement, immediate impact)
- Dependency: None (can start immediately)
- Paper potential: "Hardness-Aware Training for Balanced Medical Image Understanding"

## Expected Results
- Thyroid cos: 0.33 -> 0.45+
- More balanced learning across all regions
- Better clinical performance on small organs
