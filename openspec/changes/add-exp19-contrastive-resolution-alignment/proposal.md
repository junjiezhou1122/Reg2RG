# Change: Add Contrastive Resolution Alignment (Exp 19)

## Why
Compressed features lose information. Instead of trying to recover lost details post-hoc, use contrastive learning to make compressed features as similar as possible to original (uncompressed) features. This teaches the model to extract resolution-invariant representations.

## What Changes
- Add `ResolutionContrastiveLoss` module
- Positive pairs: (resized_features, original_features) from same region
- Negative pairs: features from different regions
- Train ViT to produce similar embeddings regardless of input resolution

## Impact
- Affected specs: resolution-invariant-features (new)
- Affected code:
  - `src/Model/resolution_contrastive.py` (new)
  - `src/train_radgenome.py` (loss integration)
- Priority: Medium
- Paper potential: "Resolution-Invariant Feature Learning via Contrastive Alignment"
