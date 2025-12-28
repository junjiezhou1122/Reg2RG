# Change: Add Adaptive Patch Size (Exp 25)

## Why
Fixed patch size is suboptimal. Small patches capture details but lose context; large patches capture context but lose details. Use adaptive patch sizes based on content complexity: simple areas → large patches, complex areas → small patches.

## What Changes
- Add `AdaptivePatcher` module that predicts optimal patch size per region
- Support multiple patch sizes: [16, 32, 64]
- Use complexity predictor to select patch size
- Combine multi-scale features via cross-attention

## Impact
- Affected specs: dynamic-patching (new)
- Affected code:
  - `src/Model/adaptive_patcher.py` (new)
  - `src/Model/vit_3d.py` (multi-scale support)
- Priority: Low
- Complexity: High (architectural change to ViT)
- Paper potential: "Adaptive Patching for Medical Vision Transformers"
