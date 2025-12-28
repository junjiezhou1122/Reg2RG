# Change: Add Multi-Crop Encoding (Exp 14)

## Why
Resolution mismatch is fundamental: RadFM pretrained on 256³ but region masks may be larger. Rather than naive downsampling (losing details), use multi-crop strategy: encode overlapping crops and merge. This preserves fine details at the cost of more compute.

## What Changes
- Add `MultiCropEncoder` that splits large regions into overlapping crops
- Encode each crop with ViT, then merge using attention-based pooling
- Support configurable crop size and overlap ratio
- Add feature merging strategies: concat, mean, attention-weighted

## Impact
- Affected specs: multi-scale-vision (new)
- Affected code:
  - `src/Model/multi_crop_encoder.py` (new)
  - `src/Model/my_embedding_layer.py` (integration)
- Priority: Medium-High
- Trade-off: More compute, but preserves details
- Paper potential: "Multi-Crop Visual Encoding for High-Resolution Medical VLMs"
