# Tasks: Exp 14 - Multi-Crop Encoding

## 1. Implementation
- [ ] 1.1 Implement `crop_volume(volume, crop_size, overlap)` utility
- [ ] 1.2 Implement `MultiCropEncoder` class
- [ ] 1.3 Add crop position encoding (which crop came from where)
- [ ] 1.4 Implement feature merging strategies:
  - [ ] 1.4.1 Concat (most information, but larger)
  - [ ] 1.4.2 Mean pooling (simplest)
  - [ ] 1.4.3 Attention-weighted (learnable)

## 2. Attention-Based Merging
- [ ] 2.1 Implement `CropAttentionMerger` with cross-attention
- [ ] 2.2 Use learnable query tokens to aggregate across crops
- [ ] 2.3 Add positional encoding for crop locations

## 3. Training
- [ ] 3.1 Handle variable number of crops per region
- [ ] 3.2 Adjust batch size for increased memory
- [ ] 3.3 Gradient checkpointing for crop encoding

## 4. Experiments
- [ ] 4.1 Config A: Single crop, 256³ (baseline)
- [ ] 4.2 Config B: 2×2×1 crops with 50% overlap
- [ ] 4.3 Config C: 2×2×2 crops with 50% overlap
- [ ] 4.4 Compare merging strategies

## 5. Evaluation
- [ ] 5.1 Reconstruction cosine per crop count
- [ ] 5.2 Report generation metrics
- [ ] 5.3 Compute efficiency (FLOPs, memory, time)
- [ ] 5.4 Identify optimal crop configuration
