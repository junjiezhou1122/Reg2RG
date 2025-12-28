# Tasks: Exp 27 - Region-Specific Compression Strategy

## 1. Implementation
- [ ] 1.1 Implement `RegionCompressionStrategy` module
- [ ] 1.2 Define compression parameters: {tokens, depth, downsample_factor}
- [ ] 1.3 Create strategy lookup: region_name → compression_config
- [ ] 1.4 Make strategies learnable or use expert-defined

## 2. Strategy Definition
- [ ] 2.1 Lung: 16 tokens, depth=6, downsample=2x (complex)
- [ ] 2.2 Bone: 4 tokens, depth=2, downsample=4x (simple)
- [ ] 2.3 Heart: 8 tokens, depth=4, downsample=3x (medium)
- [ ] 2.4 Allow per-region override

## 3. Training
- [ ] 3.1 Apply different compression per region during training
- [ ] 3.2 Learn optimal strategy from reconstruction loss
- [ ] 3.3 Regularize for efficiency (prefer fewer tokens)

## 4. Evaluation
- [ ] 4.1 Per-region reconstruction quality
- [ ] 4.2 Total tokens used vs baseline
- [ ] 4.3 Report generation quality per region
