# Tasks: Exp 25 - Adaptive Patch Size

## 1. Implementation
- [ ] 1.1 Implement `ComplexityPredictor` for patch size selection
- [ ] 1.2 Implement multi-scale patching (16/32/64)
- [ ] 1.3 Add `AdaptivePatcher` module
- [ ] 1.4 Implement cross-attention for multi-scale fusion

## 2. ViT Modification
- [ ] 2.1 Support variable patch sizes in ViT-3D
- [ ] 2.2 Handle different token counts from different patch sizes
- [ ] 2.3 Add positional encoding adaptation

## 3. Patch Size Selection
- [ ] 3.1 Complexity metric: local variance, edge density
- [ ] 3.2 Train complexity predictor to select optimal size
- [ ] 3.3 Gumbel-softmax for differentiable selection

## 4. Evaluation
- [ ] 4.1 Compare: Fixed patch vs Adaptive patch
- [ ] 4.2 Compute cost: Tokens per region
- [ ] 4.3 Quality: Reconstruction and report generation
