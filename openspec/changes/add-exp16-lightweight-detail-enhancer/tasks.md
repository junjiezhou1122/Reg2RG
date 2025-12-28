# Tasks: Exp 16 - Lightweight Detail Enhancer

## 1. Implementation
- [ ] 1.1 Implement `DetailEnhancer` CNN (3-5 conv layers)
- [ ] 1.2 Input: ViT output features [B, N, D]
- [ ] 1.3 Output: Residual features [B, N, D]
- [ ] 1.4 Final: `enhanced = input + residual`

## 2. Architecture Options
- [ ] 2.1 Option A: 1D Conv (treat tokens as sequence)
- [ ] 2.2 Option B: Reshape to 3D grid → 3D Conv → reshape back
- [ ] 2.3 Option C: Transformer layers with local attention
- [ ] 2.4 Implement all, compare

## 3. Training
- [ ] 3.1 Create paired dataset: (LR_encoded, HR_encoded)
- [ ] 3.2 LR: Downsample volume then encode
- [ ] 3.3 HR: Encode at full resolution (as reference)
- [ ] 3.4 Loss: MSE(enhanced, HR_encoded)

## 4. Integration
- [ ] 4.1 Add enhancer after ViT, before adapter
- [ ] 4.2 Make enhancer optional (toggle flag)
- [ ] 4.3 Support frozen enhancer (pretrained) mode

## 5. Experiments
- [ ] 5.1 Compare: No enhancer (baseline)
- [ ] 5.2 Compare: 1D Conv enhancer
- [ ] 5.3 Compare: 3D Conv enhancer
- [ ] 5.4 Compare: Transformer enhancer

## 6. Evaluation
- [ ] 6.1 Reconstruction cosine improvement
- [ ] 6.2 Report generation metrics
- [ ] 6.3 Compute cost (FLOPs, params)
- [ ] 6.4 Visualize residuals (what details are added?)
