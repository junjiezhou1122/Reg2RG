# Tasks: Exp 19 - Contrastive Resolution Alignment

## 1. Implementation
- [ ] 1.1 Implement `ResolutionContrastiveLoss` with InfoNCE
- [ ] 1.2 Add projection heads for resolution-invariant embedding space
- [ ] 1.3 Implement positive pair creation (same region, different resolution)
- [ ] 1.4 Implement negative pair sampling (different regions)

## 2. Data Pipeline
- [ ] 2.1 Create paired dataset: (original_crop, resized_region)
- [ ] 2.2 Handle variable compression ratios
- [ ] 2.3 Add data augmentation for robustness

## 3. Training
- [ ] 3.1 Pre-training stage: Contrastive alignment only
- [ ] 3.2 Joint training: Contrastive + Generation loss
- [ ] 3.3 Temperature scheduling for contrastive loss

## 4. Evaluation
- [ ] 4.1 Measure cosine similarity: resized vs original features
- [ ] 4.2 Downstream report generation performance
- [ ] 4.3 Transfer learning to unseen resolutions
