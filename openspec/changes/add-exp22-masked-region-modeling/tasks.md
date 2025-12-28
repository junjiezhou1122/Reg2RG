# Tasks: Exp 22 - Masked Region Modeling

## 1. Implementation
- [ ] 1.1 Implement `MaskedRegionModeling` module
- [ ] 1.2 Add random region masking (25-50% mask ratio)
- [ ] 1.3 Add mask tokens for hidden regions
- [ ] 1.4 Add decoder to reconstruct masked region features

## 2. Pre-training
- [ ] 2.1 Create pre-training script `src/pretrain_mrm.py`
- [ ] 2.2 Loss: MSE between predicted and actual masked features
- [ ] 2.3 Use larger dataset for pre-training (unlabeled CT data)
- [ ] 2.4 Save pretrained adapter weights

## 3. Fine-tuning
- [ ] 3.1 Load pretrained adapter
- [ ] 3.2 Fine-tune on RadGenome with generation loss
- [ ] 3.3 Compare with randomly initialized adapter

## 4. Evaluation
- [ ] 4.1 Downstream report generation quality
- [ ] 4.2 Feature quality: linear probe accuracy
- [ ] 4.3 Robustness to missing regions
