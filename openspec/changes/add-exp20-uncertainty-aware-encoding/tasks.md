# Tasks: Exp 20 - Uncertainty-Aware Encoding

## 1. Implementation
- [ ] 1.1 Implement `CompressionUncertaintyEstimator`
- [ ] 1.2 Compute uncertainty from: compression_ratio = original_size / encoded_size
- [ ] 1.3 Uncertainty encoding: uncertainty_emb = MLP(compression_ratio)
- [ ] 1.4 Fuse with visual features: concat or addition

## 2. Uncertainty Propagation
- [ ] 2.1 Pass uncertainty to Perceiver adapter
- [ ] 2.2 Pass uncertainty to LLM (as special token or embedding)
- [ ] 2.3 Use uncertainty in attention weighting

## 3. Training
- [ ] 3.1 Add uncertainty-calibrated loss
- [ ] 3.2 Ground truth: reconstruction error as proxy for uncertainty
- [ ] 3.3 Train to predict uncertainty from features

## 4. Evaluation
- [ ] 4.1 Correlation: predicted uncertainty vs actual information loss
- [ ] 4.2 Calibration: ECE metric
- [ ] 4.3 Impact on report generation quality
