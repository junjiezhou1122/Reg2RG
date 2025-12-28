# Tasks: Exp 8 - Adaptive Token Allocation

## 1. Implementation
- [ ] 1.1 Implement `ComplexityPredictor` network (Linear → ReLU → Linear)
- [ ] 1.2 Create multiple Perceiver variants for each token count [4, 8, 16, 32]
- [ ] 1.3 Implement Gumbel-Softmax selection (differentiable during training)
- [ ] 1.4 Implement hard selection for inference (argmax)
- [ ] 1.5 Add padding logic for variable-length outputs

## 2. Training
- [ ] 2.1 Add token efficiency loss: `mean(selected_tokens) / max_tokens`
- [ ] 2.2 Add efficiency weight hyperparameter
- [ ] 2.3 Implement curriculum learning (start with fixed, then adaptive)
- [ ] 2.4 Add Gumbel temperature annealing schedule

## 3. Simplified Version
- [ ] 3.1 Implement `AdaptivePerceiverSimple` with hard selection only
- [ ] 3.2 Test simplified version first before full implementation

## 4. Evaluation
- [ ] 4.1 Measure per-region token allocation distribution
- [ ] 4.2 Compare reconstruction quality vs token count
- [ ] 4.3 Compute total tokens used vs fixed baseline
- [ ] 4.4 Analyze if predicted complexity matches expected (Lung→high, Thyroid→low)

## 5. Experiments
- [ ] 5.1 Ablation: Token options ([4,8] vs [4,8,16] vs [4,8,16,32])
- [ ] 5.2 Ablation: Efficiency weight (0.01, 0.1, 1.0)
- [ ] 5.3 Compare with fixed 8-token baseline

## 6. Documentation
- [ ] 6.1 Record region-to-token mapping
- [ ] 6.2 Document efficiency vs quality trade-off
