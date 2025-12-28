# Tasks: Exp 21 - Anatomy Prior Enhancement

## 1. Implementation
- [ ] 1.1 Implement `AnatomyPriorBank` with per-region templates
- [ ] 1.2 Initialize from averaged features of normal samples
- [ ] 1.3 Add prior-visual fusion mechanism
- [ ] 1.4 Compute deviation score: ||visual - prior||

## 2. Prior Learning
- [ ] 2.1 Filter normal samples from RadGenome dataset
- [ ] 2.2 Aggregate features per region to form initial priors
- [ ] 2.3 Make priors learnable (EMA update during training)

## 3. Fusion Strategies
- [ ] 3.1 Additive: output = visual + alpha * prior
- [ ] 3.2 Attention-based: attend to prior based on uncertainty
- [ ] 3.3 Gated: output = gate(visual) * visual + (1-gate(visual)) * prior

## 4. Evaluation
- [ ] 4.1 Measure reconstruction quality improvement
- [ ] 4.2 Correlation: deviation score vs anomaly presence
- [ ] 4.3 Report generation quality
