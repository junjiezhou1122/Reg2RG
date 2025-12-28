# Tasks: Exp 28 - Compression-Aware Attention

## 1. Implementation
- [ ] 1.1 Implement `CompressionQualityScore` estimator
- [ ] 1.2 Score from: compression_ratio, reconstruction_error, uncertainty
- [ ] 1.3 Implement `CompressionModulatedAttention` layer
- [ ] 1.4 Modulate attention: softmax(QK^T / sqrt(d) + log(quality))

## 2. Quality Estimation
- [ ] 2.1 Quality = f(compression_ratio): inverse relationship
- [ ] 2.2 Quality from reconstruction probe (if available)
- [ ] 2.3 Learned quality predictor from features

## 3. Attention Modification
- [ ] 3.1 Add quality bias to attention logits
- [ ] 3.2 Alternative: Multiply attention weights by quality
- [ ] 3.3 Apply to cross-attention (visual → text)

## 4. Evaluation
- [ ] 4.1 Attention distribution analysis: does model attend to high-quality regions?
- [ ] 4.2 Report quality improvement
- [ ] 4.3 Ablation: With vs without modulation
