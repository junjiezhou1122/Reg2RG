# Tasks: Exp 17 - Resolution-Conditioned Perceiver

## 1. Implementation
- [ ] 1.1 Implement `ResolutionEmbedding` module
  - [ ] 1.1.1 Input: (H, W, D) original resolution tuple
  - [ ] 1.1.2 Output: embedding vector [D]
- [ ] 1.2 Support continuous resolution (Fourier encoding)
- [ ] 1.3 Support discrete resolution categories (learned embeddings)

## 2. Integration
- [ ] 2.1 Compute resolution ratio: original_size / encoded_size
- [ ] 2.2 Embed resolution: `res_emb = resolution_embedding(ratio)`
- [ ] 2.3 Option A: Prepend res_emb as first token
- [ ] 2.4 Option B: Add res_emb to all tokens
- [ ] 2.5 Option C: Concatenate res_emb to each token

## 3. Training
- [ ] 3.1 Ensure resolution varies in training data
- [ ] 3.2 Data augmentation: random resolution variations
- [ ] 3.3 Curriculum: start with fixed res, then vary

## 4. Experiments
- [ ] 4.1 Baseline: No resolution conditioning
- [ ] 4.2 Discrete: 3 resolution categories (small/medium/large)
- [ ] 4.3 Continuous: Fourier positional encoding of resolution
- [ ] 4.4 Compare integration methods (prepend vs add vs concat)

## 5. Evaluation
- [ ] 5.1 Performance across different input resolutions
- [ ] 5.2 Robustness to resolution variations
- [ ] 5.3 Generalization to unseen resolutions
