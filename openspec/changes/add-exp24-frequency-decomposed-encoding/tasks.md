# Tasks: Exp 24 - Frequency-Decomposed Encoding

## 1. Implementation
- [ ] 1.1 Implement `FrequencyDecomposer` with FFT/DCT
- [ ] 1.2 Alternative: Implement wavelet decomposition (PyWavelets)
- [ ] 1.3 Separate low-freq and high-freq components
- [ ] 1.4 Add frequency band indicators to embeddings

## 2. Frequency-Aware Encoding
- [ ] 2.1 Encode low-freq with standard ViT (structure)
- [ ] 2.2 Encode high-freq with specialized head (details)
- [ ] 2.3 Add "frequency present" embedding to indicate available bands

## 3. Compression Simulation
- [ ] 3.1 Simulate different compression levels via frequency filtering
- [ ] 3.2 Drop high-freq progressively to simulate lossy compression
- [ ] 3.3 Train model to be robust to missing high-freq

## 4. Evaluation
- [ ] 4.1 Reconstruction quality per frequency band
- [ ] 4.2 Report generation quality vs frequency content
- [ ] 4.3 Robustness to frequency dropout
