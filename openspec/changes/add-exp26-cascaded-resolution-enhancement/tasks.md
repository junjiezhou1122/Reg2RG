# Tasks: Exp 26 - Cascaded Resolution Enhancement

## 1. Implementation
- [ ] 1.1 Implement `CascadedEnhancer` with N stages
- [ ] 1.2 Stage 1: Coarse encoding (low-res features)
- [ ] 1.3 Stage 2-N: Refinement layers adding details
- [ ] 1.4 Each stage: residual addition to previous

## 2. Progressive Decoding
- [ ] 2.1 Allow early-exit at any stage
- [ ] 2.2 Return features at requested detail level
- [ ] 2.3 Trade-off: speed vs quality

## 3. Training
- [ ] 3.1 Multi-stage loss: supervise each stage
- [ ] 3.2 Progressive training: easy to hard
- [ ] 3.3 Random stage dropout for robustness

## 4. Evaluation
- [ ] 4.1 Quality vs stage number
- [ ] 4.2 Speed vs stage number
- [ ] 4.3 Optimal stage for different tasks
