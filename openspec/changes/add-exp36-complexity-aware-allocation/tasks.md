# Tasks: Exp 36 - Complexity-Aware Token Allocation

## Implementation Tasks

- [ ] Create `src/Model/complexity_predictor.py`
  - [ ] Implement `ComplexityPredictor` class
  - [ ] Multi-factor prediction: structure, boundary, contrast, consistency
  - [ ] Output single complexity score [0, 1]
  - [ ] Support both region-level and token-level prediction

- [ ] Modify Adaptive Perceiver
  - [ ] Replace volume-based routing with complexity-based
  - [ ] Map complexity to token counts: high -> more tokens
  - [ ] Gumbel-Softmax for differentiable selection

- [ ] Training with pseudo-labels
  - [ ] Use 1 - cos as complexity label
  - [ ] Add complexity prediction loss
  - [ ] Joint training: reconstruction + complexity prediction

- [ ] Testing
  - [ ] Verify complexity correlates with hardness
  - [ ] Compare token allocation: volume-based vs complexity-based

## Evaluation Tasks

- [ ] Train complexity predictor
- [ ] Analyze predicted complexity vs actual cos
- [ ] Compare reconstruction quality:
  - [ ] Volume-based allocation (Exp 8)
  - [ ] Complexity-based allocation (Exp 36)
- [ ] Visualize token allocation per region
