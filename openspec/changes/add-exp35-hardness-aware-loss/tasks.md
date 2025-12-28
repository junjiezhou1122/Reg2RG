# Tasks: Exp 35 - Hardness-Aware Loss Weighting

## Implementation Tasks

- [ ] Create `src/Model/hardness_aware_loss.py`
  - [ ] Implement `HardnessAwareLoss` class
  - [ ] Add EMA tracking for per-region cos
  - [ ] Compute dynamic weights based on hardness
  - [ ] Support configurable EMA momentum (default 0.99)
  - [ ] Support configurable weight range (default [1.0, 2.0])

- [ ] Integrate into training pipeline
  - [ ] Modify `src/lit_recon_probe.py` to use `HardnessAwareLoss`
  - [ ] Add logging for per-region weights
  - [ ] Add wandb metrics for weight visualization

- [ ] Testing
  - [ ] Unit test for weight computation
  - [ ] Verify weights converge to expected values
  - [ ] Compare training curves with/without hardness weighting

## Evaluation Tasks

- [ ] Run baseline training (without hardness weighting)
- [ ] Run hardness-aware training
- [ ] Compare:
  - [ ] Per-region cos improvement
  - [ ] Especially thyroid cos change
  - [ ] Training stability
  - [ ] Convergence speed
