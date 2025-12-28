# Tasks: Exp 37 - Context-Aware Adapter

## Implementation Tasks

- [ ] Define anatomical neighborhood graph
  - [ ] thyroid <-> trachea, esophagus
  - [ ] heart <-> lung, mediastinum, pleura
  - [ ] esophagus <-> trachea, mediastinum
  - [ ] Store as config dict

- [ ] Create `src/Model/context_aware_adapter.py`
  - [ ] Implement `ContextAwareAdapter` class
  - [ ] Cross-attention between target and context regions
  - [ ] Residual connection: enhanced = target + alpha * context_info
  - [ ] Learnable alpha per region pair

- [ ] Create routing logic
  - [ ] Identify which regions use context-aware adapter
  - [ ] Route based on: small + hard regions
  - [ ] Pass neighbor features to adapter

- [ ] Testing
  - [ ] Verify cross-attention is learning
  - [ ] Visualize attention weights
  - [ ] Compare thyroid with/without context

## Evaluation Tasks

- [ ] Baseline: Standard adapter for thyroid
- [ ] Context-aware: Thyroid with trachea context
- [ ] Compare:
  - [ ] Reconstruction cos
  - [ ] Feature quality
  - [ ] Localization accuracy
