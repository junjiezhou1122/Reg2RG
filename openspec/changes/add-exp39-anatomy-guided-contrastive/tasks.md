# Tasks: Exp 39 - Anatomy-Guided Contrastive Learning

## Implementation Tasks

- [ ] Define anatomy adjacency graph
  - [ ] Create adjacency dict from medical knowledge
  - [ ] Assign edge weights (stronger connection = higher weight)
  - [ ] Document anatomical reasoning

- [ ] Create `src/Model/anatomy_contrastive_loss.py`
  - [ ] Implement `AnatomyContrastiveLoss` class
  - [ ] Positive pairs: adjacent regions from same patient
  - [ ] Negative pairs: non-adjacent or different patients
  - [ ] Temperature parameter for softmax
  - [ ] Weighted by anatomical distance

- [ ] Integrate into training
  - [ ] Compute features for all regions in batch
  - [ ] Compute contrastive loss
  - [ ] Add to total loss with weight lambda

- [ ] Testing
  - [ ] Verify loss decreases over training
  - [ ] Check representations become more similar for adjacent regions

## Evaluation Tasks

- [ ] Baseline: No contrastive loss
- [ ] Anatomy-guided: With contrastive loss
- [ ] Compare:
  - [ ] Per-region cos (especially thyroid)
  - [ ] Feature similarity between adjacent regions
  - [ ] Downstream task performance
