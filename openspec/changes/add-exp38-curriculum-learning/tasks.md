# Tasks: Exp 38 - Curriculum Learning

## Implementation Tasks

- [ ] Create `src/curriculum_scheduler.py`
  - [ ] Implement `CurriculumScheduler` class
  - [ ] Define difficulty ordering from training data
  - [ ] Support configurable unlock schedule
  - [ ] Track which regions are active per epoch

- [ ] Integrate into training loop
  - [ ] Modify dataloader to filter inactive regions
  - [ ] OR: Mask loss for inactive regions
  - [ ] Log active regions per epoch

- [ ] Schedule configurations
  - [ ] Conservative: 1 region per 3 epochs
  - [ ] Moderate: 1 region per 2 epochs (default)
  - [ ] Aggressive: 1 region per epoch

- [ ] Testing
  - [ ] Verify schedule follows plan
  - [ ] Check loss is properly masked for inactive regions

## Evaluation Tasks

- [ ] Run baseline (all regions from start)
- [ ] Run curriculum (gradual unlock)
- [ ] Compare:
  - [ ] Final cos per region
  - [ ] Training stability (loss variance)
  - [ ] Convergence speed for hard regions
  - [ ] Knowledge transfer effects
