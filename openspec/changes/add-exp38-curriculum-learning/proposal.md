# Change: Add Curriculum Learning by Region Difficulty (Exp 38)

## Why
Our data shows huge variance in learning difficulty:

```
Training Difficulty Ranking (from training logs):
  1. Trachea:     cos = 0.70  (easiest, learns fastest)
  2. Mediastinum: cos = 0.63
  3. Breast:      cos = 0.57
  4. Esophagus:   cos = 0.57
  5. Heart:       cos = 0.55
  6. Abdomen:     cos = 0.49
  7. Bone:        cos = 0.46
  8. Pleura:      cos = 0.45
  9. Lung:        cos = 0.44
  10. Thyroid:    cos = 0.33  (hardest, learns slowest)
```

**Problem**: Training all regions simultaneously may cause:
1. Easy regions dominate early learning
2. Hard regions never catch up
3. Representations for easy regions don't help hard regions

**Solution**: Curriculum Learning - start with easy regions, gradually add hard ones.

## What Changes
- Implement curriculum scheduler that controls which regions are active
- Start with top-3 easiest regions (trachea, mediastinum, breast)
- Add one new region every 2 epochs
- End with all regions active by epoch 20

## Impact
- Affected specs: curriculum-training (new)
- Affected code:
  - `src/curriculum_scheduler.py` (new)
  - `src/lit_recon_probe.py` (integrate scheduler)
- Priority: Medium-High
- Dependency: Exp 35 (hardness measurement)
- Paper potential: "Curriculum Learning for Multi-Organ Medical Image Understanding"

## Expected Results
- More stable training
- Better final performance on hard regions
- Knowledge transfer from easy to hard regions
