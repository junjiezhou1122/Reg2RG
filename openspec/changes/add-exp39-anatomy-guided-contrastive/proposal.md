# Change: Add Anatomy-Guided Contrastive Learning (Exp 39)

## Why
Our data reveals an interesting pattern:

```
Anatomically Adjacent Regions:
  Trachea (0.70) <-> Thyroid (0.33)   - Adjacent but VERY different cos!
  Lung (0.44) <-> Pleura (0.45)       - Adjacent and similar cos
  Heart (0.55) <-> Mediastinum (0.63) - Adjacent, moderate difference
```

**Key Insight**: Anatomically adjacent regions should have related representations!
- They share spatial context
- One can help localize the other
- Easy neighbors can "teach" hard neighbors

**Current Problem**: Each region is learned independently.
No explicit encouragement for anatomical relationships.

## What Changes
- Define anatomical adjacency graph
- Add contrastive loss encouraging adjacent regions to have similar representations
- Positive pairs: same patient, adjacent regions
- Negative pairs: different patients, same region

## Impact
- Affected specs: anatomy-contrastive (new)
- Affected code:
  - `src/Model/anatomy_contrastive_loss.py` (new)
  - `src/lit_recon_probe.py` (add loss term)
- Priority: Medium-High
- Dependency: Basic training working
- Paper potential: "Anatomy-Aware Representation Learning for Medical Vision-Language Models"

## Expected Results
- Better thyroid learning (guided by trachea)
- More anatomically consistent representations
- Improved localization for small organs
