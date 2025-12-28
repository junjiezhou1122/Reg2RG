# Change: Add Context-Aware Adapter for Hard Regions (Exp 37)

## Why
Some regions are inherently hard to learn in isolation:

```
Thyroid Problem:
  - Very small (2K voxels) -> weak signal
  - Low contrast with surrounding tissue
  - Variable shape between patients
  - cos = 0.33 (worst!)

BUT: Thyroid is ALWAYS next to trachea!
  - Trachea: cos = 0.70 (best!)
  - Trachea has clear boundaries
  - Trachea can serve as "anchor" for thyroid

Key Insight: Use anatomical neighbors as context to help hard regions
```

## What Changes
- Add `ContextAwareAdapter` for difficult small regions
- Use cross-attention to incorporate neighbor region features
- Define anatomical neighborhood graph
- Apply to thyroid, esophagus, and other hard small regions

## Impact
- Affected specs: context-adapter (new)
- Affected code:
  - `src/Model/context_aware_adapter.py` (new)
  - `src/Model/my_embedding_layer.py` (routing logic)
- Priority: High
- Dependency: Basic adapter training working
- Paper potential: "Anatomy-Aware Context for Small Organ Representation in Medical VLMs"

## Expected Results
- Thyroid cos: 0.33 -> 0.50+
- Better localization for small organs
- More anatomically consistent representations
