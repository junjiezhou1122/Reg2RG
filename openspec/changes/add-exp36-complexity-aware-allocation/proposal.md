# Change: Add Complexity-Aware Token Allocation (Exp 36)

## Why
Exp 8 assumes complexity ~ volume, but our data proves this is WRONG:

```
Volume-based prediction vs Reality:
  Thyroid:  2K volume  -> predict "simple"  -> actually HARDEST (cos=0.33)
  Trachea:  3K volume  -> predict "simple"  -> actually EASIEST (cos=0.70)
  Lung:     392K volume -> predict "complex" -> hard but not hardest (cos=0.44)
```

**Key Insight**: Learning difficulty depends on multiple factors:
1. Structure complexity (lung has complex textures)
2. Boundary clarity (trachea has clear air-tissue boundary)
3. Contrast (thyroid has low contrast with surrounding tissue)
4. Shape consistency (thyroid varies between patients)

Using volume as a proxy for complexity is fundamentally flawed.

## What Changes
- Replace volume-based complexity estimation with **learned ComplexityPredictor**
- Predict complexity from visual features directly
- Use actual cos as training signal (pseudo-label)
- Allocate tokens based on predicted complexity, not volume

## Impact
- Affected specs: complexity-predictor (new)
- Affected code:
  - `src/Model/complexity_predictor.py` (new)
  - `src/Model/adaptive_perceiver.py` (modify to use predictor)
- Priority: High
- Dependency: Exp 35 (can share hardness/complexity concept)
- Paper potential: "Learning Visual Complexity for Adaptive Medical Image Compression"

## Expected Results
- More accurate token allocation
- Thyroid gets MORE tokens (not fewer!)
- Better reconstruction for hard regions
