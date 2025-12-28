# Change: Add Anatomy Prior Enhancement (Exp 21)

## Why
CT anatomy is highly structured (bones here, lungs there, liver position is predictable). Use anatomical priors to guide what the model should look for in each region, even if visual details are lost to compression.

## What Changes
- Add `AnatomyPriorBank` with learned templates for each region
- Combine visual features with anatomical prior: `output = visual + alpha * prior`
- Prior provides "expected" content, visual provides "actual" observations
- Deviation from prior = clinically relevant finding

## Impact
- Affected specs: anatomy-priors (new)
- Affected code:
  - `src/Model/anatomy_priors.py` (new)
  - `src/Model/my_embedding_layer.py` (integration)
- Priority: Medium
- Paper potential: "Anatomy-Guided Feature Enhancement for Medical VLMs"
