# Change: Add Anomaly-Aware Report Generation (Exp 12)

## Why
Current VLMs generate reports without explicit awareness of which regions are abnormal. By integrating anomaly scores and deviation features from Exp 10-11, the LLM can focus attention on clinically important regions and generate more accurate descriptions.

## What Changes
- Integrate `AnomalyScorePredictor` and `NormalTemplateBank` into VLM
- Fuse original features with deviation features weighted by anomaly score
- Add anomaly-aware attention weighting
- Build anomaly prompt to guide LLM generation

## Impact
- Affected specs: anomaly-aware-vlm (new)
- Affected code:
  - `src/Model/anomaly_aware_vlm.py` (new)
  - `src/Model/Reg2RG.py` (integration)
  - `src/train_radgenome.py` (combined loss)
- Priority: Medium-High
- Dependency: Exp 10, Exp 11
- Part of: Anomaly-Centric Design (Exp 10-12)
- Paper potential: "Deviation-Guided Report Generation: Focusing on What Matters"
