# Change: Add Normal Template Learning (Exp 11)

## Why
To detect anomalies, the model needs to understand "what is normal". By learning prototype templates for each anatomical region from normal samples, the model can identify abnormalities as deviations from these templates.

## What Changes
- Add `NormalTemplateBank` with learnable per-region templates
- Implement deviation computation (current features vs template)
- Add EMA-based template update using normal samples
- Add deviation encoding for downstream use

## Impact
- Affected specs: normal-template-bank (new)
- Affected code:
  - `src/Model/normal_template_bank.py` (new)
  - `src/Dataset/radgenome_dataset_train.py` (filter normal samples)
  - `scripts/extract_normal_samples.py` (new)
- Priority: High
- Dependency: Normal sample identification from Exp 10
- Part of: Anomaly-Centric Design (Exp 10-12)
- Paper potential: "Learning What's Normal: Template-Based Anomaly Detection for Medical VLMs"
