# Change: Add Contrastive Vision-Language Alignment (Exp 7)

## Why
Current model achieves ~85% reconstruction cosine but only ~25% report generation F1. This gap suggests visual tokens and text are not well-aligned in the embedding space. Contrastive learning can bridge this gap by forcing visual tokens to contain report-relevant information.

## What Changes
- Add `ContrastiveAlignmentLoss` module for global V-L alignment
- Add `RegionContrastiveLoss` for region-level V-L alignment
- Implement two-stage training: pre-alignment then joint training
- Add V→T and T→V retrieval accuracy metrics

## Impact
- Affected specs: vision-language-alignment (new)
- Affected code:
  - `src/Model/contrastive_loss.py` (new)
  - `src/train_radgenome.py` (loss integration)
  - `src/lit_recon_probe.py` (evaluation)
- Priority: Medium-High
- Paper potential: "Bridging the Gap: Contrastive Alignment for Medical VLMs"
