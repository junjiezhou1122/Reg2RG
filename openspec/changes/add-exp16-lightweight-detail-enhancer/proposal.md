# Change: Add Lightweight Detail Enhancer (Exp 16)

## Why
Diffusion is expensive and risky. Alternative: train a lightweight CNN to predict high-frequency residuals from ViT features. Learns what details are typically lost during compression and adds them back.

## What Changes
- Add `DetailEnhancer` CNN module
- Train to predict high-freq residuals: `residual = HR_features - LR_features`
- Add residual to ViT output: `enhanced = vit_output + enhancer(vit_output)`
- Much cheaper than diffusion, fully trainable end-to-end

## Impact
- Affected specs: detail-enhancement (new)
- Affected code:
  - `src/Model/detail_enhancer.py` (new)
  - `src/Model/my_embedding_layer.py` (integration)
- Priority: Medium-High
- Advantage: Cheap, trainable, no external dependencies
- Paper potential: "Residual Detail Enhancement for Compressed Medical Vision"
