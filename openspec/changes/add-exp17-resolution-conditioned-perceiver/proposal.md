# Change: Add Resolution-Conditioned Perceiver (Exp 17)

## Why
Current Perceiver treats all inputs the same regardless of original resolution. Low-res inputs are fundamentally different from high-res inputs that were downsampled. Teaching the adapter to handle different resolutions explicitly may improve robustness.

## What Changes
- Add `ResolutionEmbedding` to encode input resolution
- Concatenate resolution embedding with ViT features before Perceiver
- Perceiver learns resolution-aware compression strategies
- Support continuous resolution values, not just discrete categories

## Impact
- Affected specs: resolution-aware-adapter (new)
- Affected code:
  - `src/Model/resolution_embedding.py` (new)
  - `src/Model/helpers.py` (Perceiver modification)
- Priority: Medium
- Low complexity, easy to implement and test
- Paper potential: "Resolution-Aware Compression for Variable-Resolution Medical Inputs"
