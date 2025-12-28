# Change: Add Cascaded Resolution Enhancement (Exp 26)

## Why
Instead of one-shot compression, use progressive refinement. Start with coarse representation, iteratively add details. Like progressive JPEG decoding - show blurry image first, then sharpen progressively.

## What Changes
- Add `CascadedEnhancer` with multiple refinement stages
- Stage 1: Coarse features (global structure)
- Stage 2-N: Add finer details progressively
- LLM can query at any stage for trade-off between speed and detail

## Impact
- Affected specs: progressive-enhancement (new)
- Affected code:
  - `src/Model/cascaded_enhancer.py` (new)
  - `src/Model/my_embedding_layer.py` (integration)
- Priority: Low
- Complexity: Medium
- Paper potential: "Progressive Refinement for Medical Vision Compression"
