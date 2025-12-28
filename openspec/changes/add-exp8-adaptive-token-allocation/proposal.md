# Change: Add Adaptive Token Allocation (Exp 8)

## Why
Different anatomical regions have different complexity levels. Lung (complex, many structures) needs more tokens than thyroid (simple, small). Fixed 8 tokens per region is suboptimal - complex organs are under-represented while simple organs waste capacity.

## What Changes
- Add `AdaptivePerceiver` that predicts optimal token count per region
- Implement complexity predictor network
- Support variable token options: [4, 8, 16, 32]
- Add token efficiency loss to encourage minimal token usage
- Use Gumbel-Softmax for differentiable selection during training

## Impact
- Affected specs: adaptive-compression (new)
- Affected code:
  - `src/Model/adaptive_perceiver.py` (new)
  - `src/Model/my_embedding_layer.py` (integration)
  - `src/train_radgenome.py` (efficiency loss)
- Priority: High
- Dependency: Exp 6 results (region size analysis)
- Paper potential: "Content-Adaptive Compression for Medical VLMs"
