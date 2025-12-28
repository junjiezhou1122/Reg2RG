# Change: Add Uncertainty-Aware Encoding (Exp 20)

## Why
Different regions have different compression ratios, hence different information loss levels. The model should know this! Add uncertainty estimation based on compression ratio to inform downstream modules about feature reliability.

## What Changes
- Add `CompressionUncertaintyEstimator` module
- Estimate epistemic uncertainty from compression ratio
- Encode uncertainty as additional feature dimension
- LLM can use uncertainty to calibrate confidence in descriptions

## Impact
- Affected specs: compression-uncertainty (new)
- Affected code:
  - `src/Model/compression_uncertainty.py` (new)
  - `src/Model/my_embedding_layer.py` (integration)
- Priority: Medium
- Paper potential: "Compression-Aware Uncertainty for Medical VLMs"
