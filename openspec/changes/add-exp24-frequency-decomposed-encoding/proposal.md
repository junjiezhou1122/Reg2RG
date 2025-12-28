# Change: Add Frequency-Decomposed Encoding (Exp 24)

## Why
High-frequency details are lost during downsampling, but low-frequency structure is preserved. Decompose input into frequency bands, process separately, and encode with awareness of which frequencies were lost.

## What Changes
- Add `FrequencyDecomposer` module (FFT-based or wavelet)
- Separate encoding for low-freq and high-freq components
- Frequency-aware tokens indicate which information is present
- LLM understands what resolution of details are available

## Impact
- Affected specs: frequency-domain (new)
- Affected code:
  - `src/Model/frequency_decomposer.py` (new)
  - `src/Model/my_embedding_layer.py` (integration)
- Priority: Low-Medium
- Paper potential: "Frequency-Aware Medical Vision Encoding"
