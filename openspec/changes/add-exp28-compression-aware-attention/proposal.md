# Change: Add Compression-Aware Attention (Exp 28)

## Why
Compressed features are less reliable than original features. The LLM's attention mechanism should know this! Modulate attention weights based on compression quality - attend more to high-quality (low-compression) features.

## What Changes
- Add `CompressionModulatedAttention` module
- Compute compression quality score per region
- Modulate LLM attention: `attention_weight *= quality_score`
- High-quality regions get more influence in report generation

## Impact
- Affected specs: compression-modulated-attention (new)
- Affected code:
  - `src/Model/compression_attention.py` (new)
  - `src/Model/Reg2RG.py` (LLM attention modification)
- Priority: Medium
- Paper potential: "Quality-Aware Attention for Compressed Medical Features"
