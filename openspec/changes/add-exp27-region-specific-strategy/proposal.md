# Change: Add Region-Specific Compression Strategy (Exp 27)

## Why
Different organs have different compression tolerances. Bones are simple (high compression OK), lungs are complex (low compression needed). Learn per-region optimal compression strategy rather than one-size-fits-all.

## What Changes
- Add `RegionCompressionStrategy` module
- Learn compression parameters per region type
- Lung: More tokens, less aggressive downsampling
- Bone: Fewer tokens, aggressive downsampling OK
- Configurable strategy per anatomy

## Impact
- Affected specs: anatomy-aware-compression (new)
- Affected code:
  - `src/Model/region_compression.py` (new)
  - `src/Model/helpers.py` (adapter modification)
- Priority: Medium
- Paper potential: "Anatomy-Aware Compression for Medical VLMs"
