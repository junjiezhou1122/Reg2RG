# Change: Add Anti-Aliased Downsampling (Exp 18)

## Why
Standard downsampling (bilinear, nearest) introduces aliasing artifacts that destroy high-frequency information. Anti-aliased downsampling (blur before subsample) preserves more information in the low-frequency bands that survive compression.

## What Changes
- Replace standard resize with anti-aliased downsample
- Implement Gaussian blur + subsample pipeline
- Support configurable blur kernel size
- Optionally learn the blur kernel

## Impact
- Affected specs: anti-aliasing (new)
- Affected code:
  - `src/Dataset/radgenome_dataset_train.py` (preprocessing change)
  - `src/Model/anti_alias.py` (new, optional learned version)
- Priority: Medium
- Very low complexity, easy win if effective
- Paper potential: Could be a "trick" mentioned in methods section
