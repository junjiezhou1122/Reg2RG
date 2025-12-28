# Region-Level Statistics for Size-Cosine Correlation Analysis

**Date**: 2025-12-27
**Author**: Implemented with Claude assistance
**Feature**: Per-region metrics tracking with mask-based size calculation and compression ratio analysis

---

## Overview

This document describes the region-level statistics feature added to `lit_recon_probe.py` to analyze the relationship between anatomical region size, compression ratio, and reconstruction quality.

### Key Discovery Leading to This Feature

During experiments, we observed a counterintuitive result:
- **Global reconstruction cosine**: ~0.65
- **Region reconstruction cosine**: ~0.74

This finding suggests that **smaller, focused regions might be easier to reconstruct** than the full CT volume. This feature enables systematic analysis of this phenomenon.

---

## NEW: Compression Ratio Tracking (2025-12-27)

### Why Compression Ratio Matters

The data pipeline resizes all regions to a fixed size (256×256×64):

```
Original Region (variable size)
       ↓
Bbox Crop → Resize to (256×256×64)
       ↓
ViT → Perceiver → LLM

Problems:
  - Small regions get EXPANDED: info "diluted" but not lost
  - Large regions get COMPRESSED: info LOST (high-freq details smoothed)
```

### Compression Ratio Definition

```python
compression_ratio = original_bbox_voxels / target_voxels  # target = 256*256*64

# Interpretation:
#   ratio > 1.0: Region was COMPRESSED (info potentially lost) 🔻
#   ratio < 1.0: Region was EXPANDED (no info loss) 🔺
#   ratio = 1.0: No resize needed
```

### Key Hypothesis

**If compression causes info loss, we expect:**
- Pearson(compression_ratio, cosine) < 0
- Higher compression → worse reconstruction

---

## Implementation Summary

### 1. Data Structure (`run_epoch()`)

Added `region_stats` dictionary to track per-region metrics:

```python
region_stats = {
    region_name: {
        "cos_values": [],          # List of cosine values per step
        "mse_values": [],          # List of MSE values per step
        "top1_values": [],         # List of top-1% error values per step
        "volume_sizes": [],        # List of mask-based sizes (voxels)
        "compression_ratios": [],  # NEW: List of compression ratios
        "cos_sum": 0.0,            # Running sum for epoch average
        "mse_sum": 0.0,
        "top1_sum": 0.0,
        "size_sum": 0,
        "compression_ratio_sum": 0.0,  # NEW
        "count": 0,
    }
    for region_name in REGIONS
}
```

### 2. Compression Ratio Calculation

In `radgenome_dataset_train.py` `__getitem__`:

```python
# Calculate original bbox dimensions
bbox_h = h1 - h0 + 1
bbox_w = w1 - w0 + 1
bbox_d = d1 - d0 + 1
original_bbox_voxels = bbox_h * bbox_w * bbox_d
compression_ratio = original_bbox_voxels / target_voxels

# Store in bbox_sizes dict
bbox_sizes[region_name] = {
    "original_shape": (bbox_h, bbox_w, bbox_d),
    "original_voxels": original_bbox_voxels,
    "target_voxels": target_voxels,
    "compression_ratio": compression_ratio,
}
```

### 3. Epoch-Level Summary (Updated)

Prints comprehensive table with compression ratio:

```
================================================================================
[train] EPOCH 1 REGION STATISTICS
================================================================================
Region               Cos      MSE     Top1   Size(K)   Comp.Ratio  Count
--------------------------------------------------------------------------------
abdomen             0.7543   0.0234   1.2345    1234.5       2.35x🔻    100
bone                0.6823   0.0345   1.5678     567.8       1.12x🔻    100
thyroid             0.8234   0.0123   0.8765      45.2       0.13x🔺    100
...
--------------------------------------------------------------------------------
📊 CORRELATION ANALYSIS:
   Compression Ratio vs Cos (Pearson r): -0.4523
      → ⚠️  Higher compression = WORSE reconstruction (info loss confirmed!)
   Size vs Cos (Pearson r): +0.2134
      → No strong correlation
================================================================================
```

Legend:
- 🔻 = Compressed (ratio > 1, potential info loss)
- 🔺 = Expanded (ratio < 1, no info loss)

### 4. CSV Output (Updated)

`region_metrics.csv` now includes compression_ratio:

| Column | Description |
|--------|-------------|
| epoch | Training epoch number |
| split | "train" or "val" |
| region | Anatomical region name |
| cos | Average cosine similarity |
| mse | Average mean squared error |
| top1 | Average top-1% L2 error |
| avg_size | Average mask-based size (voxels) |
| **compression_ratio** | **NEW: Average compression ratio** |
| count | Number of samples |

---

## Correlation Analysis

### Two Correlations Computed

1. **Compression Ratio vs Cosine**
   - Expected: r < 0 (higher compression = worse quality)
   - Confirms info loss hypothesis

2. **Size vs Cosine**
   - Original analysis (unchanged)
   - May show different pattern than compression ratio

### Interpretation Guide

| Compression r | Size r | Interpretation |
|---------------|--------|----------------|
| r < -0.3 | - | Info loss confirmed, need mitigation |
| r > 0.3 | - | Unexpected, needs investigation |
| |r| < 0.3 | |r| < 0.3 | Compression not the main factor |

---

## Files Modified

### `src/Dataset/radgenome_dataset_train.py`

| Location | Change |
|----------|--------|
| Lines 469-501 | Added bbox_sizes calculation |
| Line 567 | Added bbox_sizes to return dict |

### `src/lit_recon_probe.py`

| Location | Change |
|----------|--------|
| Lines 1675-1692 | Added compression_ratios to region_stats |
| Lines 1718-1724 | Extract bbox_sizes from batch |
| Lines 1803-1832 | Added compression ratio extraction and recording |
| Lines 1980-2067 | Updated epoch summary with correlation analysis |
| Line 2094 | Updated CSV header |
| Lines 2306-2329 | Updated CSV writing |
| Lines 415-416 | Updated collator to handle bbox_sizes |

---

## Expected Results by Region

| Region | Expected Compression Ratio | Info Loss Risk |
|--------|---------------------------|----------------|
| lung | 2-8x | HIGH 🔻 |
| abdomen | 2-5x | HIGH 🔻 |
| mediastinum | 1-3x | MEDIUM |
| heart | 0.8-2x | LOW-MEDIUM |
| bone | 0.5-2x | LOW-MEDIUM |
| breast | 0.3-1x | LOW |
| pleura | 0.2-0.8x | LOW 🔺 |
| esophagus | 0.1-0.5x | LOW 🔺 |
| thyroid | 0.05-0.2x | LOW 🔺 |
| trachea | 0.1-0.4x | LOW 🔺 |

---

## Analysis Example

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("region_metrics.csv")
val_df = df[df["split"] == "val"]

# Plot compression_ratio vs cos
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(val_df["compression_ratio"], val_df["cos"], alpha=0.7)
plt.xlabel("Compression Ratio")
plt.ylabel("Reconstruction Cosine")
plt.axvline(x=1.0, color='r', linestyle='--', label='ratio=1 (no resize)')
plt.title("Compression Ratio vs Reconstruction Quality")
plt.legend()

# Calculate correlation
r = np.corrcoef(val_df["compression_ratio"], val_df["cos"])[0, 1]
plt.annotate(f'Pearson r = {r:.3f}', xy=(0.7, 0.95), xycoords='axes fraction')

plt.subplot(1, 2, 2)
# Color by region
for region in val_df["region"].unique():
    rdata = val_df[val_df["region"] == region]
    plt.scatter(rdata["compression_ratio"], rdata["cos"], label=region[:8], alpha=0.7)
plt.xlabel("Compression Ratio")
plt.ylabel("Reconstruction Cosine")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Compression Ratio vs Cos (by Region)")

plt.tight_layout()
plt.savefig("compression_analysis.png", dpi=150, bbox_inches='tight')
```

---

## Future Work: Mitigation Strategies

If compression ratio strongly correlates with info loss (r < -0.3), consider:

### Option 1: Resolution-Conditioned Encoding
```python
class ResolutionConditionedPerceiver(nn.Module):
    def __init__(self, dim=768):
        self.ratio_embed = nn.Linear(1, dim)
        self.perceiver = PerceiverResampler(...)

    def forward(self, tokens, compression_ratio):
        ratio_token = self.ratio_embed(compression_ratio.unsqueeze(-1))
        tokens_with_ratio = torch.cat([ratio_token, tokens], dim=1)
        return self.perceiver(tokens_with_ratio)
```

### Option 2: Multi-Crop for Large Regions
- Split large regions into overlapping crops
- Process each through ViT
- Aggregate with Perceiver

### Option 3: Anti-Aliased Downsampling
- Apply Gaussian blur before resize for compressed regions
- Reduces aliasing artifacts

---

## Limitations

### Current Implementation

The compression ratio is computed from the **resized mask bbox**, not the original NIfTI:

```
Pipeline:
  Original NIfTI → CropForeground → Resize(256×256×64) → Cache
                                           ↓
  __getitem__: Find bbox in 256×256×64 space → Crop → Resize again
```

The compression ratio measures the **second resize step** only, not the total compression from original to final. However, this still provides valuable relative information about which regions undergo more processing.

### To Get True Original Size

Would require modifying cache to store original dimensions before any resize. This needs a cache rebuild (slow) and is deferred for future work.

---

**Last Updated**: 2025-12-27
**Major Update**: Added compression ratio tracking and dual correlation analysis
