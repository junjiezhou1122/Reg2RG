#!/usr/bin/env python3
"""
Debug script to verify MONAI cache behavior.
Tests if transform outputs are correctly cached as tensors.
"""

import sys
import os
import glob
import torch

sys.path.insert(0, 'src')

from Dataset.radgenome_dataset_train import RadGenomeDataset_Train

print("=" * 60)
print("MONAI Cache Debug Script")
print("=" * 60)

# Create a test cache directory (force fresh cache with timestamp)
import time
test_cache = f"/tmp/test_cache_debug_{int(time.time())}"
os.makedirs(test_cache, exist_ok=True)

print(f"\n1. Creating dataset with test cache: {test_cache}")
ds = RadGenomeDataset_Train(
    text_tokenizer="/mnt/home/zhoujunjie/models/Llama-2-7b-chat-hf",
    data_folder="/mnt2/ct/RadGenome-ChestCT/dataset/train_preprocessed",
    mask_folder="/mnt2/ct/RadGenome-ChestCT/dataset/train_region_mask",
    csv_file="/mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/train_region_report.csv",
    cache_dir=test_cache
)

print(f"   Dataset length: {len(ds)}")

# Test MONAI's internal _transform method directly
print("\n2. Testing MONAI _transform on first sample...")
try:
    # MONAI PersistentDataset's _transform takes an index, not a sample
    result = ds._transform(0)
    print(f"   Output keys: {list(result.keys())}")
    
    # Check if tensors are in output
    has_tensors = False
    print("\n3. Checking _transform output:")
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            print(f"   ✓ {k}: Tensor {tuple(v.shape)} {v.dtype} ({v.numel() * v.element_size() / 1e6:.1f} MB)")
            has_tensors = True
        else:
            print(f"   ✗ {k}: {type(v).__name__} ({len(str(v))} chars)")
    
    if not has_tensors:
        print("\n   ❌ ERROR: No tensors in _transform output!")
    else:
        print("\n   ✓ Transform outputs contain tensors")
except Exception as e:
    print(f"   ✗ Error during _transform: {e}")
    import traceback
    traceback.print_exc()

# Test actual dataset __getitem__ (triggers MONAI caching)
print("\n4. Calling dataset[0] (triggers MONAI PersistentDataset)...")
try:
    item = ds[0]
    print("   ✓ Successfully loaded item")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Check cache files
print("\n5. Checking cache files...")
cache_files = glob.glob(f"{test_cache}/*.pt")
print(f"   Found {len(cache_files)} cache file(s)")

if cache_files:
    # Check first cache file
    cache_file = cache_files[0]
    file_size = os.path.getsize(cache_file)
    print(f"\n   File: {os.path.basename(cache_file)}")
    print(f"   Size: {file_size:,} bytes ({file_size / 1e6:.2f} MB)")
    
    # Load and inspect
    cached = torch.load(cache_file, map_location="cpu")
    print(f"   Keys: {list(cached.keys())}")
    
    cached_tensors = []
    print("\n   Content:")
    for k, v in cached.items():
        if isinstance(v, torch.Tensor):
            size_mb = v.numel() * v.element_size() / 1e6
            print(f"     ✓ {k}: Tensor {tuple(v.shape)} {v.dtype} ({size_mb:.1f} MB)")
            cached_tensors.append(k)
        else:
            print(f"     ✗ {k}: {type(v).__name__}")
    
    # Final verdict
    print("\n" + "=" * 60)
    if file_size > 10_000_000:  # > 10 MB
        print("✓✓✓ CACHE IS WORKING CORRECTLY ✓✓✓")
        print(f"Cache file is {file_size / 1e6:.1f} MB (expected ~100-200 MB)")
        print(f"Contains {len(cached_tensors)} tensor(s): {cached_tensors}")
    else:
        print("❌❌❌ CACHE IS BROKEN ❌❌❌")
        print(f"Cache file is only {file_size:,} bytes (should be ~100-200 MB)")
        print("Cached data contains strings instead of tensors!")
        print("\nPossible causes:")
        print("1. Transform is not modifying input dict in-place")
        print("2. MONAI PersistentDataset is not using the transform output")
        print("3. Python module cache issue (try: find . -name '*.pyc' -delete)")
    print("=" * 60)
else:
    print("   ❌ No cache files found!")

print("\nCleanup: rm -rf", test_cache)
