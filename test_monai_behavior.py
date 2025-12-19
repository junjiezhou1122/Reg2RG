#!/usr/bin/env python3
"""Test MONAI PersistentDataset's actual caching behavior."""

import os
import torch
from monai.data import PersistentDataset

# Simple test transform
def simple_transform(data):
    print(f"[Transform] Input keys: {list(data.keys())}")
    result = {
        "original_path": data["path"],
        "tensor": torch.randn(10, 10),  # Create a tensor
    }
    print(f"[Transform] Output keys: {list(result.keys())}")
    return result

# Test data
test_data = [
    {"path": "/fake/path1.nii", "label": "a"},
    {"path": "/fake/path2.nii", "label": "b"},
]

# Create dataset with cache
cache_dir = "/tmp/monai_test_cache"
os.makedirs(cache_dir, exist_ok=True)

print("=" * 60)
print("Creating PersistentDataset...")
ds = PersistentDataset(
    data=test_data,
    transform=simple_transform,
    cache_dir=cache_dir,
)

print(f"\nDataset length: {len(ds)}")

print("\n" + "=" * 60)
print("Calling ds[0] (should trigger transform and cache)...")
item = ds[0]
print(f"Returned keys: {list(item.keys())}")
for k, v in item.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: Tensor {tuple(v.shape)}")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")

print("\n" + "=" * 60)
print("Checking cache file...")
import glob
cache_files = glob.glob(f"{cache_dir}/*.pt")
if cache_files:
    cache_file = cache_files[0]
    print(f"Cache file: {cache_file}")
    print(f"Size: {os.path.getsize(cache_file)} bytes")
    
    cached = torch.load(cache_file, map_location="cpu")
    print(f"Cached keys: {list(cached.keys())}")
    for k, v in cached.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: Tensor {tuple(v.shape)}")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")
else:
    print("No cache files found!")

print("\n" + "=" * 60)
print("Cleanup:", cache_dir)
