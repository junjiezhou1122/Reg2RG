#!/usr/bin/env python3
"""
Test script to verify cache skipping logic works correctly.
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from Dataset.radgenome_dataset_train import RadGenomeDataset_Train

def test_cache_skip():
    """Test that cache_exists correctly identifies cached samples."""
    # Example paths (adjust these to match your setup)
    text_tokenizer = "/mnt/home/zhoujunjie/models/Llama-2-7b-chat-hf"
    data_folder = "/path/to/data"  # Replace with your path
    mask_folder = "/path/to/masks"  # Replace with your path
    csv_file = "/path/to/reports.csv"  # Replace with your path
    cache_dir = "/path/to/cache"  # Replace with your path

    print("Loading dataset...")
    dataset = RadGenomeDataset_Train(
        text_tokenizer=text_tokenizer,
        data_folder=data_folder,
        mask_folder=mask_folder,
        csv_file=csv_file,
        cache_dir=cache_dir,
    )

    print(f"Total samples: {len(dataset)}")

    # Check how many samples have existing cache
    cached_count = 0
    for i in range(len(dataset)):
        if dataset.cache_exists(i):
            cached_count += 1

    print(f"Cached samples: {cached_count}/{len(dataset)}")
    print(f"Uncached samples: {len(dataset) - cached_count}")

    if cached_count > 0:
        print("\n✓ Cache skip logic is working! Existing caches detected.")
    else:
        print("\n✗ No cached samples found. Either cache is empty or detection is not working.")

if __name__ == "__main__":
    test_cache_skip()
