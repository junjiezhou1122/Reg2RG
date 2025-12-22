#!/usr/bin/env python3
"""
Diagnostic script to debug cache filename generation issue.

This script helps identify why cache files are being double-hex-encoded.
Run this on the server to check the hash function behavior.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from Dataset.radgenome_dataset_train import RadGenomeDataset_Train
import torch

def diagnose_hash_issue():
    """Diagnose the hash generation issue."""

    print("=" * 70)
    print("🔍 RadGenome Cache Hash Diagnostic Tool")
    print("=" * 70)
    print()

    # Create a minimal dataset instance
    print("📦 Creating minimal dataset instance...")
    try:
        dataset = RadGenomeDataset_Train(
            text_tokenizer="/mnt/home/zhoujunjie/models/Llama-2-7b-chat-hf",
            data_folder="/mnt2/ct/RadGenome-ChestCT/dataset/valid_preprocessed",
            mask_folder="/mnt2/ct/RadGenome-ChestCT/dataset/valid_region_mask",
            csv_file="/mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv",
            cache_dir="/mnt2/ct/RadGenome-ChestCT/cache_lit",
        )
        print("✅ Dataset created successfully")
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return

    print()
    print("=" * 70)
    print("🧪 Testing hash function behavior")
    print("=" * 70)
    print()

    # Get first data item
    if len(dataset.data) == 0:
        print("❌ Dataset is empty!")
        return

    test_item = dataset.data[0]
    print(f"📋 Test item keys: {list(test_item.keys())}")
    print()

    # Test hash_func
    print("🔬 Testing hash_func(item)...")
    try:
        hash_val = dataset.hash_func(test_item)
        print(f"   Type: {type(hash_val)}")
        print(f"   Value: {repr(hash_val)}")
        print(f"   Length: {len(hash_val) if hasattr(hash_val, '__len__') else 'N/A'}")

        if isinstance(hash_val, bytes):
            print(f"   → As hex: {hash_val.hex()}")
            print(f"   → Hex length: {len(hash_val.hex())}")
        elif isinstance(hash_val, str):
            print(f"   → Is hex string: {all(c in '0123456789abcdefABCDEF' for c in hash_val)}")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    print()

    # Test _get_hashfile
    print("🔬 Testing _get_hashfile(item)...")
    try:
        hashfile = dataset._get_hashfile(test_item)
        filename = os.path.basename(hashfile)
        print(f"   Full path: {hashfile}")
        print(f"   Filename: {filename}")
        print(f"   Filename length: {len(filename)}")

        # Check if double-encoded
        if len(filename) > 40:  # Normal MD5 hex is 32 chars + ".pt" = 35 chars
            print("   ⚠️  WARNING: Filename looks too long (possible double encoding)")

            # Try to decode
            try:
                name_without_ext = filename.replace('.pt', '')
                decoded = bytes.fromhex(name_without_ext).decode('ascii')
                print(f"   → Decoded: {decoded}")
                print(f"   → This looks like double-hex-encoding!")
            except:
                print("   → Could not decode as hex")
        else:
            print("   ✅ Filename length looks normal")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    print()
    print("=" * 70)
    print("💡 Recommendations")
    print("=" * 70)
    print()

    if isinstance(hash_val, bytes):
        if len(hash_val.hex()) == 32:
            print("✅ hash_func returns bytes (normal)")
            print("   → Fixed code should handle this correctly")
        else:
            print("⚠️  hash_func returns bytes but unusual length")
    elif isinstance(hash_val, str):
        is_hex = all(c in '0123456789abcdefABCDEF' for c in hash_val)
        if is_hex and len(hash_val) == 32:
            print("✅ hash_func returns hex string (normal)")
            print("   → Fixed code should handle this correctly")
        elif is_hex and len(hash_val) == 64:
            print("⚠️  hash_func returns hex string but double length (64 chars)")
            print("   → This is already double-encoded!")
        else:
            print("⚠️  hash_func returns non-hex string")
            print("   → This is unusual and may cause issues")

    print()
    print("🔧 If you see double-encoding:")
    print("   1. Make sure you ran 'git pull' to get the latest code")
    print("   2. Stop the Python process completely (Ctrl+C)")
    print("   3. Re-run the warmup command from scratch")
    print("   4. Python may have cached the old module - full restart required")
    print()

if __name__ == "__main__":
    diagnose_hash_issue()
