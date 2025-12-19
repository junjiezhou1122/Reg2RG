#!/usr/bin/env python3
"""
Utility script to validate and clean corrupted cache files.

Usage:
    python cleanup_cache.py --cache_dir /path/to/cache --mode check
    python cleanup_cache.py --cache_dir /path/to/cache --mode clean
    python cleanup_cache.py --cache_dir /path/to/cache --mode deep_clean
"""
import os
import argparse
import torch
from pathlib import Path
from tqdm import tqdm


def check_cache_file(filepath: str, deep_check: bool = False) -> tuple[bool, str]:
    """
    Check if a cache file is valid.

    Args:
        filepath: Path to the .pt cache file
        deep_check: If True, try to load the file to verify integrity

    Returns:
        (is_valid, error_message)
    """
    # Check if file exists
    if not os.path.exists(filepath):
        return False, "File does not exist"

    # Check if file has non-zero size
    try:
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            return False, "Empty file (0 bytes)"
    except (OSError, IOError) as e:
        return False, f"Cannot access file: {e}"

    # Deep check: try to load the file
    if deep_check:
        try:
            torch.load(filepath, map_location="cpu")
            return True, ""
        except Exception as e:
            return False, f"Cannot load file: {e}"

    return True, ""


def scan_cache_directory(cache_dir: str, mode: str = "check"):
    """
    Scan cache directory and report/clean corrupted files.

    Args:
        cache_dir: Path to the cache directory
        mode: "check" (report only), "clean" (basic), "deep_clean" (load each file)
    """
    if not os.path.exists(cache_dir):
        print(f"Error: Cache directory does not exist: {cache_dir}")
        return

    # Find all .pt and .tmp files
    pt_files = list(Path(cache_dir).glob("*.pt"))
    tmp_files = list(Path(cache_dir).glob("*.tmp"))

    print(f"Found {len(pt_files)} .pt files and {len(tmp_files)} .tmp files")

    # Handle .tmp files (incomplete writes)
    if tmp_files:
        print(f"\n⚠️  Found {len(tmp_files)} temporary files (.tmp) from interrupted writes")
        if mode in ["clean", "deep_clean"]:
            for tmp_file in tmp_files:
                try:
                    os.remove(tmp_file)
                    print(f"  Removed: {tmp_file.name}")
                except Exception as e:
                    print(f"  Failed to remove {tmp_file.name}: {e}")
        else:
            print("  Run with --mode clean to remove them")

    # Check .pt files
    corrupted_files = []
    valid_files = []

    deep_check = (mode == "deep_clean")
    desc = "Deep checking cache files" if deep_check else "Checking cache files"

    for pt_file in tqdm(pt_files, desc=desc):
        is_valid, error = check_cache_file(str(pt_file), deep_check=deep_check)
        if is_valid:
            valid_files.append(pt_file)
        else:
            corrupted_files.append((pt_file, error))

    # Report results
    print(f"\n{'='*60}")
    print(f"Cache Validation Report")
    print(f"{'='*60}")
    print(f"Total files:      {len(pt_files)}")
    print(f"Valid files:      {len(valid_files)}")
    print(f"Corrupted files:  {len(corrupted_files)}")

    if corrupted_files:
        print(f"\n⚠️  Corrupted cache files:")
        for filepath, error in corrupted_files:
            print(f"  - {filepath.name}: {error}")

        if mode in ["clean", "deep_clean"]:
            print(f"\nCleaning corrupted files...")
            for filepath, error in corrupted_files:
                try:
                    os.remove(filepath)
                    print(f"  ✓ Removed: {filepath.name}")
                except Exception as e:
                    print(f"  ✗ Failed to remove {filepath.name}: {e}")
        else:
            print(f"\nRun with --mode clean or --mode deep_clean to remove corrupted files")
    else:
        print(f"\n✓ All cache files are valid!")

    # Calculate space used
    total_size = sum(f.stat().st_size for f in valid_files)
    print(f"\nCache size: {total_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Validate and clean MONAI persistent cache files"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Path to the cache directory"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["check", "clean", "deep_clean"],
        default="check",
        help=(
            "check: Report only (fast, checks file size)\n"
            "clean: Remove corrupted files (fast, checks file size)\n"
            "deep_clean: Load each file to verify integrity (slow but thorough)"
        )
    )

    args = parser.parse_args()

    print(f"Mode: {args.mode}")
    print(f"Cache directory: {args.cache_dir}\n")

    if args.mode in ["clean", "deep_clean"]:
        response = input("⚠️  This will delete corrupted cache files. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    scan_cache_directory(args.cache_dir, args.mode)


if __name__ == "__main__":
    main()
