#!/bin/bash
#
# Cleanup Failed Cache Files Script
#
# Purpose: Remove corrupted cache files from validation set warmup failure
# - Files with literal b'...' in filename
# - Leftover .tmp files
#
# Usage:
#   bash scripts/cleanup_failed_cache.sh /mnt2/ct/RadGenome-ChestCT/cache_lit
#

set -e  # Exit on error

CACHE_DIR="${1:-/mnt2/ct/RadGenome-ChestCT/cache_lit}"

if [ ! -d "$CACHE_DIR" ]; then
    echo "❌ Error: Cache directory not found: $CACHE_DIR"
    echo "Usage: bash $0 <cache_directory>"
    exit 1
fi

echo "🔍 Scanning cache directory: $CACHE_DIR"
echo ""

# Count total cache files
TOTAL_FILES=$(find "$CACHE_DIR" -name "*.pt" -type f 2>/dev/null | wc -l)
echo "📊 Total cache files (.pt): $TOTAL_FILES"

# Find files with literal b'...' in filename (corrupted)
echo ""
echo "🔍 Searching for corrupted files (with b'...' in filename)..."
CORRUPTED_FILES=$(find "$CACHE_DIR" -name "b'*'.pt" -type f 2>/dev/null || true)
CORRUPTED_COUNT=$(echo "$CORRUPTED_FILES" | grep -c "\.pt$" || echo "0")

if [ "$CORRUPTED_COUNT" -gt 0 ]; then
    echo "⚠️  Found $CORRUPTED_COUNT corrupted files:"
    echo "$CORRUPTED_FILES" | head -10
    if [ "$CORRUPTED_COUNT" -gt 10 ]; then
        echo "   ... and $((CORRUPTED_COUNT - 10)) more"
    fi

    read -p "❓ Delete these corrupted files? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Deleting corrupted files..."
        find "$CACHE_DIR" -name "b'*'.pt" -type f -delete
        echo "✅ Deleted $CORRUPTED_COUNT corrupted files"
    else
        echo "⏭️  Skipped deletion"
    fi
else
    echo "✅ No corrupted files found (good!)"
fi

# Find leftover .tmp files
echo ""
echo "🔍 Searching for leftover .tmp files..."
TMP_FILES=$(find "$CACHE_DIR" -name "*.tmp" -type f 2>/dev/null || true)
TMP_COUNT=$(echo "$TMP_FILES" | grep -c "\.tmp$" || echo "0")

if [ "$TMP_COUNT" -gt 0 ]; then
    echo "⚠️  Found $TMP_COUNT leftover .tmp files:"
    echo "$TMP_FILES" | head -10
    if [ "$TMP_COUNT" -gt 10 ]; then
        echo "   ... and $((TMP_COUNT - 10)) more"
    fi

    read -p "❓ Delete these .tmp files? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Deleting .tmp files..."
        find "$CACHE_DIR" -name "*.tmp" -type f -delete
        echo "✅ Deleted $TMP_COUNT .tmp files"
    else
        echo "⏭️  Skipped deletion"
    fi
else
    echo "✅ No leftover .tmp files found"
fi

# Final statistics
echo ""
echo "📊 Final cache statistics:"
FINAL_COUNT=$(find "$CACHE_DIR" -name "*.pt" -type f 2>/dev/null | wc -l)
echo "   Total valid cache files: $FINAL_COUNT"
CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
echo "   Total cache size: $CACHE_SIZE"

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. The code fix is already applied to radgenome_dataset_train.py"
echo "   2. Training set cache (24,126 files) is intact - no re-warmup needed"
echo "   3. Run validation warmup: python src/lit_recon_probe.py --precache_splits val --precache_only True"
echo "   4. Then start training: python src/lit_recon_probe.py --decoder_layers 2 --num_train_epochs 15"
