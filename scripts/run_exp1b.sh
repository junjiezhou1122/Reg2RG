#!/bin/bash
#
# Experiment 1b: Decoder with 2 layers (current default)
#
# Usage:
#   bash scripts/run_exp1b.sh [GPU_ID]
#
# Example:
#   bash scripts/run_exp1b.sh 0     # Run on GPU 0
#   bash scripts/run_exp1b.sh 1     # Run on GPU 1
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================================
# Configuration
# ============================================================================

# GPU device (can be overridden by command line argument)
CUDA_DEVICE="${1:-0}"  # Default: GPU 0

# Experiment details
EXP_NAME="exp1b"
DECODER_LAYERS=2

# Directories
OUTPUT_DIR="/mnt/home/zhoujunjie/outputs/LIT_exp1/${EXP_NAME}"
CACHE_DIR="/mnt2/ct/RadGenome-ChestCT/cache_lit"

# Training hyperparameters
NUM_EPOCHS=5
BATCH_SIZE=1
GRAD_ACCUM_STEPS=8
LEARNING_RATE=1e-4
NUM_WORKERS=4

# Validation and early stopping
VAL_CHECK_INTERVAL=10000
EARLY_STOPPING_PATIENCE=1

# W&B settings
USE_WANDB=true
WANDB_PROJECT="Reg2RG-LIT-Exp1"
WANDB_RUN_NAME="${EXP_NAME}_decoder${DECODER_LAYERS}layer"

# ============================================================================
# Setup
# ============================================================================

echo "════════════════════════════════════════════════════════════════════════"
echo "  Experiment 1b: Decoder with ${DECODER_LAYERS} layers (current default)"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  - Decoder layers: $DECODER_LAYERS"
echo "  - Epochs: $NUM_EPOCHS"
echo "  - Validation interval: $VAL_CHECK_INTERVAL steps"
echo "  - Early stopping patience: $EARLY_STOPPING_PATIENCE"
echo "  - CUDA device: $CUDA_DEVICE"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - W&B run: $WANDB_RUN_NAME"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/training.log"

# Check cache
CACHE_COUNT=$(find "$CACHE_DIR" -name "*.pt" -type f 2>/dev/null | wc -l)
echo "Cache status: $CACHE_COUNT files found"

if [ "$CACHE_COUNT" -lt 20000 ]; then
    echo "⚠️  Warning: Insufficient cached files. Expected ~25000, found $CACHE_COUNT"
    echo "Consider running cache warmup first:"
    echo "  python src/lit_recon_probe.py --precache_splits both --precache_only True"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ============================================================================
# Run Training
# ============================================================================

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training..."
echo ""

START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python src/lit_recon_probe.py \
    --decoder_layers "$DECODER_LAYERS" \
    --num_train_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --dataloader_num_workers "$NUM_WORKERS" \
    --output_dir "$OUTPUT_DIR" \
    --val_check_interval "$VAL_CHECK_INTERVAL" \
    --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
    --use_wandb "$USE_WANDB" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --save_top_k 3 \
    --monitor_metric "reg_cos" \
    --monitor_mode "max" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# ============================================================================
# Summary
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_HUMAN=$(printf '%02dh:%02dm:%02ds' $((DURATION/3600)) $((DURATION%3600/60)) $((DURATION%60)))

echo ""
echo "════════════════════════════════════════════════════════════════════════"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ${EXP_NAME} completed successfully in $DURATION_HUMAN"

    # Extract best metrics
    BEST_REG_COS=$(grep -E "val_reg_cos=" "$LOG_FILE" | tail -1 | grep -oP 'val_reg_cos=\K[0-9.]+' || echo "N/A")
    echo "   Best reg_cos: $BEST_REG_COS"
else
    echo "❌ ${EXP_NAME} failed with exit code $EXIT_CODE"
fi

echo "   Duration: $DURATION_HUMAN"
echo "   Log: $LOG_FILE"
echo "   Checkpoints: $OUTPUT_DIR/checkpoints/"
echo "════════════════════════════════════════════════════════════════════════"

exit $EXIT_CODE
