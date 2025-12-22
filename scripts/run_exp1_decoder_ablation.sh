#!/bin/bash
#
# Experiment 1: Decoder Depth Ablation Study
#
# This script runs 4 variants of the LIT probe with different decoder depths
# to understand whether the pretrained adapter preserves sufficient information.
#
# Variants:
#   - exp1a: decoder_layers=1 (baseline, weakest)
#   - exp1b: decoder_layers=2 (current default)
#   - exp1c: decoder_layers=4 (deeper)
#   - exp1d: decoder_layers=6 (deepest)
#
# Configuration:
#   - 5 epochs (reduced from 15 for faster iteration)
#   - Validation every epoch
#   - Early stopping with patience=3 (stop if no improvement for 3 epochs)
#   - All experiments logged to W&B
#

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================================
# Configuration
# ============================================================================

# Base directories
PROJECT_DIR="$HOME/Reg2RG"
OUTPUT_BASE="/mnt/home/zhoujunjie/outputs/LIT_exp1"
CACHE_DIR="/mnt2/ct/RadGenome-ChestCT/cache_lit"

# Training hyperparameters
NUM_EPOCHS=5
BATCH_SIZE=1
GRAD_ACCUM_STEPS=8
LEARNING_RATE=1e-4
NUM_WORKERS=4

# Validation and early stopping
VAL_CHECK_INTERVAL=10000  # Run validation every 10K steps
EARLY_STOPPING_PATIENCE=1  # Stop if no improvement for 1 validation

# CUDA device (set to specific GPU, e.g., "0" or "0,1")
CUDA_DEVICE="0"  # Change this to your desired GPU

# W&B settings
USE_WANDB=true
WANDB_PROJECT="Reg2RG-LIT-Exp1"

# Decoder layer variants to test
DECODER_LAYERS=(1 2 4 6)
EXP_NAMES=("exp1a" "exp1b" "exp1c" "exp1d")

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ℹ️  $*"
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $*" >&2
}

log_section() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  $*"
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

log_section "Pre-flight Checks"

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    log_error "Project directory not found: $PROJECT_DIR"
    exit 1
fi
log_info "Project directory: $PROJECT_DIR"

# Check if cache directory exists
if [ ! -d "$CACHE_DIR" ]; then
    log_error "Cache directory not found: $CACHE_DIR"
    log_error "Please run cache warmup first!"
    exit 1
fi

# Count cached files
TRAIN_CACHE_COUNT=$(find "$CACHE_DIR" -name "*.pt" -type f 2>/dev/null | wc -l)
log_info "Found $TRAIN_CACHE_COUNT cached files"

if [ "$TRAIN_CACHE_COUNT" -lt 20000 ]; then
    log_error "Insufficient cached files. Expected ~25000, found $TRAIN_CACHE_COUNT"
    log_error "Please complete cache warmup first!"
    exit 1
fi

# Check if conda environment is activated
if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    log_error "No conda environment activated!"
    log_error "Please run: conda activate reg2rg"
    exit 1
fi
log_info "Conda environment: $CONDA_DEFAULT_ENV"

# Change to project directory
cd "$PROJECT_DIR"
log_info "Working directory: $(pwd)"

log_success "All pre-flight checks passed!"

# ============================================================================
# Main Experiment Loop
# ============================================================================

log_section "Starting Experiment 1: Decoder Depth Ablation"

# Create output base directory
mkdir -p "$OUTPUT_BASE"

# Summary file
SUMMARY_FILE="$OUTPUT_BASE/exp1_summary.txt"
echo "Experiment 1: Decoder Depth Ablation" > "$SUMMARY_FILE"
echo "Started: $(date)" >> "$SUMMARY_FILE"
echo "======================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Track overall start time
OVERALL_START_TIME=$(date +%s)

# Run each variant
for i in "${!DECODER_LAYERS[@]}"; do
    LAYERS="${DECODER_LAYERS[$i]}"
    EXP_NAME="${EXP_NAMES[$i]}"

    log_section "Running $EXP_NAME: decoder_layers=$LAYERS"

    # Output directory for this variant
    OUTPUT_DIR="$OUTPUT_BASE/${EXP_NAME}"
    mkdir -p "$OUTPUT_DIR"

    # Log file
    LOG_FILE="$OUTPUT_DIR/training.log"

    # W&B run name
    WANDB_RUN_NAME="${EXP_NAME}_decoder${LAYERS}layer"

    log_info "Configuration:"
    log_info "  - Decoder layers: $LAYERS"
    log_info "  - Epochs: $NUM_EPOCHS"
    log_info "  - Batch size: $BATCH_SIZE"
    log_info "  - Gradient accumulation: $GRAD_ACCUM_STEPS"
    log_info "  - Learning rate: $LEARNING_RATE"
    log_info "  - Validation interval: $VAL_CHECK_INTERVAL steps"
    log_info "  - Early stopping patience: $EARLY_STOPPING_PATIENCE"
    log_info "  - CUDA device: $CUDA_DEVICE"
    log_info "  - Output directory: $OUTPUT_DIR"
    log_info "  - Log file: $LOG_FILE"
    log_info "  - W&B run: $WANDB_RUN_NAME"

    # Record start time
    START_TIME=$(date +%s)

    # Run training
    log_info "Starting training..."

    python src/lit_recon_probe.py \
        --decoder_layers "$LAYERS" \
        --num_train_epochs "$NUM_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
        --learning_rate "$LEARNING_RATE" \
        --dataloader_num_workers "$NUM_WORKERS" \
        --output_dir "$OUTPUT_DIR" \
        --val_check_interval "$VAL_CHECK_INTERVAL" \
        --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
        --cuda_visible_devices "$CUDA_DEVICE" \
        --use_wandb "$USE_WANDB" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$WANDB_RUN_NAME" \
        --save_top_k 3 \
        --monitor_metric "reg_cos" \
        --monitor_mode "max" \
        2>&1 | tee "$LOG_FILE"

    # Check exit status
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}

    # Record end time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_HUMAN=$(printf '%02dh:%02dm:%02ds' $((DURATION/3600)) $((DURATION%3600/60)) $((DURATION%60)))

    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        log_success "$EXP_NAME completed successfully in $DURATION_HUMAN"

        # Extract best validation metrics from log
        BEST_REG_COS=$(grep -E "val_reg_cos=" "$LOG_FILE" | tail -1 | grep -oP 'val_reg_cos=\K[0-9.]+' || echo "N/A")
        BEST_COS=$(grep -E "val_cos=" "$LOG_FILE" | tail -1 | grep -oP 'val_cos=\K[0-9.]+' || echo "N/A")

        log_info "Best metrics: reg_cos=$BEST_REG_COS, cos=$BEST_COS"

        # Write to summary
        echo "$EXP_NAME (decoder_layers=$LAYERS):" >> "$SUMMARY_FILE"
        echo "  Status: SUCCESS" >> "$SUMMARY_FILE"
        echo "  Duration: $DURATION_HUMAN" >> "$SUMMARY_FILE"
        echo "  Best reg_cos: $BEST_REG_COS" >> "$SUMMARY_FILE"
        echo "  Best cos: $BEST_COS" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
    else
        log_error "$EXP_NAME failed with exit code $TRAIN_EXIT_CODE"

        # Write to summary
        echo "$EXP_NAME (decoder_layers=$LAYERS):" >> "$SUMMARY_FILE"
        echo "  Status: FAILED (exit code $TRAIN_EXIT_CODE)" >> "$SUMMARY_FILE"
        echo "  Duration: $DURATION_HUMAN" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"

        # Ask user if they want to continue
        read -p "⚠️  Training failed. Continue with next variant? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_error "Stopping experiment due to failure"
            exit 1
        fi
    fi

    log_info "Waiting 10 seconds before next variant..."
    sleep 10
done

# ============================================================================
# Final Summary
# ============================================================================

OVERALL_END_TIME=$(date +%s)
OVERALL_DURATION=$((OVERALL_END_TIME - OVERALL_START_TIME))
OVERALL_DURATION_HUMAN=$(printf '%02dh:%02dm:%02ds' $((OVERALL_DURATION/3600)) $((OVERALL_DURATION%3600/60)) $((OVERALL_DURATION%60)))

log_section "Experiment 1 Complete!"

echo "Finished: $(date)" >> "$SUMMARY_FILE"
echo "Total duration: $OVERALL_DURATION_HUMAN" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "======================================" >> "$SUMMARY_FILE"
echo "Summary:" >> "$SUMMARY_FILE"
cat "$SUMMARY_FILE"

log_success "All experiments completed in $OVERALL_DURATION_HUMAN"
log_info "Summary saved to: $SUMMARY_FILE"
log_info "Individual logs: $OUTPUT_BASE/exp1*/training.log"
log_info "Checkpoints: $OUTPUT_BASE/exp1*/checkpoints/"
log_info "Metrics: $OUTPUT_BASE/exp1*/lit_metrics.csv"

# ============================================================================
# Optional: Compare Results
# ============================================================================

log_section "Quick Results Comparison"

echo "Decoder Layers | Best reg_cos | Best cos"
echo "---------------|--------------|----------"

for i in "${!DECODER_LAYERS[@]}"; do
    EXP_NAME="${EXP_NAMES[$i]}"
    LAYERS="${DECODER_LAYERS[$i]}"
    METRICS_FILE="$OUTPUT_BASE/${EXP_NAME}/lit_metrics.csv"

    if [ -f "$METRICS_FILE" ]; then
        # Extract best validation metrics from CSV
        BEST_REG_COS=$(awk -F',' '$2=="val" {print $6}' "$METRICS_FILE" | sort -rn | head -1)
        BEST_COS=$(awk -F',' '$2=="val" {print $4}' "$METRICS_FILE" | sort -rn | head -1)
        printf "%14s | %12s | %8s\n" "$LAYERS" "$BEST_REG_COS" "$BEST_COS"
    else
        printf "%14s | %12s | %8s\n" "$LAYERS" "N/A" "N/A"
    fi
done

echo ""
log_info "For detailed analysis, check W&B dashboard: https://wandb.ai"
log_info "Or view CSV files: $OUTPUT_BASE/exp1*/lit_metrics.csv"

log_success "Done! 🎉"
