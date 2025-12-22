#!/bin/bash
#
# Quick start: Launch all 4 Experiment 1 variants in parallel on different GPUs
#
# Usage:
#   bash scripts/run_all_exp1_parallel.sh [GPU0] [GPU1] [GPU2] [GPU3]
#
# Examples:
#   bash scripts/run_all_exp1_parallel.sh              # Use GPUs 0,1,2,3
#   bash scripts/run_all_exp1_parallel.sh 0 1 2 3      # Explicit GPU assignment
#   bash scripts/run_all_exp1_parallel.sh 4 5 6 7      # Use GPUs 4,5,6,7
#   bash scripts/run_all_exp1_parallel.sh 0 0 1 1      # Run 2 on GPU 0, 2 on GPU 1
#

set -e

# ============================================================================
# Configuration
# ============================================================================

# GPU assignments (can be overridden by command line)
GPU_EXP1A="${1:-0}"
GPU_EXP1B="${2:-1}"
GPU_EXP1C="${3:-2}"
GPU_EXP1D="${4:-3}"

# ============================================================================
# Pre-flight Checks
# ============================================================================

echo "════════════════════════════════════════════════════════════════════════"
echo "  Starting Experiment 1: 4 Decoder Variants in Parallel"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Check if scripts exist
for script in run_exp1a.sh run_exp1b.sh run_exp1c.sh run_exp1d.sh; do
    if [ ! -f "scripts/$script" ]; then
        echo "❌ Error: scripts/$script not found"
        exit 1
    fi
done

# Check cache
CACHE_DIR="/mnt2/ct/RadGenome-ChestCT/cache_lit"
if [ -d "$CACHE_DIR" ]; then
    CACHE_COUNT=$(find "$CACHE_DIR" -name "*.pt" -type f 2>/dev/null | wc -l)
    echo "✅ Cache status: $CACHE_COUNT files"
    if [ "$CACHE_COUNT" -lt 20000 ]; then
        echo "⚠️  Warning: Cache might be incomplete (expected ~25,000)"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "❌ Error: Cache directory not found: $CACHE_DIR"
    exit 1
fi

# Display GPU assignment
echo ""
echo "GPU Assignment:"
echo "  - exp1a (1 layer):  GPU $GPU_EXP1A"
echo "  - exp1b (2 layers): GPU $GPU_EXP1B"
echo "  - exp1c (4 layers): GPU $GPU_EXP1C"
echo "  - exp1d (6 layers): GPU $GPU_EXP1D"
echo ""

read -p "Start all 4 experiments? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# ============================================================================
# Launch Experiments
# ============================================================================

echo ""
echo "🚀 Launching experiments in background..."
echo ""

# Create output directory for logs
OUTPUT_BASE="/mnt/home/zhoujunjie/outputs/LIT_exp1"
mkdir -p "$OUTPUT_BASE"

# Launch each experiment in background
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting exp1a on GPU $GPU_EXP1A..."
bash scripts/run_exp1a.sh "$GPU_EXP1A" > "$OUTPUT_BASE/exp1a_launcher.log" 2>&1 &
PID_EXP1A=$!

sleep 2  # Small delay between launches

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting exp1b on GPU $GPU_EXP1B..."
bash scripts/run_exp1b.sh "$GPU_EXP1B" > "$OUTPUT_BASE/exp1b_launcher.log" 2>&1 &
PID_EXP1B=$!

sleep 2

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting exp1c on GPU $GPU_EXP1C..."
bash scripts/run_exp1c.sh "$GPU_EXP1C" > "$OUTPUT_BASE/exp1c_launcher.log" 2>&1 &
PID_EXP1C=$!

sleep 2

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting exp1d on GPU $GPU_EXP1D..."
bash scripts/run_exp1d.sh "$GPU_EXP1D" > "$OUTPUT_BASE/exp1d_launcher.log" 2>&1 &
PID_EXP1D=$!

echo ""
echo "✅ All experiments launched!"
echo ""
echo "Process IDs:"
echo "  - exp1a: $PID_EXP1A"
echo "  - exp1b: $PID_EXP1B"
echo "  - exp1c: $PID_EXP1C"
echo "  - exp1d: $PID_EXP1D"
echo ""

# ============================================================================
# Save PIDs for later reference
# ============================================================================

PID_FILE="$OUTPUT_BASE/experiment_pids.txt"
cat > "$PID_FILE" <<EOF
# Experiment 1 Process IDs
# Started: $(date)
exp1a=$PID_EXP1A
exp1b=$PID_EXP1B
exp1c=$PID_EXP1C
exp1d=$PID_EXP1D
EOF

echo "Process IDs saved to: $PID_FILE"
echo ""

# ============================================================================
# Monitoring Instructions
# ============================================================================

echo "════════════════════════════════════════════════════════════════════════"
echo "  Monitoring & Control"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Monitor logs:"
echo "   tail -f $OUTPUT_BASE/exp1a/training.log"
echo "   tail -f $OUTPUT_BASE/exp1b/training.log"
echo "   tail -f $OUTPUT_BASE/exp1c/training.log"
echo "   tail -f $OUTPUT_BASE/exp1d/training.log"
echo ""
echo "📈 Check progress:"
echo "   watch -n 5 'grep \"epoch\" $OUTPUT_BASE/exp*/training.log | tail -4'"
echo ""
echo "🖥️  Monitor GPUs:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "🛑 Stop all experiments:"
echo "   kill $PID_EXP1A $PID_EXP1B $PID_EXP1C $PID_EXP1D"
echo ""
echo "📊 W&B Dashboard:"
echo "   https://wandb.ai (Project: Reg2RG-LIT-Exp1)"
echo ""

# ============================================================================
# Wait for completion (optional)
# ============================================================================

read -p "Wait for all experiments to complete? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "⏳ Waiting for all experiments to complete..."
    echo "   (This may take 10-15 hours. You can Ctrl+C to stop waiting.)"
    echo "   (Experiments will continue running in background)"
    echo ""

    START_TIME=$(date +%s)

    # Wait for all processes
    wait $PID_EXP1A $PID_EXP1B $PID_EXP1C $PID_EXP1D 2>/dev/null

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_HUMAN=$(printf '%02dh:%02dm:%02ds' $((DURATION/3600)) $((DURATION%3600/60)) $((DURATION%60)))

    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo "✅ All experiments completed in $DURATION_HUMAN"
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""

    # Display results summary
    echo "Results Summary:"
    echo ""
    for exp in exp1a exp1b exp1c exp1d; do
        LOG_FILE="$OUTPUT_BASE/$exp/training.log"
        if [ -f "$LOG_FILE" ]; then
            BEST_REG_COS=$(grep -E "val_reg_cos=" "$LOG_FILE" | tail -1 | grep -oP 'val_reg_cos=\K[0-9.]+' || echo "N/A")
            echo "  $exp: reg_cos = $BEST_REG_COS"
        else
            echo "  $exp: Log not found"
        fi
    done
    echo ""
    echo "Full logs: $OUTPUT_BASE/exp*/training.log"
    echo "Checkpoints: $OUTPUT_BASE/exp*/checkpoints/"
else
    echo ""
    echo "✅ Experiments running in background."
    echo "   Check status anytime with:"
    echo "   ps -p $PID_EXP1A $PID_EXP1B $PID_EXP1C $PID_EXP1D"
    echo ""
    echo "   Or view logs:"
    echo "   tail -f $OUTPUT_BASE/exp1a/training.log"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════"
