# Experiment 1 - Individual Scripts Usage Guide

## 📁 Available Scripts

Each experiment has its own independent script:

- `run_exp1a.sh` - Decoder with **1 layer** (baseline)
- `run_exp1b.sh` - Decoder with **2 layers** (current default)
- `run_exp1c.sh` - Decoder with **4 layers** (deeper)
- `run_exp1d.sh` - Decoder with **6 layers** (deepest)

## 🚀 Basic Usage

### Run a single experiment

```bash
# Make scripts executable (first time only)
chmod +x scripts/run_exp1*.sh

# Run on default GPU (GPU 0)
bash scripts/run_exp1a.sh

# Or specify GPU device
bash scripts/run_exp1a.sh 0    # Run on GPU 0
bash scripts/run_exp1b.sh 1    # Run on GPU 1
bash scripts/run_exp1c.sh 2    # Run on GPU 2
bash scripts/run_exp1d.sh 3    # Run on GPU 3
```

## 🔥 Parallel Execution (Multiple GPUs)

Run all 4 experiments in parallel on different GPUs:

```bash
# Method 1: Background jobs
bash scripts/run_exp1a.sh 0 &
bash scripts/run_exp1b.sh 1 &
bash scripts/run_exp1c.sh 2 &
bash scripts/run_exp1d.sh 3 &

# Wait for all to complete
wait

echo "All experiments completed!"
```

```bash
# Method 2: Using nohup (survives terminal disconnection)
nohup bash scripts/run_exp1a.sh 0 > exp1a.out 2>&1 &
nohup bash scripts/run_exp1b.sh 1 > exp1b.out 2>&1 &
nohup bash scripts/run_exp1c.sh 2 > exp1c.out 2>&1 &
nohup bash scripts/run_exp1d.sh 3 > exp1d.out 2>&1 &

# Check status
jobs
tail -f exp1a.out  # Monitor progress
```

```bash
# Method 3: Using tmux (recommended for long-running experiments)
# Terminal 1
tmux new -s exp1a
bash scripts/run_exp1a.sh 0
# Ctrl+B, D to detach

# Terminal 2
tmux new -s exp1b
bash scripts/run_exp1b.sh 1
# Ctrl+B, D to detach

# Terminal 3
tmux new -s exp1c
bash scripts/run_exp1c.sh 2
# Ctrl+B, D to detach

# Terminal 4
tmux new -s exp1d
bash scripts/run_exp1d.sh 3
# Ctrl+B, D to detach

# Reattach to monitor:
tmux attach -t exp1a
```

## 📊 Monitor Progress

### Check logs
```bash
# View real-time logs
tail -f /mnt/home/zhoujunjie/outputs/LIT_exp1/exp1a/training.log

# Search for validation metrics
grep "val_reg_cos=" /mnt/home/zhoujunjie/outputs/LIT_exp1/exp1a/training.log

# Check if training is still running
ps aux | grep lit_recon_probe
```

### Check GPU usage
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use gpustat (if installed)
watch -n 1 gpustat
```

### Check W&B dashboard
Visit: https://wandb.ai

Project: `Reg2RG-LIT-Exp1`

Runs:
- `exp1a_decoder1layer`
- `exp1b_decoder2layer`
- `exp1c_decoder4layer`
- `exp1d_decoder6layer`

## 📈 Compare Results

After completion:

```bash
# Quick comparison
for exp in exp1a exp1b exp1c exp1d; do
    echo "=== $exp ==="
    grep "Best reg_cos:" /mnt/home/zhoujunjie/outputs/LIT_exp1/$exp/training.log || echo "Not completed"
done

# View CSV metrics
for exp in exp1a exp1b exp1c exp1d; do
    echo "=== $exp ==="
    tail -3 /mnt/home/zhoujunjie/outputs/LIT_exp1/$exp/lit_metrics.csv
done

# Check saved checkpoints
for exp in exp1a exp1b exp1c exp1d; do
    echo "=== $exp ==="
    ls -lh /mnt/home/zhoujunjie/outputs/LIT_exp1/$exp/checkpoints/
done
```

## ⚙️ Customization

To modify hyperparameters, edit the script directly:

```bash
# Open script for editing
vim scripts/run_exp1a.sh

# Key parameters (lines 27-42):
NUM_EPOCHS=5                    # Change to 10 for longer training
VAL_CHECK_INTERVAL=10000        # Change to 5000 for more frequent validation
EARLY_STOPPING_PATIENCE=1       # Change to 2 for less aggressive stopping
```

## 🛑 Stopping Experiments

```bash
# Find running processes
ps aux | grep lit_recon_probe

# Kill specific experiment (replace PID)
kill -SIGINT <PID>  # Graceful stop (saves checkpoint)

# Or kill all
pkill -f "lit_recon_probe.py --decoder_layers"

# If using tmux
tmux attach -t exp1a
# Then Ctrl+C to stop
```

## 🔄 Resume Failed Experiment

```bash
# Find the latest checkpoint
ls -lt /mnt/home/zhoujunjie/outputs/LIT_exp1/exp1a/checkpoints/

# Edit script to add resume parameter
vim scripts/run_exp1a.sh

# Add this line before the python command:
RESUME_CKPT="/mnt/home/zhoujunjie/outputs/LIT_exp1/exp1a/checkpoints/epoch=003_val_reg_cos=0.654321.pt"

# Add to python command:
    --resume_from_checkpoint "$RESUME_CKPT" \

# Run again
bash scripts/run_exp1a.sh 0
```

## 📋 Expected Timeline

With cache warmed up:

- **Per epoch**: ~2-3 hours
- **Per experiment**: ~10-15 hours (5 epochs max, may stop earlier)
- **Early stopping**: Can finish in 2-3 epochs if not improving

Running all 4 in parallel on separate GPUs: ~10-15 hours total

## ⚠️ Troubleshooting

### Cache not ready
```bash
# Check cache
ls /mnt2/ct/RadGenome-ChestCT/cache_lit/*.pt | wc -l
# Should be ~25,000

# Warm up if needed (takes ~12-16 hours)
python src/lit_recon_probe.py --precache_splits both --precache_only True
```

### Out of memory
```bash
# Edit script to reduce workers
vim scripts/run_exp1a.sh
# Change: NUM_WORKERS=4 → NUM_WORKERS=2
```

### GPU not available
```bash
# Check GPUs
nvidia-smi

# Change GPU in command
bash scripts/run_exp1a.sh 1  # Use GPU 1 instead
```

## 📚 Next Steps

After all experiments complete:

1. **Compare results** using W&B dashboard
2. **Analyze best checkpoints** for each decoder depth
3. **Document findings** in the experiment log
4. **Decide on next experiment** (Exp2: joint training)

---

**Quick Start** (cache already warmed up):
```bash
# Run all 4 experiments in parallel
bash scripts/run_exp1a.sh 0 &
bash scripts/run_exp1b.sh 1 &
bash scripts/run_exp1c.sh 2 &
bash scripts/run_exp1d.sh 3 &
wait
echo "Done! Check results in /mnt/home/zhoujunjie/outputs/LIT_exp1/"
```
