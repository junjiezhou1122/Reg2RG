# LIT Probe Experiment Design v3: Minimal-Capacity Adapter Study

**Date**: 2025-12-25
**Author**: Junjie Zhou
**Status**: Implementation Phase
**Based on**: LIT_Experiment_Design_v2.md + First Principles Analysis

---

## 📋 New Experiments Overview

This document extends v2 with **Experiment 5: Minimal-Capacity Adapter Comparison**, addressing a critical research question inspired by FAE (Feature Adaptation Encoder):

> **Core Question**: Does a 1-layer adapter provide more "honest" compression than deep adapters for information preservation evaluation?

---

## 🎯 Motivation: The "Deep Layer Laziness" Hypothesis

### Problem Statement

Current findings:
- **85% reconstruction quality** → **25% F1 score** (huge gap!)
- Global reconstruction > Region reconstruction (adapter prioritizes wrong features)
- 6-layer Perceiver: high params (~3.5M), unclear if depth helps or hurts

### Hypothesis (from FAE + First Principles)

```python
Deep adapters suffer from "layer laziness":
- Layer 1 learns: "Just do coarse extraction, later layers will refine"
- Gradient dilution: ∂Loss/∂Layer1 passes through 5 layers
- Functional division: Each layer optimizes for next layer, not final task

1-layer adapter is "forced to be honest":
- No subsequent layers to rely on
- Direct gradient: ∂Loss/∂Layer1 (no dilution)
- Must preserve all critical information in one pass
```

**Key insight**: Extracting Layer 1 from 6-layer ≠ Training 1-layer from scratch

---

## Experiment 5: Minimal-Capacity Adapter Comparison

### 🎯 Research Questions

1. **Can we use the 1st layer of pretrained 6-layer adapter?**
   - Does it preserve enough information when used alone?

2. **Does true 1-layer training improve ROI preservation?**
   - Compared to 6-layer baseline
   - Compared to extracted 1st layer

3. **Which is the best adapter for evaluation probing?**
   - For creating a diagnostic tool (APR: Adapter Probing via Reconstruction)

---

### ⚙️ Experimental Design

#### Comparison Matrix

| Exp ID | Adapter Config | Initialization | Training | Purpose |
|--------|---------------|---------------|----------|---------|
| **Exp1 (Baseline)** | 6-layer Perceiver | Pretrained | Frozen | Current best performance |
| **Exp2 (Joint)** | 6-layer Perceiver | Pretrained | Joint (adapter+decoder) | Can joint training improve? |
| **Exp5a (Extracted)** | 1st layer only | Extract from pretrained 6-layer | Joint (1st layer+decoder) | Test "deep layer laziness" |
| **Exp5b (Minimal)** | True 1-layer | Random init | Joint (1-layer+decoder) | Test "honest compression" |

---

### Exp5a: Extracted First-Layer Joint Training

#### 🎯 Objective

**Test whether the first layer of a pretrained 6-layer Perceiver can serve as a standalone adapter when jointly trained with decoder.**

This tests the "functional division" hypothesis: if Layer 1 was trained expecting Layer 2-6 to follow, it may have learned a "lazy" strategy that's suboptimal when used alone.

#### ⚙️ Model Configuration

```python
vision_encoder: Frozen ❄️           # Pretrained ViT3D
adapter: 1st layer of 6-layer 🔥    # Extracted + Trainable
  ↓ Load first layer weights from pretrained 6-layer checkpoint
  ↓ Trainable during this experiment
fc: Frozen ❄️                       # Projection layer
decoder: Trainable 🔥               # ProbeDecoder
```

#### Implementation Details

```python
# Adapter initialization
adapter = OneLayerAdapter(dim=768, num_latents=32, heads=8)

# Load weights from pretrained 6-layer checkpoint
load_first_layer_from_6layer_checkpoint(
    adapter,
    checkpoint_path="/mnt/home/zhoujunjie/models/Reg2RG/RadFM_perceiver_fc.pth"
)

# Training
set_requires_grad(adapter, True)  # Unlock for joint training
set_requires_grad(decoder, True)
```

#### Training Command

```bash
CUDA_VISIBLE_DEVICES=1 python3 src/lit_recon_probe.py \
    --tokenizer_path /mnt/home/zhoujunjie/models/Llama-2-7b-chat-hf \
    --pretrained_visual_encoder /mnt/home/zhoujunjie/models/Reg2RG/RadFM_vit3d.pth \
    --pretrained_adapter /mnt/home/zhoujunjie/models/Reg2RG/RadFM_perceiver_fc.pth \
    --adapter_depth 1 \
    --load_first_layer_from_pretrained True \
    --train_adapter True \
    --decoder_layers 4 \
    --decode_mode pre_proj \
    --data_folder /mnt2/ct/RadGenome-ChestCT/dataset/train_preprocessed \
    --mask_folder /mnt2/ct/RadGenome-ChestCT/dataset/train_region_mask \
    --report_file /mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/train_region_report.csv \
    --val_data_folder /mnt2/ct/RadGenome-ChestCT/dataset/valid_preprocessed \
    --val_mask_folder /mnt2/ct/RadGenome-ChestCT/dataset/valid_region_mask \
    --val_report_file /mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv \
    --monai_cache_dir /mnt2/ct/RadGenome-ChestCT/cache_lit \
    --output_dir /mnt/home/zhoujunjie/outputs/LIT_exp5a \
    --num_train_epochs 20 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --val_check_interval 0 \
    --save_top_k 3 \
    --monitor_metric reg_cos \
    --monitor_mode max \
    --use_wandb True \
    --wandb_project Reg2RG-LIT-Minimal-Capacity \
    --wandb_run_name exp5a_extracted_1st_layer_joint \
    --seed 42
```

#### Expected Outcome

**Hypothesis 1** (Functional division confirmed):
```
If Exp5a performance << Exp2 (6-layer joint):
  → Layer 1 was trained to "pass the baton" to Layer 2-6
  → Suboptimal when used standalone
  → Supports "deep layer laziness" hypothesis
```

**Hypothesis 2** (Layer 1 is versatile):
```
If Exp5a performance ≈ Exp2:
  → Layer 1 already captures most critical features
  → Later layers only do refinement
  → Surprising but valuable finding!
```

---

### Exp5b: True 1-Layer Joint Training (Minimal-Capacity)

#### 🎯 Objective

**Train a 1-layer adapter from scratch jointly with decoder, testing whether minimal capacity forces more honest, task-critical compression.**

This is the **core experiment** for the "Minimal-Capacity Probing Principle":
> Use the shallowest adapter that can compress. If it fails to preserve critical features under reconstruction, the compression is fundamentally unsafe.

#### ⚙️ Model Configuration

```python
vision_encoder: Frozen ❄️      # Pretrained ViT3D
adapter: 1-layer (new) 🔥      # Randomly initialized + Trainable
  ↓ Built from scratch (no pretrained weights)
  ↓ Learns to compress in one pass
fc: Frozen ❄️                  # Projection layer
decoder: Trainable 🔥          # ProbeDecoder
```

#### Implementation Details

```python
# Adapter initialization (random)
adapter = OneLayerAdapter(dim=768, num_latents=32, heads=8)
# No weight loading - starts from random initialization

# Training
set_requires_grad(adapter, True)
set_requires_grad(decoder, True)

# Hypothesis: 1-layer will be forced to:
# - Allocate attention to critical features (small lesions, boundaries)
# - Cannot "defer" to later layers
# - More "honest" compression
```

#### Training Command

```bash
CUDA_VISIBLE_DEVICES=1 python3 src/lit_recon_probe.py \
    --tokenizer_path /mnt/home/zhoujunjie/models/Llama-2-7b-chat-hf \
    --pretrained_visual_encoder /mnt/home/zhoujunjie/models/Reg2RG/RadFM_vit3d.pth \
    --adapter_depth 1 \
    --random_init_adapter True \
    --train_adapter True \
    --decoder_layers 4 \
    --decode_mode pre_proj \
    --data_folder /mnt2/ct/RadGenome-ChestCT/dataset/train_preprocessed \
    --mask_folder /mnt2/ct/RadGenome-ChestCT/dataset/train_region_mask \
    --report_file /mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/train_region_report.csv \
    --val_data_folder /mnt2/ct/RadGenome-ChestCT/dataset/valid_preprocessed \
    --val_mask_folder /mnt2/ct/RadGenome-ChestCT/dataset/valid_region_mask \
    --val_report_file /mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv \
    --monai_cache_dir /mnt2/ct/RadGenome-ChestCT/cache_lit \
    --output_dir /mnt/home/zhoujunjie/outputs/LIT_exp5b \
    --num_train_epochs 20 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --val_check_interval 0 \
    --save_top_k 3 \
    --monitor_metric reg_cos \
    --monitor_mode max \
    --use_wandb True \
    --wandb_project Reg2RG-LIT-Minimal-Capacity \
    --wandb_run_name exp5b_true_1layer_joint \
    --seed 42
```

#### Expected Outcome

**Hypothesis 1** (Minimal-capacity is better for ROI):
```
If Exp5b ROI preservation > Exp2 ROI preservation:
  → 1-layer forced to preserve critical features
  → Deep adapters "waste" capacity on easy-to-reconstruct features
  → Core finding for the paper! ✨
```

**Hypothesis 2** (Depth is necessary):
```
If Exp5b performance << Exp2:
  → 1 layer insufficient for this compression ratio (32:1)
  → Deep architecture is justified
  → But: Check if ROI/Global ratio is still better in Exp5b
```

---

### 📊 Comprehensive Comparison Table

| Exp | Adapter | Params | Train | Global Cos | ROI Cos | Small Lesion | ROI/Global | Notes |
|-----|---------|--------|-------|-----------|---------|--------------|-----------|-------|
| **Exp1** | 6-layer frozen | 3.5M | Decoder only | 0.88 | 0.70 | 0.45 | 0.795 | Baseline |
| **Exp2** | 6-layer joint | 3.5M | Joint | 0.90 | 0.78 | 0.55 | 0.867 | Current best |
| **Exp5a** | 1st layer joint | 0.6M | Joint | 0.85? | 0.72? | 0.50? | 0.847? | Test extraction |
| **Exp5b** | 1-layer joint | 0.6M | Joint | **0.87?** | **0.82?** ✨ | **0.68?** ✨ | **0.943?** ✅ | **Minimal-capacity** |

**Key Metric**: **ROI/Global Ratio**
- Higher = better balance between global and fine-grained features
- Hypothesis: Exp5b should have highest ratio (most "honest")

---

### 📈 Analysis Dimensions

#### 1. Reconstruction Quality by Region Type

```python
# Analyze per-region performance
regions = ["lung_left", "lung_right", "heart", "mediastinum",
           "thyroid", "esophagus", "trachea", "aorta"]

for region in regions:
    region_cos[exp][region] = ...

# Check:
# - Are small organs (thyroid, esophagus) better preserved in Exp5b?
# - Is the variance across regions lower in Exp5b? (more balanced)
```

#### 2. Attention Pattern Analysis

```python
# Visualize where 1-layer adapter attends
# Hypothesis: Should attend more to lesion/boundary regions

attention_maps = extract_attention_weights(adapter, CT_scan)

# Compare:
# - Exp2 (6-layer): diffuse attention?
# - Exp5b (1-layer): focused on critical regions?
```

#### 3. Training Dynamics

```python
# Compare convergence speed and stability
plot_metrics = [
    "train/loss",
    "val/reg_cos",
    "val/cos",
    "val/small_lesion_cos"  # New metric!
]

# Questions:
# - Does Exp5b converge faster? (fewer params)
# - Is training more stable? (no deep gradients)
```

---

### ✅ Success Criteria

#### Strong Success (Supports Minimal-Capacity Principle)

```
Conditions:
1. Exp5b ROI Cos > Exp2 ROI Cos (e.g., 0.82 > 0.78)
2. Exp5b Small Lesion Cos >> Exp2 Small Lesion (e.g., 0.68 > 0.55)
3. Exp5b ROI/Global ratio > Exp2 (more balanced)
4. Exp5a performance < Exp5b (confirms "layer laziness")

Conclusion:
✅ 1-layer adapter provides more honest compression
✅ Suitable as standard evaluation probe
✅ Supports FAE's finding in a new domain

Action:
→ Publish as "Minimal-Capacity Probing for VLM Adapters"
→ Establish as evaluation protocol
```

#### Partial Success (Trade-off)

```
Conditions:
1. Exp5b Global Cos < Exp2 (e.g., 0.87 < 0.90)
2. BUT Exp5b ROI Cos ≈ Exp2 or slightly better
3. Exp5b trains much faster (6× fewer params)

Conclusion:
✅ 1-layer sufficient for most information
✅ Useful for fast prototyping
⚠️ May need 6-layer for final deployment

Action:
→ Recommend 1-layer for evaluation, 6-layer for production
```

#### Failure (Depth is necessary)

```
Conditions:
1. Exp5b performance << Exp2 across all metrics
2. Exp5a ≈ Exp5b (no difference between extracted vs fresh)

Conclusion:
❌ 1-layer insufficient for 32:1 compression
❌ Need iterative refinement

Action:
→ Test with lower compression ratio (e.g., 64 latents instead of 32)
→ Or use 2-layer as minimal capacity
```

---

## 🔬 Technical Implementation Plan

### Phase 1: Code Infrastructure (Days 1-2)

#### File Structure
```
Model/
  ├── one_layer_adapter.py        # NEW: 1-layer adapter implementation
  ├── adapter_utils.py            # NEW: Weight loading utilities
  └── perceiver_resampler.py      # EXISTING

src/
  └── lit_recon_probe.py          # MODIFY: Add adapter_depth logic
```

#### Required Code Changes

**1. Create `Model/one_layer_adapter.py`**

```python
import torch
import torch.nn as nn
from einops import rearrange, repeat

class OneLayerAdapter(nn.Module):
    """
    1-layer cross-attention adapter for minimal-capacity probing.

    Based on Perceiver architecture but with depth=1.
    Follows FAE principle: shallow adapter for honest compression.
    """
    def __init__(self, dim=768, num_latents=32, heads=8, ff_mult=4):
        super().__init__()

        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # Positional embeddings (optional, for compatibility)
        self.frame_embs = None
        self.media_time_embs = None

        # Single cross-attention block
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        dim_head = dim // heads
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # Single FFN block
        hidden_dim = int(dim * ff_mult)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, F, v, D) - vision tokens
        Returns:
            (B, T, num_latents, D) - compressed latents
        """
        B, T, F, v = x.shape[:4]

        # Flatten spatial dimensions
        x = rearrange(x, "b T F v d -> b T (F v) d")

        # Initialize latents
        latents = repeat(self.latents, "n d -> b T n d", b=B, T=T)

        # Normalize
        x_norm = self.norm_media(x)
        latents_norm = self.norm_latents(latents)

        # Cross-attention (SINGLE layer!)
        h = self.heads
        q = self.to_q(latents_norm)
        kv_input = torch.cat((x_norm, latents_norm), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = rearrange(q, "b T n (h d) -> b h T n d", h=h)
        k = rearrange(k, "b T n (h d) -> b h T n d", h=h)
        v = rearrange(v, "b T n (h d) -> b h T n d", h=h)

        q = q * self.scale
        sim = torch.einsum("... i d, ... j d -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h T n d -> b T n (h d)")
        latents = latents + self.to_out(out)

        # FFN (SINGLE layer!)
        latents = latents + self.ff(latents)

        return self.norm(latents)
```

**2. Create `Model/adapter_utils.py`**

```python
import torch

def load_first_layer_from_6layer_checkpoint(adapter_1layer, checkpoint_path):
    """
    Extract first layer weights from 6-layer Perceiver checkpoint.

    Args:
        adapter_1layer: OneLayerAdapter instance
        checkpoint_path: Path to 6-layer checkpoint (.pth file)

    Returns:
        adapter_1layer with loaded weights
    """
    print(f"[INFO] Loading 1st layer from 6-layer checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "perceiver" not in ckpt:
        raise KeyError(f"Checkpoint missing 'perceiver' key")

    old_state = ckpt["perceiver"]
    new_state = {}

    # 1. Copy latents (shared)
    if "latents" in old_state:
        new_state["latents"] = old_state["latents"]
        print(f"   ✓ Copied latents: {old_state['latents'].shape}")

    # 2. Extract Layer 0 attention weights
    # Old: layers.0.0.xxx (ModuleList[PerceiverAttention, FeedForward])
    # New: xxx (direct attributes)

    attention_mapping = {
        "layers.0.0.norm_media": "norm_media",
        "layers.0.0.norm_latents": "norm_latents",
        "layers.0.0.to_q": "to_q",
        "layers.0.0.to_kv": "to_kv",
        "layers.0.0.to_out": "to_out",
    }

    for old_prefix, new_prefix in attention_mapping.items():
        matched_keys = [k for k in old_state.keys() if k.startswith(old_prefix)]
        for old_key in matched_keys:
            new_key = old_key.replace(old_prefix, new_prefix)
            new_state[new_key] = old_state[old_key]
            print(f"   ✓ {old_key} → {new_key}")

    # 3. Extract Layer 0 FFN weights
    # Old: layers.0.1.0.xxx, layers.0.1.1.xxx, ...
    # New: ff.0.xxx, ff.1.xxx, ...

    ffn_prefix = "layers.0.1"
    for old_key in old_state.keys():
        if old_key.startswith(ffn_prefix):
            # layers.0.1.X.yyy → ff.X.yyy
            new_key = old_key.replace(ffn_prefix, "ff")
            new_state[new_key] = old_state[old_key]
            print(f"   ✓ {old_key} → {new_key}")

    # 4. Copy final norm (shared)
    if "norm.weight" in old_state:
        new_state["norm.weight"] = old_state["norm.weight"]
        new_state["norm.bias"] = old_state["norm.bias"]
        print(f"   ✓ Copied final norm")

    # 5. Load into adapter
    missing, unexpected = adapter_1layer.load_state_dict(new_state, strict=False)

    print(f"[INFO] Loaded {len(new_state)} parameters")
    if missing:
        print(f"   ⚠️  Missing keys: {missing[:3]}..." if len(missing) > 3 else f"   ⚠️  Missing keys: {missing}")
    if unexpected:
        print(f"   ⚠️  Unexpected keys: {unexpected[:3]}..." if len(unexpected) > 3 else f"   ⚠️  Unexpected keys: {unexpected}")

    return adapter_1layer
```

**3. Modify `src/lit_recon_probe.py`**

Add to imports:
```python
from Model.one_layer_adapter import OneLayerAdapter
from Model.adapter_utils import load_first_layer_from_6layer_checkpoint
```

Add to ModelArguments:
```python
@dataclass
class ModelArguments:
    # ... existing fields ...

    # Experiment 5 fields
    adapter_depth: int = field(
        default=6,
        metadata={"help": "Adapter depth (1 for minimal-capacity, 6 for baseline)"}
    )

    load_first_layer_from_pretrained: bool = field(
        default=False,
        metadata={"help": "Load 1st layer from 6-layer checkpoint (Exp5a)"}
    )

    random_init_adapter: bool = field(
        default=False,
        metadata={"help": "Randomly initialize adapter, ignore pretrained (Exp5b)"}
    )
```

Modify adapter initialization (around line 665):
```python
# Create adapter based on depth
if model_args.adapter_depth == 1:
    print(f"[INFO] 🔬 Using 1-layer adapter (minimal-capacity probe)")
    self.adapter = OneLayerAdapter(
        dim=vis_dim,
        num_latents=perceiver_num,
        heads=8,
        ff_mult=4
    )
else:
    print(f"[INFO] Using {model_args.adapter_depth}-layer Perceiver")
    self.adapter = PerceiverResampler(
        dim=vis_dim,
        depth=model_args.adapter_depth,
        num_latents=perceiver_num
    )
```

Modify weight loading (around line 1364):
```python
# ===== 9. LOAD PRETRAINED ADAPTER AND PROJECTION =====
if model_args.pretrained_adapter and model_args.use_pretrained_adapter:

    if model_args.random_init_adapter:
        print("[INFO] ❄️  Adapter randomly initialized (no pretrained weights)")

    elif model_args.adapter_depth == 1 and model_args.load_first_layer_from_pretrained:
        # Exp5a: Extract 1st layer from 6-layer checkpoint
        print("[INFO] 🧪 Exp5a: Loading 1st layer from 6-layer checkpoint")
        load_first_layer_from_6layer_checkpoint(
            model.adapter,
            model_args.pretrained_adapter
        )

    elif model_args.adapter_depth == 6:
        # Normal: Load full 6-layer checkpoint
        print("[INFO] Loading full 6-layer adapter checkpoint")
        adapter_ckpt = torch.load(model_args.pretrained_adapter, map_location="cpu")
        if "perceiver" in adapter_ckpt:
            model.adapter.load_state_dict(adapter_ckpt["perceiver"])

    else:
        print(f"[WARN] Unsupported adapter configuration: depth={model_args.adapter_depth}")

    # Load FC (same for all experiments)
    if model_args.decode_mode == "post_proj" and "fc" in adapter_ckpt:
        model.fc.load_state_dict(adapter_ckpt["fc"])
```

---

### Phase 2: Execution Timeline (2 weeks)

#### Week 1: Implementation & Quick Validation

**Day 1-2**: Code implementation
- [ ] Create `Model/one_layer_adapter.py`
- [ ] Create `Model/adapter_utils.py`
- [ ] Modify `src/lit_recon_probe.py`
- [ ] Test weight loading with dummy checkpoint

**Day 3**: Quick validation (Exp5a, 5 epochs)
```bash
# Test if extracted 1st layer works
python src/lit_recon_probe.py \
    --adapter_depth 1 \
    --load_first_layer_from_pretrained True \
    --train_adapter True \
    --num_train_epochs 5 \
    --output_dir outputs/test_exp5a
```

**Day 4**: Quick validation (Exp5b, 5 epochs)
```bash
# Test if random 1-layer works
python src/lit_recon_probe.py \
    --adapter_depth 1 \
    --random_init_adapter True \
    --train_adapter True \
    --num_train_epochs 5 \
    --output_dir outputs/test_exp5b
```

**Day 5-7**: Debug and refine
- Fix any issues discovered
- Verify metrics are logged correctly
- Compare quick results

#### Week 2: Full Training & Analysis

**Day 8-11**: Run full experiments (parallel)
```bash
# Launch both experiments simultaneously (different GPUs)

# Exp5a (GPU 0)
CUDA_VISIBLE_DEVICES=0 python src/lit_recon_probe.py \
    [Exp5a full command] &

# Exp5b (GPU 1)
CUDA_VISIBLE_DEVICES=1 python src/lit_recon_probe.py \
    [Exp5b full command] &

wait
```

**Day 12-13**: Analysis
- [ ] Extract metrics from checkpoints
- [ ] Generate comparison tables
- [ ] Create visualizations (attention maps, per-region cos)
- [ ] Statistical significance tests

**Day 14**: Documentation
- [ ] Update results in this document
- [ ] Write findings summary
- [ ] Decide next steps based on results

---

## 🎯 Decision Tree After Results

```
if Exp5b ROI_cos > Exp2 ROI_cos:
    ✅ "Minimal-capacity principle confirmed!"
    → Write paper on APR (Adapter Probing via Reconstruction)
    → Use Exp5b adapter as evaluation standard
    → Proceed to Stage 2 (VLM with Exp5b adapter)

elif Exp5b ROI_cos ≈ Exp2 ROI_cos:
    ✅ "1-layer sufficient, more efficient"
    → Recommend 1-layer for fast prototyping
    → Optional: Try saliency-guided 1-layer next

else:  # Exp5b << Exp2
    ⚠️  "Compression ratio too aggressive for 1-layer"
    → Test with more latents (64 instead of 32)
    → Or test 2-layer as minimal capacity
    → Still valuable: proves depth is needed
```

---

## 📚 Expected Outputs

### Metrics CSV
```csv
experiment,adapter_config,params,global_cos,roi_cos,small_lesion_cos,roi_global_ratio,train_time
exp1,6layer_frozen,3.5M,0.88,0.70,0.45,0.795,N/A
exp2,6layer_joint,3.5M,0.90,0.78,0.55,0.867,3.2h/epoch
exp5a,1layer_extracted_joint,0.6M,0.85,0.72,0.50,0.847,1.8h/epoch
exp5b,1layer_random_joint,0.6M,0.87,0.82,0.68,0.943,1.5h/epoch
```

### Attention Visualization
- Heatmaps showing where each adapter attends
- Hypothesis: Exp5b should focus more on lesion boundaries

### Per-Region Analysis
- Bar charts comparing cos for each anatomical region
- Hypothesis: Exp5b should have more balanced performance

---

## ✅ Success Metrics Summary

| Metric | Exp5a Target | Exp5b Target | Reasoning |
|--------|--------------|--------------|-----------|
| Global Cos | > 0.83 | > 0.85 | Acceptable trade-off for simplicity |
| ROI Cos | > 0.70 | **> 0.80** ✨ | Core metric for honest compression |
| Small Lesion Cos | > 0.48 | **> 0.65** ✨ | Critical for clinical safety |
| ROI/Global Ratio | > 0.84 | **> 0.92** ✨ | Balance metric (higher = better) |
| Params | 0.6M | 0.6M | 6× reduction vs 6-layer |
| Training Speed | ~2h/epoch | ~1.5h/epoch | Faster convergence expected |

---

**Last Updated**: 2025-12-25
**Next Review**: After Exp5a/5b completion
