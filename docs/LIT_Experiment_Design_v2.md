# LIT Probe Experiment Design: Adapter & Decoder Optimization

**Date**: 2025-12-22
**Author**: Junjie Zhou
**Status**: Planning Phase

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Experiment 1: Decoder Depth Ablation](#experiment-1-decoder-depth-ablation)
3. [Experiment 2: Joint Adapter-Decoder Training](#experiment-2-joint-adapter-decoder-training)
4. [Experiment 3: Simplified Single-Layer Adapter](#experiment-3-simplified-single-layer-adapter)
5. [Experiment 4: Gated Cross-Attention Fusion](#experiment-4-gated-cross-attention-fusion)
6. [Execution Timeline](#execution-timeline)
7. [Technical Implementation](#technical-implementation)

---

## Overview

### Research Questions

This document outlines 4 experiments designed to answer critical questions about the LIT (Linear Information Theory) probe architecture:

1. **Information Preservation**: Does the pretrained Perceiver adapter effectively compress CT features while preserving anatomical information?
2. **Trainable Compression**: Can joint training of adapter + decoder learn better compression representations?
3. **Architecture Complexity**: Is the complex Perceiver Resampler necessary, or can a simple single-layer cross-attention achieve similar performance?
4. **Feature Fusion**: Can gated cross-attention fusion between global and region features outperform naive concatenation?

### Experiment Summary Table

| Exp | Research Question | Key Modification | Training Stages | Expected Outcome |
|-----|------------------|------------------|----------------|------------------|
| **1** | Is pretrained adapter compression effective? | Train decoder only, vary depth | 1 stage | Baseline performance ceiling |
| **2** | Can joint training improve compression? | Unlock adapter + decoder | 2 stages | Learnable vs frozen adapter comparison |
| **3** | Is Perceiver complexity necessary? | Replace with 1-layer cross-attn | 2 stages | Architecture complexity vs performance |
| **4** | Can gated fusion improve performance? | Add gating mechanism for global+region | 1 stage | Fusion strategy impact |

---

## Experiment 1: Decoder Depth Ablation

### 🎯 Objective

**Validate whether the pretrained Perceiver adapter preserves sufficient anatomical information by varying decoder depth.**

If shallow decoders (1-2 layers) perform similarly to deep decoders (6 layers), it indicates the adapter has already preserved most information linearly accessible. Conversely, if deep decoders significantly outperform shallow ones, it suggests the adapter loses information that requires complex non-linear reconstruction.

### ⚙️ Experimental Setup

#### Model Configuration

```python
vision_encoder: Frozen ❄️   # Pretrained ViT3D (frozen)
adapter: Frozen ❄️          # Pretrained Perceiver (frozen)
fc: Frozen ❄️               # Pretrained projection layer (frozen)
decoder: Trainable 🔥       # ProbeDecoder (trainable, varying depth)
```

#### Ablation Groups

| Group | decoder_layers | decoder_heads | decoder_ff_mult | Params | Expected Performance |
|-------|---------------|---------------|----------------|---------|---------------------|
| Exp1-a | 1 | 8 | 4 | ~1M | Low (limited expressiveness) |
| Exp1-b | 2 | 8 | 4 | ~2M | Medium (current default) |
| Exp1-c | 4 | 8 | 4 | ~4M | High |
| Exp1-d | 6 | 8 | 4 | ~6M | Highest (potential overfitting) |

#### Training Command

```bash
# Exp1-a: 1 layer
python src/lit_recon_probe.py \
  --decoder_layers 1 \
  --output_dir /mnt/home/zhoujunjie/outputs/LIT_exp1a \
  --num_train_epochs 15 \
  --use_wandb True \
  --wandb_run_name "exp1a_decoder1layer"

# Exp1-b: 2 layers (baseline)
python src/lit_recon_probe.py \
  --decoder_layers 2 \
  --output_dir /mnt/home/zhoujunjie/outputs/LIT_exp1b \
  --num_train_epochs 15 \
  --use_wandb True \
  --wandb_run_name "exp1b_decoder2layer"

# Exp1-c: 4 layers
python src/lit_recon_probe.py \
  --decoder_layers 4 \
  --output_dir /mnt/home/zhoujunjie/outputs/LIT_exp1c \
  --num_train_epochs 15 \
  --use_wandb True \
  --wandb_run_name "exp1c_decoder4layer"

# Exp1-d: 6 layers
python src/lit_recon_probe.py \
  --decoder_layers 6 \
  --output_dir /mnt/home/zhoujunjie/outputs/LIT_exp1d \
  --num_train_epochs 15 \
  --use_wandb True \
  --wandb_run_name "exp1d_decoder6layer"
```

### 📊 Evaluation Metrics

**Primary Metric**:
- `val/reg_cos`: Region-specific cosine similarity (higher = better)

**Secondary Metrics**:
- `val/cos`: Global cosine similarity
- `val/mse`: Mean squared error
- `val/top1`: Top-1% worst reconstruction error

### ✅ Success Criteria

**Hypothesis 1** (Adapter is effective):
```
If: decoder=2 reg_cos ≈ decoder=6 reg_cos (difference < 0.05)
Then: Adapter preserves sufficient information; decoder only performs linear decoding
Conclusion: Current architecture is well-designed
```

**Hypothesis 2** (Adapter loses information):
```
If: decoder=6 reg_cos >> decoder=2 reg_cos (difference > 0.15)
Then: Adapter compresses too aggressively; deep decoder compensates by "guessing"
Conclusion: Need to improve adapter (→ proceed to Exp2)
```

---

## Experiment 2: Joint Adapter-Decoder Training

### 🎯 Objective

**Investigate whether unlocking the adapter for joint training with the decoder can learn a better compression representation that preserves more anatomical information.**

This experiment tests whether the adapter, when trained specifically for reconstruction tasks, can learn to prioritize preserving region-specific features over generic visual features.

### ⚙️ Experimental Setup (2-Stage Training)

#### Stage 1: LIT Probe Training (Adapter + Decoder Joint Training)

```python
# Model Configuration
vision_encoder: Frozen ❄️      # Pretrained ViT3D (frozen)
adapter: Trainable 🔥          # Perceiver (UNLOCKED!)
fc: Frozen ❄️                  # Projection layer (frozen)
decoder: Trainable 🔥          # ProbeDecoder (trainable)

# Training Objective
loss = global_recon_loss + λ_region * region_recon_loss

# Dataset: RadGenome CT + region masks
# Epochs: 15-20
# Learning Rate: 1e-4 (both adapter and decoder)
```

**Goal**: Train adapter to optimize for reconstruction quality, learning a compression that preserves decodable information.

#### Training Command (Stage 1)

```bash
python src/lit_recon_probe.py \
  --pretrained_adapter /mnt/home/zhoujunjie/models/Reg2RG/RadFM_perceiver_fc.pth \
  --use_pretrained_adapter True \
  --train_adapter True \  # New flag: unlock adapter
  --decoder_layers 2 \
  --output_dir /mnt/home/zhoujunjie/outputs/LIT_exp2_stage1 \
  --num_train_epochs 20 \
  --learning_rate 1e-4 \
  --use_wandb True \
  --wandb_run_name "exp2_joint_training_stage1"
```

---

#### Stage 2: VLM Training (Remove Decoder, Add LLM)

```python
# Model Configuration
vision_encoder: Frozen ❄️      # ViT3D (frozen)
adapter: Frozen ❄️             # Use Stage1-trained adapter (FROZEN!)
fc: Trainable 🔥               # Projection layer (trainable)
LLM: LoRA Fine-tune 🔥         # Llama-2-7B (LoRA)
decoder: Removed ❌            # Delete decoder

# Training Objective
loss = CrossEntropy(LLM_output, region_report_text)

# Dataset: RadGenome CT + region reports (text)
# Epochs: 5-10
```

**Goal**: Validate whether the improved adapter from Stage 1 leads to better VLM performance.

#### Training Command (Stage 2)

```bash
# Use the original VLM training script, but load Stage1 adapter
python src/train_vlm.py \
  --pretrained_adapter /mnt/home/zhoujunjie/outputs/LIT_exp2_stage1/checkpoints/best_adapter.pt \
  --use_lora True \
  --lora_rank 8 \
  --num_train_epochs 10 \
  --output_dir /mnt/home/zhoujunjie/outputs/VLM_exp2_stage2 \
  --wandb_run_name "exp2_vlm_stage2"
```

### 📊 Evaluation & Comparison

#### Stage 1 Comparison

| Method | Adapter Training | val/reg_cos | val/cos | Notes |
|--------|-----------------|-------------|---------|-------|
| **Exp1 (Baseline)** | Pretrained + Frozen | 0.85 | 0.88 | Current approach |
| **Exp2 (Joint)** | Joint Training | **0.90+** ✨ | 0.90+ | Expected improvement |

#### Stage 2 Comparison

| Method | Adapter Source | BLEU-4 | METEOR | CIDEr | Notes |
|--------|---------------|--------|--------|-------|-------|
| **Original VLM** | Pretrained adapter | X | X | X | Baseline |
| **Exp2 VLM** | Stage1-trained adapter | **+5%?** | **+3%?** | **+10%?** | Expected improvement |

### ✅ Success Criteria

**Strong Success**:
```
Stage 1: Exp2 reg_cos > Exp1 reg_cos (e.g., 0.90 vs 0.85)
Stage 2: Exp2 VLM metrics > Original VLM metrics
→ Conclusion: Joint training improves both reconstruction and downstream VLM
```

**Partial Success**:
```
Stage 1: Exp2 reg_cos >> Exp1 reg_cos (much better)
Stage 2: Exp2 VLM ≈ Original VLM (similar)
→ Conclusion: Better reconstruction ≠ better VLM (overfitting to visual features?)
```

**Failure**:
```
Stage 1: Exp2 reg_cos ≈ Exp1 reg_cos (no improvement)
→ Conclusion: Pretrained adapter is already optimal; joint training doesn't help
```

---

## Experiment 3: Simplified Single-Layer Adapter

### 🎯 Objective

**Test whether the complex Perceiver Resampler (6-layer iterative cross-attention + self-attention) is necessary, or if a simple single-layer cross-attention can achieve comparable performance.**

This experiments the trade-off between architectural complexity and performance, following Occam's Razor: prefer simpler models if performance is similar.

### ⚙️ Architecture Comparison

#### Current: Perceiver Resampler (Complex)

```python
class PerceiverResampler(nn.Module):
    """
    6-layer iterative cross-attention + self-attention
    Latents gradually extract information from input tokens
    """
    def __init__(self, dim=768, num_latents=32, depth=6, heads=8):
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim))
        self.layers = nn.ModuleList([
            PerceiverBlock(dim, heads)  # Cross-attn + Self-attn
            for _ in range(depth)
        ])

    def forward(self, x):  # x: (B, N, 768) where N=1024 tokens
        latents = self.latents.expand(B, -1, -1)
        for layer in self.layers:
            latents = layer(latents, x)  # 6 iterations
        return latents  # (B, 32, 768)

# Parameters: ~3M
# Computation: O(6 × 32 × 1024) = 6 iterations
```

---

#### Proposed: SimpleCrossAttentionAdapter (Simple)

```python
class SimpleCrossAttentionAdapter(nn.Module):
    """
    Single-layer cross-attention
    Queries directly attend to input tokens in one pass
    """
    def __init__(self, dim=768, num_queries=32, heads=8):
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):  # x: (B, N, 768) where N=1024 tokens
        q = self.queries.expand(B, -1, -1)
        out, _ = self.cross_attn(q, x, x)  # Q=queries, K=V=x
        return self.norm(out)  # (B, 32, 768)

# Parameters: ~0.5M (6× fewer!)
# Computation: O(1 × 32 × 1024) = single pass (6× faster!)
```

### ⚙️ Experimental Setup (2-Stage, Same as Exp2)

#### Stage 1: LIT Probe

```python
vision_encoder: Frozen ❄️
adapter: SimpleCrossAttentionAdapter 🔥  # Replace with simple adapter
decoder: Trainable 🔥

loss = global_recon_loss + λ_region * region_recon_loss
```

#### Stage 2: VLM Training

```python
vision_encoder: Frozen ❄️
adapter: Frozen ❄️  # Use Stage1-trained simple adapter
LLM: LoRA Fine-tune 🔥
```

### 📊 Comparison Table

| Method | Adapter | Params | Training Speed | val/reg_cos | VLM BLEU-4 |
|--------|---------|--------|---------------|-------------|-----------|
| **Exp2** | Perceiver (6-layer) | 3M | 1× (baseline) | 0.90 | X |
| **Exp3** | 1-layer Cross-Attn | 0.5M | **6× faster** ⚡ | 0.87? | X-2%? |

### ✅ Success Criteria

**Strong Success** (Simple is sufficient):
```
If: Exp3 reg_cos ≈ Exp2 reg_cos (difference < 0.03)
    AND Exp3 VLM ≈ Exp2 VLM
    AND Exp3 trains 6× faster
→ Conclusion: Simple architecture is preferable (Occam's Razor)
→ Action: Replace Perceiver with SimpleCrossAttentionAdapter
```

**Partial Success** (Trade-off exists):
```
If: Exp3 reg_cos = 0.87 vs Exp2 reg_cos = 0.90 (3% worse)
    BUT Exp3 trains 6× faster with 6× fewer params
→ Conclusion: Trade-off between efficiency and performance
→ Action: Use simple adapter for fast experiments, Perceiver for final model
```

**Failure** (Complex is necessary):
```
If: Exp3 reg_cos << Exp2 reg_cos (e.g., 0.75 vs 0.90)
→ Conclusion: Iterative refinement in Perceiver is essential
→ Action: Keep Perceiver architecture
```

---

## Experiment 4: Gated Cross-Attention Fusion

### 🎯 Objective

**Investigate whether dynamically gating the fusion of global and region features via learned attention mechanisms outperforms naive concatenation or independent processing.**

Current approach processes global and region features independently. This experiment proposes a gating mechanism to let the model dynamically decide how much to weight global context vs region-specific information.

### ⚙️ Current Approach (Independent Processing)

```python
# Current: Process global and regions separately
global_features = encode(CT_image)         # (B, 1024, 768)
region_features = [encode(region_i) for region_i in regions]

# Compute losses independently
loss_global = recon_loss(decode(compress(global_features)), global_features)
loss_regions = mean([recon_loss(decode(compress(r)), r) for r in region_features])

total_loss = loss_global + λ_region * loss_regions
```

**Limitation**:
- Global and region information are processed in isolation
- No information exchange between global context and region-specific features
- Equal weighting for all regions (some may be less informative)

---

### ⚙️ Proposed Approach (Gated Cross-Attention Fusion)

```python
class GatedFusion(nn.Module):
    """
    Gated cross-attention fusion module
    Dynamically fuses global and region features with learned gating
    """
    def __init__(self, dim=768, heads=8):
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()  # Output ∈ [0, 1]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, global_feat, region_feat):
        """
        Args:
            global_feat: (B, N, 768) - global CT features
            region_feat: (B, N, 768) - region-specific features

        Returns:
            fused_feat: (B, N, 768) - gated fusion
        """
        # 1. Cross-attention: region attends to global
        region_enhanced, attn_weights = self.cross_attn(
            query=region_feat,
            key=global_feat,
            value=global_feat
        )  # (B, N, 768)

        # 2. Compute gating weights
        concat = torch.cat([global_feat, region_enhanced], dim=-1)  # (B, N, 1536)
        gate = self.gate_net(concat)  # (B, N, 768) ∈ [0, 1]

        # 3. Gated fusion
        fused = gate * global_feat + (1 - gate) * region_enhanced

        return self.norm(fused), attn_weights, gate

# Usage in training loop
fusion_module = GatedFusion(dim=768)

global_feat = encode(CT_image)
region_feats = [encode(region_i) for region_i in regions]

fused_feats = []
for region_feat in region_feats:
    fused, attn, gate = fusion_module(global_feat, region_feat)
    fused_feats.append(fused)
    # Can visualize gate values to understand which regions rely more on global context

# Reconstruction loss on fused features
loss_regions = mean([recon_loss(decode(compress(f)), target) for f in fused_feats])
```

**Advantages**:
- ✅ **Dynamic weighting**: Different samples/regions adaptively balance global vs region info
- ✅ **Information exchange**: Regions can extract relevant information from global context
- ✅ **Interpretability**: Gate values can be visualized (which regions depend more on global?)

---

### ⚙️ Experimental Setup

```python
# Model Configuration
vision_encoder: Frozen ❄️
adapter: Frozen ❄️
fc: Frozen ❄️
decoder: Trainable 🔥
fusion_module: GatedFusion 🔥  # NEW!

# Training
for batch in train_loader:
    global_feat = vision_encoder(batch["image"])
    region_feats = [vision_encoder(batch[region]) for region in REGIONS]

    # Apply gated fusion
    fused_feats = [fusion_module(global_feat, r)[0] for r in region_feats]

    # Compress and reconstruct
    compressed = [adapter(f) for f in fused_feats]
    reconstructed = [decoder(c, grid) for c in compressed]

    # Loss
    loss = sum([recon_loss(recon, target) for recon, target in ...])
```

### 📊 Comparison

| Method | Fusion Strategy | Params | val/reg_cos | Interpretability |
|--------|----------------|--------|-------------|-----------------|
| **Exp1 (Baseline)** | Independent | - | 0.85 | Low |
| **Concat Fusion** | Simple concat | +0.5M | 0.86? | Low |
| **Exp4 (Gated)** | Gated cross-attn | +1M | **0.88+** ✨ | High (gate vis) |

### 📊 Visualization

After training, analyze gate values:

```python
# Analyze which regions rely more on global context
gate_stats = {}
for region_name in REGIONS:
    gate_values = []  # Collect gate values for this region across val set
    # ...
    gate_stats[region_name] = {
        "mean": np.mean(gate_values),
        "std": np.std(gate_values)
    }

# Hypothesis:
# - Small regions (thyroid, esophagus): high gate → rely more on global context
# - Large regions (lung, heart): low gate → rely more on region-specific features
```

### ✅ Success Criteria

**Strong Success**:
```
If: Exp4 reg_cos > Exp1 reg_cos (e.g., 0.88 vs 0.85)
    AND gate visualization shows meaningful patterns
    (e.g., small organs have higher gate values)
→ Conclusion: Gated fusion is beneficial and interpretable
```

**Partial Success**:
```
If: Exp4 reg_cos ≈ Exp1 reg_cos (similar performance)
    BUT gate visualization shows interpretable patterns
→ Conclusion: Fusion doesn't improve performance, but provides insights
```

**Failure**:
```
If: Exp4 reg_cos ≈ Exp1 reg_cos
    AND gate values are random/uninformative
→ Conclusion: Added complexity not justified; keep simple approach
```

---

## Execution Timeline

### Phase 1: Foundation (Weeks 1-4) - **CRITICAL**

**Week 1-2: Exp1 (Decoder Ablation)**
- [ ] Run 4 ablation groups (decoder_layers = 1, 2, 4, 6)
- [ ] Analyze results and establish baseline
- [ ] **Deliverable**: Baseline performance metrics + decision on adapter quality

**Week 3-4: Exp2 Stage 1 (Joint Training)**
- [ ] Implement adapter unlocking flag
- [ ] Train adapter + decoder jointly
- [ ] Compare with Exp1 baseline
- [ ] **Deliverable**: Adapter checkpoint + reconstruction metrics

**Decision Point** (End of Week 4):
```
If Exp2 Stage 1 shows significant improvement (reg_cos +0.05):
  → Proceed to Exp2 Stage 2 (VLM training)
  → Proceed to Exp3 (architecture simplification)
Else:
  → Analyze failure (why didn't joint training help?)
  → Re-evaluate approach before continuing
```

---

### Phase 2: Architecture Optimization (Weeks 5-6) - **CONDITIONAL**

**Week 5-6: Exp3 (Simple Adapter)**
- [ ] Implement SimpleCrossAttentionAdapter
- [ ] Run 2-stage training (same as Exp2)
- [ ] Compare efficiency vs performance trade-off
- [ ] **Deliverable**: Efficiency analysis + architecture recommendation

**Trigger Condition**:
- Only execute if Exp2 showed improvement (otherwise, skip to analysis)

---

### Phase 3: Advanced Fusion (Weeks 7-8) - **OPTIONAL**

**Week 7-8: Exp4 (Gated Fusion)**
- [ ] Implement GatedFusion module
- [ ] Train and evaluate
- [ ] Visualize gate values for interpretability
- [ ] **Deliverable**: Fusion strategy analysis + gate visualization

**Trigger Condition**:
- Only execute if time permits and Exp1-3 completed successfully
- This is a "bonus" experiment, not critical path

---

### Phase 4: VLM Validation (Weeks 5-9) - **PARALLEL**

**Exp2 Stage 2: VLM Training**
- [ ] Train VLM with Exp2 adapter
- [ ] Train VLM with Exp3 adapter (if applicable)
- [ ] Compare with baseline VLM
- [ ] **Deliverable**: End-to-end performance validation

---

## Technical Implementation

### Required Code Changes

#### 1. Exp1: Minimal Changes (Parameter Only)

**File**: `src/lit_recon_probe.py`

```python
# Already implemented - just run with different --decoder_layers
```

**Action**: None (just run experiments with different CLI arguments)

---

#### 2. Exp2: Add Adapter Training Flag

**File**: `src/lit_recon_probe.py`

**Changes**:

```python
@dataclass
class TrainArguments:
    # ... existing fields ...

    # New field
    train_adapter: bool = field(
        default=False,
        metadata={"help": "Whether to unlock adapter for training (Exp2)."}
    )

# In main():
# Modify line 1353-1356
set_requires_grad(model.vision_encoder, False)  # Keep frozen
set_requires_grad(model.adapter, train_args.train_adapter)  # Unlock if flag=True
set_requires_grad(model.fc, False)  # Keep frozen
set_requires_grad(model.decoder, True)  # Always train

# Modify optimizer (line 1358-1365)
if train_args.train_adapter:
    # Optimize both adapter and decoder
    trainable_params = list(model.adapter.parameters()) + list(model.decoder.parameters())
else:
    # Only optimize decoder
    trainable_params = model.decoder.parameters()

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=train_args.learning_rate,
    weight_decay=train_args.weight_decay
)
```

---

#### 3. Exp3: Implement SimpleCrossAttentionAdapter

**New File**: `Model/simple_adapter.py`

```python
import torch
import torch.nn as nn

class SimpleCrossAttentionAdapter(nn.Module):
    """
    Single-layer cross-attention adapter as a simpler alternative to Perceiver.

    Args:
        dim: Token dimension (768)
        num_queries: Number of output queries (32)
        heads: Number of attention heads (8)
    """
    def __init__(self, dim: int = 768, num_queries: int = 32, heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tokens from ViT, shape (B, num_media, num_frames, N, dim)

        Returns:
            Compressed representation, shape (B, num_media, num_queries, dim)
        """
        B, num_media, num_frames, N, dim = x.shape

        # Reshape to (B*num_media*num_frames, N, dim)
        x = x.reshape(B * num_media * num_frames, N, dim)

        # Expand queries
        q = self.queries.expand(x.shape[0], -1, -1)

        # Cross-attention
        attn_out, _ = self.cross_attn(q, x, x)
        q = self.norm1(q + attn_out)

        # FFN
        ffn_out = self.ffn(q)
        q = self.norm2(q + ffn_out)

        # Reshape back
        num_queries = q.shape[1]
        q = q.reshape(B, num_media, num_frames, num_queries, dim)

        return q
```

**Modify**: `src/lit_recon_probe.py`

```python
# Add import
from Model.simple_adapter import SimpleCrossAttentionAdapter

@dataclass
class ModelArguments:
    # ... existing fields ...

    # New field
    use_simple_adapter: bool = field(
        default=False,
        metadata={"help": "Use SimpleCrossAttentionAdapter instead of Perceiver (Exp3)."}
    )

# In LITProbeModel.__init__() (line 665-668)
if decode_mode == "pre_proj":
    if model_args.use_simple_adapter:
        self.adapter = SimpleCrossAttentionAdapter(
            dim=vis_dim,
            num_queries=perceiver_num,
            heads=8
        )
    else:
        self.adapter = PerceiverResampler(
            dim=vis_dim,
            num_latents=perceiver_num
        )
```

---

#### 4. Exp4: Implement GatedFusion

**New File**: `Model/gated_fusion.py`

```python
import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    """
    Gated cross-attention fusion for global and region features.

    Dynamically weights the contribution of global context vs region-specific
    information using learned gating.
    """
    def __init__(self, dim: int = 768, heads: int = 8):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        global_feat: torch.Tensor,
        region_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            global_feat: (B, N, dim) - global CT features
            region_feat: (B, N, dim) - region-specific features

        Returns:
            fused_feat: (B, N, dim) - gated fusion result
            attn_weights: (B, N, N) - cross-attention weights
            gate: (B, N, dim) - gating values ∈ [0, 1]
        """
        # Cross-attention: region attends to global
        region_enhanced, attn_weights = self.cross_attn(
            query=region_feat,
            key=global_feat,
            value=global_feat,
            need_weights=True,
            average_attn_weights=False
        )

        # Compute gating weights
        concat = torch.cat([global_feat, region_enhanced], dim=-1)
        gate = self.gate_net(concat)  # (B, N, dim) ∈ [0, 1]

        # Gated fusion
        fused = gate * global_feat + (1 - gate) * region_enhanced

        return self.norm(fused), attn_weights, gate
```

**Modify**: `src/lit_recon_probe.py` (extensive changes needed)

---

### Checkpoint Management

For all experiments, save:

```python
# Best checkpoint (by val/reg_cos)
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "train_metrics": train_metrics,
    "val_metrics": val_metrics,
    "config": {
        "model_args": vars(model_args),
        "data_args": vars(data_args),
        "train_args": vars(train_args),
    },
    "experiment_id": "exp1a",  # Track which experiment
}
```

---

## Appendix: Expected Results

### Hypothetical Results Table

| Experiment | Setting | val/reg_cos | val/cos | Training Time | Notes |
|-----------|---------|-------------|---------|--------------|-------|
| Exp1-a | decoder=1 | 0.78 | 0.82 | 2h/epoch | Too shallow |
| Exp1-b | decoder=2 | 0.85 | 0.88 | 2.5h/epoch | Current baseline |
| Exp1-c | decoder=4 | 0.87 | 0.89 | 3h/epoch | Marginal improvement |
| Exp1-d | decoder=6 | 0.87 | 0.89 | 3.5h/epoch | No further gain |
| Exp2 Stage1 | joint training | **0.91** | **0.92** | 3h/epoch | Adapter improves! |
| Exp3 Stage1 | simple adapter | 0.88 | 0.90 | **0.5h/epoch** | Good trade-off |
| Exp4 | gated fusion | 0.89 | 0.91 | 3.5h/epoch | Fusion helps |

---

## References

- Original LIT paper: [Linear Information Theory Probes]
- Perceiver architecture: [Perceiver: General Perception with Iterative Attention]
- RadGenome dataset: [Region-specific CT report generation]

---

**Last Updated**: 2025-12-22
**Next Review**: After Exp1 completion (Week 2)
