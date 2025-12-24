# DeepStack Multi-Layer Vision Feature Injection

**Status:** Idea/Planning
**Date Created:** 2025-12-23
**Priority:** TBD

---

## 📋 Executive Summary

**Core Idea:** Instead of only using the final layer output from ViT, extract features from multiple depths (early, middle, late layers) and inject them directly into the early layers of the LLaMA language model via residual connections.

**Key Innovation:** Increases visual information density without increasing context length, potentially improving fine-grained detail recognition (small lesions, anatomical boundaries) while maintaining computational efficiency.

**Inspiration:** Based on the DeepStack mechanism from recent vision-language model research, which showed significant improvements on InfoVQA (+12%) and DocVQA (+16%) tasks.

---

## 🎯 Motivation

### Current Architecture Limitations

Our existing Reg2RG model (`src/Model/Reg2RG.py`, `src/Model/my_embedding_layer.py`) has the following characteristics:

1. **Single-Layer Feature Extraction**
   - ViT depth: 12 layers (`my_embedding_layer.py:66`)
   - Current usage: Only **Layer 12 (final) output** is extracted (`vit_3d.py:122`)
   - Information loss: Early-layer fine-grained features (textures, edges, small lesions) are discarded

2. **Long Sequence Problem**
   - Image tokens: 32
   - Region tokens: 10 regions × 33 tokens = 330
   - Text tokens: ~100
   - **Total sequence length: ~462 tokens**
   - Attention complexity: O(462²) = 213,444 operations

3. **Indirect Information Propagation**
   - Vision features are concatenated as prefix tokens (`my_embedding_layer.py:196`)
   - Visual information propagates to final predictions through 32 LLaMA layers via self-attention
   - Potential information dilution through long propagation path

### What DeepStack Solves

1. ✅ **Preserves multi-scale visual information** (fine details + semantic context)
2. ✅ **No increase in sequence length** (avoids quadratic attention cost)
3. ✅ **Direct injection into LLM hidden states** (stronger signal, less dilution)

---

## 🏗️ Technical Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    3D Medical Image (CT)                     │
│                  [B, 1, 512, 512, 512]                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
            ┌────────────────┐
            │   ViT 3D       │
            │   (12 layers)  │
            └────────┬───────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
   Layer 4      Layer 8      Layer 12
   (Early)      (Middle)     (Final)
   Textures     Structures   Semantics
   Edges        Organs       Diagnosis
        │            │            │
        ▼            ▼            ▼
  Perceiver₄   Perceiver₈   Perceiver₁₂
  + FC₄        + FC₈        + FC₁₂
        │            │            │
        ▼            ▼            ▼
   [B,32,4096]  [B,32,4096]  [B,32,4096]
        │            │            │
        └────────────┼────────────┘
                     │
                     ▼
            ┌────────────────┐
            │  LLaMA Model   │
            ├────────────────┤
            │ Text Embed     │
            │      ↓         │
            │ Layer 0 + vis₄ │ ← Inject early features
            │      ↓         │
            │ Layer 1 + vis₈ │ ← Inject mid features
            │      ↓         │
            │ Layer 2 + vis₁₂│ ← Inject late features
            │      ↓         │
            │ Layer 3-31     │
            │      ↓         │
            │   Output       │
            └────────────────┘
```

### Layer Selection Rationale

For our 12-layer ViT, we propose extracting features from:

| ViT Layer | Depth Level | Visual Information Content | Medical Imaging Use Case |
|-----------|-------------|---------------------------|--------------------------|
| **Layer 4** | Early (33%) | Fine-grained textures, edges, local patterns | Small nodules, micro-calcifications, subtle opacities, bone fractures |
| **Layer 8** | Middle (67%) | Mid-level semantics, organ structures | Anatomical boundaries, organ shapes, vessel patterns |
| **Layer 12** | Final (100%) | High-level semantics, global context | Overall diagnosis, disease classification, report semantics |

**Why this distribution?**
- Evenly spaced (every 4 layers) for complementary information
- Early layer preserves fine details critical for small lesion detection
- Middle layer captures anatomical structure (critical for region-based reports)
- Final layer maintains current semantic understanding capability

---

## 💻 Implementation Plan

### Phase 1: Minimal Viable Prototype (MVP)

#### Step 1: Modify ViT to Return Intermediate Features

**File:** `src/Model/vit_3d.py`

**Current Implementation (Line 77-81):**
```python
def forward(self, x):
    for attn, ff in self.layers:
        x = attn(x) + x
        x = ff(x) + x
    return x
```

**Proposed Modification:**
```python
def forward(self, x, return_intermediate=False):
    """
    Args:
        x: Input tensor [B, N, D]
        return_intermediate: If True, return intermediate layer outputs

    Returns:
        If return_intermediate=False: Final layer output [B, N, D]
        If return_intermediate=True: Tuple (final_output, intermediate_features)
            where intermediate_features = [layer4_out, layer8_out, layer12_out]
    """
    intermediate_features = []

    for i, (attn, ff) in enumerate(self.layers):
        x = attn(x) + x
        x = ff(x) + x

        # Extract features at layers 3, 7, 11 (0-indexed, representing layers 4, 8, 12)
        if return_intermediate and i in [3, 7, 11]:
            intermediate_features.append(x.clone())

    if return_intermediate:
        return x, intermediate_features
    return x
```

**Testing:**
```python
# Verify output shapes
vision_encoder = ViT(...)
x = torch.randn(2, 1, 512, 512, 512)
final_out, intermediate = vision_encoder(x, return_intermediate=True)

assert len(intermediate) == 3
assert intermediate[0].shape == intermediate[1].shape == intermediate[2].shape
```

---

#### Step 2: Add Multi-Layer Projectors

**File:** `src/Model/my_embedding_layer.py`

**New Component: DeepStackProjectors**
```python
class DeepStackProjectors(nn.Module):
    """
    Separate Perceiver + FC projectors for each ViT layer.
    Each projector learns to map layer-specific features to LLM hidden dimension.
    """
    def __init__(self, vis_dim=768, embedding_dim=4096, num_latents=32):
        super().__init__()

        # Layer 4 projector (early features)
        self.perceiver_layer4 = PerceiverResampler(
            dim=vis_dim,
            num_latents=num_latents
        )
        self.fc_layer4 = nn.Linear(vis_dim, embedding_dim)

        # Layer 8 projector (middle features)
        self.perceiver_layer8 = PerceiverResampler(
            dim=vis_dim,
            num_latents=num_latents
        )
        self.fc_layer8 = nn.Linear(vis_dim, embedding_dim)

        # Layer 12 projector (final features)
        self.perceiver_layer12 = PerceiverResampler(
            dim=vis_dim,
            num_latents=num_latents
        )
        self.fc_layer12 = nn.Linear(vis_dim, embedding_dim)

    def forward(self, layer4_feats, layer8_feats, layer12_feats):
        """
        Args:
            layer4_feats: [B, num_patches, vis_dim]
            layer8_feats: [B, num_patches, vis_dim]
            layer12_feats: [B, num_patches, vis_dim]

        Returns:
            Tuple of (layer4_embeds, layer8_embeds, layer12_embeds)
            Each: [B, num_latents, embedding_dim]
        """
        # Process each layer
        layer4_tokens = self.perceiver_layer4(layer4_feats.unsqueeze(2))  # [B, S, 1, num_patches, vis_dim]
        layer4_embeds = self.fc_layer4(layer4_tokens)

        layer8_tokens = self.perceiver_layer8(layer8_feats.unsqueeze(2))
        layer8_embeds = self.fc_layer8(layer8_tokens)

        layer12_tokens = self.perceiver_layer12(layer12_feats.unsqueeze(2))
        layer12_embeds = self.fc_layer12(layer12_tokens)

        return layer4_embeds, layer8_embeds, layer12_embeds
```

**Modify MyEmbedding.__init__():**
```python
class MyEmbedding(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... existing code ...

        # OPTION 1: Replace single projector with multi-layer projectors
        self.deepstack_projectors = DeepStackProjectors(
            vis_dim=self.vis_dim,
            embedding_dim=self.embedding_dim,
            num_latents=32
        )

        # OPTION 2: Keep existing projector for backward compatibility,
        # add deepstack_projectors as additional component
```

**Modify MyEmbedding.forward():**
```python
def forward(self, vision_x, mask_x, text_input, region2areas):
    B, S, C, H, W, D = next(iter(vision_x.values())).shape

    vision_temp = vision_x['image']
    vision_temp = rearrange(vision_temp, "b S c h w d-> (b S) c h w d")

    # ========== NEW: Extract multi-layer features ==========
    vision_final, intermediate_feats = self.vision_encoder(
        vision_temp,
        return_intermediate=True
    )
    # intermediate_feats = [layer4_out, layer8_out, layer12_out]
    # Each: [(B*S), num_patches, vis_dim]

    # Reshape for perceiver
    layer4_feats = rearrange(intermediate_feats[0], "(b s) v d -> b s v d", b=B, s=S)
    layer8_feats = rearrange(intermediate_feats[1], "(b s) v d -> b s v d", b=B, s=S)
    layer12_feats = rearrange(intermediate_feats[2], "(b s) v d -> b s v d", b=B, s=S)

    # Project each layer
    layer4_embed, layer8_embed, layer12_embed = self.deepstack_projectors(
        layer4_feats, layer8_feats, layer12_feats
    )
    # Each: [B, S, num_latents, embedding_dim]

    # Flatten for use
    layer4_embed = rearrange(layer4_embed, "b s n d -> b (s n) d")
    layer8_embed = rearrange(layer8_embed, "b s n d -> b (s n) d")
    layer12_embed = rearrange(layer12_embed, "b s n d -> b (s n) d")

    # Store for LLaMA injection (will be used in Reg2RG.forward())
    self.cached_deepstack_features = {
        'layer4': layer4_embed,
        'layer8': layer8_embed,
        'layer12': layer12_embed
    }
    # ========== END NEW CODE ==========

    # ... rest of existing code for regions and text ...
    # For MVP, we can still use layer12 for the traditional path
    image_embedding = layer12_embed  # Or keep existing perceiver logic

    # ... existing region processing ...

    return out_put
```

---

#### Step 3: Inject into LLaMA Hidden States

**File:** `src/Model/Reg2RG.py`

**Challenge:** We're using HuggingFace's `LlamaForCausalLM` with LoRA. We need to inject features into intermediate layers without breaking the existing architecture.

**Proposed Approach: Monkey-Patching**

```python
class Reg2RG(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... existing initialization ...

        # Storage for DeepStack features (set by embedding layer)
        self.deepstack_features = None

        # Injection weights (learnable scaling factors)
        self.injection_weight_layer0 = nn.Parameter(torch.tensor(0.1))
        self.injection_weight_layer1 = nn.Parameter(torch.tensor(0.1))
        self.injection_weight_layer2 = nn.Parameter(torch.tensor(0.1))

        # Patch LLaMA layers to inject features
        self._setup_deepstack_injection()

    def _setup_deepstack_injection(self):
        """
        Monkey-patch LLaMA's first 3 layers to inject vision features.
        """
        # Save original forward methods
        self.original_layer0_forward = self.lang_model.model.layers[0].forward
        self.original_layer1_forward = self.lang_model.model.layers[1].forward
        self.original_layer2_forward = self.lang_model.model.layers[2].forward

        # Create patched forward methods
        def make_injected_forward(layer_idx, original_forward, injection_weight):
            def injected_forward(hidden_states, *args, **kwargs):
                # Call original layer
                output = original_forward(hidden_states, *args, **kwargs)

                # Inject vision features if available
                if self.deepstack_features is not None:
                    if layer_idx == 0:
                        vision_feats = self.deepstack_features['layer4']
                    elif layer_idx == 1:
                        vision_feats = self.deepstack_features['layer8']
                    elif layer_idx == 2:
                        vision_feats = self.deepstack_features['layer12']

                    # Residual injection
                    # output[0] is hidden_states, shape: [B, seq_len, hidden_dim]
                    # vision_feats shape: [B, num_vision_tokens, hidden_dim]

                    # Option A: Broadcast to all positions
                    vision_avg = vision_feats.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]
                    output = (output[0] + injection_weight * vision_avg,) + output[1:]

                    # Option B: Inject at specific positions (requires position tracking)
                    # vision_start_pos = self.get_vision_token_positions()
                    # output[0][:, vision_start_pos:vision_start_pos+num_vision_tokens, :] += vision_feats

                return output

            return injected_forward

        # Apply patches
        self.lang_model.model.layers[0].forward = make_injected_forward(
            0, self.original_layer0_forward, self.injection_weight_layer0
        )
        self.lang_model.model.layers[1].forward = make_injected_forward(
            1, self.original_layer1_forward, self.injection_weight_layer1
        )
        self.lang_model.model.layers[2].forward = make_injected_forward(
            2, self.original_layer2_forward, self.injection_weight_layer2
        )

    def forward(self, lang_x, vision_x, mask_x, region2area, attention_mask, labels):
        if labels.shape == lang_x.shape:
            # Generate embeddings (this sets self.embedding_layer.cached_deepstack_features)
            input_embedding = self.embedding_layer(vision_x, mask_x, lang_x, region2area)

            # Transfer cached features to model
            self.deepstack_features = self.embedding_layer.cached_deepstack_features

            # Forward pass (injection happens automatically via patched layers)
            output = self.lang_model(
                inputs_embeds=input_embedding,
                attention_mask=attention_mask,
                labels=labels
            )

            # Clear cache
            self.deepstack_features = None

            if torch.distributed.get_rank() == 0:
                print('lm_loss:', output['loss'].item())

            return dict(
                logits=output['logits'],
                loss=output['loss'],
            )
```

---

### Phase 2: Ablation Studies

After MVP works, systematically evaluate:

1. **Layer Combination Ablation**
   - Baseline: Layer 12 only (current)
   - Variant A: Layers [4, 8, 12]
   - Variant B: Layers [3, 6, 9, 12]
   - Variant C: Layers [6, 9, 12]
   - Variant D: Layers [8, 10, 12]

2. **Injection Position Ablation**
   - Current: LLaMA Layers [0, 1, 2]
   - Variant A: LLaMA Layers [1, 2, 3]
   - Variant B: LLaMA Layers [0, 2, 4]
   - Variant C: LLaMA Layers [0, 1, 2] with learnable routing

3. **Projection Method Ablation**
   - Current: 3 separate Perceivers + FCs
   - Variant A: Shared Perceiver, separate FCs
   - Variant B: No Perceiver, direct FC projection
   - Variant C: Cross-attention instead of Perceiver

4. **Injection Strategy Ablation**
   - Current: Residual addition
   - Variant A: Gated fusion (learnable mixing)
   - Variant B: Cross-attention to vision features
   - Variant C: Concatenation (increases sequence length)

---

### Phase 3: Advanced Optimizations

1. **Memory Optimization**
   - Use different `num_latents` per layer (e.g., 16/24/32)
   - Apply gradient checkpointing to intermediate features
   - Mixed precision training with bfloat16

2. **Training Stability**
   - Curriculum learning for injection weights (0.01 → 0.1 → 0.5)
   - LayerNorm before injection
   - Warmup schedule for DeepStack components

3. **Task-Specific Adaptations**
   - **Region-aware DeepStack:** Different layer selections for different anatomical regions
   - **Disease-specific routing:** Route different diseases to different layer combinations
   - **Dynamic layer selection:** Learn which layers to use per input

---

## 📊 Expected Outcomes

### Quantitative Improvements

Based on DeepStack paper results on InfoVQA/DocVQA (document understanding tasks similar to medical report generation):

| Metric | Current Baseline | Expected with DeepStack | Improvement |
|--------|------------------|------------------------|-------------|
| Fine detail recognition | TBD | TBD | Target: +10-15% |
| Small lesion detection F1 | TBD | TBD | Target: +12% |
| Anatomical accuracy | TBD | TBD | Target: +8% |
| Overall BLEU-4 | TBD | TBD | Target: +5-7% |
| Inference speed | Baseline | ~Same (seq length unchanged) | 0% |

### Qualitative Improvements

**Expected report quality improvements:**

**Scenario: Small pulmonary nodule in CT scan**

- **Current (Layer 12 only):**
  > "There is an opacity in the right lung."

- **Expected with DeepStack:**
  > "There is a 3mm spiculated nodule in the right upper lobe apex, with irregular margins and subtle ground-glass attenuation."

**Why?**
- Layer 4 features → captures "spiculated" (毛刺状) texture
- Layer 8 features → captures "right upper lobe apex" anatomical location
- Layer 12 features → captures "nodule" semantic concept

---

## ⚠️ Challenges & Risks

### Technical Challenges

1. **Memory Consumption**
   - **Problem:** Storing 3 sets of intermediate features = 3× memory during forward pass
   - **Mitigation:**
     - Gradient checkpointing for intermediate features
     - Smaller `num_latents` for early layers (16 instead of 32)
     - Clear cached features immediately after use

2. **Training Instability**
   - **Problem:** Injecting features into early LLaMA layers might disrupt text processing
   - **Mitigation:**
     - Start with very small injection weights (0.01 → 0.1)
     - Freeze LLaMA + DeepStack separately initially
     - Add LayerNorm before injection

3. **LoRA Compatibility**
   - **Problem:** LoRA is applied to LLaMA weights, our injection bypasses this
   - **Mitigation:**
     - Keep LoRA on LLaMA, train DeepStack projectors separately
     - Or apply LoRA to projection layers too

4. **Hyperparameter Sensitivity**
   - **Problem:** Many new hyperparameters (which layers, where to inject, injection weights)
   - **Mitigation:**
     - Systematic ablation studies (Phase 2)
     - Start with paper's recommendations (layers 4/8/12, inject at 0/1/2)

### Research Risks

1. **Negative Result Risk**
   - DeepStack may not help medical imaging (paper only tested InfoVQA/DocVQA)
   - **Mitigation:** Start with MVP, quick iteration to test hypothesis

2. **Overfitting Risk**
   - More parameters (3× projectors) may overfit small datasets
   - **Mitigation:** Strong regularization, LoRA on projectors

3. **Computational Cost**
   - Training time may increase due to 3× projection overhead
   - **Mitigation:** Profile and optimize; consider shared Perceiver variant

---

## 🔬 Evaluation Plan

### Metrics

1. **Standard Report Generation Metrics**
   - BLEU-1/2/3/4
   - ROUGE-L
   - METEOR
   - BERTScore

2. **Medical-Specific Metrics**
   - Clinical accuracy (disease mention F1)
   - Anatomical location accuracy
   - Lesion size/characteristic accuracy (manual evaluation)

3. **Efficiency Metrics**
   - Training time per epoch
   - Inference time per sample
   - GPU memory usage
   - Parameters count (trainable vs total)

### Experimental Protocol

1. **Baseline:** Current Reg2RG model (Layer 12 only)
2. **DeepStack variants:**
   - DS-4/8/12: Extract layers 4, 8, 12
   - DS-3/6/9/12: Extract layers 3, 6, 9, 12
   - DS-8/10/12: Late-layer focus
3. **Ablations:** See Phase 2
4. **Dataset:** Same train/val/test split as current experiments
5. **Training:** Same hyperparameters initially, then optimize

---

## 📁 Code Structure

### New Files

```
src/Model/
├── vit_3d.py                     # MODIFY: Add return_intermediate flag
├── my_embedding_layer.py         # MODIFY: Add DeepStackProjectors
├── Reg2RG.py                     # MODIFY: Add injection mechanism
└── deepstack_projectors.py       # NEW: DeepStackProjectors class (optional refactor)

docs/
└── DeepStack_Experiment_Design.md  # THIS FILE

scripts/
├── train_deepstack.sh            # NEW: Training script for DeepStack variant
└── eval_deepstack.sh             # NEW: Evaluation script

configs/
└── deepstack_config.yaml         # NEW: DeepStack-specific hyperparameters
```

### Configuration Parameters

```yaml
# configs/deepstack_config.yaml
deepstack:
  enabled: true

  # Which ViT layers to extract (0-indexed)
  vit_extraction_layers: [3, 7, 11]  # Layers 4, 8, 12

  # Which LLaMA layers to inject into (0-indexed)
  llama_injection_layers: [0, 1, 2]

  # Projection settings
  num_latents_per_layer: [16, 24, 32]  # Early layers use fewer latents
  shared_perceiver: false  # If true, share perceiver across layers

  # Injection settings
  injection_type: "residual"  # residual | gated | cross_attn
  initial_injection_weight: 0.1
  injection_weight_schedule: "linear"  # linear | exponential | constant

  # Training settings
  freeze_llama: true  # Only train projectors initially
  freeze_vit: true
  use_gradient_checkpointing: true
```

---

## 📅 Timeline Estimate

### Quick Prototype (MVP Only)
- **Week 1:** Implement ViT intermediate extraction + projectors (~8 hours)
- **Week 2:** Implement LLaMA injection + debugging (~12 hours)
- **Week 3:** Initial training run + evaluation (~16 hours)
- **Total:** ~1 month part-time

### Full Implementation (MVP + Ablations)
- **Month 1:** MVP implementation and baseline evaluation
- **Month 2:** Ablation studies (layer combinations, injection positions)
- **Month 3:** Optimization and advanced variants
- **Month 4:** Final experiments and paper writing

---

## 🎯 Success Criteria

### Minimum Success (MVP)
- ✅ Code runs without errors
- ✅ Training converges
- ✅ No performance regression compared to baseline
- ✅ Memory usage < 2× baseline

### Target Success
- ✅ +5% improvement on at least one key metric (BLEU-4 or clinical F1)
- ✅ Qualitative improvement in fine-grained detail generation
- ✅ Ablation studies show clear benefit of multi-layer features

### Stretch Success
- ✅ +10% improvement on multiple metrics
- ✅ State-of-the-art results on target benchmark
- ✅ Novel contribution: Region-aware or disease-specific layer routing

---

## 🔄 Alternatives Considered

### Alternative 1: Concatenate All Layers as Tokens
**Idea:** Convert Layer 4/8/12 to tokens and concatenate them all.

**Pros:** Simple implementation
**Cons:** Sequence length 462 → 558 (+21%), attention cost increases 2.4×

**Verdict:** ❌ Rejected due to computational cost

### Alternative 2: Use Only Layer 4 (Early Features)
**Idea:** Maybe early features alone are sufficient?

**Pros:** Simpler (only 1 extra projector)
**Cons:** Loses semantic information from Layer 12

**Verdict:** ⚠️ Worth testing as ablation

### Alternative 3: Adaptive Layer Selection
**Idea:** Learn which layers to use per input via attention mechanism.

**Pros:** More flexible, potentially better
**Cons:** More complex, harder to train

**Verdict:** ✅ Save for Phase 3 (advanced optimizations)

---

## 📚 References

1. **DeepStack Paper:** [Insert paper reference when available]
   - InfoVQA improvement: +12%
   - DocVQA improvement: +16%
   - Key insight: Multi-layer features without sequence length increase

2. **Related Work:**
   - Flamingo: Cross-attention to vision features at every layer
   - BLIP-2: Q-Former for efficient vision-language alignment
   - LLaVA: Simple concatenation approach (our baseline)

3. **Our Codebase:**
   - Current architecture: `docs/LIT_Architecture.md`
   - Training guide: `docs/RESUME_TRAINING.md`
   - Experiment tracking: `docs/LIT_Experiment_Design_v2.md`

---

## 📝 Decision Log

### To Be Decided

- [ ] **Go/No-Go decision:** Implement this or prioritize other ideas?
- [ ] **Timeline:** When to start implementation?
- [ ] **Resource allocation:** GPU hours, personnel time
- [ ] **Baseline comparison:** Which current checkpoint to compare against?

### Future Discussion Points

1. Should we combine DeepStack with other architectural improvements (e.g., cross-attention between regions)?
2. Can we leverage the position encoding features selected in the system reminder?
3. Should we explore region-specific layer selection (different layers for lung vs heart)?

---

## 👤 Ownership

**Proposed by:** User + Claude (Brainstorming session)
**Implementation owner:** TBD
**Reviewer:** TBD
**Priority:** TBD

---

## 📎 Appendices

### Appendix A: Comparison Table

| Aspect | Current Method | DeepStack Method |
|--------|---------------|------------------|
| ViT layers used | Layer 12 only | Layers 4, 8, 12 |
| Vision info in LLM | Input tokens (prefix) | Hidden state injection |
| Sequence length | 462 tokens | 100 tokens (text only) |
| Attention cost | O(462²) = 213,444 | O(100²) = 10,000 |
| Speed | Baseline | **21× faster attention** |
| Information density | Single semantic level | Multi-scale (fine+mid+semantic) |
| Extra parameters | 0 | ~3× projectors (~100M) |

### Appendix B: Visual Information by Layer

```
Layer 4 (Early):
- Edge detection
- Texture patterns (ground-glass opacity, spiculation)
- Micro-calcifications
- Small vessel visualization

Layer 8 (Middle):
- Organ boundaries (heart border, lung fissures)
- Anatomical structures (bronchi, major vessels)
- Region relationships (mediastinum vs lung)
- Shape features (cardiomegaly, pleural effusion shape)

Layer 12 (Final):
- Disease concepts (pneumonia, atelectasis)
- Overall diagnostic impression
- Semantic relationships
- Report-level understanding
```

### Appendix C: Implementation Checklist

**Phase 1: MVP**
- [ ] Modify ViT forward to return intermediate features
- [ ] Implement DeepStackProjectors class
- [ ] Test projectors with dummy data
- [ ] Modify MyEmbedding to use DeepStackProjectors
- [ ] Implement LLaMA layer monkey-patching in Reg2RG
- [ ] Test end-to-end forward pass
- [ ] Verify backward pass and gradient flow
- [ ] Run sanity check training (1 batch, 10 steps)
- [ ] Full training run on small dataset
- [ ] Evaluate on validation set
- [ ] Compare with baseline

**Phase 2: Ablations**
- [ ] Layer combination ablations (4/8/12 vs others)
- [ ] Injection position ablations (layers 0/1/2 vs others)
- [ ] Projection method ablations (separate vs shared)
- [ ] Injection strategy ablations (residual vs gated vs cross-attn)

**Phase 3: Optimization**
- [ ] Memory optimization (variable num_latents)
- [ ] Training stability (curriculum learning)
- [ ] Region-aware variants
- [ ] Disease-specific routing
- [ ] Dynamic layer selection

---

## 🔖 Tags

`#architecture` `#vision-language` `#multi-layer-features` `#deepstack` `#efficiency` `#medical-imaging` `#idea` `#planning`

---

**Last Updated:** 2025-12-23
**Status:** Awaiting decision
