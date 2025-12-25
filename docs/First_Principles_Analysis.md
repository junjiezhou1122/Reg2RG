# CT Radiology Report Generation: First Principles Analysis

**Date**: 2025-12-25
**Author**: Junjie Zhou
**Status**: Research Rethink

---

## 🚨 The Core Problem

### Current Situation
```python
Original Reg2RG F1 Score: 0.25  # ❌ Very poor!

Your Findings:
- Frozen adapter reconstruction: 79% (cos ≈ 0.79)
- Joint-trained adapter reconstruction: 85% (cos ≈ 0.85)
- But: F1 score still ~0.25-0.30?

Conclusion: 85% reconstruction → 0.25 F1
→ Information bottleneck is NOT in reconstruction quality!
```

**The disconnect**: We're optimizing the wrong objective.

---

## 🧠 First Principles: What IS Radiology Report Generation?

### Not a Vision-Language Problem (Common Misconception)

**Wrong framing**:
```
CT Image → Vision Encoder → Adapter → LLM → Report
        ↓
    "Just like image captioning!"
```

**Why this fails**:
1. **Sparse Information Density**
   - Natural images: dense semantic info everywhere (cat, sofa, window...)
   - CT scans: 95% normal tissue, 5% abnormalities
   - Current approach: compresses everything equally → dilutes critical findings

2. **Multi-Scale Reasoning**
   - Need global context: "Is this heart size normal relative to patient's build?"
   - Need local details: "Is this 3mm nodule spiculated?"
   - Current approach: single compression level → loses either context or detail

3. **3D Spatial Structure**
   - Findings are defined by 3D location + morphology
   - "5mm nodule in right upper lobe posterior segment"
   - Current approach: flattens to tokens → spatial relationships lost

---

## 🔍 Root Cause Analysis: Why 85% Reconstruction ≠ Good Reports?

### Problem 1: **Compression Ratio is Too Aggressive**

```python
Input:  1024 tokens (16×16×4 patches from 512³ volume)
Output: 32 tokens (Perceiver compression)

Compression ratio: 32:1 (97% information thrown away!)

# What's preserved?
Low-frequency global features: ✅ (easy to compress)
  - Overall organ shapes
  - Tissue density distributions
  - Smooth intensity gradients

High-frequency local features: ❌ (lost in compression)
  - Small nodules (< 5mm)
  - Subtle infiltrates
  - Lesion margins (spiculated vs smooth)
  - Calcifications
```

**Evidence from your data**:
```python
global_cos < region_cos  # You observed this!

Why?
- Global features are smooth → easy to compress/reconstruct
- Region features have more detail → harder to compress
- But: region details are what radiologists write about!
```

**Test**: Can you reconstruct a 3mm nodule from 32 tokens?
- Answer: Probably not! It's averaged out with surrounding tissue.

---

### Problem 2: **Reconstruction Objective ≠ Report Objective**

```python
# What reconstruction optimizes:
MSE + (1 - cosine)
→ Minimize pixel-level error
→ Preserves what's EASY to reconstruct (smooth features)

# What reports need:
F1 score on clinical findings
→ Detect what's CLINICALLY RELEVANT (abnormalities)
→ Preserves what's HARD to find (sparse, subtle)
```

**Concrete example**:
```python
CT Scan: [95% normal lung tissue + 5% with nodule]

Reconstruction Loss optimizes:
- 95% weight on normal tissue → minimize error here
- 5% weight on nodule → "acceptable" to lose

Report Quality requires:
- Nodule detection = critical (0% vs 100% matters)
- Normal tissue = "just say normal" (pixel-perfect not needed)
```

**This is why 85% reconstruction still gives 0.25 F1!**

---

### Problem 3: **Fixed Region Segmentation is Suboptimal**

```python
Current: 8 predefined anatomical regions
- lung_left, lung_right, heart, thyroid, trachea, ...

Problems:
1. Findings don't respect anatomical boundaries
   - "Mass crossing from lung into mediastinum"

2. Most regions are normal (wasted compression budget)
   - 7/8 regions: "normal" → still compress to 32 tokens each

3. Important findings might be between regions
   - Pleural effusion (lung-chest wall interface)
   - Lymphadenopathy (between organs)
```

**Better approach**: Dynamic attention to abnormal regions

---

### Problem 4: **Perceiver Might Not Suit Medical Imaging**

**Perceiver design philosophy**:
- Designed for: Multi-modal fusion (images + audio + text)
- Strength: Handling variable-length inputs
- Mechanism: Iterative cross-attention (6 layers)

**Medical imaging requirements**:
- Need: Preserve spatial hierarchy (low-res context + high-res details)
- Challenge: 3D structure (not just 2D)
- Critical: Fine-grained localization

**Potential issue**:
```python
Perceiver latents (32 tokens) act as "bottleneck"
→ All information must flow through these 32 vectors
→ Acts like a soft clustering (32 cluster centers)
→ Small findings get merged with background
```

**Alternative architectures** (might preserve better):
1. **Pyramid/Multi-scale**: Different resolution levels
2. **Sparse attention**: Focus on abnormal regions
3. **Hierarchical**: Coarse-to-fine refinement

---

## 💡 First Principles Solution Design

### Principle 1: **Information Should Match Task Priority**

**Current**: Equal compression for all spatial regions (uniform)
**Needed**: More bits for clinically important regions (adaptive)

**Proposal**: **Saliency-Guided Compression**

```python
# Step 1: Identify salient regions (abnormality detector)
saliency_map = abnormality_detector(CT_scan)  # (H, W, D) → [0, 1]
# High values = likely abnormal (nodules, masses, etc.)

# Step 2: Allocate compression budget by saliency
for region in salient_regions:
    tokens_allocated = saliency[region] * total_token_budget
    # Abnormal areas get more tokens, normal areas get fewer

# Example allocation:
Normal lung (90% of volume):  10 tokens  # "normal" needs few tokens
Nodule region (1% of volume): 15 tokens  # abnormality needs detail!
Heart (9% of volume):         7 tokens
```

**Key insight**: A 3mm nodule should get as many tokens as the entire heart if it's clinically significant!

---

### Principle 2: **Preserve Multi-Scale Structure**

**Current**: Single-level compression (1024 → 32)
**Needed**: Hierarchical representation (like radiologist's reasoning)

**Proposal**: **Multi-Scale Adapter**

```python
# Radiologist's mental process:
1. Global view (low-res): Overall scan quality, patient positioning
2. Organ level (medium-res): Organ sizes, major abnormalities
3. Lesion level (high-res): Lesion characteristics, margins

# Architecture:
encoder_output = ViT(CT_scan)  # (1024, 768)

# Multi-scale compression
global_tokens = coarse_adapter(encoder_output)      # 8 tokens: global context
organ_tokens = medium_adapter(encoder_output)       # 16 tokens: organ-level
lesion_tokens = fine_adapter(salient_regions)       # 16 tokens: lesion details

final_representation = concat([global_tokens, organ_tokens, lesion_tokens])
# Total: 40 tokens (slightly more than current 32, but much more efficient!)
```

**Benefits**:
- Global context: preserved in 8 tokens (for impressions like "overall normal")
- Local details: preserved in 16 tokens (for findings like "5mm nodule")
- Better than: 32 tokens trying to do both (and failing at both)

---

### Principle 3: **Learn What Matters for Reports, Not Reconstruction**

**Current**: Optimize reconstruction → hope it transfers to reports
**Needed**: Optimize report quality directly

**Proposal**: **Report-Aware Compression Training**

```python
# Two-stage with feedback

# Stage 1: End-to-end training (not just probe!)
for batch in train_loader:
    CT_scan, report = batch

    # Forward pass
    vision_tokens = encoder(CT_scan)
    compressed = adapter(vision_tokens)  # ← Train this
    generated_report = LLM(compressed)   # ← Train this too

    # Loss: Report quality (not reconstruction!)
    loss = CrossEntropy(generated_report, report)

    # Adapter learns: "What features help LLM generate good reports?"
    loss.backward()  # Gradient flows through adapter!

# Stage 2: Analysis via probe
# Now test: Can we reconstruct from this report-optimized adapter?
# (Might be worse reconstruction, but better reports!)
```

**Key difference from your Exp2**:
- Your Exp2: Train adapter to reconstruct well, then use for VLM
- This: Train adapter to help VLM directly (reconstruction is just diagnostic)

**Why this is risky but promising**:
- Risk: Requires more GPU memory (full VLM in training)
- Promise: Adapter learns exactly what LLM needs

---

### Principle 4: **Spatial Reasoning Should Be Explicit**

**Current**: Spatial info is implicit in token embeddings
**Needed**: Explicit 3D localization

**Proposal**: **Spatial-Semantic Tokens**

```python
# Each token carries both content and location

class SpatialSemanticToken:
    content: Tensor (768,)      # What: feature embedding
    location: Tensor (3,)       # Where: (x, y, z) coordinates
    scale: float                # Scale: region size
    confidence: float           # Abnormality probability

# Benefits for report generation:
LLM_input = [
    {content: [lesion_features], location: (120, 80, 45), confidence: 0.95},
    # → LLM can generate: "5mm nodule in right upper lobe"

    {content: [normal_features], location: (150, 150, 50), confidence: 0.1},
    # → LLM can generate: "left lung clear"
]

# Current approach loses location:
# All tokens are just (768,) embeddings → LLM has no spatial grounding
```

**Implementation**:
- Add positional encoding as explicit features (not just in attention)
- Train LLM to decode spatial coordinates into anatomical language

---

## 🧪 Proposed Experiment Suite (From First Principles)

### Experiment A: **Ablate the Compression Ratio**

**Hypothesis**: 32 tokens is too few; quality improves with more tokens

```python
# Test multiple compression levels
Exp A1: 16 tokens  (64:1 compression)
Exp A2: 32 tokens  (32:1 compression) ← current baseline
Exp A3: 64 tokens  (16:1 compression)
Exp A4: 128 tokens (8:1 compression)

# Evaluate both:
- Reconstruction quality (your LIT probe)
- Report F1 score (VLM on each)

# Expected result:
If F1 increases significantly with more tokens:
  → Bottleneck is compression ratio
Else:
  → Bottleneck is elsewhere (architecture, training)
```

**Timeline**: 2 weeks (4 compression levels × 2 stages each)

---

### Experiment B: **Saliency-Guided Compression**

**Hypothesis**: Allocating tokens by clinical importance improves reports

```python
# Compare:
Baseline: Uniform compression (all regions equal)
Proposed: Saliency-weighted compression

# Implementation:
1. Train simple abnormality detector:
   - Input: ViT tokens
   - Output: Saliency map (which tokens are abnormal)

2. Use saliency to weight Perceiver queries:
   - More queries for salient regions
   - Fewer queries for normal regions

# Evaluation:
- Reconstruction quality (global vs lesion-specific)
- Report F1 (especially for lesion detection)
```

**Timeline**: 2-3 weeks

---

### Experiment C: **Multi-Scale Adapter**

**Hypothesis**: Hierarchical compression preserves both context and details

```python
# Architecture:
Global adapter:  8 tokens  (for "impression" level)
Organ adapter:   16 tokens (for "findings" level)
Lesion adapter:  16 tokens (for "details" level)
Total: 40 tokens (vs 32 baseline)

# Training:
- Use hierarchical loss:
  - Global loss: Match overall impression
  - Organ loss: Match organ-specific findings
  - Lesion loss: Match lesion descriptions

# Evaluation:
- Does this improve fine-grained findings (nodule size, margin)?
- Does this improve hierarchical report structure?
```

**Timeline**: 3 weeks

---

### Experiment D: **End-to-End Report Training** (Most Radical)

**Hypothesis**: Optimizing adapter for report quality beats reconstruction-based training

```python
# Method:
Skip the probe! Train adapter + LLM together from scratch.

# Comparison:
Your Exp2: Adapter trained for reconstruction → frozen for VLM
Exp D:     Adapter trained jointly with VLM (gradient flows through LLM)

# Training:
for epoch in epochs:
    for CT_scan, report in train_data:
        tokens = vision_encoder(CT_scan)  # frozen
        compressed = adapter(tokens)       # trainable
        generated = LLM(compressed)        # trainable (LoRA)

        loss = CrossEntropy(generated, report)
        loss.backward()  # Adapter learns what LLM needs!

# Then test reconstruction (curiosity):
reconstruction_quality = probe_test(adapter)
# Might be WORSE reconstruction, but BETTER reports!
```

**Why this could work**:
- Adapter learns: "What visual features predict rare clinical terms?"
- Might compress: "This looks spiculated" (useful for report)
- Might discard: "Exact HU values of normal tissue" (useless for report)

**Risk**: Requires training full VLM (expensive)

**Timeline**: 4 weeks (long training)

---

## 🔬 Diagnostic Experiments (Fast, Do First)

### Diagnostic 1: **Reconstruction Quality by Region Type**

**Question**: Which regions reconstruct poorly? Are they clinically important?

```python
# In your current Exp2 validation, add:
for region in [lung, heart, thyroid, ...]:
    cos_by_region[region] = compute_cos(region)

# Then check clinical importance:
important_regions = [lung, mediastinum]  # Where findings are
unimportant_regions = [bone, air]         # Less relevant for reports

if cos_by_region[important] < cos_by_region[unimportant]:
    print("❌ Adapter prioritizes wrong regions!")
```

**Timeline**: 1 day (just add logging)

---

### Diagnostic 2: **Token Saliency Analysis**

**Question**: Do compressed tokens correspond to clinical findings?

```python
# Method:
1. Take a CT with known lesion (e.g., 5mm nodule at location (x,y,z))
2. Visualize which input tokens contribute most to each output latent
3. Check: Do latents "attend to" the nodule region?

# If no:
  → Adapter is ignoring small findings!
  → Explains poor F1 score despite good reconstruction
```

**Timeline**: 2-3 days (visualization + analysis)

---

### Diagnostic 3: **Compression Budget vs F1 Score**

**Question**: Is 32 tokens enough? How much is needed?

```python
# Quick test (before full training):
1. Use pretrained adapter to get 32 tokens
2. Randomly drop N tokens (simulate lower compression)
3. See which tokens are "expendable" vs critical

# Analysis:
if dropping any 4 tokens → F1 drops significantly:
    "32 tokens is already tight! Need more."
elif dropping 16 tokens → F1 stays similar:
    "32 tokens is wasteful! Architecture is the problem."
```

**Timeline**: 1 day

---

## 📊 Observations to Investigate

### Observation 1: **Global cos < Region cos**

**Your finding**: This is fascinating and counterintuitive!

**Hypothesis**: Global features are "easier" but less informative

```python
# Possible explanation:
Global features (low-frequency):
  - Smooth organ shapes
  - Overall density
  → Easy to compress (low entropy)
  → But: "Patient has lungs" is not useful for report

Region features (high-frequency):
  - Lesion textures
  - Boundary details
  → Hard to compress (high entropy)
  → But: "Nodule is spiculated" is CRITICAL for report!

# Test:
Compare reconstruction loss vs report F1:
- If regions with worse reconstruction → better F1
  → Confirms: reconstruction quality ≠ report quality
```

---

### Observation 2: **85% Reconstruction → 25% F1**

**This massive gap reveals the core problem!**

**Calculation**:
```python
Reconstruction preserves: 85% of information
But reports capture: 25% of findings

Implication:
- The 85% preserved info is NOT the 25% needed for reports!
- We're preserving the WRONG 85%

# Analogy:
You're writing a detective report:
- You remember 85% of the day: breakfast, commute, lunch...
- But you forget the 15%: the murder scene!
→ Your report is useless despite 85% memory
```

**Solution direction**:
- Don't optimize for "average information preservation"
- Optimize for "critical information preservation"
- Use report labels to identify what's critical!

---

## 🎯 Recommended Research Path

### Phase 1: **Fast Diagnostics** (Week 1)

Do the 3 diagnostic experiments above to identify bottleneck:

```
Is it compression ratio? → Do Experiment A
Is it attention to findings? → Do Experiment B
Is it architecture? → Do Experiment C
Is it training objective? → Do Experiment D
```

### Phase 2: **Targeted Fix** (Weeks 2-4)

Based on Phase 1 results, implement the most promising fix.

### Phase 3: **Validate End-to-End** (Weeks 5-6)

Train full VLM with improved adapter, measure F1 improvement.

---

## 🧠 Philosophical Takeaway

**The Lesson**: First principles thinking reveals:

```
Current approach:
  "Can we compress CT → reconstruct CT?"
  (Answer: Yes, 85%)
  But: Wrong question!

Right question:
  "Can we compress CT → generate accurate reports?"
  (Answer: No, 25% F1)

Solution:
  Stop optimizing reconstruction.
  Start optimizing reports.
```

**Your intuition is correct**: We need to design experiments that target the ACTUAL task, not proxy metrics.

---

## 📚 Related Work to Read

1. **Sparse Attention for Medical Imaging**
   - "Swin Transformer for Medical Image Analysis" (hierarchical)

2. **Task-Specific Compression**
   - "Conditional Autoencoders" (compress based on task)

3. **Multi-Scale Vision-Language**
   - "CLIP with Pyramid Features"

4. **Radiology-Specific**
   - How do radiologists actually read CTs? (eye-tracking studies)
   - What visual features predict clinical terms?

---

**Next Steps**:
1. Run Diagnostic Experiments 1-3 (3 days)
2. Based on results, choose Experiment A, B, C, or D
3. Document findings and iterate

**Remember**: High reconstruction ≠ good reports. Optimize what matters!

---

**Last Updated**: 2025-12-25
