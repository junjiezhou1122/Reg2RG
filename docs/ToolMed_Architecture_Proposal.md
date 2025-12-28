# ToolMed: Tool-Augmented Medical Vision-Language Architecture

**Project**: A New Paradigm for Medical VLMs
**Created**: 2025-12-28
**Author**: Junjie Zhou
**Status**: Brainstorming / Proposal

---

## 1. Executive Summary

### The Vision

We propose **ToolMed**, a fundamentally new architecture paradigm for medical vision-language models that:

1. **Decomposes** the monolithic VLM into explicit, interpretable **tools**
2. **Leverages** existing pretrained models instead of reinventing the wheel
3. **Enables** true modularity - add new tools without retraining the system
4. **Provides** full interpretability - every decision traceable to specific tool outputs

### Key Innovation

```
Traditional VLM:     Image → [Black Box] → Report
                              ???

ToolMed:             Image → [Internal Tools] → [Fusion Hub] → [LLM] → Report
                              ↓ Interpretable    ↓ Translates   ↓ Reasons
                              outputs            languages      over findings
```

---

## 2. Motivation: Why We Need This

### 2.1 The Current Problems

#### Problem 1: Garbage In, Garbage Out

```
Medical images often require resize before processing:

Original CT: 512×512×300 (large lung region)
     ↓ resize
Target: 256×256×64

Compression ratio = 4x+ for large organs!
Small nodules (2-5mm) are LOST in this compression.

Current models: Learn from corrupted input, produce corrupted output.
```

#### Problem 2: Reinventing the Wheel

```
We already have excellent pretrained models:

- GPT-4 / Claude / LLaMA: Great reasoning, medical knowledge
- SAM-Med3D: Excellent segmentation
- TotalSegmentator: Robust organ segmentation
- CT-CLIP: Good medical image features
- Specialized detectors: Nodule detection, fracture detection, etc.

Yet every new paper builds from scratch!
Why not COMPOSE these existing tools?
```

#### Problem 3: Black Box Models

```
Current medical VLMs:
- Can't explain WHY they made a decision
- Can't audit which image regions influenced output
- Can't debug when they fail
- Clinicians don't trust them

We need INTERPRETABILITY by design, not as an afterthought.
```

#### Problem 4: No Modularity

```
Current approach:
- Paper A: ViT + Adapter1 + LLM1
- Paper B: ViT + Adapter2 + LLM2
- Paper C: CNN + Adapter3 + LLM1

Each paper builds from scratch.
Can't combine innovations.
Can't swap components.
Not comparable.
```

### 2.2 The Core Insight

```
Radiologists don't work as black boxes!

They use "mental tools":
1. Segment regions (identify organs)
2. Detect abnormalities (spot issues)
3. Measure (size, density)
4. Characterize (texture, margins)
5. Reason (integrate findings)
6. Consult (call specialist for hard cases)

We should make these tools EXPLICIT in the architecture!
```

---

## 3. Architecture Overview

### 3.1 The Two-Level Tool System

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   Level 2: EXTERNAL TOOLS (Agent-Controlled)                        │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  SAM-Med3D │ TotalSeg │ Specialist Models │ Prior Compare   │   │
│   │  (Separate pretrained models, called on-demand via MCP)     │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              ↑                                      │
│                         Tool Calls (when uncertain)                 │
│                              │                                      │
│   ┌──────────────────────────┴──────────────────────────────────┐   │
│   │                      LLM Agent                              │   │
│   │              (Reasoning, Report Generation)                 │   │
│   └──────────────────────────┬──────────────────────────────────┘   │
│                              │                                      │
│                     Structured Output                               │
│                              ↑                                      │
│   Level 1: INTERNAL TOOLS (Built into Architecture)                 │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                                                             │   │
│   │   ┌─────────┬─────────┬─────────┬─────────┬─────────┐       │   │
│   │   │ Organ   │ Anomaly │  Size   │ Texture │ Uncert- │       │   │
│   │   │ Router  │ Detector│ Estimat │ Encoder │  ainty  │       │   │
│   │   └─────────┴─────────┴─────────┴─────────┴─────────┘       │   │
│   │                                                             │   │
│   │   Differentiable, fast, always-on, interpretable            │   │
│   │                                                             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              ↑                                      │
│                         ViT Features                                │
│                              │                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                    ViT Encoder                               │  │
│   │               (Pretrained, e.g., CT-CLIP)                    │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              ↑                                      │
│                          CT Image                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Internal Tools vs External Tools

| Aspect | Internal Tools | External Tools |
|--------|---------------|----------------|
| **Location** | Inside model architecture | Separate pretrained models |
| **Training** | End-to-end differentiable | Frozen, pretrained |
| **Speed** | Fast (one forward pass) | Slower (API calls) |
| **When used** | Always (core capabilities) | On-demand (complex cases) |
| **Examples** | Organ attention, anomaly score | SAM-Med3D, prior comparison |
| **Interpretability** | Attention maps, scores | Full segmentation masks |

### 3.3 Why This Mirrors Radiologists

```
Radiologist Workflow:              ToolMed Architecture:
────────────────────              ─────────────────────

1. Quick scan of image        →   ViT backbone (fast encoding)
   "What am I looking at?"

2. Identify organs/regions    →   Organ Router (internal tool)
   "Lungs, heart, liver..."       (attention to each organ)

3. Spot abnormalities         →   Anomaly Detector (internal tool)
   "Something in RUL"             (highlight suspicious areas)

4. Measure and characterize   →   Attribute Predictors (internal tools)
   "8mm, spiculated, solid"       (size, texture, density)

5. Assess confidence          →   Uncertainty Estimator (internal tool)
   "Pretty sure, but..."          (know when unsure)

6. Write report               →   LLM Decoder
   Synthesize all findings        (generate coherent text)

7. Consult specialist         →   External Tools
   For difficult cases            (call when uncertain)
```

---

## 4. The Alignment Problem (Critical Challenge)

### 4.1 The Problem

Different tools output different "languages" (representations):

```
OrganRouter:    768-dim attention maps
AnomalyDet:     256-dim features + scalar score
TextureNet:     128-dim embedding + category
SizeHead:       3 floats (dimensions in mm)
External SAM:   Binary segmentation mask

These are COMPLETELY DIFFERENT formats!
How do they communicate?
How does LLM understand all of them?
```

### 4.2 The Core Dilemma

```
To UNDERSTAND something → Need TRAINING
To be MODULAR          → Don't want to RETRAIN

These are in conflict!

If Fusion Hub is trainable:
  → Can understand tools
  → But must RETRAIN when adding new tool
  → NOT modular ❌

If Fusion Hub is frozen:
  → Truly modular
  → But CAN'T understand new tool's language
  → Doesn't work ❌
```

### 4.3 Our Solution: Adapter + Reconstruction

```
Key Insight: Each tool learns to speak Fusion Hub's language
             via a lightweight adapter.
             Reconstruction loss ensures information is preserved.

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   New Tool                                                      │
│      ↓                                                          │
│   Output (any dim, e.g., 512-dim)                               │
│      ↓                                                          │
│   ┌─────────────────┐                                           │
│   │  ADAPTER LAYER  │  ← Only train THIS! (1 layer)             │
│   └────────┬────────┘                                           │
│            ↓                                                    │
│   Fusion Hub Language (256-dim)                                 │
│            ↓                                                    │
│   ┌─────────────────┐                                           │
│   │   FUSION HUB    │  ← FROZEN (never changes)                 │
│   └────────┬────────┘                                           │
│            ↓                                                    │
│   ┌─────────────────┐                                           │
│   │    DECODER      │  ← Train to reconstruct (1 layer)         │
│   └────────┬────────┘                                           │
│            ↓                                                    │
│   Reconstructed Output (512-dim)                                │
│            ↓                                                    │
│   Loss = ||Original - Reconstructed||²                          │
│                                                                 │
│   If reconstruction works → Adapter learned the language!       │
│   After training: Discard decoder, keep adapter                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Why this works:**
1. **Simple**: Just 2 small layers (adapter + decoder)
2. **Fast**: Train in minutes, not hours
3. **Modular**: Add new tool without touching Fusion Hub
4. **Self-supervised**: Reconstruction loss, no labels needed
5. **Preserves information**: If decoder reconstructs → info preserved

---

## 5. Text Output from Tools

### 5.1 The Challenge

LLM understands text naturally. How do we get text from tools?

```
Some tools have structured output:
  → Easy: Use templates

Some tools only output embeddings:
  → Challenge: How to generate text?
```

### 5.2 Solutions

#### Solution A: Template-Based (No Training)

```python
# Tool predicts structured values
output = {
    "type": "nodule",       # Classification head
    "location": "RUL",      # Classification head
    "size": 8.2,            # Regression head
    "anomaly_score": 0.87,  # Regression head
}

# Template converts to text (no neural network!)
text = f"Found {output['type']} in {output['location']}. " \
       f"Size: {output['size']:.1f}mm. " \
       f"Anomaly score: {output['anomaly_score']:.2f}."

# Result: "Found nodule in RUL. Size: 8.2mm. Anomaly score: 0.87."
```

#### Solution B: Probe Heads (Light Training)

```
For pure embedding tools, add simple linear probes:

Embedding [128-dim]
    ↓
├─→ Linear → "normal/abnormal"
├─→ Linear → "small/medium/large"
└─→ Linear → "solid/ground-glass/mixed"
    ↓
Template → Text

Probes are tiny! Easy to train.
```

#### Solution C: Clustering (No Training)

```
1. Cluster the embedding space (K-means, offline)
2. Manually label each cluster with text description
3. At inference: find nearest cluster → return label

No neural text generation!
Just nearest neighbor lookup.
```

#### Solution D: Text is Optional

```
Not all tools need text output.

Tools WITH text:    Embedding + Text → Fusion Hub + LLM
Tools WITHOUT text: Embedding only → Adapter → Fusion Hub → LLM

System works with both!
```

---

## 6. Hierarchical ROI Finding

### 6.1 The Challenge

To analyze findings, we need to know WHERE to look (ROI).
But finding ROI seems like a chicken-and-egg problem.

### 6.2 Solution: Hierarchical Approach

```
Level 1: ORGANS (Easy, we have TotalSegmentator)
─────────────────────────────────────────────────
TotalSegmentator → lung, heart, liver, ...
Each organ = one coarse ROI


Level 2: ATTENTION within organ (From ViT)
─────────────────────────────────────────────────
Within lung ROI, where does ViT attend most?
High attention = candidate sub-region


Level 3: SPECIFIC FINDINGS (Specialized detectors)
─────────────────────────────────────────────────
Use nodule detector within high-attention region
Get precise finding location
```

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CT IMAGE                                     │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │  TotalSegmentator     │  Level 1: Organ ROIs
                  └───────────┬───────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ↓                  ↓                  ↓
       ┌───────┐          ┌───────┐          ┌───────┐
       │ Lung  │          │ Heart │          │ Liver │
       └───┬───┘          └───────┘          └───────┘
           │
           ↓
   ┌───────────────┐
   │ ViT Attention │  Level 2: Sub-region ROIs
   │ within Lung   │
   └───────┬───────┘
           │
           ↓
   High-attention area
           │
           ↓
   ┌───────────────┐
   │ Nodule Det    │  Level 3: Finding ROIs
   └───────┬───────┘
           │
           ↓
   Precise nodule location
```

---

## 7. Component Protocol Architecture

### 7.1 The Vision: A Protocol, Not Just a Model

```
Instead of: "Here's our model architecture"
We say:     "Here's a PROTOCOL for building medical VLMs"
            "Plug in any components that follow the protocol"
```

### 7.2 The Protocol Specification

```python
class ToolProtocol:
    """
    Every tool must follow this interface.
    """

    # Identity
    name: str                    # "OrganRouter", "AnomalyDetector", etc.
    tool_type: ToolType          # ENCODER, SEGMENTOR, DETECTOR, CLASSIFIER, etc.

    # Dimensions
    input_type: str              # "image", "features", "region", etc.
    output_dim: int              # Must project to standard dim (e.g., 256)

    # Capabilities
    provides: List[str]          # ["organ_segmentation", "anomaly_score", ...]
    requires: List[str]          # ["vit_features"] or ["image"], etc.

    # Methods
    def forward(self, x) -> ToolOutput:
        """Process input, return standardized output."""
        pass

    def to_text(self, output) -> Optional[str]:
        """Convert output to text (if available)."""
        pass


@dataclass
class ToolOutput:
    """Standardized output format for all tools."""

    embedding: Tensor           # Always 256-dim (projected by tool itself)
    structured: Optional[Dict]  # Structured values (type, size, etc.)
    text: Optional[str]         # Text description (if available)
    confidence: float           # 0.0 to 1.0
    tool_type: ToolType         # What kind of tool produced this
```

### 7.3 Component Library

```
Anyone can contribute components that follow the protocol:

ENCODERS              ANALYZERS             REASONERS
├─ ViT-3D             ├─ OrganRouter        ├─ LLaMA
├─ CT-CLIP            ├─ AnomalyDetector    ├─ GPT-4
├─ ResNet3D           ├─ TextureNet         ├─ Mistral
├─ SwinUNETR          ├─ SizeEstimator      ├─ Med-LLM
└─ [Your encoder]     ├─ DensityNet         └─ [Your LLM]
                      └─ [Your analyzer]

EXTERNAL TOOLS        ADAPTERS              FUSION HUBS
├─ SAM-Med3D          ├─ Perceiver          ├─ CrossAttentionHub
├─ TotalSegmentator   ├─ Q-Former           ├─ TransformerHub
├─ NoduleDetector     ├─ LinearProbe        └─ [Your hub]
└─ [Your tool]        └─ [Your adapter]
```

### 7.4 Composition Engine

```python
# User specifies components
config = {
    "encoder": "CT-CLIP",
    "internal_tools": ["OrganRouter", "AnomalyDetector", "TextureNet"],
    "fusion_hub": "CrossAttentionHub",
    "reasoner": "LLaMA-3",
    "external_tools": ["SAM-Med3D", "TotalSegmentator"],
}

# Engine automatically builds the model
model = CompositionEngine.build(config, registry)

# Or auto-compose based on requirements
model = CompositionEngine.auto_compose(
    input="3D CT",
    output="radiology report",
    required_capabilities=["organ_segmentation", "anomaly_detection"],
)
```

---

## 8. Implementation Details

### 8.1 Fusion Hub with Type Embeddings

```python
class FusionHub(nn.Module):
    """
    Universal translator using cross-attention.
    Uses type embeddings to understand different tool outputs.
    """

    def __init__(self, num_types=6, embed_dim=256, num_queries=32):
        super().__init__()

        # Type embeddings (learned)
        self.type_embed = nn.Embedding(num_types, embed_dim)

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim))

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8)

        # To LLM dimension
        self.to_llm = nn.Linear(embed_dim, 4096)

    def forward(self, tool_outputs: List[ToolOutput]):
        all_embeddings = []

        for output in tool_outputs:
            # Add type information
            type_emb = self.type_embed(output.tool_type.value)
            combined = output.embedding + type_emb
            all_embeddings.append(combined)

        # Concatenate all tool outputs
        all_kv = torch.cat(all_embeddings, dim=1)

        # Cross-attend with learnable queries
        fused, _ = self.cross_attn(
            self.queries.unsqueeze(0).expand(batch_size, -1, -1),
            all_kv, all_kv
        )

        return self.to_llm(fused)
```

### 8.2 Tool Adapter Training

```python
def train_adapter_for_new_tool(tool, fusion_hub, dataloader, epochs=10):
    """
    Train adapter for new tool using reconstruction loss.
    Only trains adapter and decoder, everything else frozen.
    """

    # Create adapter and decoder
    adapter = nn.Sequential(
        nn.Linear(tool.output_dim, 256),
        nn.LayerNorm(256),
        nn.GELU(),
    )

    decoder = nn.Sequential(
        nn.Linear(256, tool.output_dim),
        nn.LayerNorm(tool.output_dim),
    )

    # Freeze tool and fusion hub
    tool.eval()
    fusion_hub.eval()
    for p in tool.parameters():
        p.requires_grad = False
    for p in fusion_hub.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(decoder.parameters()),
        lr=1e-3
    )

    for epoch in range(epochs):
        for batch in dataloader:
            # Forward through frozen tool
            with torch.no_grad():
                tool_output = tool(batch)

            # Through trainable adapter
            hub_input = adapter(tool_output)

            # Through frozen fusion hub
            with torch.no_grad():
                hub_output = fusion_hub.encode(hub_input)

            # Through trainable decoder
            reconstructed = decoder(hub_output)

            # Reconstruction loss
            loss = F.mse_loss(reconstructed, tool_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    return adapter  # Decoder discarded after training
```

### 8.3 Complete Forward Pass

```python
class ToolMed(nn.Module):
    """
    Complete ToolMed architecture.
    """

    def __init__(self, encoder, internal_tools, fusion_hub, llm, external_tools):
        super().__init__()
        self.encoder = encoder
        self.internal_tools = nn.ModuleDict(internal_tools)
        self.adapters = nn.ModuleDict()  # One adapter per tool
        self.fusion_hub = fusion_hub
        self.llm = llm
        self.external_tools = external_tools

    def forward(self, image, max_external_calls=3):
        # Step 1: Encode image
        features = self.encoder(image)

        # Step 2: Run all internal tools
        tool_outputs = []
        texts = []

        for name, tool in self.internal_tools.items():
            output = tool(features)

            # Apply adapter to project to fusion hub language
            if name in self.adapters:
                output.embedding = self.adapters[name](output.embedding)

            tool_outputs.append(output)

            if output.text:
                texts.append(f"[{name}] {output.text}")

        # Step 3: Fusion hub combines all tool outputs
        fused_tokens = self.fusion_hub(tool_outputs)

        # Step 4: Check if external tools needed
        uncertainty = self.get_uncertainty(tool_outputs)

        external_calls = 0
        while uncertainty > 0.5 and external_calls < max_external_calls:
            # LLM decides which external tool to call
            tool_to_call = self.llm.decide_tool(fused_tokens, self.external_tools)

            if tool_to_call:
                external_output = self.external_tools[tool_to_call](image)
                texts.append(f"[{tool_to_call}] {external_output.text}")

                # Update fused representation
                fused_tokens = self.fusion_hub.update(fused_tokens, external_output)

            external_calls += 1
            uncertainty = self.get_uncertainty(tool_outputs)

        # Step 5: LLM generates report
        text_context = "\n".join(texts)
        report = self.llm.generate(fused_tokens, text_context)

        return report, {
            "tool_outputs": tool_outputs,
            "texts": texts,
            "external_calls": external_calls,
        }
```

---

## 9. Training Strategy

### 9.1 Three-Phase Training

```
Phase 1: Train Core System (Once)
──────────────────────────────────
- Train encoder (or use pretrained)
- Train internal tools with auxiliary supervision
- Train fusion hub to combine tool outputs
- Train/finetune LLM for report generation

Duration: Days to weeks
Do once, then freeze.


Phase 2: Add New Tools (As Needed)
──────────────────────────────────
For each new tool:
- Train adapter using reconstruction loss
- Only adapter trained, everything else frozen

Duration: Minutes to hours per tool
Repeat as needed.


Phase 3: End-to-End Finetuning (Optional)
─────────────────────────────────────────
- Unfreeze adapters
- Finetune on downstream task
- Keep fusion hub and LLM frozen (or LoRA)

Duration: Hours
For performance boost.
```

### 9.2 Loss Functions

```python
# Phase 1: Core system training
loss = (
    # Main task
    report_generation_loss +

    # Internal tool supervision
    0.1 * organ_segmentation_loss +      # From TotalSegmentator pseudo-labels
    0.1 * anomaly_detection_loss +       # From report NLP extraction
    0.1 * size_estimation_loss +         # From report NLP extraction
    0.1 * texture_classification_loss +  # From report keywords

    # Uncertainty calibration
    0.05 * uncertainty_calibration_loss
)

# Phase 2: Adapter training
adapter_loss = reconstruction_loss  # MSE(original, reconstructed)
```

---

## 10. Evaluation Plan

### 10.1 Metrics

```
Performance Metrics:
- BLEU, ROUGE, F1 for report generation
- Detection accuracy for internal tools
- Segmentation IoU for organ routing

Interpretability Metrics:
- Tool attribution accuracy (which tool found which finding)
- Attention alignment with radiologist annotations
- Explanation quality (human evaluation)

Modularity Metrics:
- Time to add new tool
- Performance retention after adding tool
- Zero-shot performance of new tool
```

### 10.2 Ablation Studies

```
1. Internal tools only vs External tools only vs Both
2. With vs without text output
3. With vs without adapter reconstruction
4. Different fusion hub architectures
5. Different numbers of internal tools
```

---

## 11. Paper Publication Strategy

### Paper 1: Architecture Paper (Main Contribution)

**Title**: "ToolMed: Tool-Augmented Vision-Language Models for Interpretable Medical Image Analysis"

**Venue**: CVPR / ICCV / MICCAI

**Key Contributions**:
1. Internal + External tool architecture
2. Adapter + reconstruction for modularity
3. Fusion hub for multi-tool integration
4. Full interpretability by design

### Paper 2: Protocol/Framework Paper

**Title**: "OpenMedVL: An Open Protocol for Composable Medical Vision-Language Systems"

**Venue**: NeurIPS / Nature Methods

**Key Contributions**:
1. Protocol specification for medical VLM components
2. Component library with interchangeable parts
3. Composition engine for automatic model building

### Paper 3: Clinical Interpretability Paper

**Title**: "From Black Box to Glass Box: Explainable AI for Radiology through Tool Decomposition"

**Venue**: Radiology / Nature Medicine

**Key Contributions**:
1. Clinical interpretability framework
2. User study with radiologists
3. Trust and adoption metrics

### Paper 4: Analysis Paper

**Title**: "What Do Medical Vision-Language Models Actually See? A Tool-Based Analysis"

**Venue**: ICLR / Medical Image Analysis

**Key Contributions**:
1. Systematic analysis via tool decomposition
2. Failure mode analysis per tool
3. Recommendations for improvement

---

## 12. Novelty Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    NOVELTY MATRIX                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                  Internal  External  Modular  Interpretable     │
│                   Tools     Tools    Adding    by Design        │
│                  ────────  ────────  ────────  ─────────────    │
│ Med-VLP             ✗         ✗         ✗          ✗            │
│ RadFM               ✗         ✗         ✗          ✗            │
│ LLaVA-Med           ✗         ✗         ✗          ✗            │
│ M3D                 ✗         ✗         ✗          ✗            │
│ GPT-4V (medical)    ✗         ~         ✗          ✗            │
│                                                                 │
│ ToolMed (Ours)      ✓         ✓         ✓          ✓            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13. Open Questions & Future Work

### 13.1 Open Questions

1. **Optimal number of internal tools?** Too few = not enough capability, too many = complexity
2. **When to call external tools?** Uncertainty threshold tuning
3. **Text vs embedding tradeoff?** How much to rely on each
4. **Training data for internal tools?** Bootstrapping from existing tools

### 13.2 Future Directions

1. **Multi-modal extension**: X-ray, MRI, ultrasound
2. **Temporal analysis**: Prior comparison as internal tool
3. **Interactive refinement**: User feedback to improve tools
4. **Federated learning**: Train tools across institutions

---

## 14. Timeline

```
Month 1-2:   Implement core architecture
             - Internal tools layer
             - Fusion hub
             - Basic integration

Month 3-4:   Train and evaluate
             - Phase 1 training
             - Ablation studies
             - Performance optimization

Month 5:     Add modularity
             - Adapter + reconstruction
             - External tools integration
             - Modularity experiments

Month 6:     Paper writing
             - Paper 1 (Architecture)
             - Experiments finalization
             - Submission preparation
```

---

## 15. References & Related Work

### Medical VLMs
- RadFM, LLaVA-Med, Med-Flamingo, MedVInT

### Tool-Augmented LLMs
- Toolformer, Gorilla, ToolLLM

### Modular Networks
- Neural Module Networks, Routing Networks

### Medical Image Analysis
- TotalSegmentator, SAM-Med3D, CT-CLIP

---

**Document Version**: v1.0
**Created**: 2025-12-28
**Last Updated**: 2025-12-28

---

## Appendix A: Terminology

| Term | Definition |
|------|------------|
| Internal Tool | Differentiable module built into the model architecture |
| External Tool | Separate pretrained model called on-demand |
| Fusion Hub | Module that combines outputs from multiple tools |
| Adapter | Small network that projects tool output to fusion hub language |
| Tool Protocol | Standardized interface that all tools must follow |
| SIR | Structured Intermediate Representation |

## Appendix B: Tool Types

```python
class ToolType(Enum):
    ENCODER = 0      # Visual feature extraction
    SEGMENTOR = 1    # Region/organ segmentation
    DETECTOR = 2     # Anomaly/finding detection
    CLASSIFIER = 3   # Category prediction
    REGRESSOR = 4    # Measurement prediction
    DESCRIBER = 5    # Texture/appearance description
```

---

## Appendix C: Detailed Internal Tool Implementations

### C.1 Organ Router (Internal Tool)

```python
class OrganRouter(nn.Module):
    """
    Routes attention to different organs.
    Learns to focus on relevant anatomical regions.

    Input: ViT features [B, N, D]
    Output: Per-organ attention weights and features
    """

    def __init__(self, vit_dim=768, num_organs=10, num_queries_per_organ=4):
        super().__init__()
        self.tool_type = ToolType.SEGMENTOR
        self.num_organs = num_organs

        # Organ names for text generation
        self.organ_names = [
            "lung", "heart", "liver", "spleen", "kidney",
            "pancreas", "stomach", "esophagus", "trachea", "spine"
        ]

        # Learnable queries for each organ
        self.organ_queries = nn.Parameter(
            torch.randn(num_organs, num_queries_per_organ, vit_dim)
        )

        # Cross-attention to extract organ-specific features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=vit_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Organ presence classifier
        self.organ_classifier = nn.Sequential(
            nn.Linear(vit_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Project to standard output dim
        self.projector = nn.Linear(vit_dim, 256)

    def forward(self, vit_features):
        """
        Args:
            vit_features: [B, N, D] features from ViT

        Returns:
            ToolOutput with organ attention and features
        """
        B, N, D = vit_features.shape

        organ_features = []
        organ_attentions = []
        organ_presence = []

        for organ_idx in range(self.num_organs):
            # Get queries for this organ
            queries = self.organ_queries[organ_idx].unsqueeze(0).expand(B, -1, -1)

            # Cross-attend to ViT features
            organ_feat, attn_weights = self.cross_attention(
                query=queries,
                key=vit_features,
                value=vit_features,
                need_weights=True
            )

            # Pool organ features
            pooled = organ_feat.mean(dim=1)  # [B, D]

            # Classify if organ is present
            presence = self.organ_classifier(pooled)  # [B, 1]

            organ_features.append(pooled)
            organ_attentions.append(attn_weights)
            organ_presence.append(presence)

        # Stack results
        organ_features = torch.stack(organ_features, dim=1)  # [B, num_organs, D]
        organ_presence = torch.cat(organ_presence, dim=1)    # [B, num_organs]

        # Project to standard dim
        embedding = self.projector(organ_features.mean(dim=1))  # [B, 256]

        # Generate structured output
        structured = {}
        for i, name in enumerate(self.organ_names):
            if organ_presence[0, i] > 0.5:
                structured[name] = {
                    "present": True,
                    "confidence": organ_presence[0, i].item(),
                    "attention": organ_attentions[i][0].detach().cpu()
                }

        # Generate text
        present_organs = [n for i, n in enumerate(self.organ_names)
                        if organ_presence[0, i] > 0.5]
        text = f"Identified organs: {', '.join(present_organs)}"

        return ToolOutput(
            embedding=embedding,
            structured=structured,
            text=text,
            confidence=organ_presence.mean().item(),
            tool_type=self.tool_type
        )
```

### C.2 Anomaly Detector (Internal Tool)

```python
class AnomalyDetector(nn.Module):
    """
    Detects abnormal regions in the image.
    Uses learned normal distribution to identify outliers.

    Input: ViT features [B, N, D]
    Output: Per-token anomaly scores and overall anomaly assessment
    """

    def __init__(self, vit_dim=768, hidden_dim=256):
        super().__init__()
        self.tool_type = ToolType.DETECTOR

        # Anomaly scoring network
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Spatial aggregation for finding localization
        self.spatial_attention = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        # Feature extractor for anomalous regions
        self.anomaly_feature_extractor = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256)  # Standard output dim
        )

        # Finding classifier
        self.finding_classifier = nn.Linear(256, 10)
        self.finding_names = [
            "nodule", "mass", "consolidation", "ground_glass", "effusion",
            "cardiomegaly", "fracture", "calcification", "cavity", "other"
        ]

    def forward(self, vit_features, organ_mask=None):
        """
        Args:
            vit_features: [B, N, D] features from ViT
            organ_mask: Optional [B, N] mask to focus on specific region

        Returns:
            ToolOutput with anomaly scores and localization
        """
        B, N, D = vit_features.shape

        # Compute per-token anomaly scores
        token_anomaly_scores = self.anomaly_scorer(vit_features)  # [B, N, 1]
        token_anomaly_scores = token_anomaly_scores.squeeze(-1)   # [B, N]

        # Apply organ mask if provided
        if organ_mask is not None:
            token_anomaly_scores = token_anomaly_scores * organ_mask

        # Find top-k anomalous tokens
        k = min(10, N)
        top_scores, top_indices = torch.topk(token_anomaly_scores, k, dim=1)

        # Extract features from anomalous regions
        top_features = torch.gather(
            vit_features, 1,
            top_indices.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, k, D]

        # Aggregate anomaly features
        anomaly_features = self.anomaly_feature_extractor(
            top_features.mean(dim=1)
        )  # [B, 256]

        # Classify finding type
        finding_logits = self.finding_classifier(anomaly_features)
        finding_probs = F.softmax(finding_logits, dim=-1)
        finding_idx = finding_probs.argmax(dim=-1)

        # Overall anomaly score
        overall_score = token_anomaly_scores.max(dim=1)[0]  # [B]

        # Generate structured output
        structured = {
            "overall_anomaly_score": overall_score[0].item(),
            "max_anomaly_locations": top_indices[0].tolist(),
            "finding_type": self.finding_names[finding_idx[0]],
            "finding_confidence": finding_probs[0, finding_idx[0]].item(),
            "per_token_scores": token_anomaly_scores[0].detach().cpu()
        }

        # Generate text based on anomaly level
        score = overall_score[0].item()
        finding = self.finding_names[finding_idx[0]]

        if score > 0.8:
            text = f"HIGH anomaly detected (score: {score:.2f}). " \
                   f"Likely finding: {finding}. Recommend detailed review."
        elif score > 0.5:
            text = f"MODERATE anomaly detected (score: {score:.2f}). " \
                   f"Possible finding: {finding}."
        else:
            text = f"LOW anomaly score ({score:.2f}). Likely normal."

        return ToolOutput(
            embedding=anomaly_features,
            structured=structured,
            text=text,
            confidence=1 - abs(score - 0.5) * 2,  # Confidence in prediction
            tool_type=self.tool_type
        )
```

### C.3 Size Estimator (Internal Tool)

```python
class SizeEstimator(nn.Module):
    """
    Estimates physical size of findings in millimeters.
    Uses learned spatial calibration.

    Input: ViT features + region mask
    Output: Size measurements in 3D
    """

    def __init__(self, vit_dim=768, hidden_dim=256):
        super().__init__()
        self.tool_type = ToolType.REGRESSOR

        # Size regression network
        self.size_regressor = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # x, y, z dimensions
            nn.Softplus()  # Ensure positive values
        )

        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Project to standard output dim
        self.projector = nn.Linear(vit_dim + 3, 256)

        # Learned scale factor (mm per feature unit)
        self.scale_factor = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

    def forward(self, vit_features, region_features=None):
        """
        Args:
            vit_features: [B, N, D] features from ViT
            region_features: Optional [B, D] features from specific region

        Returns:
            ToolOutput with size measurements
        """
        # Use region features if provided, otherwise use pooled features
        if region_features is not None:
            features = region_features
        else:
            features = vit_features.mean(dim=1)  # [B, D]

        # Predict size
        raw_size = self.size_regressor(features)  # [B, 3]
        size_mm = raw_size * self.scale_factor * 10  # Scale to mm

        # Estimate confidence
        confidence = self.confidence_estimator(features)  # [B, 1]

        # Create combined features for embedding
        combined = torch.cat([features, size_mm], dim=-1)
        embedding = self.projector(combined)  # [B, 256]

        # Extract values
        x, y, z = size_mm[0].tolist()
        max_dim = max(x, y, z)

        # Generate structured output
        structured = {
            "size_x_mm": x,
            "size_y_mm": y,
            "size_z_mm": z,
            "max_dimension_mm": max_dim,
            "volume_mm3": x * y * z * 0.523,  # Approximate ellipsoid
            "confidence": confidence[0].item()
        }

        # Generate text with size category
        if max_dim < 6:
            category = "small"
        elif max_dim < 10:
            category = "medium"
        elif max_dim < 30:
            category = "large"
        else:
            category = "very large"

        text = f"Size: {max_dim:.1f}mm (max dimension). " \
               f"Dimensions: {x:.1f} × {y:.1f} × {z:.1f} mm. " \
               f"Category: {category}."

        return ToolOutput(
            embedding=embedding,
            structured=structured,
            text=text,
            confidence=confidence[0].item(),
            tool_type=self.tool_type
        )
```

### C.4 Texture Analyzer (Internal Tool)

```python
class TextureAnalyzer(nn.Module):
    """
    Analyzes texture patterns in regions.
    Classifies margins, density, and internal patterns.

    Input: ViT features from region
    Output: Texture classification and description
    """

    def __init__(self, vit_dim=768, hidden_dim=256):
        super().__init__()
        self.tool_type = ToolType.CLASSIFIER

        # Texture feature extractor with multi-scale
        self.texture_encoder = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Margin classifier
        self.margin_classifier = nn.Linear(hidden_dim, 4)
        self.margin_labels = ["smooth", "lobulated", "spiculated", "irregular"]

        # Density classifier
        self.density_classifier = nn.Linear(hidden_dim, 4)
        self.density_labels = ["solid", "part-solid", "ground-glass", "cystic"]

        # Internal pattern classifier
        self.pattern_classifier = nn.Linear(hidden_dim, 5)
        self.pattern_labels = ["homogeneous", "heterogeneous", "calcified",
                               "cavitary", "necrotic"]

        # Enhancement pattern (for contrast studies)
        self.enhancement_classifier = nn.Linear(hidden_dim, 3)
        self.enhancement_labels = ["none", "uniform", "rim"]

        # Project to standard output dim
        self.projector = nn.Linear(hidden_dim, 256)

    def forward(self, vit_features, region_mask=None):
        """
        Args:
            vit_features: [B, N, D] features from ViT
            region_mask: Optional [B, N] mask for specific region

        Returns:
            ToolOutput with texture classification
        """
        # Apply mask if provided
        if region_mask is not None:
            mask = region_mask.unsqueeze(-1)
            masked_features = vit_features * mask
            features = masked_features.sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        else:
            features = vit_features.mean(dim=1)

        # Extract texture features
        texture_feat = self.texture_encoder(features)  # [B, hidden_dim]

        # Classify all attributes
        margin_logits = self.margin_classifier(texture_feat)
        density_logits = self.density_classifier(texture_feat)
        pattern_logits = self.pattern_classifier(texture_feat)
        enhancement_logits = self.enhancement_classifier(texture_feat)

        # Get predictions
        margin_idx = margin_logits.argmax(dim=-1)
        density_idx = density_logits.argmax(dim=-1)
        pattern_idx = pattern_logits.argmax(dim=-1)
        enhancement_idx = enhancement_logits.argmax(dim=-1)

        margin = self.margin_labels[margin_idx[0]]
        density = self.density_labels[density_idx[0]]
        pattern = self.pattern_labels[pattern_idx[0]]
        enhancement = self.enhancement_labels[enhancement_idx[0]]

        # Get confidence scores
        margin_conf = F.softmax(margin_logits, dim=-1).max(dim=-1)[0]
        density_conf = F.softmax(density_logits, dim=-1).max(dim=-1)[0]

        # Project to standard output
        embedding = self.projector(texture_feat)  # [B, 256]

        # Generate structured output
        structured = {
            "margin": margin,
            "margin_confidence": margin_conf[0].item(),
            "density": density,
            "density_confidence": density_conf[0].item(),
            "internal_pattern": pattern,
            "enhancement": enhancement,
        }

        # Generate descriptive text
        text = f"Texture analysis: {margin} margins, {density} density, " \
               f"{pattern} internal pattern."

        # Add clinical significance for concerning features
        if margin == "spiculated":
            text += " Spiculated margins are concerning for malignancy."
        if density == "part-solid":
            text += " Part-solid density warrants follow-up."

        return ToolOutput(
            embedding=embedding,
            structured=structured,
            text=text,
            confidence=(margin_conf[0].item() + density_conf[0].item()) / 2,
            tool_type=self.tool_type
        )
```

### C.5 Uncertainty Estimator (Internal Tool)

```python
class UncertaintyEstimator(nn.Module):
    """
    Estimates model uncertainty to decide if external tools are needed.
    Uses ensemble disagreement and feature density estimation.

    Input: Features from all other internal tools
    Output: Uncertainty score and recommendation
    """

    def __init__(self, input_dim=256, num_tools=4, hidden_dim=128):
        super().__init__()
        self.tool_type = ToolType.REGRESSOR
        self.num_tools = num_tools

        # Per-tool uncertainty estimator
        self.tool_uncertainty = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ) for _ in range(num_tools)
        ])

        # Agreement checker (are tools consistent?)
        self.agreement_checker = nn.Sequential(
            nn.Linear(input_dim * num_tools, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # External tool recommender
        self.external_recommender = nn.Sequential(
            nn.Linear(input_dim * num_tools + num_tools, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)  # 5 external tools
        )

        self.external_tool_names = [
            "SAM-Med3D", "TotalSegmentator", "NoduleDetector",
            "PriorComparison", "SpecialistConsult"
        ]

        # Project to standard output dim
        self.projector = nn.Linear(num_tools + 1, 256)

    def forward(self, tool_outputs: List[ToolOutput]):
        """
        Args:
            tool_outputs: List of outputs from internal tools

        Returns:
            ToolOutput with uncertainty assessment
        """
        # Extract embeddings from all tools
        embeddings = [out.embedding for out in tool_outputs]

        # Compute per-tool uncertainty
        tool_uncertainties = []
        for i, (emb, estimator) in enumerate(zip(embeddings, self.tool_uncertainty)):
            unc = estimator(emb)
            tool_uncertainties.append(unc)

        tool_uncertainties = torch.cat(tool_uncertainties, dim=-1)  # [B, num_tools]

        # Check agreement between tools
        combined_emb = torch.cat(embeddings, dim=-1)  # [B, num_tools * 256]
        agreement = self.agreement_checker(combined_emb)  # [B, 1]

        # Overall uncertainty
        overall_uncertainty = tool_uncertainties.mean(dim=-1, keepdim=True)
        # Disagreement increases uncertainty
        overall_uncertainty = overall_uncertainty + (1 - agreement) * 0.3
        overall_uncertainty = overall_uncertainty.clamp(0, 1)

        # Recommend external tools if uncertain
        rec_input = torch.cat([combined_emb, tool_uncertainties], dim=-1)
        external_scores = self.external_recommender(rec_input)
        external_probs = F.softmax(external_scores, dim=-1)

        # Project to standard output
        unc_features = torch.cat([tool_uncertainties, overall_uncertainty], dim=-1)
        embedding = self.projector(unc_features)  # [B, 256]

        # Generate structured output
        unc_value = overall_uncertainty[0, 0].item()

        structured = {
            "overall_uncertainty": unc_value,
            "tool_uncertainties": {
                f"tool_{i}": tool_uncertainties[0, i].item()
                for i in range(self.num_tools)
            },
            "agreement_score": agreement[0, 0].item(),
            "recommended_external_tools": [
                self.external_tool_names[i]
                for i in range(5)
                if external_probs[0, i] > 0.3
            ]
        }

        # Generate text
        if unc_value > 0.7:
            text = f"HIGH uncertainty ({unc_value:.2f}). " \
                   f"Recommend: {', '.join(structured['recommended_external_tools'])}."
        elif unc_value > 0.4:
            text = f"MODERATE uncertainty ({unc_value:.2f}). " \
                   f"Consider additional analysis."
        else:
            text = f"LOW uncertainty ({unc_value:.2f}). " \
                   f"Internal analysis sufficient."

        return ToolOutput(
            embedding=embedding,
            structured=structured,
            text=text,
            confidence=1 - unc_value,
            tool_type=self.tool_type
        )
```

---

## Appendix D: Detailed Data Flow

### D.1 Complete Forward Pass Diagram

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                                  INPUT                                         │
│                           CT Volume [1, 1, 256, 256, 64]                        │
└────────────────────────────────────┬───────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                               ViT ENCODER                                      │
│                                                                                │
│  1. Patch Embedding:    [1, 1, 256, 256, 64] → [1, 4096, 768]                  │
│  2. Position Embedding: Add learned positions                                  │
│  3. Transformer:        12 layers self-attention                               │
│  4. Output:             [1, 4096, 768] (N=4096 tokens, D=768)                  │
│                                                                                │
└────────────────────────────────────┬───────────────────────────────────────────┘
                                     │
                                     │ vit_features [1, 4096, 768]
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│   ORGAN ROUTER    │    │  ANOMALY DETECTOR │    │  TEXTURE ANALYZER │
│                   │    │                   │    │                   │
│ Input: vit_feat   │    │ Input: vit_feat   │    │ Input: vit_feat   │
│ Output:           │    │ Output:           │    │ Output:           │
│  - organ_emb[256] │    │  - anom_emb[256]  │    │  - text_emb[256]  │
│  - organ_text     │    │  - anom_score     │    │  - margin_class   │
│  - organ_attn     │    │  - anom_text      │    │  - density_class  │
│                   │    │                   │    │  - texture_text   │
└─────────┬─────────┘    └─────────┬─────────┘    └─────────┬─────────┘
          │                        │                        │
          │                        ▼                        │
          │              ┌───────────────────┐              │
          │              │  SIZE ESTIMATOR   │              │
          │              │                   │              │
          │              │ Input: vit_feat   │              │
          │              │   + anom_region   │              │
          │              │ Output:           │              │
          │              │  - size_emb[256]  │              │
          │              │  - size_mm        │              │
          │              │  - size_text      │              │
          │              └─────────┬─────────┘              │
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │   UNCERTAINTY ESTIMATOR     │
                    │                             │
                    │ Input: all tool embeddings  │
                    │ Output:                     │
                    │  - uncertainty_score        │
                    │  - need_external_tools?     │
                    │  - recommended_tools        │
                    └──────────────┬──────────────┘
                                   │
                                   │ All tool outputs
                                   ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                              ADAPTER LAYER                                     │
│                                                                                │
│  For each tool output:                                                         │
│    tool_emb [256] → Adapter (if needed) → hub_emb [256]                        │
│                                                                                │
│  All tools now speak the same "Fusion Hub language"                            │
│                                                                                │
└────────────────────────────────────┬───────────────────────────────────────────┘
                                     │
                                     │ List of [B, 256] embeddings
                                     ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                               FUSION HUB                                       │
│                                                                                │
│  1. Add Type Embeddings:                                                       │
│     organ_emb + type_emb[SEGMENTOR]                                            │
│     anom_emb + type_emb[DETECTOR]                                              │
│     size_emb + type_emb[REGRESSOR]                                             │
│     text_emb + type_emb[CLASSIFIER]                                            │
│                                                                                │
│  2. Concatenate all: [B, num_tools, 256]                                       │
│                                                                                │
│  3. Cross-Attention with Learnable Queries:                                    │
│     queries [32, 256] × tool_outputs → fused [32, 256]                         │
│                                                                                │
│  4. Project to LLM dimension:                                                  │
│     fused [32, 256] → llm_tokens [32, 4096]                                    │
│                                                                                │
└────────────────────────────────────┬───────────────────────────────────────────┘
                                     │
                                     │ fused_tokens [1, 32, 4096]
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 │
        ┌───────────────────────┐                     │
        │  UNCERTAINTY CHECK    │                     │
        │                       │                     │
        │  if uncertainty > 0.5:│                     │
        │    → Call External    │                     │
        │      Tools            │                     │
        └───────────┬───────────┘                     │
                    │                                 │
        ┌───────────┴───────────┐                     │
        │                       │                     │
        ▼                       ▼                     │
┌───────────────┐       ┌───────────────┐             │
│  SAM-Med3D    │       │ Prior Compare │             │
│               │       │               │             │
│  Detailed     │       │  Check        │             │
│  Segmentation │       │  Stability    │             │
└───────┬───────┘       └───────┬───────┘             │
        │                       │                     │
        └───────────┬───────────┘                     │
                    │                                 │
                    │ external_outputs                │
                    │                                 │
                    ▼                                 │
        ┌───────────────────────┐                     │
        │  UPDATE FUSION        │                     │
        │                       │                     │
        │  fused_tokens +=      │                     │
        │    external_info      │                     │
        └───────────┬───────────┘                     │
                    │                                 │
                    └─────────────┬───────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                            TEXT AGGREGATION                                    │
│                                                                                │
│  Collect all text outputs:                                                     │
│                                                                                │
│  "[OrganRouter] Identified organs: lung, heart, liver"                         │
│  "[AnomalyDet] HIGH anomaly detected (0.87). Likely finding: nodule."          │
│  "[SizeEstimator] Size: 8.2mm (max dimension). Category: medium."              │
│  "[TextureAnalyzer] Texture: spiculated margins, part-solid density."          │
│  "[SAM-Med3D] Detailed segmentation confirms RUL nodule, volume 245mm³."       │
│                                                                                │
└────────────────────────────────────┬───────────────────────────────────────────┘
                                     │
                                     │ fused_tokens + text_context
                                     ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                                 LLM                                            │
│                                                                                │
│  Input:                                                                        │
│    - Visual tokens: fused_tokens [32, 4096]                                    │
│    - Text context: tool descriptions                                           │
│    - Instruction: "Generate radiology report"                                  │
│                                                                                │
│  Processing:                                                                   │
│    - Attend to visual tokens                                                   │
│    - Integrate tool descriptions                                               │
│    - Generate coherent report                                                  │
│                                                                                │
│  Output:                                                                       │
│    "FINDINGS: There is an 8.2mm part-solid nodule in the right upper          │
│     lobe with spiculated margins. The nodule demonstrates..."                  │
│                                                                                │
└────────────────────────────────────┬───────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                                OUTPUT                                          │
│                                                                                │
│  1. Report Text (for clinical use)                                             │
│  2. Tool Outputs (for interpretability)                                        │
│  3. Attention Maps (for visualization)                                         │
│  4. Uncertainty Scores (for confidence)                                        │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix E: Experimental Design Details

### E.1 Datasets

```
Primary Dataset: RadGenome-ChestCT
─────────────────────────────────────
- 1,000+ CT volumes with region-level annotations
- Reports for each anatomical region
- Ground truth: organ masks, finding locations, report text

Secondary Datasets:
─────────────────────────────────────
1. MIMIC-CXR (X-ray, for generalization)
2. CT-RATE (CT reports, large scale)
3. TotalSegmentator dataset (organ segmentation)
4. LIDC-IDRI (lung nodule detection)

Dataset Splits:
─────────────────────────────────────
- Training: 70%
- Validation: 15%
- Test: 15%
```

### E.2 Evaluation Metrics

```
A. Report Generation Quality
──────────────────────────────────────────────────────
Metric          | Description                    | Target
─────────────────────────────────────────────────────
BLEU-4          | N-gram overlap                 | > 0.15
ROUGE-L         | Longest common subsequence     | > 0.30
F1-RadGraph     | Clinical entity extraction     | > 0.35
BERTScore       | Semantic similarity            | > 0.85
Clinical Acc.   | Key finding accuracy           | > 0.80


B. Internal Tool Performance
──────────────────────────────────────────────────────
Tool            | Metric                | Target
─────────────────────────────────────────────────────
OrganRouter     | Dice Score            | > 0.85
AnomalyDetector | AUROC                 | > 0.80
SizeEstimator   | MAE (mm)              | < 2.0
TextureAnalyzer | Accuracy              | > 0.75
UncertaintyEst  | ECE (calibration)     | < 0.10


C. Modularity Metrics
──────────────────────────────────────────────────────
Metric                          | Target
─────────────────────────────────────────────────────
Time to add new tool            | < 1 hour
Adapter training epochs         | < 10
Performance retention           | > 95%
Zero-shot new tool performance  | > 60% of finetuned


D. Interpretability Metrics
──────────────────────────────────────────────────────
Metric                          | Method
─────────────────────────────────────────────────────
Tool Attribution Accuracy       | Which tool found which finding
Attention-GT Alignment          | IoU with radiologist annotations
Explanation Quality             | Human evaluation (1-5 scale)
Failure Traceability            | Can identify failing tool
```

### E.3 Ablation Studies

```
Ablation 1: Number of Internal Tools
────────────────────────────────────────
Configurations:
- 2 tools (OrganRouter + AnomalyDet)
- 4 tools (+ Size + Texture)
- 6 tools (+ Uncertainty + Density)

Hypothesis: More tools = better performance but diminishing returns


Ablation 2: With vs Without External Tools
────────────────────────────────────────
Configurations:
- Internal only
- External only (no internal tools)
- Both (full system)

Hypothesis: Both > Internal only > External only


Ablation 3: Fusion Hub Architecture
────────────────────────────────────────
Configurations:
- Simple concatenation
- Cross-attention (ours)
- Transformer layers
- Perceiver

Hypothesis: Cross-attention best balance of performance/efficiency


Ablation 4: Adapter vs Direct Connection
────────────────────────────────────────
Configurations:
- No adapter (direct tool output)
- Linear adapter
- MLP adapter (ours)
- Transformer adapter

Hypothesis: MLP adapter best for modularity


Ablation 5: Text Output Importance
────────────────────────────────────────
Configurations:
- No text (embedding only)
- Template text (ours)
- Generated text (small LM)

Hypothesis: Template text sufficient, generated text marginal gain
```

### E.4 Baseline Comparisons

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BASELINE COMPARISON                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Model               │ Architecture      │ Interpretable │ Modular         │
│  ──────────────────────────────────────────────────────────────────────    │
│  LLaVA-Med           │ CLIP + Vicuna     │ ✗             │ ✗               │
│  RadFM               │ ViT + LLaMA       │ ✗             │ ✗               │
│  Med-Flamingo        │ Flamingo-based    │ ✗             │ ✗               │
│  MedVInT             │ ViT + GPT-2       │ ✗             │ ✗               │
│  M3D                 │ 3D ViT + LLM      │ ✗             │ ✗               │
│  ──────────────────────────────────────────────────────────────────────    │
│  ToolMed (Ours)      │ Tools + Fusion    │ ✓             │ ✓               │
│                                                                             │
│                                                                             │
│  Expected Results:                                                          │
│  ───────────────────                                                        │
│  • Performance: Comparable or slightly better than baselines                │
│  • Interpretability: Significantly better (tool outputs visible)            │
│  • Modularity: Only ours supports adding tools without retraining           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix F: Edge Cases and Failure Handling

### F.1 Handling Missing Organs

```python
def handle_missing_organ(organ_router_output, expected_organs):
    """
    Handle cases where expected organs are not detected.
    """
    detected = set(organ_router_output.structured.keys())
    expected = set(expected_organs)

    missing = expected - detected

    if missing:
        # Option 1: Use default region (whole image)
        # Option 2: Call external segmentation tool
        # Option 3: Report with uncertainty

        return {
            "status": "incomplete",
            "missing_organs": list(missing),
            "action": "call_external_segmentor",
            "fallback": "use_whole_image_analysis"
        }

    return {"status": "complete"}
```

### F.2 Handling Tool Disagreement

```python
def handle_tool_disagreement(tool_outputs):
    """
    Handle cases where tools give conflicting information.
    """
    # Example: AnomalyDetector says abnormal, TextureAnalyzer says normal

    anomaly_score = tool_outputs['anomaly'].structured['overall_anomaly_score']
    texture_concern = tool_outputs['texture'].structured['margin'] in ['spiculated', 'irregular']

    if anomaly_score > 0.7 and not texture_concern:
        # Disagreement detected
        return {
            "status": "disagreement",
            "conflicting_tools": ["AnomalyDetector", "TextureAnalyzer"],
            "action": "increase_uncertainty",
            "recommendation": "call_external_specialist"
        }

    return {"status": "agreement"}
```

### F.3 Handling External Tool Failure

```python
def handle_external_tool_failure(tool_name, error):
    """
    Graceful degradation when external tool fails.
    """
    fallback_strategies = {
        "SAM-Med3D": "use_internal_organ_router",
        "TotalSegmentator": "use_internal_organ_router",
        "NoduleDetector": "rely_on_internal_anomaly_detector",
        "PriorComparison": "skip_temporal_analysis",
    }

    return {
        "failed_tool": tool_name,
        "error": str(error),
        "fallback": fallback_strategies.get(tool_name, "continue_without"),
        "uncertainty_increase": 0.2
    }
```

---

## Appendix G: Computational Requirements

### G.1 Model Size Estimates

```
Component                    │ Parameters    │ Memory (fp16)
────────────────────────────────────────────────────────────
ViT Encoder (CT-CLIP)        │ 300M          │ 600MB
Internal Tools (5 tools)     │ 50M           │ 100MB
Fusion Hub                   │ 20M           │ 40MB
Adapters (5 adapters)        │ 5M            │ 10MB
LLM (LLaMA-7B)               │ 7B            │ 14GB
────────────────────────────────────────────────────────────
Total                        │ ~7.4B         │ ~15GB
```

### G.2 Training Requirements

```
Phase 1: Core System Training
────────────────────────────────
GPU: 4× A100 (80GB)
Batch Size: 4 per GPU
Training Time: ~3 days
Data: 10,000 CT volumes

Phase 2: Adapter Training (per tool)
────────────────────────────────
GPU: 1× A100 (80GB)
Batch Size: 16
Training Time: ~30 minutes
Data: 1,000 samples (tool outputs)

Inference
────────────────────────────────
GPU: 1× A100 (40GB) or 2× RTX 4090
Batch Size: 1 (clinical use)
Latency: ~2 seconds per CT
```

### G.3 Optimization Strategies

```
1. Gradient Checkpointing
   - Reduce memory for ViT encoder
   - Trade compute for memory

2. Mixed Precision (fp16/bf16)
   - 50% memory reduction
   - Faster training

3. LoRA for LLM
   - Only train 0.1% of LLM parameters
   - Significant memory savings

4. Tool Parallelization
   - Run internal tools in parallel
   - Reduce latency

5. External Tool Caching
   - Cache segmentation results
   - Avoid redundant computation
```

---

## Appendix H: Clinical Integration Considerations

### H.1 DICOM Integration

```python
class DICOMInterface:
    """
    Interface for clinical PACS integration.
    """

    def load_from_dicom(self, dicom_path):
        """Load CT from DICOM format."""
        # Read DICOM series
        # Extract pixel data
        # Apply windowing
        # Normalize to model input range
        pass

    def export_to_dicom_sr(self, report, tool_outputs):
        """
        Export report as DICOM Structured Report.
        Includes tool outputs as coded entries.
        """
        # Create SR document
        # Add finding codes (RadLex)
        # Add measurements
        # Add confidence scores
        pass
```

### H.2 HL7 FHIR Output

```json
{
  "resourceType": "DiagnosticReport",
  "status": "final",
  "code": {
    "coding": [{
      "system": "http://loinc.org",
      "code": "24331-1",
      "display": "CT Chest"
    }]
  },
  "conclusion": "8.2mm spiculated nodule in RUL...",
  "extension": [{
    "url": "http://toolmed.ai/tool-outputs",
    "valueString": {
      "organ_router": {"organs": ["lung", "heart"]},
      "anomaly_detector": {"score": 0.87},
      "size_estimator": {"size_mm": 8.2},
      "texture_analyzer": {"margin": "spiculated"},
      "uncertainty": {"score": 0.23}
    }
  }]
}
```

### H.3 Audit Trail

```python
@dataclass
class AuditEntry:
    """Complete audit trail for clinical use."""

    timestamp: datetime
    input_hash: str           # Hash of input CT
    model_version: str        # ToolMed version

    # Tool-level audit
    tools_executed: List[str]
    tool_outputs: Dict[str, Any]
    tool_confidences: Dict[str, float]

    # External tools
    external_tools_called: List[str]
    external_tool_reasons: List[str]

    # Final output
    report_text: str
    overall_confidence: float

    # Interpretability
    attention_maps: Dict[str, np.ndarray]
    decision_explanation: str
```

---

## Appendix I: Future Research Directions

### I.1 Multi-Modal Extension

```
Current: CT only

Future:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐                │
│  │  CT   │  │ X-ray │  │  MRI  │  │ US    │                │
│  └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘                │
│      │          │          │          │                     │
│      └──────────┴──────────┴──────────┘                     │
│                      │                                      │
│                      ▼                                      │
│             Modality-Specific Encoders                      │
│                      │                                      │
│                      ▼                                      │
│             Shared Internal Tools                           │
│             (work across modalities)                        │
│                      │                                      │
│                      ▼                                      │
│                 Fusion Hub                                  │
│                      │                                      │
│                      ▼                                      │
│                    LLM                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### I.2 Temporal Analysis Tool

```python
class TemporalAnalysisTool(nn.Module):
    """
    Compare current scan with prior studies.
    Track lesion stability, growth, new findings.
    """

    def forward(self, current_features, prior_features, time_delta):
        # Register scans
        # Compare findings
        # Compute growth rates
        # Assess stability

        return ToolOutput(
            embedding=...,
            structured={
                "new_findings": [...],
                "resolved_findings": [...],
                "changed_findings": [
                    {"id": 1, "growth_mm": 2.3, "growth_rate": "rapid"}
                ],
                "stable_findings": [...]
            },
            text="Comparison with prior: Nodule increased from 6mm to 8mm..."
        )
```

### I.3 Interactive Refinement

```
Radiologist: "Focus more on the left lower lobe"
     │
     ▼
System: Rerun OrganRouter with attention bias to LLL
     │
     ▼
Updated analysis with LLL focus
     │
     ▼
Radiologist: "Measure that lesion more carefully"
     │
     ▼
System: Call external high-precision measurement tool
     │
     ▼
Refined measurement with uncertainty bounds
```

---

**Document Version**: v2.0
**Created**: 2025-12-28
**Last Updated**: 2025-12-28
**Major Additions**:
- Appendix C: Detailed internal tool implementations
- Appendix D: Complete data flow diagram
- Appendix E: Experimental design details
- Appendix F: Edge cases and failure handling
- Appendix G: Computational requirements
- Appendix H: Clinical integration
- Appendix I: Future research directions
