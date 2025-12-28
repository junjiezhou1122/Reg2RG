# Project Context

## Purpose

**Reg2RG** (Region-Guided Referring and Grounding) is a multimodal AI system for generating comprehensive clinical radiology reports from 3D CT scans.

**Core Goals:**
- Process 3D CT volumes using a Vision Transformer encoder
- Extract region-specific anatomical information from 10 organ/body regions
- Generate structured, clinically-accurate medical reports using LLaMA-2-7b
- Combine global image understanding with region-level grounding for detailed analysis

**Publication:** IEEE Transactions on Medical Imaging, 2025
**Citation:** Chen et al., "Large Language Model with Region-Guided Referring and Grounding for CT Report Generation"

## Tech Stack

### Core ML/Deep Learning
- **Python**: 3.9
- **PyTorch**: 2.0.1 (neural network operations)
- **PyTorch Lightning**: 1.9.0 (training orchestration)
- **Transformers**: 4.28.1 (LLaMA-2-7b-chat-hf)
- **PEFT**: 0.7.1 (LoRA for parameter-efficient fine-tuning)

### Medical Imaging
- **MONAI**: 1.3.0 (medical image transforms, caching, preprocessing)
- **nibabel**: 5.1.0 (NIfTI file I/O)
- **SimpleITK**: 2.2.1 (image processing)

### Infrastructure
- **DeepSpeed**: 0.12.6 (distributed training)
- **Weights & Biases**: 0.16.1 (experiment tracking)
- **einops**: Tensor rearrangement utilities

### Evaluation
- **rouge_score**: ROUGE metrics for text generation
- **RaTEScore**: Medical text evaluation
- **pycocotools**: COCO evaluation metrics

## Project Conventions

### Code Style

**File Naming:**
- Snake_case for Python files: `radgenome_dataset_train.py`, `lit_recon_probe.py`
- CamelCase for classes: `Reg2RG`, `MyEmbedding`, `PerceiverResampler`

**Variable Conventions:**
- Tensor suffixes: `vision_x`, `mask_x`, `lang_x`
- Batch dimensions: `B` (batch), `S` (sequence), `C` (channel), `H/W/D` (spatial)
- Region keys as strings: `'lung'`, `'heart'`, `'abdomen'`

**Tensor Operations:**
- Use `einops.rearrange()` with explicit patterns: `"b S c h w d -> (b S) c h w d"`
- Include shape comments in forward passes

**Imports:**
- Standard library → third-party → local modules
- Relative imports within package: `from Model.Reg2RG import Reg2RG`

### Architecture Patterns

**Model Pipeline:**
```
CT Volume (B, S, C, H, W, D)
    ↓
[ViT-3D Encoder]  ← Frozen, pretrained from RadFM
    ↓
[Region-Specific Processing]  ← Same encoder for each anatomical region
    ↓
[Perceiver Resampler]  ← Compress 768-dim → 32 latent tokens
    ↓
[Projection Layer]  ← 768-dim → 4096-dim LLM space
    ↓
[LLaMA-2 7B + LoRA]  ← Generate report tokens
    ↓
Clinical Report (text)
```

**Configuration:**
- Dataclasses for argument groups (`ModelArguments`, `DataArguments`, `TrainingArguments`)
- Shell scripts in `configs/` for experiment configurations
- DeepSpeed configs in `ds_configs/`

**Data Pipeline:**
- MONAI `PersistentDataset` for automatic caching of expensive transforms
- `RobustPersistentDataset` with atomic writes for crash safety
- Graceful handling of corrupted cache files

### Testing Strategy

**Current Approach:**
- Integration tests via experiment scripts (`scripts/run_exp1*.sh`)
- Cache diagnostics (`scripts/diagnose_cache_hash.py`)
- Manual validation through inference on test splits

**Evaluation Pipeline:**
- `evaluation/hf_nlg_evaluation.py` - Whole-report NLG metrics (BLEU, ROUGE)
- `evaluation/hf_nlg_evaluation_region.py` - Region-level metrics
- `evaluation/region_pred_acc.py` - Region prediction accuracy
- `evaluation/ce_evaluator_ct2rep/` - Clinical efficacy evaluation

### Git Workflow

**Commit Convention:** Conventional commits
- `feat(scope): description` - New features
- `fix(scope): description` - Bug fixes
- `docs(scope): description` - Documentation changes

**Branch Strategy:**
- `main` branch for stable code
- Feature development on topic branches

## Domain Context

### Anatomical Regions (10 total)
```python
REGIONS = [
    'abdomen', 'bone', 'breast', 'esophagus', 'heart',
    'lung', 'mediastinum', 'pleura', 'thyroid', 'trachea and bronchie'
]
```

Each region has:
- A binary segmentation mask
- Separate ViT encoding
- Individual embeddings combined with global image embedding
- Region-specific evaluation metrics

### Model Components
- **Vision Encoder**: ViT(image_size=512, frames=512, patch=32, depth=12, dim=768)
- **Mask Encoder**: ViT(image_size=256, frames=64, patch=32, depth=3, dim=255)
- **Perceiver Resampler**: 6-layer cross-attention (compresses to 32 latents)
- **LLM**: LLaMA-2-7b-chat-hf with LoRA (r=8, α=32, dropout=0.1)

### Current Research Focus
1. **Adapter Experiments (Exp5)**: Single-layer minimal-capacity adapters
2. **Reconstruction Probing (LIT)**: Information preservation analysis
3. **Joint Training**: Adapter + decoder co-training modes
4. **Information Flow**: Analyzing what gets preserved through compression

## Important Constraints

### Technical Constraints
- **GPU Memory**: Large 3D volumes require efficient batching and gradient checkpointing
- **Cache Integrity**: Distributed training requires atomic cache writes to prevent corruption
- **Frozen Encoder**: Vision encoder is pretrained (RadFM) and kept frozen during training
- **LoRA Only**: LLM updates limited to low-rank adapters (not full fine-tuning)

### Data Constraints
- **Dataset**: RadGenome-ChestCT from Hugging Face Hub
- **Input Format**: NIfTI volumes with corresponding segmentation masks
- **Preprocessing**: HU clamping, foreground cropping, resize to [256, 256, 64]

### Evaluation Constraints
- Reports must be evaluated at both whole-report and region-level granularity
- Clinical efficacy requires medical entity extraction and accuracy assessment

## External Dependencies

### Pretrained Models
- **LLM**: `meta-llama/Llama-2-7b-chat-hf` (Hugging Face Hub)
- **Vision Encoder**: `RadFM_vit3d.pth` (3D ViT from RadFM project)
- **Adapter Weights**: `RadFM_perceiver_fc.pth` (Perceiver + projection)
- **Full Reg2RG**: `Trusure/Reg2RG` (Hugging Face Hub)

### Dataset
- **RadGenome-ChestCT**: Hugging Face Hub dataset with region-level annotations

### Infrastructure
- **Weights & Biases**: Experiment tracking and visualization
- **Hugging Face Hub**: Model and dataset hosting
