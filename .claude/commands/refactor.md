---
description: Perform code refactoring for Reg2RG medical imaging codebase. Use when restructuring code, reducing complexity, fixing naming, extracting functions, or eliminating duplicates.
allowed-tools:
  - Read
  - Edit
  - Write
  - Grep
  - Glob
  - Bash
---

# Reg2RG Code Refactoring

Systematic code refactoring tailored to this medical imaging codebase.

## User Request

$ARGUMENTS

---

## Codebase Context

### Architecture Overview
```
src/
├── lit_recon_probe.py      # Main training (2,500 LOC) ⚠️ needs decomposition
├── Dataset/
│   └── radgenome_dataset_train.py  # Data loading (571 LOC)
└── Model/
    ├── vit_3d.py           # 3D Vision Transformer (frozen encoder)
    ├── helpers.py          # PerceiverResampler (6-layer adapter)
    ├── one_layer_adapter.py # OneLayerAdapter (Exp5 minimal)
    └── adapter_utils.py    # Weight loading utilities
```

### Experiment Variants
| Exp | Mode | Components |
|-----|------|------------|
| 1 | Decoder only | `adapter` frozen, `decoder` trained |
| 2 | Joint | `adapter` + `decoder` trained |
| 5 | Minimal | 1-layer adapter |
| 9 | Separate | `global_adapter` + `local_adapter` + 2 decoders |

---

## Refactoring Rules (Project-Specific)

### 1. Function Size Limits

**Rule:** Functions > 50 lines should be decomposed.

**Priority Targets:**
| Function | Lines | Action |
|----------|-------|--------|
| `main()` | 1,200+ | Extract: `setup_device()`, `load_datasets()`, `initialize_model()`, `setup_optimizer()`, `training_loop()` |
| `run_epoch()` | 400+ | Extract: `process_batch()`, `compute_region_metrics()`, `compute_correlation_analysis()` |
| `prepare_samples()` | 60 | Extract: `_build_single_sample()` |

**Template for extraction:**
```python
# Before (in main):
# 50 lines of device setup...

# After:
def setup_device(train_args: TrainArguments) -> torch.device:
    """Initialize CUDA device and set random seeds."""
    if train_args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = train_args.cuda_visible_devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_args.seed)
    return device
```

---

### 2. Naming Conventions

**Variables:**
| Bad | Good | Reason |
|-----|------|--------|
| `x_g`, `z_g` | `tokens_global`, `latent_global` | Single letters unclear |
| `rname`, `rs` | `region_name`, `region_stats` | Avoid abbreviations |
| `vision_xs` | `vision_batch` | Use `_batch` suffix for collections |
| `key, vol` | `region_name, region_volume` | Descriptive loop vars |

**Functions:**
| Pattern | Convention | Example |
|---------|------------|---------|
| Actions | `verb_noun()` | `run_epoch()`, `compute_loss()` |
| Getters | `get_noun()` | `get_active_adapter()` |
| Predicates | `is_/should_/has_` | `is_improvement()`, `should_validate()` |
| Private | `_prefix` | `_build_single_sample()` |

**Classes:**
- Models: `LITProbeModel`, `ProbeDecoder`
- Data: `RadGenomeDataset`, `LITDataCollator`
- Config: `ModelArguments`, `TrainArguments`

---

### 3. Duplicate Code Patterns

**Pattern A: Adapter Selection (4+ locations)**
```python
# BEFORE (repeated):
if self.separate_adapters:
    adapter = self.global_adapter if is_global else self.local_adapter
else:
    adapter = self.adapter

# AFTER (extract method):
def get_adapter(self, is_global: bool = True) -> nn.Module:
    """Return appropriate adapter based on experiment mode."""
    if self.separate_adapters:
        return self.global_adapter if is_global else self.local_adapter
    return self.adapter
```

**Pattern B: Compression Ratio Calculation (2 files)**
```python
# BEFORE (in lit_recon_probe.py AND radgenome_dataset_train.py):
bbox_h, bbox_w, bbox_d = h1 - h0 + 1, w1 - w0 + 1, d1 - d0 + 1
original_voxels = bbox_h * bbox_w * bbox_d
compression_ratio = original_voxels / target_voxels

# AFTER (create src/utils/metrics.py):
def compute_compression_ratio(
    bbox: Tuple[int, int, int, int, int, int],
    target_size: Tuple[int, int, int] = (256, 256, 64)
) -> float:
    """Calculate volume compression ratio from bounding box."""
    h0, h1, w0, w1, d0, d1 = bbox
    original = (h1 - h0 + 1) * (w1 - w0 + 1) * (d1 - d0 + 1)
    target = target_size[0] * target_size[1] * target_size[2]
    return original / target
```

**Pattern C: Metric Accumulation**
```python
# BEFORE (repeated dict pattern):
agg = {"loss": 0.0, "mse": 0.0, "cos": 0.0, "count": 0}
# ... accumulate ...
final = {k: v / agg["count"] for k, v in agg.items() if k != "count"}

# AFTER (create class):
class MetricsAccumulator:
    def __init__(self, keys: List[str]):
        self.sums = {k: 0.0 for k in keys}
        self.count = 0

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self.sums[k] += v
        self.count += 1

    def average(self) -> Dict[str, float]:
        return {k: v / self.count for k, v in self.sums.items()}
```

---

### 4. Magic Numbers → Constants

**Create `src/config/constants.py`:**
```python
"""Project-wide constants for Reg2RG."""

class DataConfig:
    """Data preprocessing constants."""
    HU_MIN = -1000
    HU_MAX = 200
    REGION_SIZE = (256, 256, 64)  # Target region volume
    CACHE_VERSION = "lit_img_hu_seg_v1"

class ModelConfig:
    """Model architecture constants."""
    VIT_IMAGE_SIZE = 512
    VIT_FRAMES = 512
    VIT_DEPTH = 12
    PATCH_SIZE = 32
    FRAME_PATCH_SIZE = 4
    VIS_DIM = 768
    LLM_DIM = 4096
    DEFAULT_LATENTS = 32

class TrainingConfig:
    """Training defaults."""
    DEFAULT_LR = 1e-4
    DEFAULT_WEIGHT_DECAY = 0.01
    DEFAULT_GRAD_ACCUM = 8

# Anatomical regions for chest CT
REGIONS = [
    "abdomen", "bone", "breast", "esophagus", "heart",
    "lung", "mediastinum", "pleura", "thyroid", "trachea and bronchie"
]
```

---

### 5. Reduce Nesting (Exit Early)

**Before (4 levels):**
```python
for step, batch in enumerate(loader):
    for key, vol in vision_x.items():
        if key != "image":
            present = vol.abs().sum(dim=(1,2,3,4,5)) > 0
            if present.any():
                # ... 50 lines of processing ...
```

**After (exit early + extract):**
```python
for step, batch in enumerate(loader):
    global_loss = process_global_image(batch, model, ln)
    region_losses = process_all_regions(batch, model, ln)
    total_loss = global_loss + sum(region_losses)

def process_all_regions(batch, model, ln) -> List[torch.Tensor]:
    """Process all anatomical regions in batch."""
    losses = []
    for region_name, region_volume in batch["vision_x"].items():
        if region_name == "image":
            continue
        loss = process_single_region(region_name, region_volume, model, ln)
        if loss is not None:
            losses.append(loss)
    return losses

def process_single_region(name, volume, model, ln) -> Optional[torch.Tensor]:
    """Process one region, return None if not present."""
    present = volume.abs().sum(dim=(1,2,3,4,5)) > 0
    if not present.any():
        return None
    # ... processing ...
    return loss
```

---

### 6. Experiment Mode Handling

**Current (scattered if/else):**
```python
# In __init__:
if separate_adapters:
    self.global_adapter = ...
    self.local_adapter = ...
elif adapter_depth == 1:
    self.adapter = OneLayerAdapter(...)
else:
    self.adapter = PerceiverResampler(...)

# In training setup:
if model_args.separate_adapters:
    set_requires_grad(model.global_adapter, ...)
    set_requires_grad(model.local_adapter, ...)
else:
    set_requires_grad(model.adapter, ...)
```

**Better (factory pattern):**
```python
from enum import Enum

class ExperimentMode(Enum):
    EXP1_DECODER_ONLY = "decoder_only"
    EXP2_JOINT = "joint"
    EXP5_MINIMAL = "minimal_1layer"
    EXP9_SEPARATE = "separate_global_local"

def create_adapters(config: ModelArguments) -> Dict[str, nn.Module]:
    """Factory for experiment-specific adapter configuration."""
    if config.separate_adapters:
        return {
            "global": OneLayerAdapter(dim=config.vis_dim, ...),
            "local": OneLayerAdapter(dim=config.vis_dim, ...),
        }
    elif config.adapter_depth == 1:
        return {"shared": OneLayerAdapter(dim=config.vis_dim, ...)}
    else:
        return {"shared": PerceiverResampler(dim=config.vis_dim, ...)}
```

---

## Refactoring Workflow

### Step 1: Identify Scope
```bash
# Find all occurrences
Grep(pattern="TARGET_PATTERN", output_mode="files_with_matches")

# Count occurrences
Grep(pattern="TARGET_PATTERN", output_mode="count")

# View with context
Grep(pattern="TARGET_PATTERN", output_mode="content", -n=true, -B=2, -A=2)
```

### Step 2: Plan Changes
- List all files affected
- Identify dependencies
- Check for test coverage
- Estimate impact

### Step 3: Execute
```python
# For single file, use Edit with replace_all
Edit(
    file_path="src/lit_recon_probe.py",
    old_string="x_g",
    new_string="tokens_global",
    replace_all=True
)

# For new utilities, use Write
Write(
    file_path="src/utils/metrics.py",
    content="..."
)
```

### Step 4: Verify
```bash
# Syntax check
python3 -m py_compile src/lit_recon_probe.py

# Run tests (if available)
pytest tests/

# Quick smoke test
python3 src/lit_recon_probe.py --help
```

---

## Quick Reference: Common Refactors

| Task | Command |
|------|---------|
| Rename variable | `Edit(old="x_g", new="tokens_global", replace_all=True)` |
| Extract function | Read → identify block → Write new function → Edit call site |
| Remove duplicate | Grep both locations → Write utility → Edit both to import |
| Add constant | Write to `config/constants.py` → Edit usages |
| Fix naming | Grep pattern → Review → Edit with replace_all |

---

## Priority Refactoring Targets

### Critical (Do First)
1. Extract `main()` helpers (~1,200 LOC → 6 functions)
2. Create `src/config/constants.py` for magic numbers
3. Consolidate compression ratio calculation

### High Priority
4. Extract `run_epoch()` helpers (~400 LOC → 3 functions)
5. Implement `MetricsAccumulator` class
6. Standardize variable naming (x_g → tokens_global)

### Medium Priority
7. Adapter factory pattern for Exp variants
8. Extract region processing to reduce nesting
9. Create `src/utils/` module structure

---

## Files to Create

```
src/
├── config/
│   └── constants.py      # Magic numbers, REGIONS list
├── utils/
│   ├── __init__.py
│   ├── metrics.py        # MetricsAccumulator, compression_ratio
│   ├── checkpoint.py     # save/load checkpoint utilities
│   └── device.py         # setup_device(), set_requires_grad()
└── training/
    ├── __init__.py
    └── loop.py           # run_epoch(), process_batch()
```
