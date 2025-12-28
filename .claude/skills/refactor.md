---
description: Code refactoring for Reg2RG medical imaging codebase. Use when restructuring code, reducing complexity, fixing naming conventions, extracting functions, eliminating duplicates, or improving code quality.
---

# Reg2RG Code Refactoring Skill

## Overview

This skill provides systematic code refactoring knowledge tailored to the Reg2RG medical imaging codebase. It covers:
- Function decomposition for large training modules
- Naming convention standardization
- Duplicate code elimination
- Magic number extraction
- Experiment mode handling (Exp1/2/5/9)

---

## Codebase Architecture

```
src/
├── lit_recon_probe.py           # Main LIT training (2,500 LOC) ⚠️
├── Dataset/
│   └── radgenome_dataset_train.py  # Data loading with MONAI caching
└── Model/
    ├── vit_3d.py                # 3D Vision Transformer (frozen)
    ├── helpers.py               # PerceiverResampler (6-layer)
    ├── one_layer_adapter.py     # OneLayerAdapter (1-layer, Exp5)
    └── adapter_utils.py         # Weight loading utilities
```

### Experiment Variants
| Exp | Description | Components |
|-----|-------------|------------|
| 1 | Decoder only | `adapter` frozen, `decoder` trained |
| 2 | Joint training | `adapter` + `decoder` trained |
| 5 | Minimal capacity | 1-layer adapter |
| 9 | Separate paths | `global_adapter` + `local_adapter` + 2 decoders |

---

## Refactoring Principles

### 1. Function Size Limits

Functions exceeding 50 lines should be decomposed.

**Priority Targets:**

| Function | Location | Lines | Recommended Extraction |
|----------|----------|-------|----------------------|
| `main()` | lit_recon_probe.py:1273 | 1,200+ | `setup_device()`, `load_datasets()`, `initialize_model()`, `setup_optimizer()`, `setup_logging()`, `training_loop()` |
| `run_epoch()` | lit_recon_probe.py:1783 | 400+ | `process_batch()`, `compute_region_metrics()`, `compute_correlation_analysis()` |
| `prepare_samples()` | radgenome_dataset_train.py:345 | 60 | `_build_single_sample()` |

**Extraction Pattern:**
```python
# Before: 50+ lines inline in main()
def main():
    # ... device setup code ...
    # ... 50 lines ...

# After: Extract to focused function
def setup_device(train_args: TrainArguments) -> torch.device:
    """Initialize CUDA device and random seeds."""
    if train_args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = train_args.cuda_visible_devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(train_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_args.seed)
    return device

def main():
    device = setup_device(train_args)
```

---

### 2. Naming Conventions

#### Variables
| Current (Bad) | Refactored (Good) | Reason |
|---------------|-------------------|--------|
| `x_g`, `z_g`, `x_hat_g` | `tokens_global`, `latent_global`, `reconstructed_global` | Single letters unclear |
| `x_r`, `z_r` | `tokens_region`, `latent_region` | Consistency |
| `rname`, `rs`, `rstats` | `region_name`, `region_stats` | No abbreviations |
| `vision_xs`, `mask_xs` | `vision_batch`, `mask_batch` | Use `_batch` suffix |
| `key, vol` (in loops) | `region_name, region_volume` | Descriptive |

#### Functions
| Pattern | Convention | Examples |
|---------|------------|----------|
| Actions | `verb_noun()` | `run_epoch()`, `compute_loss()`, `save_checkpoint()` |
| Getters | `get_noun()` | `get_active_adapter()`, `get_decoder()` |
| Predicates | `is_/should_/has_` | `is_improvement()`, `should_validate()`, `has_region()` |
| Private helpers | `_prefix` | `_build_single_sample()`, `_compute_bbox()` |
| Factory methods | `create_noun()` | `create_adapters()`, `create_optimizer()` |

#### Classes
- Models: PascalCase with descriptive suffix - `LITProbeModel`, `ProbeDecoder`
- Data: `RadGenomeDataset`, `LITDataCollator`
- Config: `ModelArguments`, `TrainArguments`, `DataArguments`

---

### 3. Duplicate Code Patterns

#### Pattern A: Adapter/Decoder Selection
**Locations:** lit_recon_probe.py lines 912-916, 950-953, 1562-1565

```python
# DUPLICATED (4+ times):
if self.separate_adapters:
    adapter = self.global_adapter if is_global else self.local_adapter
else:
    adapter = self.adapter

# REFACTORED (add method to LITProbeModel):
def get_adapter(self, is_global: bool = True) -> nn.Module:
    """Return appropriate adapter based on experiment mode."""
    if self.separate_adapters:
        return self.global_adapter if is_global else self.local_adapter
    return self.adapter

def get_decoder(self, is_global: bool = True) -> nn.Module:
    """Return appropriate decoder based on experiment mode."""
    if self.separate_adapters:
        return self.global_decoder if is_global else self.local_decoder
    return self.decoder
```

#### Pattern B: Compression Ratio Calculation
**Locations:** lit_recon_probe.py:1950-1965, radgenome_dataset_train.py:486-501

```python
# DUPLICATED in 2 files:
bbox_h, bbox_w, bbox_d = h1 - h0 + 1, w1 - w0 + 1, d1 - d0 + 1
original_voxels = bbox_h * bbox_w * bbox_d
compression_ratio = original_voxels / target_voxels

# REFACTORED (create src/utils/metrics.py):
def compute_compression_ratio(
    bbox: Tuple[int, int, int, int, int, int],
    target_size: Tuple[int, int, int] = (256, 256, 64)
) -> float:
    """Calculate volume compression ratio from bounding box coordinates."""
    h0, h1, w0, w1, d0, d1 = bbox
    original = (h1 - h0 + 1) * (w1 - w0 + 1) * (d1 - d0 + 1)
    target = target_size[0] * target_size[1] * target_size[2]
    return original / target
```

#### Pattern C: Metric Accumulation
**Locations:** lit_recon_probe.py lines 1810-1840, 2143-2214

```python
# DUPLICATED pattern:
agg = {"loss": 0.0, "mse": 0.0, "cos": 0.0, "count": 0}
for batch in loader:
    agg["loss"] += loss.item()
    agg["count"] += 1
final = {k: v / agg["count"] for k, v in agg.items() if k != "count"}

# REFACTORED (create utility class):
class MetricsAccumulator:
    """Accumulate and average training/validation metrics."""

    def __init__(self, keys: List[str]):
        self.sums = {k: 0.0 for k in keys}
        self.count = 0

    def add(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if k in self.sums:
                self.sums[k] += v
        self.count += 1

    def average(self) -> Dict[str, float]:
        if self.count == 0:
            return {k: 0.0 for k in self.sums}
        return {k: v / self.count for k, v in self.sums.items()}

    def reset(self) -> None:
        self.sums = {k: 0.0 for k in self.sums}
        self.count = 0
```

---

### 4. Magic Numbers → Constants

**Create `src/config/constants.py`:**

```python
"""
Project-wide constants for Reg2RG medical imaging.

Usage:
    from config.constants import DataConfig, ModelConfig, REGIONS
"""

from typing import Tuple


class DataConfig:
    """Data preprocessing and loading constants."""
    HU_MIN: int = -1000
    HU_MAX: int = 200
    REGION_SIZE: Tuple[int, int, int] = (256, 256, 64)
    REGION_VOXELS: int = 256 * 256 * 64  # 4,194,304
    CACHE_VERSION: str = "lit_img_hu_seg_v1"


class ModelConfig:
    """Model architecture constants."""
    # Vision Transformer
    VIT_IMAGE_SIZE: int = 512
    VIT_FRAMES: int = 512
    VIT_DEPTH: int = 12
    VIT_HEADS: int = 8
    VIT_MLP_DIM: int = 2048

    # Patch sizes
    PATCH_SIZE: int = 32
    FRAME_PATCH_SIZE: int = 4

    # Dimensions
    VIS_DIM: int = 768
    LLM_DIM: int = 4096

    # Adapter defaults
    DEFAULT_LATENTS: int = 32
    DEFAULT_HEADS: int = 8
    DEFAULT_DIM_HEAD: int = 64
    DEFAULT_FF_MULT: int = 4


class TrainingConfig:
    """Training hyperparameter defaults."""
    DEFAULT_LR: float = 1e-4
    DEFAULT_WEIGHT_DECAY: float = 0.01
    DEFAULT_GRAD_ACCUM: int = 8
    DEFAULT_BATCH_SIZE: int = 1


# Anatomical regions for chest CT analysis
REGIONS: Tuple[str, ...] = (
    "abdomen",
    "bone",
    "breast",
    "esophagus",
    "heart",
    "lung",
    "mediastinum",
    "pleura",
    "thyroid",
    "trachea and bronchie",
)
```

**Usage after refactoring:**
```python
# Before:
grid = self._patch_grid(256, 256, 64)  # Magic numbers
hu_min, hu_max = -1000, 200            # Magic numbers

# After:
from config.constants import DataConfig, ModelConfig

grid = self._patch_grid(*DataConfig.REGION_SIZE)
hu_min, hu_max = DataConfig.HU_MIN, DataConfig.HU_MAX
```

---

### 5. Reduce Nesting (Exit Early Pattern)

**Before (4+ levels of nesting):**
```python
for step, batch in enumerate(loader):
    for key, vol in vision_x.items():
        if key != "image":
            present = vol.abs().sum(dim=(1,2,3,4,5)) > 0
            if present.any():
                with torch.no_grad():
                    x_r, reg_grid = model.encode_tokens(vol[present])
                # ... 50 more lines ...
```

**After (exit early + extract):**
```python
def process_batch(batch, model, ln, device) -> Tuple[torch.Tensor, Dict]:
    """Process single batch for global and region reconstruction."""
    vision_x = {k: v.to(device) for k, v in batch["vision_x"].items()}

    # Global path
    global_loss, global_metrics = compute_reconstruction(
        model, vision_x["image"], ln, is_global=True
    )

    # Region paths
    region_losses = []
    for region_name, region_volume in vision_x.items():
        if region_name == "image":
            continue

        loss = process_region(model, region_name, region_volume, ln)
        if loss is not None:
            region_losses.append(loss)

    total_loss = global_loss + sum(region_losses)
    return total_loss, global_metrics


def process_region(model, name, volume, ln) -> Optional[torch.Tensor]:
    """Process single region. Returns None if region not present."""
    present = volume.abs().sum(dim=(1,2,3,4,5)) > 0
    if not present.any():
        return None

    with torch.no_grad():
        tokens, grid = model.encode_tokens(volume[present])

    latent = model.compress_tokens(tokens, is_global=False)
    reconstructed = model.decode_tokens(latent, grid, is_global=False)
    loss, _, _, _ = recon_loss(reconstructed, tokens, ln)
    return loss
```

---

### 6. Experiment Mode Factory Pattern

**Current (scattered conditionals):**
```python
# In __init__ (lines 758-797):
if separate_adapters:
    self.global_adapter = OneLayerAdapter(...)
    self.local_adapter = OneLayerAdapter(...)
elif adapter_depth == 1:
    self.adapter = OneLayerAdapter(...)
else:
    self.adapter = PerceiverResampler(...)

# In training setup (lines 1560-1568):
if model_args.separate_adapters:
    set_requires_grad(model.global_adapter, train_args.train_adapter)
    set_requires_grad(model.local_adapter, train_args.train_adapter)
else:
    set_requires_grad(model.adapter, train_args.train_adapter)
```

**Refactored (factory pattern):**
```python
from enum import Enum
from typing import Dict, NamedTuple

class ExperimentMode(Enum):
    """Supported experiment configurations."""
    DECODER_ONLY = "exp1_decoder"      # Exp1: Decoder only
    JOINT = "exp2_joint"               # Exp2: Joint adapter+decoder
    MINIMAL = "exp5_minimal"           # Exp5: 1-layer adapter
    SEPARATE = "exp9_separate"         # Exp9: Separate global/local


class AdapterConfig(NamedTuple):
    """Configuration for adapter creation."""
    adapters: Dict[str, nn.Module]
    decoders: Dict[str, nn.Module]
    trainable_adapter: bool


def create_experiment_config(
    model_args: ModelArguments,
    train_args: TrainArguments,
    vis_dim: int,
    num_latents: int,
    decoder_config: dict,
) -> AdapterConfig:
    """Factory for experiment-specific model configuration."""

    if model_args.separate_adapters:
        # Exp9: Separate global/local paths
        adapters = {
            "global": OneLayerAdapter(dim=vis_dim, num_latents=num_latents),
            "local": OneLayerAdapter(dim=vis_dim, num_latents=num_latents),
        }
        decoders = {
            "global": ProbeDecoder(**decoder_config),
            "local": ProbeDecoder(**decoder_config),
        }
    elif model_args.adapter_depth == 1:
        # Exp5: Minimal 1-layer adapter
        adapters = {"shared": OneLayerAdapter(dim=vis_dim, num_latents=num_latents)}
        decoders = {"shared": ProbeDecoder(**decoder_config)}
    else:
        # Exp1/2: Standard Perceiver
        adapters = {"shared": PerceiverResampler(dim=vis_dim, num_latents=num_latents)}
        decoders = {"shared": ProbeDecoder(**decoder_config)}

    return AdapterConfig(
        adapters=adapters,
        decoders=decoders,
        trainable_adapter=train_args.train_adapter,
    )
```

---

## Recommended File Structure After Refactoring

```
src/
├── config/
│   ├── __init__.py
│   └── constants.py          # DataConfig, ModelConfig, REGIONS
├── utils/
│   ├── __init__.py
│   ├── metrics.py            # MetricsAccumulator, compute_compression_ratio
│   ├── checkpoint.py         # save_checkpoint, load_checkpoint
│   └── device.py             # setup_device, set_requires_grad
├── training/
│   ├── __init__.py
│   ├── loop.py               # run_epoch, process_batch
│   └── experiment.py         # ExperimentMode, create_experiment_config
├── Dataset/
│   └── radgenome_dataset_train.py
├── Model/
│   └── ... (unchanged)
└── lit_recon_probe.py        # Simplified main entry point
```

---

## Refactoring Priority

### 🔴 Critical (High Impact, Do First)
1. Create `src/config/constants.py` - Extract all magic numbers
2. Extract `main()` into 6 focused functions
3. Consolidate compression ratio calculation

### 🟠 High Priority
4. Extract `run_epoch()` into 3 functions
5. Implement `MetricsAccumulator` class
6. Standardize variable naming (`x_g` → `tokens_global`)

### 🟡 Medium Priority
7. Add `get_adapter()`/`get_decoder()` methods
8. Implement experiment factory pattern
9. Create `src/utils/` module structure

### 🟢 Low Priority (Nice to Have)
10. Add type hints to all functions
11. Improve error messages
12. Add docstrings to extracted functions

---

## Verification Checklist

After each refactoring:

```bash
# 1. Syntax check
python3 -m py_compile src/lit_recon_probe.py

# 2. Import check
python3 -c "from src.lit_recon_probe import main"

# 3. Help command (smoke test)
python3 src/lit_recon_probe.py --help

# 4. Quick training test (1 step)
python3 src/lit_recon_probe.py --num_train_epochs 1 --precache_only True
```
