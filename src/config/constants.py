"""
Project-wide constants for Reg2RG medical imaging.

Usage:
    from config.constants import REGIONS, ModelConfig, DataConfig
"""

from typing import Tuple


# ============================================================================
# ANATOMICAL REGIONS
# ============================================================================

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


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ModelConfig:
    """Model architecture constants."""

    # Vision Transformer (ViT-3D)
    VIT_IMAGE_SIZE: int = 512
    VIT_FRAMES: int = 512
    VIT_DEPTH: int = 12
    VIT_HEADS: int = 8
    VIT_MLP_DIM: int = 2048
    VIT_DROPOUT: float = 0.1

    # Patch sizes
    PATCH_SIZE: int = 32
    FRAME_PATCH_SIZE: int = 4

    # Embedding dimensions
    VIS_DIM: int = 768       # ViT output dimension
    LLM_DIM: int = 4096      # LLaMA embedding dimension

    # Adapter defaults
    PERCEIVER_DEPTH: int = 6           # Standard Perceiver layers
    PERCEIVER_LATENTS: int = 32        # Compressed tokens
    PERCEIVER_HEADS: int = 8
    PERCEIVER_DIM_HEAD: int = 64
    PERCEIVER_FF_MULT: int = 4

    # Decoder defaults
    DECODER_LAYERS: int = 4
    DECODER_HEADS: int = 8
    DECODER_FF_MULT: int = 4


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

class DataConfig:
    """Data preprocessing constants."""

    # Hounsfield Unit (HU) normalization
    HU_MIN: int = -1000
    HU_MAX: int = 200

    # Region volume size (after resize)
    REGION_SIZE: Tuple[int, int, int] = (256, 256, 64)
    REGION_VOXELS: int = 256 * 256 * 64  # 4,194,304

    # Global image size
    IMAGE_SIZE: Tuple[int, int, int] = (512, 512, 512)

    # Cache versioning
    CACHE_VERSION: str = "lit_img_hu_seg_v1"


# ============================================================================
# TRAINING DEFAULTS
# ============================================================================

class TrainingConfig:
    """Training hyperparameter defaults."""

    # Optimization
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 0.01
    GRAD_ACCUM_STEPS: int = 8

    # Batching
    BATCH_SIZE: int = 1
    NUM_WORKERS: int = 2

    # Training
    NUM_EPOCHS: int = 20
    SEED: int = 42

    # Checkpointing
    SAVE_TOP_K: int = 3
    MONITOR_METRIC: str = "reg_cos"
    MONITOR_MODE: str = "max"
