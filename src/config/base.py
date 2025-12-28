"""
Base configuration dataclasses for Reg2RG experiments.

These classes define the structure of experiment configurations.
Specific experiments override these with their own defaults.

Usage:
    from config.base import ModelArgs, DataArgs, TrainArgs
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from .paths import Paths
from .constants import ModelConfig, DataConfig, TrainingConfig


# ============================================================================
# EXPERIMENT MODES
# ============================================================================

class ExperimentType(Enum):
    """Supported experiment types."""

    # LIT Probing experiments
    LIT_BASELINE = "lit_baseline"           # Exp1: Decoder only
    LIT_JOINT = "lit_joint"                 # Exp2: Adapter + Decoder
    LIT_MINIMAL = "lit_minimal"             # Exp5: 1-layer adapter
    LIT_SEPARATE = "lit_separate"           # Exp9: Separate global/local

    # Full Reg2RG training
    REG2RG_BASE = "reg2rg_base"             # Base Reg2RG training
    REG2RG_LORA = "reg2rg_lora"             # Reg2RG with LoRA


class AdapterType(Enum):
    """Adapter architecture types."""
    PERCEIVER_6L = "perceiver_6layer"       # Standard 6-layer Perceiver
    PERCEIVER_1L = "perceiver_1layer"       # Minimal 1-layer
    SEPARATE = "separate_global_local"       # Exp9: Two 1-layer adapters


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass
class ModelArgs:
    """
    Model architecture configuration.

    Shared across LIT probing and Reg2RG training.
    """

    # ===== Pretrained Weights =====
    tokenizer_path: str = field(
        default_factory=lambda: str(Paths.LLAMA_7B)
    )
    pretrained_visual_encoder: str = field(
        default_factory=lambda: str(Paths.VIT_3D)
    )
    pretrained_adapter: str = field(
        default_factory=lambda: str(Paths.PERCEIVER_FC)
    )
    use_pretrained_adapter: bool = True

    # ===== Architecture =====
    vis_dim: int = ModelConfig.VIS_DIM
    llm_dim: int = ModelConfig.LLM_DIM
    perceiver_num: int = ModelConfig.PERCEIVER_LATENTS

    # ===== Adapter Configuration =====
    adapter_type: AdapterType = AdapterType.PERCEIVER_6L
    adapter_depth: int = ModelConfig.PERCEIVER_DEPTH
    random_init_adapter: bool = False
    load_first_layer_from_pretrained: bool = False

    # ===== Exp9: Separate Adapters =====
    separate_adapters: bool = False

    # ===== Decoder Configuration =====
    decoder_layers: int = ModelConfig.DECODER_LAYERS
    decoder_heads: int = ModelConfig.DECODER_HEADS
    decoder_ff_mult: int = ModelConfig.DECODER_FF_MULT
    decode_mode: str = "pre_proj"  # "pre_proj" or "post_proj"

    # ===== LLM (for Reg2RG) =====
    lang_encoder_path: Optional[str] = field(
        default_factory=lambda: str(Paths.LLAMA_7B)
    )

    def __post_init__(self):
        """Derive adapter_type from other settings."""
        if self.separate_adapters:
            self.adapter_type = AdapterType.SEPARATE
            self.adapter_depth = 1
        elif self.adapter_depth == 1:
            self.adapter_type = AdapterType.PERCEIVER_1L


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

@dataclass
class DataArgs:
    """
    Dataset paths and loading configuration.

    Shared across LIT probing and Reg2RG training.
    """

    # ===== Training Data =====
    data_folder: str = field(
        default_factory=lambda: str(Paths.TRAIN_DATA)
    )
    mask_folder: str = field(
        default_factory=lambda: str(Paths.TRAIN_MASK)
    )
    report_file: str = field(
        default_factory=lambda: str(Paths.TRAIN_REPORT)
    )

    # ===== Validation Data =====
    val_data_folder: Optional[str] = field(
        default_factory=lambda: str(Paths.VAL_DATA)
    )
    val_mask_folder: Optional[str] = field(
        default_factory=lambda: str(Paths.VAL_MASK)
    )
    val_report_file: Optional[str] = field(
        default_factory=lambda: str(Paths.VAL_REPORT)
    )

    # ===== Caching =====
    monai_cache_dir: str = field(
        default_factory=lambda: str(Paths.MONAI_CACHE)
    )

    # ===== Data Splitting =====
    val_split: float = 0.05  # Only used if val_* paths not set


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass
class TrainArgs:
    """
    Training hyperparameters and logging configuration.

    Shared across LIT probing and Reg2RG training.
    """

    # ===== Output =====
    output_dir: str = field(
        default_factory=lambda: str(Paths.OUTPUTS / "LIT")
    )

    # ===== Training =====
    num_train_epochs: int = TrainingConfig.NUM_EPOCHS
    learning_rate: float = TrainingConfig.LEARNING_RATE
    weight_decay: float = TrainingConfig.WEIGHT_DECAY
    batch_size: int = TrainingConfig.BATCH_SIZE
    gradient_accumulation_steps: int = TrainingConfig.GRAD_ACCUM_STEPS
    seed: int = TrainingConfig.SEED

    # ===== Data Loading =====
    dataloader_num_workers: int = TrainingConfig.NUM_WORKERS

    # ===== Adapter Training =====
    train_adapter: bool = False  # False = freeze adapter (Exp1), True = train (Exp2+)

    # ===== Checkpointing =====
    save_top_k: int = TrainingConfig.SAVE_TOP_K
    monitor_metric: str = TrainingConfig.MONITOR_METRIC
    monitor_mode: str = TrainingConfig.MONITOR_MODE
    checkpoint_subdir: str = "checkpoints"
    resume_from_checkpoint: Optional[str] = None

    # ===== Validation =====
    val_check_interval: int = 0  # 0 = validate at epoch end
    early_stopping_patience: int = 0  # 0 = disabled

    # ===== Logging =====
    log_every: int = 10
    show_progress: bool = True

    # ===== W&B =====
    use_wandb: bool = False
    wandb_project: str = "Reg2RG"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"

    # ===== Device =====
    cuda_visible_devices: Optional[str] = None

    # ===== Caching =====
    precache_splits: str = "none"
    precache_only: bool = False
    precache_max_items: Optional[int] = None
    precache_num_workers: int = 0


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.

    Combines model, data, and training args into one object.
    """

    name: str
    experiment_type: ExperimentType
    model: ModelArgs
    data: DataArgs
    train: TrainArgs

    def __post_init__(self):
        """Set default output directory and W&B run name."""
        if self.train.wandb_run_name is None:
            self.train.wandb_run_name = self.name

    def to_cli_args(self) -> list:
        """Convert to command-line argument list for subprocess."""
        args = []
        for field_name, field_value in vars(self.model).items():
            if field_value is not None and not field_name.startswith('_'):
                args.extend([f"--{field_name}", str(field_value)])
        for field_name, field_value in vars(self.data).items():
            if field_value is not None and not field_name.startswith('_'):
                args.extend([f"--{field_name}", str(field_value)])
        for field_name, field_value in vars(self.train).items():
            if field_value is not None and not field_name.startswith('_'):
                args.extend([f"--{field_name}", str(field_value)])
        return args
