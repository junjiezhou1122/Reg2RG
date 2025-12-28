"""
HuggingFace-compatible argument dataclasses for Reg2RG experiments.

These dataclasses work with transformers.HfArgumentParser for CLI parsing.
Used by both lit_recon_probe.py and run_experiment.py for consistency.

Usage:
    from transformers import HfArgumentParser
    from config.arguments import ModelArguments, DataArguments, TrainArguments

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
"""

from dataclasses import dataclass, field
from typing import Optional

from .paths import Paths


# ============================================================================
# MODEL ARGUMENTS
# ============================================================================

@dataclass
class ModelArguments:
    """
    Model architecture and pretrained weights configuration.

    This dataclass holds all hyperparameters that define the model structure
    and paths to pretrained checkpoints.
    """

    # ===== Pretrained Weights =====
    tokenizer_path: str = field(
        default_factory=lambda: str(Paths.LLAMA_7B),
        metadata={"help": "Path to the tokenizer (for dataset preprocessing)."},
    )

    pretrained_visual_encoder: str = field(
        default_factory=lambda: str(Paths.VIT_3D),
        metadata={"help": "Path to pretrained ViT3D weights."},
    )

    pretrained_adapter: str = field(
        default_factory=lambda: str(Paths.PERCEIVER_FC),
        metadata={"help": "Path to pretrained Perceiver adapter weights."},
    )

    use_pretrained_adapter: bool = field(
        default=True,
        metadata={"help": "Load pretrained Perceiver weights if available."},
    )

    # ===== Decode Mode =====
    decode_mode: str = field(
        default="pre_proj",
        metadata={"help": "Decode from 'pre_proj' (Z) or 'post_proj' (fc(Z))."},
    )

    # ===== Dimensions =====
    perceiver_num: int = field(default=32, metadata={"help": "Latent token count."})
    vis_dim: int = field(default=768, metadata={"help": "ViT token dimension."})
    llm_dim: int = field(default=4096, metadata={"help": "Projection output dimension."})

    # ===== Decoder Configuration =====
    decoder_layers: int = field(default=4, metadata={"help": "Decoder depth."})
    decoder_heads: int = field(default=8, metadata={"help": "Decoder heads."})
    decoder_ff_mult: int = field(default=4, metadata={"help": "Decoder FFN multiplier."})

    # ===== Adapter Configuration =====
    adapter_depth: int = field(
        default=6,
        metadata={
            "help": "Adapter depth (1 for minimal-capacity Exp5, 6 for baseline). "
            "When set to 1, uses OneLayerAdapter instead of PerceiverResampler."
        }
    )

    load_first_layer_from_pretrained: bool = field(
        default=False,
        metadata={
            "help": "Load 1st layer from 6-layer checkpoint (Exp5a). "
            "Only applies when adapter_depth=1."
        }
    )

    random_init_adapter: bool = field(
        default=False,
        metadata={
            "help": "Randomly initialize adapter, ignore pretrained weights (Exp5b)."
        }
    )

    # ===== Experiment 9: Separate Global/Local Adapters =====
    separate_adapters: bool = field(
        default=False,
        metadata={
            "help": "Use separate adapters for global and local paths (Exp9). "
            "When True, creates two independent 1-layer adapters."
        }
    )


# ============================================================================
# DATA ARGUMENTS
# ============================================================================

@dataclass
class DataArguments:
    """
    Dataset paths and data loading configuration.

    Specifies where to find CT scans, region masks, and reports.
    """

    # ===== Training Data =====
    data_folder: str = field(
        default_factory=lambda: str(Paths.TRAIN_DATA),
        metadata={"help": "Directory containing preprocessed CT scan volumes."},
    )

    mask_folder: str = field(
        default_factory=lambda: str(Paths.TRAIN_MASK),
        metadata={"help": "Directory containing binary masks for anatomical regions."},
    )

    report_file: str = field(
        default_factory=lambda: str(Paths.TRAIN_REPORT),
        metadata={"help": "CSV file with region-specific radiology reports."},
    )

    # ===== Validation Data =====
    val_data_folder: Optional[str] = field(
        default_factory=lambda: str(Paths.VAL_DATA),
        metadata={"help": "Validation CT volume directory."},
    )

    val_mask_folder: Optional[str] = field(
        default_factory=lambda: str(Paths.VAL_MASK),
        metadata={"help": "Validation region mask directory."},
    )

    val_report_file: Optional[str] = field(
        default_factory=lambda: str(Paths.VAL_REPORT),
        metadata={"help": "Validation report CSV."},
    )

    # ===== Caching =====
    monai_cache_dir: str = field(
        default_factory=lambda: str(Paths.MONAI_CACHE),
        metadata={"help": "Directory for MONAI dataset caching."},
    )

    val_split: float = field(
        default=0.05,
        metadata={"help": "Fraction of data used for validation (if val_* not set)."},
    )


# ============================================================================
# TRAINING ARGUMENTS
# ============================================================================

@dataclass
class TrainArguments:
    """
    Training hyperparameters and logging configuration.

    Controls the training process: learning rate, epochs, batch size, etc.
    """

    # ===== Output =====
    output_dir: str = field(
        default_factory=lambda: str(Paths.OUTPUTS / "LIT"),
        metadata={"help": "Where to write metrics and checkpoints."},
    )

    # ===== Training Hyperparameters =====
    num_train_epochs: int = field(default=20, metadata={"help": "Total training epochs."})
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay."})
    batch_size: int = field(default=1, metadata={"help": "Batch size per step."})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": "Gradient accumulation steps."})
    seed: int = field(default=42, metadata={"help": "Random seed."})

    # ===== Data Loading =====
    dataloader_num_workers: int = field(default=2, metadata={"help": "DataLoader workers."})

    # ===== Logging =====
    log_every: int = field(default=10, metadata={"help": "Log every N steps."})
    show_progress: bool = field(default=True, metadata={"help": "Show tqdm progress bars."})

    # ===== Adapter Training =====
    train_adapter: bool = field(
        default=False,
        metadata={"help": "Train adapter jointly with decoder (Exp2+)."},
    )
    lambda_region: float = field(default=1.0, metadata={"help": "Region loss weight."})

    # ===== Checkpointing =====
    monitor_metric: str = field(default="reg_cos", metadata={"help": "Metric to monitor."})
    monitor_mode: str = field(default="max", metadata={"help": "max or min."})
    save_top_k: int = field(default=3, metadata={"help": "Keep top-k checkpoints."})
    checkpoint_subdir: str = field(default="checkpoints", metadata={"help": "Checkpoint subdirectory."})
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to checkpoint to resume from."},
    )

    # ===== Validation & Early Stopping =====
    val_check_interval: int = field(
        default=0,
        metadata={"help": "Validate every N steps (0 = epoch end only)."},
    )
    early_stopping_patience: int = field(
        default=0,
        metadata={"help": "Early stopping patience (0 = disabled)."},
    )

    # ===== Device =====
    cuda_visible_devices: Optional[str] = field(
        default=None,
        metadata={"help": "GPU IDs (e.g., '0' or '0,1')."},
    )

    # ===== W&B =====
    use_wandb: bool = field(default=False, metadata={"help": "Enable W&B logging."})
    wandb_project: str = field(default="Reg2RG-LIT", metadata={"help": "W&B project."})
    wandb_entity: Optional[str] = field(default=None, metadata={"help": "W&B entity."})
    wandb_run_name: Optional[str] = field(default=None, metadata={"help": "W&B run name."})
    wandb_mode: str = field(default="online", metadata={"help": "W&B mode."})

    # ===== Cache Warmup =====
    precache_splits: str = field(default="none", metadata={"help": "Cache warmup: none|train|val|both."})
    precache_only: bool = field(default=False, metadata={"help": "Build cache and exit."})
    precache_max_items: Optional[int] = field(default=None, metadata={"help": "Max items to cache."})
    precache_num_workers: int = field(default=0, metadata={"help": "Cache warmup workers."})
    warmup_priority: Optional[int] = field(default=None, metadata={"help": "Warmup process priority."})
