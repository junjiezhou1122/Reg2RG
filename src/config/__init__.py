"""
Reg2RG Configuration System

Unified configuration for all experiments.

Usage:
    # Quick experiment access
    from config import get_experiment, list_experiments

    exp = get_experiment("exp9")
    print(exp.model.separate_adapters)  # True

    # Access paths
    from config import Paths
    print(Paths.VIT_3D)

    # Access constants
    from config import REGIONS, ModelConfig, DataConfig

    # HuggingFace-compatible Arguments (for CLI parsing)
    from config import ModelArguments, DataArguments, TrainArguments

    # Create custom experiment
    from config import ExperimentConfig, ModelArgs, DataArgs, TrainArgs

    custom = ExperimentConfig(
        name="my_experiment",
        experiment_type=ExperimentType.LIT_SEPARATE,
        model=ModelArgs(separate_adapters=True),
        data=DataArgs(),
        train=TrainArgs(num_train_epochs=10),
    )
"""

# Paths (server-specific)
from .paths import Paths, ServerPaths

# Constants
from .constants import (
    REGIONS,
    ModelConfig,
    DataConfig,
    TrainingConfig,
)

# HuggingFace-compatible Arguments (for HfArgumentParser)
from .arguments import (
    ModelArguments,
    DataArguments,
    TrainArguments,
)

# Base configuration classes (for experiment registry)
from .base import (
    ExperimentType,
    AdapterType,
    ModelArgs,
    DataArgs,
    TrainArgs,
    ExperimentConfig,
)

# Experiment registry
from .experiments import get_experiment, list_experiments, EXPERIMENTS


__all__ = [
    # Paths
    "Paths",
    "ServerPaths",
    # Constants
    "REGIONS",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    # HuggingFace Arguments
    "ModelArguments",
    "DataArguments",
    "TrainArguments",
    # Base classes
    "ExperimentType",
    "AdapterType",
    "ModelArgs",
    "DataArgs",
    "TrainArgs",
    "ExperimentConfig",
    # Experiment access
    "get_experiment",
    "list_experiments",
    "EXPERIMENTS",
]
