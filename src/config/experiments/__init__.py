"""
Experiment configurations for LIT probing.

Usage:
    from config.experiments import get_experiment, list_experiments

    # Get a specific experiment
    exp = get_experiment("exp9_separate")

    # List all available experiments
    experiments = list_experiments()
"""

from ..base import (
    ExperimentConfig,
    ExperimentType,
    AdapterType,
    ModelArgs,
    DataArgs,
    TrainArgs,
)
from ..paths import Paths


# ============================================================================
# LIT EXPERIMENTS
# ============================================================================

def exp5b_fresh_1layer() -> ExperimentConfig:
    """
    Exp 5b: Fresh 1-layer adapter trained from scratch.

    - Adapter: 1-layer, random init, trained
    - Decoder: 4-layer, trained
    - Baseline for Exp 9 comparison
    """
    return ExperimentConfig(
        name="exp5b_fresh_1layer_joint",
        experiment_type=ExperimentType.LIT_MINIMAL,
        model=ModelArgs(
            adapter_depth=1,
            random_init_adapter=True,
            separate_adapters=False,
            decoder_layers=4,
            decode_mode="pre_proj",
        ),
        data=DataArgs(),
        train=TrainArgs(
            output_dir=str(Paths.OUTPUTS / "LIT_exp5" / "exp5b_fresh_1layer"),
            num_train_epochs=20,
            train_adapter=True,
            learning_rate=1e-4,
            early_stopping_patience=3,
            use_wandb=True,
            wandb_project="Reg2RG-LIT-Exp5",
        ),
    )


def exp9_separate() -> ExperimentConfig:
    """
    Exp 9: Separate Global/Local adapters and decoders.

    - Global Adapter: 1-layer, random init, trained
    - Local Adapter: 1-layer, random init, trained
    - Global Decoder: 4-layer, trained
    - Local Decoder: 4-layer, trained
    """
    return ExperimentConfig(
        name="exp9_separate_1layer_joint",
        experiment_type=ExperimentType.LIT_SEPARATE,
        model=ModelArgs(
            adapter_depth=1,
            random_init_adapter=True,
            separate_adapters=True,
            decoder_layers=4,
            decode_mode="pre_proj",
        ),
        data=DataArgs(),
        train=TrainArgs(
            output_dir=str(Paths.OUTPUTS / "LIT_exp9" / "exp9_separate_1layer"),
            num_train_epochs=20,
            train_adapter=True,
            learning_rate=1e-4,
            early_stopping_patience=3,
            use_wandb=True,
            wandb_project="Reg2RG-LIT-Exp9",
        ),
    )


def exp1_decoder_only() -> ExperimentConfig:
    """
    Exp 1: Decoder-only baseline (adapter frozen).

    - Adapter: 6-layer Perceiver, pretrained, frozen
    - Decoder: 4-layer, trained
    """
    return ExperimentConfig(
        name="exp1_decoder_only",
        experiment_type=ExperimentType.LIT_BASELINE,
        model=ModelArgs(
            adapter_depth=6,
            random_init_adapter=False,
            separate_adapters=False,
            decoder_layers=4,
            decode_mode="pre_proj",
        ),
        data=DataArgs(),
        train=TrainArgs(
            output_dir=str(Paths.OUTPUTS / "LIT_exp1" / "exp1_decoder_only"),
            num_train_epochs=20,
            train_adapter=False,  # Frozen!
            learning_rate=1e-4,
            use_wandb=True,
            wandb_project="Reg2RG-LIT-Exp1",
        ),
    )


def exp2_joint() -> ExperimentConfig:
    """
    Exp 2: Joint adapter + decoder training.

    - Adapter: 6-layer Perceiver, pretrained, trained
    - Decoder: 4-layer, trained
    """
    return ExperimentConfig(
        name="exp2_joint_training",
        experiment_type=ExperimentType.LIT_JOINT,
        model=ModelArgs(
            adapter_depth=6,
            random_init_adapter=False,
            separate_adapters=False,
            decoder_layers=4,
            decode_mode="pre_proj",
        ),
        data=DataArgs(),
        train=TrainArgs(
            output_dir=str(Paths.OUTPUTS / "LIT_exp2" / "exp2_joint"),
            num_train_epochs=20,
            train_adapter=True,
            learning_rate=1e-4,
            use_wandb=True,
            wandb_project="Reg2RG-LIT-Exp2",
        ),
    )


# ============================================================================
# EXPERIMENT REGISTRY
# ============================================================================

EXPERIMENTS = {
    "exp1": exp1_decoder_only,
    "exp1_decoder_only": exp1_decoder_only,
    "exp2": exp2_joint,
    "exp2_joint": exp2_joint,
    "exp5b": exp5b_fresh_1layer,
    "exp5b_fresh": exp5b_fresh_1layer,
    "exp9": exp9_separate,
    "exp9_separate": exp9_separate,
}


def get_experiment(name: str) -> ExperimentConfig:
    """
    Get experiment configuration by name.

    Args:
        name: Experiment name (e.g., "exp9", "exp5b_fresh")

    Returns:
        ExperimentConfig object

    Raises:
        KeyError: If experiment not found
    """
    if name not in EXPERIMENTS:
        available = ", ".join(sorted(set(EXPERIMENTS.keys())))
        raise KeyError(f"Unknown experiment: {name}. Available: {available}")
    return EXPERIMENTS[name]()


def list_experiments() -> dict:
    """
    List all available experiments.

    Returns:
        Dict mapping experiment names to their descriptions
    """
    return {
        "exp1": "Decoder-only baseline (adapter frozen)",
        "exp2": "Joint adapter + decoder training",
        "exp5b": "Fresh 1-layer adapter from scratch",
        "exp9": "Separate global/local adapters",
    }
