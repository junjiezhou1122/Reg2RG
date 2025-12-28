#!/usr/bin/env python3
"""
Run LIT experiments with unified configuration.

Uses HuggingFace's HfArgumentParser for consistent argument handling.

Usage:
    # List available experiments
    python run_experiment.py --list

    # Run specific experiment
    python run_experiment.py exp9

    # Run with GPU selection
    CUDA_VISIBLE_DEVICES=0 python run_experiment.py exp9

    # Dry run (show config without running)
    python run_experiment.py exp9 --dry-run

    # Override specific parameters
    python run_experiment.py exp9 --epochs 5 --lr 5e-5
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import get_experiment, list_experiments, ExperimentConfig
from config import ModelArguments, DataArguments, TrainArguments


def print_config(exp: ExperimentConfig) -> None:
    """Print experiment configuration."""
    print(f"\n{'='*60}")
    print(f"Experiment: {exp.name}")
    print(f"Type: {exp.experiment_type.value}")
    print(f"{'='*60}")

    print("\n📦 Model Configuration:")
    print(f"  adapter_depth: {exp.model.adapter_depth}")
    print(f"  separate_adapters: {exp.model.separate_adapters}")
    print(f"  random_init_adapter: {exp.model.random_init_adapter}")
    print(f"  decoder_layers: {exp.model.decoder_layers}")
    print(f"  decode_mode: {exp.model.decode_mode}")
    print(f"  perceiver_num: {exp.model.perceiver_num}")

    print("\n📂 Data Configuration:")
    print(f"  data_folder: {exp.data.data_folder}")
    print(f"  val_data_folder: {exp.data.val_data_folder}")

    print("\n🏋️ Training Configuration:")
    print(f"  output_dir: {exp.train.output_dir}")
    print(f"  num_train_epochs: {exp.train.num_train_epochs}")
    print(f"  train_adapter: {exp.train.train_adapter}")
    print(f"  learning_rate: {exp.train.learning_rate}")
    print(f"  batch_size: {exp.train.batch_size}")
    print(f"  gradient_accumulation_steps: {exp.train.gradient_accumulation_steps}")
    print(f"  wandb_project: {exp.train.wandb_project}")
    print(f"  wandb_run_name: {exp.train.wandb_run_name}")
    print()


def config_to_hf_args(exp: ExperimentConfig) -> tuple:
    """
    Convert ExperimentConfig to HuggingFace-compatible Arguments.

    Returns:
        Tuple of (ModelArguments, DataArguments, TrainArguments)
    """
    model_args = ModelArguments(
        tokenizer_path=exp.model.tokenizer_path,
        pretrained_visual_encoder=exp.model.pretrained_visual_encoder,
        pretrained_adapter=exp.model.pretrained_adapter,
        use_pretrained_adapter=exp.model.use_pretrained_adapter,
        decode_mode=exp.model.decode_mode,
        perceiver_num=exp.model.perceiver_num,
        vis_dim=exp.model.vis_dim,
        llm_dim=exp.model.llm_dim,
        decoder_layers=exp.model.decoder_layers,
        decoder_heads=exp.model.decoder_heads,
        decoder_ff_mult=exp.model.decoder_ff_mult,
        adapter_depth=exp.model.adapter_depth,
        load_first_layer_from_pretrained=exp.model.load_first_layer_from_pretrained,
        random_init_adapter=exp.model.random_init_adapter,
        separate_adapters=exp.model.separate_adapters,
    )

    data_args = DataArguments(
        data_folder=exp.data.data_folder,
        mask_folder=exp.data.mask_folder,
        report_file=exp.data.report_file,
        val_data_folder=exp.data.val_data_folder,
        val_mask_folder=exp.data.val_mask_folder,
        val_report_file=exp.data.val_report_file,
        monai_cache_dir=exp.data.monai_cache_dir,
        val_split=exp.data.val_split,
    )

    train_args = TrainArguments(
        output_dir=exp.train.output_dir,
        num_train_epochs=exp.train.num_train_epochs,
        learning_rate=exp.train.learning_rate,
        weight_decay=exp.train.weight_decay,
        batch_size=exp.train.batch_size,
        gradient_accumulation_steps=exp.train.gradient_accumulation_steps,
        seed=exp.train.seed,
        dataloader_num_workers=exp.train.dataloader_num_workers,
        log_every=exp.train.log_every,
        show_progress=exp.train.show_progress,
        train_adapter=exp.train.train_adapter,
        monitor_metric=exp.train.monitor_metric,
        monitor_mode=exp.train.monitor_mode,
        save_top_k=exp.train.save_top_k,
        checkpoint_subdir=exp.train.checkpoint_subdir,
        resume_from_checkpoint=exp.train.resume_from_checkpoint,
        val_check_interval=exp.train.val_check_interval,
        early_stopping_patience=exp.train.early_stopping_patience,
        use_wandb=exp.train.use_wandb,
        wandb_project=exp.train.wandb_project,
        wandb_entity=exp.train.wandb_entity,
        wandb_run_name=exp.train.wandb_run_name,
        wandb_mode=exp.train.wandb_mode,
    )

    return model_args, data_args, train_args


def run_experiment(model_args: ModelArguments, data_args: DataArguments, train_args: TrainArguments) -> None:
    """
    Run training with the given arguments.

    This imports and calls the training logic from lit_recon_probe.
    """
    # Import training module (delayed import to avoid loading torch at startup)
    from lit_recon_probe import run_training

    # Run training with the arguments
    run_training(model_args, data_args, train_args)


def main():
    parser = argparse.ArgumentParser(
        description="Run LIT experiments with unified configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --list              # List experiments
  python run_experiment.py exp9                # Run exp9
  python run_experiment.py exp9 --dry-run      # Show config only
  python run_experiment.py exp9 --epochs 5     # Override epochs
        """,
    )

    parser.add_argument("experiment", nargs="?", help="Experiment name (e.g., exp9, exp5b)")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print config without running")
    parser.add_argument("--epochs", type=int, help="Override num_train_epochs")
    parser.add_argument("--lr", type=float, help="Override learning_rate")
    parser.add_argument("--gpu", type=str, help="GPU ID (e.g., '0' or '0,1')")

    args = parser.parse_args()

    # List experiments
    if args.list:
        print("\n📋 Available Experiments:")
        print("-" * 40)
        for name, desc in list_experiments().items():
            print(f"  {name:12} - {desc}")
        print()
        return

    # Require experiment name
    if not args.experiment:
        parser.print_help()
        return

    # Get experiment config
    try:
        exp = get_experiment(args.experiment)
    except KeyError as e:
        print(f"❌ Error: {e}")
        return

    # Apply overrides
    if args.epochs:
        exp.train.num_train_epochs = args.epochs
    if args.lr:
        exp.train.learning_rate = args.lr

    # Print config
    print_config(exp)

    # Dry run - just print config
    if args.dry_run:
        print("🔍 Dry run - not executing")
        return

    # Convert to HuggingFace Arguments
    model_args, data_args, train_args = config_to_hf_args(exp)

    print("🚀 Starting experiment...")
    print("-" * 60)

    try:
        run_experiment(model_args, data_args, train_args)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
