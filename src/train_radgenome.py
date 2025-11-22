"""
This script trains the Reg2RG model end-to-end on the RadGenome dataset.

Feynman-style summary: Imagine we want a smart writer (LLM) to produce
medical reports for CT scans. The writer doesn't understand pixels, so we
first convert CT images (and their region masks) into language-like
embeddings, then give those embeddings and text prompts to the writer.

This file does four simple things:
1) Define how to batch items coming from the dataset (DataCollator).
2) Set random seeds so experiments are repeatable.
3) Parse command line/config arguments describing paths and training options.
4) Build dataset + model + Trainer, then call train().

We intentionally keep logic here minimal; heavy lifting is in:
- Dataset: src/Dataset/radgenome_dataset_train.py
- Model:   src/Model/Reg2RG.py and its embedding layer
"""

# tqdm gives us pretty progress bars during loops (if/when used)
import tqdm.auto as tqdm

# Functional API from PyTorch for some tensor ops (e.g., softmax, cross-entropy)
import torch.nn.functional as F

# Typing hints to make code clearer and less error-prone
from typing import List, Optional, Tuple, Union, Optional, Dict, Sequence

# Hugging Face Transformers for argument parsing and Trainer loop
import transformers

# PEFT/LoRA tooling is used inside the model; imported here for completeness
from peft import get_peft_model, LoraConfig, TaskType

# Hugging Face Trainer orchestrates the training loop
from transformers import Trainer

# Dataclass utilities for clean argument containers
from dataclasses import dataclass, field

# Our core multimodal model: language model + vision embedding fusion
from Model.Reg2RG import Reg2RG

# Training split of the RadGenome dataset (yields text + vision tensors)
from Dataset.radgenome_dataset_train import RadGenomeDataset_Train

# Structured configs: model paths, data paths, and HF TrainingArguments
from args.train_radgenome.jhcpu7 import ModelArguments, DataArguments, TrainingArguments

# Numpy for number crunching convenience
import numpy as np

# PyTorch base import (cuda moves, tensors, etc.)
import torch              

# Python's random module (used alongside numpy/pytorch seeding)
import random

# A fixed list of anatomical regions we consider. The dataset and model agree on
# these names so we can route the correct region masks/embeddings.
REGIONS = [
    'abdomen',
    'bone',
    'breast',
    'esophagus',
    'heart',
    'lung',
    'mediastinum',
    'pleura',
    'thyroid',
    'trachea and bronchie',
]

@dataclass
class DataCollator(object):
    """
    Batches items returned by RadGenomeDataset_Train into a single minibatch.

    Feynman explanation: The dataset yields a Python dict per sample with
    - lang_x: token IDs for our prompt+answer
    - vision_x: a dict of tensors for each region and a global 'image'
    - mask_x:   a dict of raw mask tensors per region
    - region2area: a mapping telling us which region index maps to which name
    - attention_mask: tells the LLM which tokens are padding
    - label: same shape as lang_x, but with -100 where we don't compute loss

    The collator stacks these along a new batch dimension, while resolving
    that some samples might not have every region. For missing regions, we
    insert zero tensors so the shapes still line up across the batch.
    """

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract each field across the list of samples into a list.
        # This one-liner builds six lists in the same order as instances.
        lang_xs, vision_xs, mask_xs, region2areas, attention_masks, labels = tuple(
            [instance[key] for instance in instances]
            for key in ('lang_x', 'vision_x', 'mask_x', 'region2area', 'attention_mask', 'label')
        )

        # Stack text-related tensors into [B, ...] shapes.
        lang_xs = torch.cat([_.unsqueeze(0) for _ in lang_xs], dim=0)
        attention_masks = torch.cat([_.unsqueeze(0) for _ in attention_masks], dim=0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)

        # Prepare per-region packing lists. We start with all regions,
        # then drop truly-unused ones (regions that no sample has).
        vision_temp = {area: [] for area in REGIONS}
        mask_temp = {area: [] for area in REGIONS}

        # Grab tensor shapes for fallback zero tensors when a region is missing.
        vision_shape = next(iter(vision_xs[0].values())).shape
        mask_shape = next(iter(mask_xs[0].values())).shape

        # We'll track regions that never appear in any sample of this batch.
        useless_regions = []

        # For every known region, iterate batch samples and collect tensors.
        for area in REGIONS:
            flag = False  # marks whether at least one sample has this region
            for i in range(len(vision_xs)):
                if area in vision_xs[i]:
                    # Sample i has this region: append real tensors
                    vision_temp[area].append(vision_xs[i][area])
                    mask_temp[area].append(mask_xs[i][area])
                    flag = True
                else:
                    # Sample i lacks this region: append zeros for alignment
                    vision_temp[area].append(torch.zeros(vision_shape))
                    mask_temp[area].append(torch.zeros(mask_shape))
            if not flag:
                # Entire batch lacks this region; we can drop it completely
                useless_regions.append(area)

        # The global image embedding is stored under 'image' inside each vision_x
        images = torch.cat([vision['image'].unsqueeze(0) for vision in vision_xs], dim=0)

        # Drop any region that did not show up in the batch (saves compute/memory)
        for area in useless_regions:
            vision_temp.pop(area)
            mask_temp.pop(area)
        useful_regions = list(vision_temp.keys())

        # Concatenate per-region lists into tensors with batch dimension first.
        vision_xs = {area: torch.cat([_.unsqueeze(0) for _ in vision_temp[area]], dim=0) for area in useful_regions}
        # Also attach the stacked global images.
        vision_xs['image'] = images

        # Same concat for the mask tensors.
        mask_xs = {area: torch.cat([_.unsqueeze(0) for _ in mask_temp[area]], dim=0) for area in useful_regions}

        # Package everything into a single dict for the Trainer to feed the model.
        return dict(
            lang_x=lang_xs,
            vision_x=vision_xs,
            mask_x=mask_xs,
            region2area=region2areas,
            attention_mask=attention_masks,
            labels=labels,
        )
        
def set_seed(seed: int):
    """
    Make results reproducible by fixing random seeds across libraries.

    Why: Without fixed seeds, randomness in data order, initialization,
    and CUDA kernels makes it hard to compare runs fairly.
    """
    # Python's own RNG
    random.seed(seed)
    # Numpy RNG
    np.random.seed(seed)
    # PyTorch CPU RNG
    torch.manual_seed(seed)
    # PyTorch CUDA RNG (all visible GPUs)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN pick deterministic algorithms where possible
    torch.backends.cudnn.deterministic = True
    # And disable heuristic benchmarking to avoid nondeterminism
    torch.backends.cudnn.benchmark = False
                 
def main():
    """
    Orchestrate the training run: parse args, build dataset + model,
    plug them into the Trainer, and run training.
    """
    # Fix seeds first so any randomness in dataset/model init is consistent.
    set_seed(42)

    # We let Hugging Face parse three dataclasses at once:
    # - ModelArguments: paths to LLM/tokenizer and vision checkpoints
    # - DataArguments:  where the CT volumes, masks, and CSV live
    # - TrainingArguments: batch size, epochs, logging, deepspeed, etc.
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Build the training dataset: this performs on-the-fly preprocessing
    # (crop foreground, resize to [256,256,64], HU clamp/normalize) and
    # assembles prompts + labels with special <image>/<region> tokens.
    print("Setup Data")
    Train_dataset = RadGenomeDataset_Train(
        text_tokenizer=model_args.tokenizer_path,
        data_folder=data_args.data_folder,
        mask_folder=data_args.mask_folder,
        csv_file=data_args.report_file,
        cache_dir=data_args.monai_cache_dir,
    )

    # Instantiate the multimodal model. Internally, Reg2RG loads the LLaMA
    # language model (wrapped with LoRA) and creates a visual embedding layer
    # that transforms CT volumes + region masks into token embeddings.
    print("Setup Model")
    model = Reg2RG(
        lang_model_path=model_args.lang_encoder_path,
        text_tokenizer_path=model_args.tokenizer_path,
        pretrained_visual_encoder=model_args.pretrained_visual_encoder,
        pretrained_adapter=model_args.pretrained_adapter,
    )

    # Hook everything up to the Trainer. We pass our custom DataCollator so
    # vision/mask dicts are properly batched.
    trainer = Trainer(
        model=model,
        train_dataset=Train_dataset,
        args=training_args,
        data_collator=DataCollator(),
    )

    # Run training according to TrainingArguments (epochs, deepspeed, etc.).
    # If a checkpoint path is provided, resume from that checkpoint.
    resume_ckpt = getattr(training_args, "resume_from_checkpoint", None)
    if resume_ckpt:
        print(f"Resuming training from checkpoint: {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()
    # Save the trainer state (useful for resuming/debugging); checkpoints are
    # handled separately according to save_strategy in TrainingArguments.
    trainer.save_state()
      
if __name__ == "__main__":
    main()
