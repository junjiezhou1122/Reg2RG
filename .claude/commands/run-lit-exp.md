---
description: Run LIT (Linear Information Transfer) reconstruction probe experiments
allowed-tools:
  - Bash
  - Read
  - Write
  - Glob
  - Grep
---

# LIT Experiment Runner

Run LIT reconstruction probe experiments for Reg2RG project.

## Available Experiments

| Exp | Name | Description |
|-----|------|-------------|
| 5b | Fresh 1-Layer | 共享 1-layer adapter, 随机初始化 |
| 9 | Separate Adapters | 分离 Global/Local adapter + decoder |

## User Request

$ARGUMENTS

## Instructions

1. **Identify the experiment** from the user's request
2. **Check if a script exists** in `scripts/run_exp*.sh`
3. **If script exists**, show it and ask if they want to run it
4. **If no script**, create one based on the experiment configuration

## Experiment Configurations

### Exp 5b: Fresh 1-Layer (Baseline)
```bash
--adapter_depth 1 \
--random_init_adapter True \
--train_adapter True \
--decoder_layers 4 \
--wandb_project Reg2RG-LIT-Exp5 \
--wandb_run_name exp5b_fresh_1layer_joint
```

### Exp 9: Separate Global/Local Adapters
```bash
--adapter_depth 1 \
--separate_adapters True \
--random_init_adapter True \
--train_adapter True \
--decoder_layers 4 \
--wandb_project Reg2RG-LIT-Exp9 \
--wandb_run_name exp9_separate_1layer_joint
```

## Common Parameters (Server Paths)

```bash
--tokenizer_path /mnt/home/zhoujunjie/models/Llama-2-7b-chat-hf
--pretrained_visual_encoder /mnt/home/zhoujunjie/models/Reg2RG/RadFM_vit3d.pth
--data_folder /mnt2/ct/RadGenome-ChestCT/dataset/train_preprocessed
--mask_folder /mnt2/ct/RadGenome-ChestCT/dataset/train_region_mask
--report_file /mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/train_region_report.csv
--val_data_folder /mnt2/ct/RadGenome-ChestCT/dataset/valid_preprocessed
--val_mask_folder /mnt2/ct/RadGenome-ChestCT/dataset/valid_region_mask
--val_report_file /mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv
--monai_cache_dir /mnt2/ct/RadGenome-ChestCT/cache_lit
```

## Output Directory Pattern

```
/mnt/home/zhoujunjie/outputs/LIT_exp{N}/exp{N}_{variant}
```

## Actions

1. Show the user the script content
2. Ask which GPU to use (default: CUDA_VISIBLE_DEVICES=1)
3. Confirm before running
4. Monitor the output and report progress
