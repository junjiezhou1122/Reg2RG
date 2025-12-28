#!/bin/bash
# Exp 9: Separate Global/Local 1-Layer Adapters
#
# 实验目的: 验证分离 Global/Local Adapter 是否比共享 Adapter 更好
#
# 架构:
#   Global Image  → [Global Adapter (1-layer, random init)] → [Global Decoder] → Global Loss
#   Local Regions → [Local Adapter (1-layer, random init)]  → [Local Decoder]  → Local Loss
#
# 对比基线: Exp 5b (共享 1-layer adapter)
#
# 预期参数量:
#   - Global Adapter: ~600K
#   - Local Adapter:  ~600K
#   - Global Decoder: ~2.4M (4层)
#   - Local Decoder:  ~2.4M (4层)
#   - Total: ~6M (是 Exp 5b 的 2x)

CUDA_VISIBLE_DEVICES=1 python3 src/lit_recon_probe.py \
    --tokenizer_path /mnt/home/zhoujunjie/models/Llama-2-7b-chat-hf \
    --pretrained_visual_encoder /mnt/home/zhoujunjie/models/Reg2RG/RadFM_vit3d.pth \
    --adapter_depth 1 \
    --separate_adapters True \
    --random_init_adapter True \
    --decode_mode pre_proj \
    --data_folder /mnt2/ct/RadGenome-ChestCT/dataset/train_preprocessed \
    --mask_folder /mnt2/ct/RadGenome-ChestCT/dataset/train_region_mask \
    --report_file /mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/train_region_report.csv \
    --val_data_folder /mnt2/ct/RadGenome-ChestCT/dataset/valid_preprocessed \
    --val_mask_folder /mnt2/ct/RadGenome-ChestCT/dataset/valid_region_mask \
    --val_report_file /mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv \
    --monai_cache_dir /mnt2/ct/RadGenome-ChestCT/cache_lit \
    --decoder_layers 4 \
    --train_adapter True \
    --output_dir /mnt/home/zhoujunjie/outputs/LIT_exp9/exp9_separate_1layer \
    --num_train_epochs 20 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --dataloader_num_workers 2 \
    --log_every 10 \
    --val_check_interval 0 \
    --early_stopping_patience 3 \
    --save_top_k 3 \
    --monitor_metric reg_cos \
    --monitor_mode max \
    --precache_splits none \
    --seed 42 \
    --use_wandb True \
    --wandb_project Reg2RG-LIT-Exp9 \
    --wandb_run_name exp9_separate_1layer_joint \
    --wandb_mode online
