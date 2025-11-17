
#!/bin/bash

# Device settings
cuda_devices="3"

# Paths (aligned with train_radgenome/jhcpu7.sh)
lang_encoder_path="/mnt/home/zhoujunjie/models/Llama-2-7b-chat-hf"
tokenizer_path="/mnt/home/zhoujunjie/models/Llama-2-7b-chat-hf"
pretrained_visual_encoder="/mnt/home/zhoujunjie/models/Reg2RG/RadFM_vit3d.pth"
pretrained_adapter="/mnt/home/zhoujunjie/models/Reg2RG/RadFM_perceiver_fc.pth"
ckpt_path="/mnt/home/zhoujunjie/Reg2RG/outputs/Reg2RG_radgenome/pytorch_model.bin"
data_folder="/mnt2/ct/RadGenome-ChestCT/dataset/valid_preprocessed"
mask_folder="/mnt2/ct/RadGenome-ChestCT/dataset/valid_region_mask"
report_file="/mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv"
monai_cache_dir="/mnt2/ct/RadGenome-ChestCT/cache"
result_path="/mnt/home/zhoujunjie/Reg2RG/results/radgenome_combined_reports.csv"
