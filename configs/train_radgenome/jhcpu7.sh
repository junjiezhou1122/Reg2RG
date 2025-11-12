# This file holds settings for a training run.
# Think of it like a packing list the training script reads.

# Experiment settings
# A short, human-friendly name for this run (used in output paths)
experiment_name="Reg2RG_radgenome"
# Use bfloat16 precision to save memory and speed up training (if supported)
bf16=True

# Device settings
# Which GPUs to use (comma-separated IDs). Example: 0,1 uses GPU 0 and 1
cuda_devices="3,4"  

# Torchrun settings
# A free TCP port that worker processes use to communicate
master_port=25368

# Paths
# Path to the LLM weights (e.g., LLaMA-2)
lang_encoder_path="/jhcnas5/chenzhixuan/checkpoints/Llama-2-7b-chat-hf"
# Path to the tokenizer files (often same directory as the LLM)
tokenizer_path="/jhcnas5/chenzhixuan/checkpoints/Llama-2-7b-chat-hf"
# Pretrained 3D vision encoder checkpoint (for CT volumes)
pretrained_visual_encoder="/jhcnas5/chenzhixuan/MyOpenSource/huggingface/Reg2RG/RadFM_vit3d.pth"
# Adapter (Perceiver/FC) checkpoint that maps vision features to language space
pretrained_adapter="/jhcnas5/chenzhixuan/MyOpenSource/huggingface/Reg2RG/RadFM_perceiver_fc.pth"
# Folder with preprocessed training CT volumes
data_folder='/data/chenzhixuan/data/RadGenome-ChestCT/dataset/train_preprocessed'
# Folder with region masks aligned to the CT volumes
mask_folder='/data/chenzhixuan/data/RadGenome-ChestCT/dataset/train_region_mask'
# CSV file that holds region-level training sentences
report_file='/data/chenzhixuan/data/RadGenome-ChestCT/dataset/radgenome_files/train_region_report.csv'
# Optional cache directory for MONAI transforms (not used by this script)
monai_cache_dir='/jhcnas5/chenzhixuan/data/RadGenome-ChestCT/cache' # not used here
# Where to save checkpoints and logs for this experiment
output_dir="/jhcnas5/chenzhixuan/checkpoints/Reg2RG/outputs/$experiment_name"
# DeepSpeed config file for memory/throughput optimizations
deepspeed_config="/home/chenzhixuan/Workspace/Reg2RG/ds_configs/stage2.json"

# Training settings
# Step size for learning; too big can diverge, too small can be slow
learning_rate=5e-5
# Number of samples per GPU per step (CT volumes are large, so keep small)
per_device_train_batch_size=1
# How many times to iterate over the whole dataset
num_train_epochs=10
# Accumulate gradients for this many steps before updating weights
gradient_accumulation_steps=8
# Whether to run eval during training ("no" means skip evaluation here)
evaluation_strategy="no"
# When to save checkpoints ("epoch" means after each full pass)
save_strategy="epoch"
# Keep only this many recent checkpoints to save disk space
save_total_limit=3
# L2 regularization strength; 0.0 means disabled
weight_decay=0.0
# Number of warmup steps before using the full learning rate
warmup_steps=20
# Learning rate schedule: warmup then keep constant
lr_scheduler_type="constant_with_warmup"
# Number of parallel data-loading worker processes
dataloader_num_workers=8
# Log training metrics every N steps
logging_steps=1

