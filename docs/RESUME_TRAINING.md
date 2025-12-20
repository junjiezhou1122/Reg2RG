# 训练断点续传功能文档

## 概述

训练脚本支持从checkpoint恢复训练，包括：
- 自动保存interrupt checkpoint（Ctrl+C中断时）
- 保存top-k最佳checkpoint（基于验证集指标）
- 从任意checkpoint恢复训练状态

## 📋 Checkpoint类型

### 1. Interrupt Checkpoint（中断保存）

**触发时机**：
- 按 `Ctrl+C` 中断训练
- 收到 SIGTERM 信号

**保存位置**：
```
{output_dir}/checkpoints/interrupt_epoch={epoch}_step={step}_{reason}_{timestamp}.pt
```

**示例文件名**：
```
interrupt_epoch=003_step=1234_sigint_1702345678.pt
```

**保存内容**：
- `epoch`: 当前完成的epoch数
- `global_step`: 全局训练步数
- `decoder_state_dict`: Decoder权重
- `optimizer_state_dict`: 优化器状态（学习率、动量等）
- `model_args`, `data_args`, `train_args`: 训练配置

### 2. Top-K Checkpoint（最佳模型）

**触发时机**：
- 每个epoch结束后
- 验证集指标达到top-k最佳

**保存位置**：
```
{output_dir}/checkpoints/epoch={epoch}_val_{metric}={score}.pt
```

**示例文件名**：
```
epoch=005_val_reg_cos=0.245600.pt
```

**保存内容**：
- `epoch`: Epoch数
- `model_state_dict`: **完整模型权重**（包括encoder、adapter、decoder）
- `optimizer_state_dict`: 优化器状态
- `train_metrics`: 训练集指标
- `val_metrics`: 验证集指标
- 所有配置参数

**配置参数**：
- `--save_top_k 3`: 保留最好的3个checkpoint
- `--monitor_metric reg_cos`: 监控指标
- `--monitor_mode max`: max表示越大越好，min表示越小越好

## 🚀 使用方法

### 方法1：从Interrupt Checkpoint恢复

**步骤1：查找最新的interrupt checkpoint**

```bash
# 查看所有interrupt checkpoint
ls -lt /mnt/home/zhoujunjie/outputs/LIT/checkpoints/ | grep interrupt

# 输出示例：
# interrupt_epoch=003_step=7854_sigint_1702345678.pt
# interrupt_epoch=002_step=5236_sigint_1702123456.pt
```

**步骤2：使用resume参数恢复训练**

```bash
CUDA_VISIBLE_DEVICES=3 python3 src/lit_recon_probe.py \
  --tokenizer_path /mnt/home/zhoujunjie/models/Llama-2-7b-chat-hf \
  --pretrained_visual_encoder /mnt/home/zhoujunjie/models/Reg2RG/RadFM_vit3d.pth \
  --pretrained_adapter /mnt/home/zhoujunjie/models/Reg2RG/RadFM_perceiver_fc.pth \
  --decode_mode pre_proj \
  --data_folder /mnt2/ct/RadGenome-ChestCT/dataset/train_preprocessed \
  --mask_folder /mnt2/ct/RadGenome-ChestCT/dataset/train_region_mask \
  --report_file /mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/train_region_report.csv \
  --val_data_folder /mnt2/ct/RadGenome-ChestCT/dataset/valid_preprocessed \
  --val_mask_folder /mnt2/ct/RadGenome-ChestCT/dataset/valid_region_mask \
  --val_report_file /mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv \
  --monai_cache_dir /mnt2/ct/RadGenome-ChestCT/cache_lit \
  --output_dir /mnt/home/zhoujunjie/outputs/LIT \
  --num_train_epochs 10 \
  --batch_size 2 \
  --dataloader_num_workers 4 \
  --gradient_accumulation_steps 4 \
  --precache_splits none \
  --use_wandb True \
  --wandb_project Reg2RG-LIT \
  --wandb_run_name LIT_train_resumed \
  --resume_from_checkpoint /mnt/home/zhoujunjie/outputs/LIT/checkpoints/interrupt_epoch=003_step=7854_sigint_1702345678.pt
```

### 方法2：从Top-K Best Checkpoint恢复

**步骤1：查找最佳checkpoint**

```bash
# 查看所有top-k checkpoint
ls -lt /mnt/home/zhoujunjie/outputs/LIT/checkpoints/ | grep epoch=

# 输出示例（按reg_cos从高到低排序）：
# epoch=008_val_reg_cos=0.312500.pt  ← 最好
# epoch=005_val_reg_cos=0.287300.pt
# epoch=003_val_reg_cos=0.245600.pt
```

**步骤2：使用最佳checkpoint恢复**

```bash
--resume_from_checkpoint /mnt/home/zhoujunjie/outputs/LIT/checkpoints/epoch=008_val_reg_cos=0.312500.pt
```

## 📊 恢复时的输出信息

成功恢复时会看到：

```
[resume] Loading checkpoint from: .../interrupt_epoch=003_step=7854_sigint_1702345678.pt
[resume] Restored decoder weights
[resume] Restored optimizer state
[resume] Resuming from epoch 4 (completed epoch 3)
[resume] Restored global_step: 7854
[resume] Resume complete. Starting training from epoch 4

[train] epoch 4/10:   0%|          | 0/24126 [00:00<?, ?it/s]
```

## ⚙️ 恢复的内容

### ✅ 会恢复的内容

1. **模型权重**：
   - Interrupt checkpoint: 仅Decoder权重
   - Top-K checkpoint: 完整模型（Encoder + Adapter + Decoder）

2. **优化器状态**：
   - 学习率
   - Adam的动量参数（m, v）
   - 优化器的内部状态

3. **训练进度**：
   - Epoch计数
   - Global step计数

### ❌ 不会恢复的内容

以下内容使用**命令行参数**提供的值：

- Learning rate（可以改变学习率继续训练）
- Batch size
- Number of workers
- 其他训练超参数

## 🔍 常见场景

### 场景1：训练被意外中断

```bash
# 原始训练运行到epoch 3被中断
# 查找interrupt checkpoint
ls -lt output/checkpoints/ | grep interrupt | head -1

# 恢复训练（保持所有参数不变）
python src/lit_recon_probe.py \
  [所有原始参数] \
  --resume_from_checkpoint output/checkpoints/interrupt_epoch=003_...pt
```

### 场景2：调整学习率继续训练

```bash
# 从epoch 8继续，但降低学习率
python src/lit_recon_probe.py \
  [所有原始参数] \
  --learning_rate 5e-5 \  # 从1e-4降到5e-5
  --resume_from_checkpoint output/checkpoints/epoch=008_val_reg_cos=0.312500.pt
```

### 场景3：从最佳checkpoint fine-tune更多epochs

```bash
# 原本训练10个epochs，现在想继续训练到15个epochs
python src/lit_recon_probe.py \
  [所有原始参数] \
  --num_train_epochs 15 \  # 从10改到15
  --resume_from_checkpoint output/checkpoints/epoch=010_val_reg_cos=0.345600.pt
```

### 场景4：调整batch size继续训练

```bash
# 从interrupt恢复，但改用更大的batch size
python src/lit_recon_probe.py \
  [所有原始参数] \
  --batch_size 4 \  # 从2改到4
  --gradient_accumulation_steps 2 \  # 相应调整，保持有效batch=8
  --resume_from_checkpoint output/checkpoints/interrupt_epoch=005_...pt
```

## ⚠️ 注意事项

### 1. Checkpoint文件路径

- 使用**绝对路径**或相对于执行目录的路径
- 确保checkpoint文件存在且可读

### 2. 模型架构必须匹配

恢复时的模型架构参数必须与保存时一致：
- `--vis_dim`
- `--llm_dim`
- `--perceiver_num`
- `--decoder_layers`
- `--decode_mode`

**错误示例**：
```bash
# 原始训练使用 --decode_mode pre_proj
# 恢复时改成 post_proj 会失败
--resume_from_checkpoint xxx.pt \
--decode_mode post_proj  # ❌ 不匹配，会报错
```

### 3. Epoch计数逻辑

- Checkpoint保存的是**已完成的epoch数**
- 恢复后从**下一个epoch**开始

**示例**：
```
interrupt_epoch=003_...pt  ← 完成了epoch 3
恢复后从epoch 4开始训练
```

### 4. W&B日志连续性

如果使用Weights & Biases：
- `global_step` 会正确恢复，保证日志连续
- 建议使用不同的 `--wandb_run_name` 区分恢复的run

**推荐做法**：
```bash
# 原始训练
--wandb_run_name LIT_train_original

# 恢复训练
--wandb_run_name LIT_train_resumed_from_epoch3
```

### 5. 数据路径必须一致

恢复训练时，确保数据路径没有变化：
- `--data_folder`
- `--mask_folder`
- `--report_file`
- `--monai_cache_dir`

## 🐛 故障排查

### 错误1: FileNotFoundError

```
FileNotFoundError: Checkpoint not found: /path/to/checkpoint.pt
```

**解决方法**：
- 检查路径是否正确
- 使用绝对路径
- 确认文件存在：`ls -lh /path/to/checkpoint.pt`

### 错误2: KeyError或RuntimeError

```
RuntimeError: Error(s) in loading state_dict for Decoder
```

**原因**：模型架构不匹配

**解决方法**：
- 确保所有模型参数与原始训练一致
- 检查 `--decode_mode`, `--decoder_layers` 等参数

### 错误3: 恢复后loss异常

**现象**：恢复后loss突然变得很大或很小

**可能原因**：
1. 学习率设置不当
2. Optimizer状态没有正确恢复

**解决方法**：
- 确认checkpoint包含 `optimizer_state_dict`
- 检查是否意外修改了 `--learning_rate`

## 📚 相关文档

- [训练参数说明](./TRAINING_ARGS.md)
- [MONAI Cache机制](./MONAI_CACHE.md)
- [性能优化指南](./PERFORMANCE.md)

## 💡 最佳实践

1. **定期检查checkpoint**：
   ```bash
   # 每天检查一次checkpoint情况
   ls -lth output/checkpoints/ | head -10
   ```

2. **备份重要checkpoint**：
   ```bash
   # 备份最佳checkpoint
   cp output/checkpoints/epoch=010_val_reg_cos=0.345600.pt \
      backup/best_model_$(date +%Y%m%d).pt
   ```

3. **使用tmux/screen**：
   ```bash
   # 防止SSH断开导致训练中断
   tmux new -s training
   python src/lit_recon_probe.py [参数]
   # Ctrl+B, D 退出tmux

   # 重新连接
   tmux attach -t training
   ```

4. **监控训练进度**：
   ```bash
   # 查看最新的interrupt checkpoint时间
   ls -lt output/checkpoints/ | grep interrupt | head -1

   # 这样可以判断训练是否还在运行
   ```

## 📝 示例脚本

创建一个便捷的恢复脚本 `resume_latest.sh`：

```bash
#!/bin/bash

# 自动找到最新的interrupt checkpoint并恢复训练

CKPT_DIR="/mnt/home/zhoujunjie/outputs/LIT/checkpoints"
LATEST_CKPT=$(ls -t $CKPT_DIR/interrupt_*.pt 2>/dev/null | head -1)

if [ -z "$LATEST_CKPT" ]; then
    echo "Error: No interrupt checkpoint found in $CKPT_DIR"
    exit 1
fi

echo "Found latest checkpoint: $LATEST_CKPT"
echo "Resuming training..."

CUDA_VISIBLE_DEVICES=3 python3 src/lit_recon_probe.py \
  --tokenizer_path /mnt/home/zhoujunjie/models/Llama-2-7b-chat-hf \
  --pretrained_visual_encoder /mnt/home/zhoujunjie/models/Reg2RG/RadFM_vit3d.pth \
  --pretrained_adapter /mnt/home/zhoujunjie/models/Reg2RG/RadFM_perceiver_fc.pth \
  --decode_mode pre_proj \
  --data_folder /mnt2/ct/RadGenome-ChestCT/dataset/train_preprocessed \
  --mask_folder /mnt2/ct/RadGenome-ChestCT/dataset/train_region_mask \
  --report_file /mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/train_region_report.csv \
  --val_data_folder /mnt2/ct/RadGenome-ChestCT/dataset/valid_preprocessed \
  --val_mask_folder /mnt2/ct/RadGenome-ChestCT/dataset/valid_region_mask \
  --val_report_file /mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv \
  --monai_cache_dir /mnt2/ct/RadGenome-ChestCT/cache_lit \
  --output_dir /mnt/home/zhoujunjie/outputs/LIT \
  --num_train_epochs 10 \
  --batch_size 2 \
  --dataloader_num_workers 4 \
  --gradient_accumulation_steps 4 \
  --precache_splits none \
  --use_wandb True \
  --wandb_project Reg2RG-LIT \
  --wandb_run_name LIT_train_auto_resumed \
  --resume_from_checkpoint "$LATEST_CKPT"
```

使用方法：
```bash
chmod +x resume_latest.sh
./resume_latest.sh
```

---

**文档版本**: v1.0
**最后更新**: 2025-12-20
**作者**: Claude + zhoujunjie
