# LIT 实验怎么跑 ✅

这个实验只做 **重建探针**（ProbeDecoder），不改主模型结构、不训练 LLM。  
目的：验证 `1024 tokens → 32 latents` 的压缩是否还能“复述”原始特征。

---

## 1) 先准备环境 🧰

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2) 直接运行（最常用）🚀

### ✅ 基线：解码 `Z`（pre-proj）

```bash
python src/lit_recon_probe.py \
  --tokenizer_path /path/to/Llama-2-7b-chat-hf \
  --pretrained_visual_encoder /path/to/RadFM_vit3d.pth \
  --pretrained_adapter /path/to/RadFM_perceiver_fc.pth \
  --decode_mode pre_proj \
  --data_folder /path/to/train_preprocessed \
  --mask_folder /path/to/train_region_mask \
  --report_file /path/to/train_region_report.csv \
  --val_data_folder /path/to/valid_preprocessed \
  --val_mask_folder /path/to/valid_region_mask \
  --val_report_file /path/to/valid_region_report.csv \
  --output_dir /path/to/results/LIT_pre_proj
```

### ✅ 对照：解码 `fc(Z)`（post-proj）

```bash
python src/lit_recon_probe.py \
  --tokenizer_path /path/to/Llama-2-7b-chat-hf \
  --pretrained_visual_encoder /path/to/RadFM_vit3d.pth \
  --pretrained_adapter /path/to/RadFM_perceiver_fc.pth \
  --decode_mode post_proj \
  --data_folder /path/to/train_preprocessed \
  --mask_folder /path/to/train_region_mask \
  --report_file /path/to/train_region_report.csv \
  --val_data_folder /path/to/valid_preprocessed \
  --val_mask_folder /path/to/valid_region_mask \
  --val_report_file /path/to/valid_region_report.csv \
  --output_dir /path/to/results/LIT_post_proj
```

### ✅ Validation 数据怎么指定

优先推荐用 **独立 valid 数据集**（上面三行 `--val_*`）。  
如果你的路径命名和默认一致（`train_preprocessed/train_region_mask/train_region_report.csv`），脚本也会自动推断出：

- `valid_preprocessed`
- `valid_region_mask`
- `valid_region_report.csv`

如果两者都没有，就会退回到 `--val_split` 从训练集里随机切分。

---

### ✅ 用 W&B 记录训练过程 + 保存最好的 3 个 epoch

```bash
python src/lit_recon_probe.py \
  --decode_mode pre_proj \
  --data_folder /path/to/train_preprocessed \
  --mask_folder /path/to/train_region_mask \
  --report_file /path/to/train_region_report.csv \
  --val_data_folder /path/to/valid_preprocessed \
  --val_mask_folder /path/to/valid_region_mask \
  --val_report_file /path/to/valid_region_report.csv \
  --use_wandb True \
  --wandb_project Reg2RG-LIT \
  --wandb_run_name LIT_preproj \
  --save_top_k 3 \
  --monitor_metric reg_cos \
  --monitor_mode max \
  --output_dir /path/to/results/LIT_pre_proj
```

> 不想联网：`--wandb_mode offline`（或 `disabled`）。  
> online 模式需要提前设置 `WANDB_API_KEY`。

---

## 3) 输出会在哪 📦

每次运行会生成：

```
<output_dir>/lit_metrics.csv
<output_dir>/checkpoints/epoch=XXX_val_*.pt   (如果开启 save_top_k)
```

字段包含：
- `cos`: Global CosSim  
- `reg_cos`: Region CosSim  
- `top1`: Top-1% token error（细节敏感）  
- `reg_top1`: Region 的 Top-1% token error  
- `decode_mode`: pre_proj / post_proj

---

## 4) 参数说明（最常用的）🧭

- `--decode_mode pre_proj | post_proj`  
  - `pre_proj`: 解码 Perceiver 输出 `Z`（768 维）
  - `post_proj`: 解码 `fc(Z)`（4096 维，再线性回 768）

- `--output_dir`  
  输出路径（会写入 `lit_metrics.csv`）

- `--num_train_epochs`  
  默认 15，ProbeDecoder 收敛很快 ⏱️

- `--val_data_folder / --val_mask_folder / --val_report_file`  
  使用独立 validation 数据（推荐）。只要设置其中任意一个，就必须三个都设置。

- `--val_split`  
  只有在没有独立 validation 数据时才会启用（从训练集随机切分）。

- `--use_wandb True`  
  启用 W&B 记录（需要安装 `wandb`）。

- `--save_top_k 3`  
  保存 val 上最好的 K 个 checkpoint（默认 3；设为 0 关闭）。

- `--monitor_metric reg_cos` / `--monitor_mode max|min`  
  选择“最好”的标准（默认最大化 `reg_cos`）。

---

## 5) 你应该怎么看结果 👀

- **Global CosSim 高 + Region CosSim 高** ✅  
  → 压缩后信息仍然保真

- **Global 高但 Region 低** ⚠️  
  → Region 细节更容易丢，值得进一步分析

- **pre_proj 明显好于 post_proj** ⚠️  
  → projection 可能在“扭曲/损失”信息

---

如果你想把这个实验扩展为 ROI 版本或 Delta Test，我可以在这个脚本基础上继续加。  
先把这两条跑通，就能得到非常清晰的“压缩是否保真”的证据了 💡
