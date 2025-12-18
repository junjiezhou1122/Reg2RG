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

---

## 6) TODO：下一步实验清单（可持续追加）🧪

> 建议每次跑完把命令 + 关键指标（`val/reg_cos`、`val/cos`、`val/mse`、`val/top1`）记录到 W&B 或 `lit_metrics.csv`，并把最好的 ckpt 路径记下来。

### A. 先把评估做“快反馈”
- [ ] 做一个 `valid_small`（例如 200–1000 例）用于快速迭代（减少每个 epoch 的等待）
- [ ] 确认 train/val 的样本数是否正确（避免 val 误配成 train）
- [ ] 固定随机种子，重复跑 2 次看方差（确认曲线/指标稳定）

### B. 核心消融：压缩到底丢没丢信息（看 val）
- [ ] `perceiver_num` 扫描：`32 → 16 → 8`（压缩更狠应显著掉分）
- [ ] `decode_mode` 对照：`pre_proj` vs `post_proj`（验证 projection 是否扭曲信息）
- [ ] 上界对照：不用压缩（直接用 `x` 或更高 `perceiver_num`）估计“理论最好能到哪”
- [ ] 破坏性对照：打乱/随机化 `z`（例如 shuffle latents）训练 decoder，应明显掉分（排除“偶然好看”）

### C. Decoder 容量与损失权重（避免 decoder 太强造成误判）
- [ ] `decoder_layers` 扫描：`1/2/4/6`（看 val 提升是否来自 decoder 变强）
- [ ] `lambda_region` 扫描：`0 / 0.1 / 0.3 / 1.0`（global vs region trade-off）
- [ ] 监控指标从 `reg_cos` 切到 `loss/mse/top1` 做对照（避免只看一个指标）

### D. 速度与资源（共享服务器友好）
- [ ] Token cache：缓存冻结 ViT 的 `x_g`（必要时再缓存 `x_region`），训练阶段直接读 cache（大幅减少 NIfTI I/O + CPU transforms）
- [ ] DataLoader 参数 sweep：`num_workers/pin_memory/persistent_workers`（目标：提高 GPU 利用率、降低 swap）
- [ ] 如需用双卡：把脚本改成 DDP（否则多卡不会自动变快）

### E. 结果可解释性（写论文/做分析用）
- [ ] 按 region 拆分指标（每个器官单独 `reg_cos/reg_top1`），找最难的部位
- [ ] 统计 “最差 token”（top-1%）集中在哪些 slice/区域（定位压缩丢失点）
- [ ] 可视化：token 的 PCA/TSNE 或 cosine 分布（global vs region 的差异）

你可以把新的实验条目继续往这个列表里加（保持一条=一个可执行的对照/消融）。  
