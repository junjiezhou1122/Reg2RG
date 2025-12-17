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
  --output_dir /path/to/results/LIT_post_proj
```

---

## 3) 输出会在哪 📦

每次运行会生成：

```
<output_dir>/lit_metrics.csv
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
  默认 10，ProbeDecoder 收敛很快 ⏱️

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
