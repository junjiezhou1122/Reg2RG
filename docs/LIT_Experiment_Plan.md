# Latent Integrity Test (LIT) 实验规划 🧪🧠

**一句话版本**：我们用一个“重建探针（ProbeDecoder）”去测试：`1024 vision tokens → 32 latents` 这种强压缩之后，信息到底还在不在。  
不看生成文本、不改主模型结构，只看“压缩后能不能把原特征复述回来” ✅

---

## 0) 我们要解决的疑问 ❓

训练报告生成模型时，大家最担心的是：
- 压缩太狠会不会把细节抹平？（尤其是 Region view）😰
- 压缩后的 latents 还能不能“代表原始视觉特征”？🤔

LIT 的目的就是：**把这个疑问变成一个可量化的实验指标** 📊

---

## 1) 实验目标（本版不做 ROI）🎯

本版先只关注两个最核心的指标：
- **Global CosSim**：global 视角的重建 cosine similarity（越高越好）🌍
- **Region CosSim**：region 视角的重建 cosine similarity（越高越好）🧩

为什么这样做？👇  
因为这两个指标：
- 不依赖 mask 对齐（避免 crop/resize 不一致带来的麻烦）🧼
- 能直接回答：**region 的压缩是不是更容易丢信息** 🧠

---

## 2) 固定不变的结构（我们不改 Adapter/Projection）🧱

我们测试的就是你主模型里现成的这条链路：

```
X      = VisionEncoder(volume)          # 原始视觉 token（GT 特征）
Z      = PerceiverResampler(X)          # 压缩 latents（长度=32，维度=768）
Z_proj = fc(Z)                          # 投影到 LLM 维（长度=32，维度=4096）
```

我们做两种重建输入（用于定位瓶颈在哪）🔍：
- **Decode(Z)**：只测 Perceiver 压缩后的信息量
- **Decode(fc(Z))**：测“Perceiver + projection”总链路的信息量

> 说明：Global 和 Region **共享同一个 PerceiverResampler**（这点很关键），所以我们也让 decoder 共享，才能检验“共享压缩语言”是否成立 🤝

---

## 3) 核心组件：ProbeDecoder（重建探针）🛠️

### 3.1 它做什么？
ProbeDecoder 的任务就是：  
给我 `Z`（32 个 latents），我尝试重建回 `X_hat`（1024 个 tokens）。

### 3.2 为什么用 Cross-Attention？
因为 `Z` 是“memory”，`X_hat` 对应的是每一个 patch 的 token。  
Cross-attention 很自然：**让每个 patch query 去“问”32 个 latents 里有没有我需要的信息** 🧲

### 3.3 结构建议（2 层就够做探针）🧩
- 输入 Query：`Q`，形状 `[B, 1024, 768]`
  - `Q = learnable_queries + pos_embed`（位置编码让每个 query 知道自己对应哪个 patch）📍
- 输入 Memory：`Z`（或 `Z_proj` 先过一个 `4096→768` 的线性层）🧠
- 输出：`X_hat`，形状 `[B, 1024, 768]`

为什么只用 2 层？👇  
因为我们要的是“信息是否存在”的测试，不希望 decoder 太强把结果“脑补出来” 🧠⚠️

---

## 4) 损失函数：HybridReconLoss 📉

我们同时用两个损失，让数值更稳：

```
L = λ_mse * MSE(LN(X_hat), LN(X))
  + λ_cos * (1 - CosSim(norm(X_hat), norm(X)))
```

推荐默认：`λ_mse=1, λ_cos=1` ✅  
为什么要 LN / norm？👇  
因为不同样本、不同 token 的尺度差异会影响 MSE；归一化后更像在比“形状/方向”而不是比“幅度” 🧽

---

## 5) 训练协议（只训练 Probe，不动主模型）🧊🔥

### 5.1 数据怎么用？
直接用你现有 training set，不需要文本标签 ✅  
每个样本我们都会产生两类视图：
- **Global view**：整幅（或全局处理后的）volume 🌍
- **Region view**：每个存在的 region volume 🧩

### 5.2 冻结/训练哪些模块？
- Freeze 🧊：VisionEncoder、PerceiverResampler（加载你最优权重）、fc
- Train 🔥：仅 ProbeDecoder

这样做的原因很简单：👇  
我们现在要测的是“**已训练好的 adapter/projection 是否保留信息**”，而不是重新训练它们 🧪

### 5.3 总 loss（global + region）
为了让两种视图都能被同一个 decoder 重建，我们用联合 loss：

```
L_total = L_global + λ_region * mean_k(L_region_k)
```

推荐：`λ_region=1`。  
为什么要对 region 做 mean_k？👇  
因为每个样本的 region 数 K 不一样，不平均会导致 region 多的样本主导训练 🧮

### 5.4 超参建议 ⚙️
- Optimizer：AdamW
- LR：`1e-4`
- Epoch：10–20（decoder 学得很快）
- Batch：保持和主任务一致（通常 1），配合 grad accumulation

---

## 6) 评估指标（本版只看 Global / Region）📊

验证集上至少记录：
- `Global CosSim`（↑ 越大越好）
- `Region CosSim`（↑ 越大越好）

建议额外记录一个“细节敏感”的长尾指标（不涉及 ROI，也很直观）🧷：
- **Top-1% 平均误差**：把 1024 个 token 的误差排序，取最大的 1%（约 10 个 token）求平均  
  - 作用：均值可能被“容易重建的 token”稀释，但长尾能更敏感地反映“小信息有没有被抹平” 🕵️

---

## 7) 对照/消融（保证结论可信）🧪

最推荐先做两个（成本低、信息量大）✅：

1) **Decode(Z) vs Decode(fc(Z))**  
   - 目的：定位 projection 是否让表示更难解码 🔍

2) **Pretrained adapter vs Random adapter**  
   - 目的：证明“用别人预训练的 Perceiver”是否真的更保真 🧠✨

---

## 8) 结果怎么读（快速决策规则）🧭

### 情况 A：Global 高，Region 也高 ✅
说明压缩后的 latents 对两种视图都足够保真：信息大概率“还在”。

### 情况 B：Global 高，但 Region 明显低 ⚠️
这是重要信号：region 压缩更容易丢信息（和我们对“小细节更难”的直觉一致）。  
下一步才值得考虑更强的分析（比如 ROI、delta test 等）。

### 情况 C：Decode(Z) 还行，但 Decode(fc(Z)) 掉得明显 ⚠️
说明 projection 可能在“改写/扭曲”表示，让它更难恢复。  
后续可以讨论：projection 学习率、是否冻结、是否加正则等（但这不是本版 LIT 的范围）。

---

## 9) 产出物（你最终能拿到什么）📦

- 一张表：Global/Region CosSim（+ Top-1% error 可选）📊
- 两条曲线：train/val 的 cosSim 与 loss 走势 📈
- 结论一句话：`32 latents` 对 global/region 的信息保真度是否足够 ✅

---

## 10) 备注（工程注意点）🧯

- 本实验前提是：你的主模型确实在用 **PerceiverResampler** 作为 adapter。  
  如果代码被临时改成了其他压缩器（比如 ConvResampler），需要切回 Perceiver 版本才能做“保留结构不变”的 LIT ✅

**方法二：** 做一个病灶 proxy
- 基于高频残差/异常激活的 pseudo-ROI
- 再用同样的 R3E 测试

---

## 9. 最终交付物

### 跑完实验应得到：

1. **一套 val 指标表**
   - global/region/ROI/R3E（含 Z vs fc(Z) 对比）

2. **2–3 个可视化样本**
   - 重建/误差热图 + ROI overlay

3. **一个明确结论**
   - 压缩是否对 ROI 产生"选择性信息丢失"
   - 损失主要来自 adapter 还是 projection

---

## 10. 下一步工作

确认按这个版本走后，需要定下：

1. **ProbeDecoder 的具体结构细节**
   - 2 层 decoder block 的具体维度
   - pos embedding 用 learnable 还是复用 3D pos

2. **Region K 的采样策略**
   - 全用 or 每步采样 K 个

---

## 实现文件结构（建议）

```
src/
├── Model/
│   ├── probe_decoder.py          # ProbeDecoder 实现
│   ├── lit_losses.py              # HybridReconLoss 实现
│   └── lit_metrics.py             # ROIMetricCalculator 实现
├── train_lit.py                   # LIT 训练脚本
└── eval_lit.py                    # LIT 评估脚本
```

---

**版本**: v1.0
**日期**: 2025-12-17
**状态**: 规划阶段
