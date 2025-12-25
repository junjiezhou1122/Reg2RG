# Adapter Usefulness Probe: Quick Test

**Goal**: 在VLM训练前，快速判断adapter压缩的向量是否包含有用语义信息

**Date**: 2025-12-25

---

## 问题背景

**核心疑问**: 重建质量好（reg_cos > 0.85）≠ 信息对VLM有用

需要一个**轻量级测试**在Stage 2前验证adapter的有效性。

---

## 探测任务设计

### Probe 1: 区域存在性分类（最简单）

**任务**: 给定compressed vector，预测8个区域是否存在病灶

```python
# 数据：使用你的训练集
Input: adapter输出的32个latent tokens (B, 32, 768)
Label: 8个二分类标签 [肺, 心脏, 甲状腺, ...]
  1 = 该区域有异常/病灶
  0 = 该区域正常

# 模型：简单的线性分类器（probe不应太复杂！）
probe = nn.Sequential(
    nn.Linear(32 * 768, 256),
    nn.ReLU(),
    nn.Linear(256, 8),
    nn.Sigmoid()
)

# 训练：只训练probe，adapter冻结
loss = BCELoss(probe(adapter_output), labels)

# 评估：
if accuracy > 75%:
    print("✅ Adapter包含区域语义信息")
else:
    print("❌ Adapter可能只有低级特征")
```

**优点**:
- 快速（1-2小时训练）
- 直接测试语义理解
- 不需要LLM

---

### Probe 2: 区域报告关键词预测（中等）

**任务**: 预测每个区域的报告中是否包含关键词

```python
# 从report中提取关键词
keywords = ["normal", "nodule", "mass", "effusion", "pneumonia", ...]

# 对每个region
Input: region的compressed vector
Label: 该region报告中的关键词（多标签分类）

# 如果probe准确率高 → adapter保留了语义
```

---

### Probe 3: 与原始VLM Adapter对比（最直接）

**任务**: 在**同一个VLM**上测试两个adapter

```python
# 对比实验
Baseline: 预训练的RadFM adapter (frozen)
Exp2:     你的joint-trained adapter

# 用同样的VLM decoder（LoRA微调）测试
if Exp2_VLM_metric > Baseline_VLM_metric:
    improvement = "Exp2 adapter更有用"
else:
    improvement = "Joint training没帮助"
```

---

## 快速判断法（无需额外实验）

### 信号1: Region Reconstruction分布

**检查每个region的重建质量是否均衡**

```python
# 在validation后，分析每个region的cos
region_cos = {
    "lung": 0.88,
    "heart": 0.85,
    "thyroid": 0.45,  # ⚠️ 太低！
    "esophagus": 0.42,
    ...
}

# 判断标准
if std(region_cos) < 0.1:
    print("✅ 所有region都保留了信息")
else:
    print("⚠️ 小器官信息丢失严重")
```

**为什么重要？**
- 小器官（甲状腺、食管）在医学报告中很重要
- 如果只重建好大器官（肺、心），VLM报告会有遗漏

---

### 信号2: Top1 Error的空间分布

**检查最差1%的token分布在哪里**

```python
# 假设你能可视化top1 error的位置
bad_tokens_distribution = analyze_top1_locations(model)

# 如果最差token集中在：
if concentrated_in("boundaries", "small_organs"):
    print("✅ 正常，边界确实难重建")
elif concentrated_in("random_noise"):
    print("⚠️ 模型在拟合噪声")
```

---

### 信号3: 对比Exp1 vs Exp2的重建"模式"

**定性分析重建结果**

```python
# 可视化几个样本的重建
for sample in val_set[:5]:
    original = sample["vision_tokens"]
    recon_exp1 = decoder(frozen_adapter(original))
    recon_exp2 = decoder(trained_adapter(original))

    # 人眼检查：
    # - Exp1: 是否重建了纹理，但丢失了结构？
    # - Exp2: 是否保留了区域轮廓/语义边界？
```

**定性判断**:
- 如果Exp2重建**更模糊但语义边界清晰** → 可能学到了抽象特征（好）
- 如果Exp2重建**更锐利但语义错乱** → 过拟合到像素（坏）

---

## 实施建议

### 最小成本方案（推荐先做）

**只需在当前Stage 1基础上增加一行代码**：

```python
# 在validation loop中，记录每个region的cos
region_metrics = {
    "lung_left": [],
    "lung_right": [],
    "heart": [],
    ...
}

for batch in val_loader:
    for region_name, region_tensor in batch.items():
        cos = compute_cos(reconstruct(region_tensor), encode(region_tensor))
        region_metrics[region_name].append(cos)

# Epoch结束后，打印
for name, scores in region_metrics.items():
    print(f"{name}: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")
```

**判断**:
```python
if all_regions_mean > 0.75 and std_across_regions < 0.15:
    confidence = "高：adapter很可能保留了有用信息"
elif small_organs_mean < 0.60:
    confidence = "低：小器官信息丢失，VLM可能表现差"
```

---

### 中等成本方案（如果有时间）

**实现Probe 1（区域存在性分类）**

预计时间：
- 数据准备：1小时（提取adapter输出 + 标注区域标签）
- 训练probe：2小时（简单MLP）
- 分析结果：1小时

总计：**半天时间**，即可提前知道adapter质量

---

### 最终答案方案（必须做）

**Stage 2 VLM训练** - 这是不可替代的最终测试

---

## 预期结果与解读

| Stage 1指标 | Probe准确率 | Stage 2 VLM | 结论 |
|-----------|-----------|-------------|------|
| reg_cos > 0.85 | >75% | 提升显著 | ✅ Perfect！Joint training有效 |
| reg_cos > 0.85 | >75% | 提升微弱 | ⚠️ 重建好，但信息对VLM不critical |
| reg_cos > 0.85 | <60% | 无提升 | ❌ 只学到低级特征 |
| reg_cos < 0.70 | <60% | 无提升 | ❌ 失败，信息丢失严重 |

---

## 哲学思考

**为什么会出现"重建好但无用"的情况？**

1. **目标不对齐**
   - Stage 1优化：MSE + Cosine（像素级目标）
   - Stage 2需要：语义级特征（报告生成目标）

2. **信息压缩的选择**
   - Adapter可能优先保留：易重建的低频信息（纹理）
   - 而丢失：难重建但重要的高频信息（病灶边界）

3. **线性可访问 ≠ 有用**
   - LIT probe测试：信息是否线性可解码
   - VLM需要：信息对语言模型是否有意义

**解决方案**: Exp2的joint training正是试图对齐这两个目标！

---

**Last Updated**: 2025-12-25
