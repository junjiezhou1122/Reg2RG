# LIT Reconstruction Probe - 架构文档

## 📋 目录
- [核心思想](#核心思想)
- [整体架构](#整体架构)
- [详细流程图](#详细流程图)
- [模块详解](#模块详解)
- [训练策略](#训练策略)
- [实验设计](#实验设计)
- [代码结构](#代码结构)

---

## 🎯 核心思想

### **一句话总结**
测试将CT的1024个特征token压缩成32个后，能保留多少信息？

### **测试方法**
训练一个decoder尝试从32个压缩token还原回1024个原始token，通过重建质量评估信息保留程度。

### **为什么叫"Probe"（探针）？**
- Decoder像一个"探针"，探测压缩表征中残留的信息
- **只训练decoder**，不修改encoder/compressor
- 测试的是"压缩表征本身"的信息量，而非整个系统的表达能力

---

## 🏗️ 整体架构

### **5个核心模块**

| 模块 | 状态 | 输入 | 输出 | 作用 |
|------|------|------|------|------|
| **1. ViT Encoder** | ❄️ 冻结 | CT图像 (512³) | 1024 tokens (768维) | 提取视觉特征 |
| **2. Perceiver Adapter** | ❄️ 冻结 | 1024 tokens | 32 tokens (768维) | 信息压缩 (1024→32) |
| **3. Projection Layer** | ❄️ 冻结 | 32 tokens (768维) | 32 tokens (4096维) | 投影到LLM空间 |
| **4. ProbeDecoder** | 🔥 训练 | 32 tokens | 1024 tokens (768维) | 重建原始特征 |
| **5. Loss Calculation** | - | 重建 vs 原始 | Loss值 | 评估重建质量 |

---

## 📐 详细流程图

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                           FORWARD PASS（前向传播）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                            原始CT图像
                          (512×512×512)
                                │
                                ↓
              ┌───────────────────────────────┐
              │     ViT 3D Encoder ❄️         │  ← 预训练，冻结
              │     ┌─────────────────────┐   │     12层Transformer
              │     │ Patch Embedding     │   │     Patch: 32×32×4
              │     │ 512³ → 16×16×128    │   │
              │     └─────────────────────┘   │
              │              │                │
              │              ↓                │
              │     ┌─────────────────────┐   │
              │     │ 12 Transformer      │   │
              │     │ Layers              │   │
              │     │ (Self-Attention +   │   │
              │     │  FeedForward)       │   │
              │     └─────────────────────┘   │
              └───────────────────────────────┘
                                │
                                ↓
                    【Ground Truth - 保存用于比较】
                      tokens_original (x_g)
                     (batch, 1024, 768)  ◄────────────────┐
                                │                          │
                                ↓                          │
              ┌───────────────────────────────┐           │
              │  Perceiver Adapter ❄️         │           │
              │  ┌─────────────────────────┐  │           │
              │  │ Learnable Latents       │  │           │  比较！
              │  │ (32 queries)            │  │           │  计算Loss
              │  └─────────────────────────┘  │           │
              │              │                 │           │
              │              ↓                 │           │
              │  ┌─────────────────────────┐  │           │
              │  │ Cross-Attention         │  │           │
              │  │ (queries attend to      │  │           │
              │  │  1024 input tokens)     │  │           │
              │  └─────────────────────────┘  │           │
              └───────────────────────────────┘           │
                                │                          │
                                ↓                          │
                      compressed (z_g)                    │
                     (batch, 32, 768)                     │
                                │                          │
              ┌─────────────────┴──────────────┐          │
              │                                 │          │
              ↓                                 ↓          │
    decode_mode="pre_proj"          decode_mode="post_proj"
              │                                 │
              ↓                                 ↓
          Identity                   ┌────────────────────┐
          (no-op)                    │ FC Layer ❄️        │
              │                      │ Linear(768→4096)   │
              │                      └────────────────────┘
              │                                 │
              │                                 ↓
              │                        (batch, 32, 4096)
              │                                 │
              │                      ┌────────────────────┐
              │                      │ mem_proj           │
              │                      │ Linear(4096→768)   │
              │                      └────────────────────┘
              │                                 │
              └─────────────────┬───────────────┘
                                ↓
                      memory (for decoder)
                     (batch, 32, 768)
                                │
                                ↓
              ┌───────────────────────────────┐
              │     ProbeDecoder 🔥            │  ← 唯一训练的部分！
              │                               │
              │  ┌─────────────────────────┐  │
              │  │ Learnable Query Tokens  │  │  随机初始化
              │  │ (1, 1024, 768)          │  │  通过训练优化
              │  │        +                │  │
              │  │ 3D Positional Encoding  │  │  编码空间位置
              │  │ (batch, 1024, 768)      │  │  (h, w, d)
              │  └─────────────────────────┘  │
              │              │                 │
              │              ↓                 │
              │  ┌─────────────────────────┐  │
              │  │ Layer 1:                │  │
              │  │ ┌─────────────────────┐ │  │
              │  │ │ Self-Attention      │ │  │  queries互相看
              │  │ │ (Q, K, V = queries) │ │  │
              │  │ └─────────────────────┘ │  │
              │  │         │               │  │
              │  │         ↓               │  │
              │  │ ┌─────────────────────┐ │  │
              │  │ │ Cross-Attention     │◄┼──┼─ 从memory提取信息
              │  │ │ Q=queries           │ │  │  (32 tokens)
              │  │ │ K,V=memory          │ │  │
              │  │ └─────────────────────┘ │  │
              │  │         │               │  │
              │  │         ↓               │  │
              │  │ ┌─────────────────────┐ │  │
              │  │ │ FeedForward         │ │  │  非线性变换
              │  │ │ (MLP: expand 4x)    │ │  │
              │  │ └─────────────────────┘ │  │
              │  └─────────────────────────┘  │
              │              │                 │
              │              ↓                 │
              │  ┌─────────────────────────┐  │
              │  │ Layer 2:                │  │
              │  │ - Self-Attention        │  │
              │  │ - Cross-Attention       │◄─┼─ 进一步精细化
              │  │ - FeedForward           │  │
              │  └─────────────────────────┘  │
              │              │                 │
              │              ↓                 │
              │  ┌─────────────────────────┐  │
              │  │ LayerNorm               │  │
              │  └─────────────────────────┘  │
              └───────────────────────────────┘
                                │
                                ↓
                    【Reconstruction】
                 tokens_reconstructed (x_hat_g)
                     (batch, 1024, 768)  ◄────────────────┘
                                │
                                ↓
              ┌───────────────────────────────┐
              │     Loss Calculation          │
              │                               │
              │  1. LayerNorm(x) & (x_hat)    │
              │     ↓                          │
              │  2. MSE Loss                  │
              │     mse = mean((x_hat - x)²)  │
              │     ↓                          │
              │  3. Cosine Similarity         │
              │     cos = <x_hat, x>          │
              │           ──────────           │
              │           |x_hat|·|x|          │
              │     ↓                          │
              │  4. Combined Loss             │
              │     loss = mse + (1 - cos)    │
              │     ↓                          │
              │  5. Diagnostic (top-k L2)     │
              │     top1% worst errors         │
              └───────────────────────────────┘
                                │
                                ↓
                             Loss
                                │
                                ↓
                    Backpropagation 🔥
                  (only update decoder)
                                │
                                ↓
                   optimizer.step()
                   (AdamW, lr=1e-4)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🔧 模块详解

### **1. ViT 3D Encoder**

```python
ViT(
    image_size=512,           # 空间分辨率 (H, W)
    frames=512,               # 深度 (D)
    image_patch_size=32,      # H,W patch大小
    frame_patch_size=4,       # D patch大小
    dim=768,                  # token维度
    depth=12,                 # Transformer层数
    heads=8,                  # 注意力头数
    mlp_dim=2048,            # MLP隐藏层维度
)
```

**输入输出**：
- 输入：`(batch, 1, 512, 512, 512)` - CT volume
- Patch化：`16×16×128 = 32,768` patches
- 实际：`1024` tokens (可能使用了更大的patch或池化)
- 输出：`(batch, 1024, 768)` - token序列

---

### **2. Perceiver Adapter**

```python
PerceiverResampler(
    dim=768,              # 输入/输出维度
    num_latents=32        # 压缩目标数量
)
```

**工作原理**：
1. 初始化32个可学习的latent queries
2. 通过Cross-Attention从1024个输入token中提取信息
3. 输出固定的32个压缩token

**数学表示**：
```
Latents = LearnableParameter(32, 768)
Output = CrossAttention(Q=Latents, K=Input, V=Input)
```

---

### **3. Projection Layer**

```python
fc = nn.Linear(768, 4096)  # 投影到LLM空间
```

**两种decode模式**：

| 模式 | 说明 | 解码来源 |
|------|------|----------|
| `pre_proj` | 从Perceiver输出解码 | (batch, 32, 768) |
| `post_proj` | 从投影后解码 | (batch, 32, 4096) → Linear → (batch, 32, 768) |

---

### **4. ProbeDecoder**

#### **4.1 架构**

```python
ProbeDecoder(
    num_tokens=1024,      # 要重建的token数量 (8×8×16)
    dim=768,              # token维度
    depth=2,              # decoder层数
    heads=8,              # 注意力头数
    ff_mult=4,            # FFN扩展系数
)
```

#### **4.2 组件**

**A. Learnable Query Tokens**
```python
self.query = nn.Parameter(torch.randn(1, 1024, 768))
```
- 随机初始化
- 通过训练优化
- 作为重建的"种子"

**B. 3D Positional Encoding**
```python
self.pos_embed = PositionEmbeddingLearned3d(
    dim // 3,              # 每个维度256维
    h_patch_num=16,        # 高度方向patch数
    w_patch_num=16,        # 宽度方向patch数
    d_patch_num=128        # 深度方向patch数
)
```
- 为每个token编码3D空间位置
- 让decoder知道重建哪个位置

**C. Decoder Layers**
```python
CrossAttentionDecoderLayer(
    dim=768,
    heads=8,
    ff_mult=4
)
```

每层包含：
1. **Self-Attention**：queries之间互相协调
2. **Cross-Attention**：从memory中提取信息
3. **FeedForward**：非线性变换

#### **4.3 Forward流程**

```python
def forward(memory, grid, batch_size):
    # 1. 准备query + 位置编码
    pos = self.pos_embed(batch_size, h, w, d, memory)
    q = self.query.expand(batch_size, -1, -1) + pos

    # 2. 通过decoder layers
    for layer in self.layers:
        q = layer(q, memory)  # 逐层精细化

    # 3. 最终归一化
    return self.norm(q)
```

---

### **5. Loss Functions**

#### **5.1 MSE Loss**
```python
x_ln = LayerNorm(x)
x_hat_ln = LayerNorm(x_hat)
mse = mean((x_hat_ln - x_ln)²)
```
- 衡量重建的**幅度误差**
- LayerNorm使loss对缩放不敏感

#### **5.2 Cosine Similarity**
```python
x_norm = normalize(x)
x_hat_norm = normalize(x_hat)
cos = mean(<x_norm, x_hat_norm>)
```
- 衡量重建的**方向一致性**
- 值域：[-1, 1]，越接近1越好

#### **5.3 Combined Loss**
```python
loss = mse + (1 - cos)
```
- MSE：关注幅度
- (1-cos)：关注方向

#### **5.4 Top-k L2 Error（诊断）**
```python
def l2_error_topk(x_hat, x, ratio=0.01):
    err = norm(x_hat - x, dim=-1)  # 每个token的L2误差
    k = int(len(err) * ratio)       # 取最差的1%
    return err.topk(k).mean()       # 平均误差
```
- 找出重建最差的token
- 帮助诊断局部问题

---

## 🎯 训练策略

### **参数冻结策略**

```
❄️ Frozen (不更新梯度):
   ├─ ViT Encoder (全部)
   ├─ Perceiver Adapter (全部)
   └─ FC Projection (全部)

🔥 Trainable (更新梯度):
   └─ ProbeDecoder
      ├─ query (learnable tokens)
      ├─ pos_embed (3D positional encoding)
      ├─ layers (decoder layers)
      └─ norm (final layernorm)
```

### **为什么只训练Decoder？**

1. **测试信息保留**：
   - 不改变压缩过程
   - 只测试"能从压缩表征中提取多少信息"

2. **避免过拟合**：
   - 如果训练整个pipeline，可能学会"绕过"压缩
   - 只训练decoder确保测试的是压缩质量

3. **公平比较**：
   - 不同decode_mode可以公平对比
   - 压缩表征保持一致

---

### **训练参数**

```python
# 推荐配置
TrainArguments(
    num_train_epochs=15,              # 训练轮数
    learning_rate=1e-4,               # 学习率
    weight_decay=0.01,                # L2正则化
    batch_size=1,                     # 批次大小
    gradient_accumulation_steps=8,    # 梯度累积（重要！）
    lambda_region=1.0,                # 区域loss权重
)
```

**关键设置**：
- `gradient_accumulation_steps=8`：有效batch size = 1×8 = 8
- 小batch size：显存限制（CT volume很大）
- 梯度累积：弥补小batch size

---

### **优化器**

```python
optimizer = torch.optim.AdamW(
    model.decoder.parameters(),  # 只优化decoder
    lr=1e-4,
    weight_decay=0.01
)
```

---

## 🧪 实验设计

### **LIT (Linear Information Theory) Probe的核心问题**

**研究问题**：
> Perceiver将1024个token压缩成32个后，丢失了多少信息？

**假设**：
- H1：如果decoder能高质量重建 → 信息保留良好
- H0：如果decoder重建质量差 → 信息大量丢失

---

### **实验设置**

#### **1. 两个重建任务**

```python
# Task 1: 全局图像重建
vision_x["image"]  # 完整CT (512×512×512)
  ↓
1024 tokens
  ↓
32 compressed
  ↓
重建 1024 tokens

# Task 2: 区域重建
vision_x["lung"]   # 肺部区域 (256×256×64)
  ↓
1024 tokens
  ↓
32 compressed
  ↓
重建 1024 tokens
```

#### **2. 两种decode模式对比**

| 模式 | 解码源 | 假设 |
|------|--------|------|
| `pre_proj` | Perceiver输出 (vis_dim) | 测试压缩前的信息保留 |
| `post_proj` | FC投影后 (llm_dim) | 测试投影后的信息保留 |

---

### **评估指标**

| 指标 | 含义 | 好的结果 | 差的结果 |
|------|------|----------|----------|
| **cos (Cosine)** | 方向一致性 | > 0.95 | < 0.70 |
| **mse** | 幅度误差 | < 0.1 | > 0.5 |
| **top1** | 最差1%误差 | < 2.0 | > 5.0 |
| **reg_cos** | 区域余弦 | > 0.90 | < 0.65 |

**重要性排序**：
1. `reg_cos` > `cos`：区域重建更能反映解剖信息保留
2. `cos` > `mse`：方向比幅度重要
3. `top1`：诊断性指标，找出问题区域

---

### **预期结果解读**

#### **情况1：高质量重建（✅ 好）**
```
Metrics:
  cos: 0.96
  reg_cos: 0.94
  mse: 0.08
  top1: 1.5

结论：
  ✅ 32个token成功保留了关键解剖信息
  ✅ Perceiver压缩是信息保留型的
  ✅ 可以安全用于下游任务
```

#### **情况2：中等质量重建（⚠️ 一般）**
```
Metrics:
  cos: 0.85
  reg_cos: 0.80
  mse: 0.25
  top1: 3.5

结论：
  ⚠️ 部分信息丢失
  ⚠️ 某些解剖区域重建较差
  ⚠️ 需要调整压缩率或方法
```

#### **情况3：低质量重建（❌ 差）**
```
Metrics:
  cos: 0.65
  reg_cos: 0.60
  mse: 0.50
  top1: 6.0

结论：
  ❌ 大量信息丢失
  ❌ 32个token不足以表示原始特征
  ❌ 需要增加压缩token数量或改进方法
```

---

## 📊 数据流与维度变化

### **完整Pipeline**

```
阶段 1: 编码
─────────────
CT Image                    ViT Encoder
(512×512×512)          →    (B, 1024, 768)
   │                           │
   │ 分成16×16×128个patch      │ 每个patch → 768维向量
   │ (patch=32×32×4)           │
   └─────────────────────────┘

阶段 2: 压缩
─────────────
(B, 1024, 768)        Perceiver        (B, 32, 768)
   │                     │                  │
   │ 1024个特征向量      │ 压缩32倍         │ 32个latent向量
   └───────────────────┴──────────────────┘

阶段 3: 投影（可选）
────────────────────
(B, 32, 768)           FC Layer         (B, 32, 4096)
   │                     │                  │
   │ Vision空间          │ 线性变换         │ LLM空间
   └───────────────────┴──────────────────┘

阶段 4: 重建
─────────────
(B, 32, X)          ProbeDecoder      (B, 1024, 768)
   │                     │                  │
   │ 压缩表征            │ Cross-Attn       │ 重建的特征
   │                     │ 2层Transformer   │
   └───────────────────┴──────────────────┘

阶段 5: 比较
─────────────
Ground Truth: (B, 1024, 768)  ─┐
                               ├─→ Loss = mse + (1-cos)
Reconstructed: (B, 1024, 768) ─┘
```

---

### **区域处理流程**

```
Full Image (512³)          →    (B, 1024, 768)
Region (lung, 256³)        →    (B, 1024, 768)  # 同样的token数
Region (heart, 256³)       →    (B, 1024, 768)
   ↓                              ↓
压缩 (32 tokens)           压缩 (32 tokens)
   ↓                              ↓
重建 (1024 tokens)         重建 (1024 tokens)
   ↓                              ↓
Loss (global)              Loss (region)

Total Loss = loss_global + λ_region × loss_region
            (λ_region = 1.0)
```

---

## 💻 代码结构

### **文件组织**

```
src/lit_recon_probe.py
├─ IMPORTS                    # 库导入
├─ REGIONS                    # 解剖区域列表
├─ CONFIGURATION CLASSES      # 配置类
│  ├─ ModelArguments          # 模型配置
│  ├─ DataArguments           # 数据配置
│  └─ TrainArguments          # 训练配置
├─ DATA COLLATION             # 数据处理
│  └─ LITDataCollator         # 批次合并
├─ DECODER ARCHITECTURE       # 解码器架构
│  ├─ CrossAttentionDecoderLayer  # 单层decoder
│  └─ ProbeDecoder            # 完整decoder
├─ MAIN MODEL                 # 主模型
│  └─ LITProbeModel           # LIT Probe模型
├─ LOSS FUNCTIONS             # 损失函数
│  ├─ l2_error_topk           # Top-k L2误差
│  ├─ recon_loss              # 重建损失
│  └─ set_requires_grad       # 冻结/解冻
└─ MAIN TRAINING LOOP         # 训练主循环
   └─ main()                  # 入口函数
```

---

### **关键类与函数**

#### **1. LITProbeModel**

```python
class LITProbeModel(nn.Module):
    def __init__(self, vis_dim, llm_dim, perceiver_num,
                 decoder_layers, decoder_heads, decoder_ff_mult,
                 decode_mode):
        # 初始化5个模块
        self.vision_encoder = ViT(...)      # ❄️
        self.adapter = PerceiverResampler(...)  # ❄️
        self.fc = nn.Linear(...)            # ❄️
        self.mem_proj = ...                 # 根据mode
        self.decoder = ProbeDecoder(...)    # 🔥

    def encode_tokens(self, volume):
        # CT volume → tokens
        tokens, grid = self.vision_encoder(volume)
        return tokens, grid

    def compress_tokens(self, tokens):
        # tokens → compressed
        z = self.adapter(tokens)
        if self.decode_mode == "post_proj":
            z = self.fc(z)
        return z

    def decode_tokens(self, memory, grid):
        # compressed → reconstructed
        memory = self.mem_proj(memory)
        return self.decoder(memory, grid, batch_size)
```

#### **2. ProbeDecoder**

```python
class ProbeDecoder(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, ff_mult):
        self.query = nn.Parameter(...)      # 可学习query
        self.pos_embed = PositionEmbeddingLearned3d(...)
        self.layers = nn.ModuleList([
            CrossAttentionDecoderLayer(...)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, memory, grid, batch_size):
        # 1. query + pos
        pos = self.pos_embed(batch_size, h, w, d, memory)
        q = self.query.expand(batch_size, -1, -1) + pos

        # 2. 通过decoder layers
        for layer in self.layers:
            q = layer(q, memory)

        # 3. 归一化
        return self.norm(q)
```

#### **3. Training Loop**

```python
def main():
    # 1. 解析参数
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    # 2. 加载数据
    train_loader, val_loader = ...

    # 3. 初始化模型
    model = LITProbeModel(...)

    # 4. 加载预训练权重
    model.vision_encoder.load_state_dict(...)
    model.adapter.load_state_dict(...)

    # 5. 冻结encoder/adapter
    set_requires_grad(model.vision_encoder, False)
    set_requires_grad(model.adapter, False)
    set_requires_grad(model.decoder, True)

    # 6. 训练循环
    for epoch in range(num_epochs):
        train_metrics = run_epoch(train_loader, train=True)
        val_metrics = run_epoch(val_loader, train=False)

        # 记录指标
        log_metrics(epoch, train_metrics, val_metrics)
```

---

## 🔬 实验要点

### **关键设计决策**

| 决策 | 选择 | 理由 |
|------|------|------|
| **只训练decoder** | ✅ | 测试信息保留，避免过拟合 |
| **使用LayerNorm** | ✅ | Loss对尺度不敏感 |
| **Cosine + MSE** | ✅ | 同时考虑方向和幅度 |
| **Region Loss** | ✅ | 关注解剖区域的重建 |
| **Top-k Error** | ✅ | 诊断局部问题 |
| **Gradient Accumulation** | ✅ | 弥补小batch size |

---

### **超参数调优建议**

#### **核心参数**

```python
# 最重要的3个参数：
gradient_accumulation_steps = 8    # 有效batch size
learning_rate = 1e-4               # 学习率
lambda_region = 1.0                # 区域权重

# 次要参数：
decoder_layers = 2                 # decoder深度
decoder_heads = 8                  # 注意力头数
weight_decay = 0.01                # L2正则化
```

#### **调参建议**

1. **如果训练不稳定**：
   - 降低learning_rate (1e-4 → 5e-5)
   - 增加weight_decay (0.01 → 0.05)
   - 增加gradient_accumulation_steps (8 → 16)

2. **如果重建质量差**：
   - 增加decoder_layers (2 → 4)
   - 增加decoder_heads (8 → 16)
   - 调整lambda_region (尝试0.5-2.0)

3. **如果过拟合**：
   - 增加weight_decay
   - 减少decoder_layers
   - 添加dropout

---

## 📈 预期结果与分析

### **成功的标志**

```
训练曲线：
  - train_cos 稳步上升到 0.95+
  - val_cos 跟随上升，无明显过拟合
  - reg_cos 与 cos 接近（差距<0.05）
  - loss 稳定下降

最终指标（Epoch 15）：
  - val_cos: 0.94 - 0.97
  - val_reg_cos: 0.90 - 0.95
  - val_mse: 0.05 - 0.15
  - val_top1: 1.5 - 2.5
```

### **如何解读结果**

#### **场景A：pre_proj vs post_proj**

```
如果 pre_proj 更好：
  → Perceiver输出已经保留了足够信息
  → 投影到LLM空间可能损失信息

如果 post_proj 更好：
  → 投影后的表征质量更高
  → 可能FC layer起到了feature refinement作用
```

#### **场景B：全局 vs 区域**

```
如果 reg_cos << cos：
  → 区域特异性信息丢失
  → 可能需要增加perceiver_num

如果 reg_cos ≈ cos：
  → 信息保留均衡
  → 压缩策略合理
```

---

## 🎓 总结

### **核心贡献**

1. **量化信息损失**：
   - 提供了评估压缩质量的定量方法
   - 不仅看downstream任务，还看信息保留

2. **区域评估**：
   - 不只评估全局，还评估解剖区域
   - 对医学应用更有意义

3. **公平对比**：
   - 冻结encoder/compressor
   - 不同decode_mode可公平比较

---

### **方法优势**

✅ **简单直接**：只训练decoder，易于理解和实现
✅ **定量评估**：提供明确的数值指标
✅ **诊断能力**：top-k error帮助找出问题区域
✅ **灵活性强**：可用于任何encoder-compressor架构

---

### **局限性**

⚠️ **线性假设**：假设信息可以线性解码（实际可能需要非线性）
⚠️ **重建目标**：重建ViT特征，不是原始CT（可能不够直观）
⚠️ **计算成本**：需要额外训练decoder

---

## 📚 参考资料

### **相关工作**

- **Linear Probes**: Evaluating learned representations
- **Perceiver**: General perception with iterative attention
- **Vision Transformers**: ViT for 3D medical imaging

### **代码文件**

- `src/lit_recon_probe.py` - 主训练脚本
- `src/Model/vit_3d.py` - 3D Vision Transformer
- `src/Model/helpers.py` - Perceiver Resampler
- `src/Dataset/radgenome_dataset_train.py` - 数据加载

---

## ✨ Quick Start

### **训练命令**

```bash
python src/lit_recon_probe.py \
    --pretrained_visual_encoder /path/to/vit.pth \
    --pretrained_adapter /path/to/perceiver.pth \
    --decode_mode pre_proj \
    --output_dir ./outputs/LIT \
    --num_train_epochs 15 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --use_wandb
```

### **评估结果**

```bash
# 查看CSV日志
cat outputs/LIT/lit_metrics.csv

# 可视化（如果使用W&B）
wandb online
# 访问 https://wandb.ai/your-project
```

---

**最后更新**：2025-12-17
**版本**：v1.0
**作者**：Reg2RG Team
