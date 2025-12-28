# Reg2RG Experiment Master Plan

**项目**: Region-Guided Radiology Report Generation
**Last Updated**: 2025-12-27
**Author**: Junjie Zhou

---

## 📊 实验总览

| Exp | 名称 | 状态 | 核心问题 | 关键发现 |
|-----|------|------|----------|----------|
| 1 | Decoder-Only Baseline | ✅ 完成 | Adapter 信息保留多少？ | cos ≈ 0.80 |
| 2 | Joint Adapter+Decoder | ✅ 完成 | 解冻 Adapter 能否提升？ | cos ≈ 0.85 (+5pts) |
| 5 | Minimal-Capacity Adapter | ✅ 完成 | 1-layer 是否更"诚实"？ | 训练中 |
| 6 | Region Size Analysis | 🔄 进行中 | Size 和重建质量的关系？ | 待分析 |
| 7 | Contrastive Alignment | 📋 计划中 | V-L 对齐能否改善生成？ | - |
| 8 | Adaptive Token Allocation | 📋 计划中 | 动态分配 tokens 数量？ | - |
| 9 | Separate Global/Local Weights | 📋 计划中 | 分离权重是否更优？ | - |
| **10** | **Anomaly Score Prediction** | 📋 计划中 | 异常检测假设验证 | - |
| **11** | **Normal Template Learning** | 📋 计划中 | 学习"正常"模板 | - |
| **12** | **Anomaly-Aware Generation** | 📋 计划中 | 异常引导报告生成 | - |
| **13** | **Medical LLM Comparison** | 📋 计划中 | 医学LLM vs 通用LLM | - |
| | | | | |
| | **=== Resolution Problem (Garbage In) ===** | | | |
| **14** | **Multi-Crop Encoding** | 📋 计划中 | 大region切块保留细节？ | - |
| **15** | **Diffusion Enhancement** | 📋 计划中 | Diffusion恢复丢失细节？ | - |
| **16** | **Lightweight Detail Enhancer** | 📋 计划中 | 轻量级残差恢复？ | - |
| **17** | **Resolution-Conditioned Perceiver** | 📋 计划中 | 让模型aware压缩程度？ | - |
| **18** | **Anti-Aliased Downsampling** | 📋 计划中 | Gaussian blur减少混叠？ | - |
| | | | | |
| | **=== New Ideas (Brainstorm) ===** | | | |
| **19** | **Contrastive Resolution Alignment** | 📋 计划中 | 压缩特征对齐原始特征？ | - |
| **20** | **Uncertainty-Aware Encoding** | 📋 计划中 | 预测压缩导致的不确定性？ | - |
| **21** | **Anatomy-Prior Enhancement** | 📋 计划中 | 解剖学先验指导恢复？ | - |
| **22** | **Masked Region Modeling** | 📋 计划中 | 自监督预测丢失信息？ | - |
| **23** | **Knowledge Distillation (High-Res)** | 📋 计划中 | 高分辨率模型蒸馏？ | - |
| | | | | |
| | **=== Advanced Resolution Ideas ===** | | | |
| **24** | **Frequency-Decomposed Encoding** | 📋 计划中 | FFT分离高低频分别处理？ | - |
| **25** | **Adaptive Patch Size** | 📋 计划中 | 高压缩区用更小patch？ | - |
| **26** | **Cascaded Resolution Enhancement** | 📋 计划中 | 渐进式多阶段恢复？ | - |
| **27** | **Region-Specific Strategy** | 📋 计划中 | 每种器官用不同策略？ | - |
| **28** | **Compression-Aware Attention** | 📋 计划中 | 注意力中加入压缩偏置？ | - |
| | | | | |
| | **=== ToolMed: New Architecture Paradigm ===** | | | |
| **30** | **Internal Tools Layer** | 📋 计划中 | 可微分内部工具实现？ | - |
| **31** | **Fusion Hub** | 📋 计划中 | 多工具输出融合？ | - |
| **32** | **Adapter + Reconstruction** | 📋 计划中 | 模块化工具添加？ | - |
| **33** | **External Tools Integration** | 📋 计划中 | 预训练模型作为工具？ | - |
| **34** | **Component Protocol** | 📋 计划中 | 标准化工具接口？ | - |

---

## ✅ 已完成实验

### Experiment 1: Decoder-Only Baseline (LIT Probe)

**日期**: 2025-12-24
**状态**: ✅ 完成

**研究问题**:
- 预训练的 Adapter (Perceiver Resampler) 保留了多少原始 ViT 特征信息？

**实验设置**:
```bash
--train_adapter False      # Adapter 冻结 ❄️
--decoder_layers 4         # 2 层 decoder
--decode_mode pre_proj     # 从 Perceiver 输出解码
```

**关键结果**:
| Metric | Global | Region |
|--------|--------|--------|
| Cosine | 0.65 | 0.80 |

**结论**:
- Adapter 保留了约 80% 的 region 信息
- Global 信息保留较差 (~65%)
- **重大发现**: Region cos > Global cos (反直觉！)

**相关文件**:
- `src/lit_recon_probe.py`

---

### Experiment 2: Joint Adapter+Decoder Training

**日期**: 2025-12-24
**状态**: ✅ 完成

**研究问题**:
- 解冻 Adapter 进行联合训练，是否能提升重建质量？

**实验设置**:
```bash
--train_adapter True       # Adapter 解冻 🔥
--decoder_layers 2
--decode_mode pre_proj
```

**关键结果**:
| Metric | Exp1 (Frozen) | Exp2 (Unfrozen) | Δ |
|--------|---------------|-----------------|---|
| Region Cos | 0.80 | 0.85 | +5 pts |
| Global Cos | 0.65 | ~0.70 | +5 pts |

**结论**:
- 解冻 Adapter 带来显著提升 (+5 points)
- 预训练 Adapter 不是最优的
- Joint training 让 Adapter 学习更适合重建的压缩

---

### Experiment 5: Minimal-Capacity Adapter

**日期**: 2025-12-25
**状态**: ✅ 代码完成，训练进行中

**研究问题**:
- 1-layer Adapter 是否比 6-layer 更"诚实"？
- 浅层网络是否强制保留更关键的信息？

**理论基础**:
- **Layer Laziness**: 深层网络早期层可能"偷懒"
- **Gradient Dilution**: 6 层反传导致梯度衰减
- **Minimal-Capacity Principle**: 最浅网络必须一次性保留关键信息

**实验设置**:

#### Exp 5a: Extracted First Layer
```bash
--adapter_depth 1
--load_first_layer_from_pretrained True
--train_adapter True
```

#### Exp 5b: Fresh 1-Layer
```bash
--adapter_depth 1
--random_init_adapter True
--train_adapter True
```

**相关文件**:
- `src/Model/one_layer_adapter.py`
- `src/Model/adapter_utils.py`
- `docs/Exp5_Implementation_Status.md`

---

## 🔄 进行中实验

### Experiment 6: Region Size vs Reconstruction Quality

**日期**: 2025-12-26
**状态**: 🔄 代码完成，等待数据

**研究问题**:
- Region size 和重建质量 (cos) 是否相关？
- "Smaller is Better" 假设验证

**已实现功能**:
- `region_stats` 数据结构
- Mask-based size 计算
- Pearson correlation 分析
- `region_metrics.csv` 输出

**预期结果**:
```
Pearson r < -0.3 → 确认 "Smaller is Better"
Pearson r > 0.3  → 更大 region 更容易重建
|r| < 0.3        → 无显著关系
```

**相关文件**:
- `src/lit_recon_probe.py`
- `docs/Region_Level_Statistics.md`

---

## 📋 计划中实验

### Experiment 7: Contrastive Vision-Language Alignment

**状态**: 📋 计划中
**优先级**: 🥈 中高
**论文卖点**: "Bridging the Gap: Contrastive Alignment for Medical VLMs"

**研究问题**:
- Visual tokens 和 text 是否在嵌入空间对齐？
- Contrastive loss 能否改善 LLM 对 visual features 的利用？

**核心动机**:
```
问题: 重建质量 85% 但报告生成 F1 只有 25%
     → Gap 可能来自 visual-text 未对齐

解决: Contrastive learning 强制对齐
     → Visual tokens 必须包含和 report 相关的信息
```

**实现方案**:

#### 7.1 Global Contrastive
```python
class ContrastiveAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        self.temperature = temperature
        self.visual_proj = nn.Linear(768, 512)
        self.text_proj = nn.Linear(4096, 512)

    def forward(self, visual_tokens, text_tokens):
        # Pool to single vector
        v_embed = F.normalize(self.visual_proj(visual_tokens.mean(1)), dim=-1)
        t_embed = F.normalize(self.text_proj(text_tokens.mean(1)), dim=-1)

        # Similarity matrix [B, B]
        logits = torch.matmul(v_embed, t_embed.T) / self.temperature
        labels = torch.arange(len(logits), device=logits.device)

        # Bidirectional loss
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.T, labels)

        return (loss_v2t + loss_t2v) / 2
```

#### 7.2 Region-Level Contrastive
```python
class RegionContrastiveLoss(nn.Module):
    """每个 region 和其对应的 report 段落对齐"""

    def forward(self, region_tokens_dict, region_reports_dict):
        total_loss = 0
        for region_name in region_tokens_dict:
            v_tokens = region_tokens_dict[region_name]
            t_tokens = self.encode_text(region_reports_dict[region_name])
            loss = self.contrastive(v_tokens, t_tokens)
            total_loss += loss
        return total_loss / len(region_tokens_dict)
```

**训练策略**:
```
阶段 1: Contrastive Pre-alignment (冻结 LLM)
  Loss = Contrastive Loss only

阶段 2: Joint Training
  Loss = Generation Loss + λ * Contrastive Loss
```

**评估指标**:
- V→T / T→V Retrieval Accuracy
- Report Generation F1 提升

---

### Experiment 8: Adaptive Token Allocation

**状态**: 📋 计划中
**优先级**: 🥇 高
**论文卖点**: "Content-Adaptive Compression for Medical VLMs"

**研究问题**:
- 不同 region 的复杂度不同，是否需要不同数量的 tokens？
- 动态分配能否提升整体效率和质量？

**核心观察**:
```
现有设计 (Fixed Allocation):
  Lung    → [8 tokens]  (复杂器官，8 tokens 可能不够)
  Heart   → [8 tokens]  (中等复杂)
  Thyroid → [8 tokens]  (简单器官，8 tokens 可能浪费)

问题:
  - 复杂器官被欠表示
  - 简单器官浪费容量
  - 总 token 数固定，分配不优
```

**实现方案**:

```python
class AdaptivePerceiver(nn.Module):
    """根据 region 复杂度动态分配 tokens"""

    def __init__(self, dim=768, token_options=[4, 8, 16, 32]):
        super().__init__()
        self.token_options = token_options

        # 复杂度预测器: 根据输入特征预测需要多少 tokens
        self.complexity_predictor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, len(token_options)),  # 输出每个选项的分数
        )

        # 为每个 token 数量创建一个 Perceiver
        self.perceivers = nn.ModuleDict({
            str(n): PerceiverResampler(dim=dim, num_latents=n, depth=2)
            for n in token_options
        })

    def forward(self, region_features):
        """
        Args:
            region_features: [B, num_tokens, dim]
        Returns:
            compressed: [B, selected_num_tokens, dim]
            selected_n: int (选择的 token 数量)
        """
        # 1. 预测复杂度
        pooled = region_features.mean(dim=1)  # [B, dim]
        complexity_logits = self.complexity_predictor(pooled)  # [B, num_options]

        # 2. 选择 token 数量 (训练时用 Gumbel-Softmax, 推理时用 argmax)
        if self.training:
            # Gumbel-Softmax for differentiable selection
            weights = F.gumbel_softmax(complexity_logits, tau=1.0, hard=True)
        else:
            # Hard selection during inference
            idx = complexity_logits.argmax(dim=-1)
            weights = F.one_hot(idx, num_classes=len(self.token_options)).float()

        # 3. 加权组合各 Perceiver 输出 (或选择单个)
        outputs = []
        for i, n in enumerate(self.token_options):
            out = self.perceivers[str(n)](region_features.unsqueeze(1).unsqueeze(1))
            out = out.squeeze(1)  # [B, n, dim]
            outputs.append(out * weights[:, i:i+1, None])

        # 4. 合并 (需要 padding 到最大长度)
        max_n = max(self.token_options)
        padded_outputs = []
        for i, out in enumerate(outputs):
            n = self.token_options[i]
            if n < max_n:
                pad = torch.zeros(out.shape[0], max_n - n, out.shape[2], device=out.device)
                out = torch.cat([out, pad], dim=1)
            padded_outputs.append(out)

        return sum(padded_outputs)  # [B, max_n, dim]
```

**简化版 (推理时选择)**:
```python
class AdaptivePerceiverSimple(nn.Module):
    """简化版: 推理时直接选择一个 Perceiver"""

    def __init__(self, dim=768):
        super().__init__()
        self.complexity_net = nn.Linear(dim, 4)  # 4 个选项

        self.perceiver_4 = PerceiverResampler(dim=dim, num_latents=4)
        self.perceiver_8 = PerceiverResampler(dim=dim, num_latents=8)
        self.perceiver_16 = PerceiverResampler(dim=dim, num_latents=16)
        self.perceiver_32 = PerceiverResampler(dim=dim, num_latents=32)

    def forward(self, x, return_complexity=False):
        # 预测复杂度
        complexity = self.complexity_net(x.mean(dim=1)).argmax(dim=-1)

        # 根据复杂度选择 Perceiver
        if complexity == 0:
            out = self.perceiver_4(x)
        elif complexity == 1:
            out = self.perceiver_8(x)
        elif complexity == 2:
            out = self.perceiver_16(x)
        else:
            out = self.perceiver_32(x)

        if return_complexity:
            return out, complexity
        return out
```

**训练策略**:
```
Loss = Reconstruction_Loss + λ * Token_Efficiency_Loss

其中 Token_Efficiency_Loss 鼓励用更少的 tokens:
  efficiency_loss = mean(selected_num_tokens) / max_tokens
```

**预期结果**:
| Region | 预期复杂度 | 预期 Tokens |
|--------|-----------|-------------|
| Lung | 高 | 16-32 |
| Heart | 中 | 8-16 |
| Liver | 中 | 8-16 |
| Thyroid | 低 | 4-8 |
| Esophagus | 低 | 4-8 |

**依赖**: Exp 6 的 region size/complexity 分析结果

---

### Experiment 9: Separate Global/Local Weights

**状态**: 📋 计划中
**优先级**: 🥇 高
**论文卖点**: "Scale-Specific Compression for Hierarchical Medical VLMs"

**研究问题**:
- Global 和 Local 的最优压缩策略是否不同？
- 分离权重是否比共享权重更好？

**核心观察**:
```
现有设计 (Shared Weights):
┌─────────────────────────────────────────────────────┐
│  Global (512³) ──┐                                  │
│                  ├──→ Shared Encoder ──→ Shared Adapter
│  Local (256³) ───┘                                  │
└─────────────────────────────────────────────────────┘

问题:
  Global 需要: 保留整体布局, 可以丢弃细节
  Local 需要:  保留局部细节, 可以丢弃全局位置

  共享权重 = 两边都不是最优的折中
```

**实现方案**:

#### 方案 A: 分离 Adapter (共享 Encoder)
```python
class SeparateAdapterModel(nn.Module):
    """Encoder 共享, Adapter 分离"""

    def __init__(self):
        # 共享的 Vision Encoder
        self.encoder = ViT3D(...)

        # 分离的 Adapters
        self.global_adapter = PerceiverResampler(
            dim=768,
            num_latents=32,
            depth=6,
        )
        self.local_adapter = PerceiverResampler(
            dim=768,
            num_latents=8,  # 每个 region 更少 tokens
            depth=4,        # 更浅，因为 local 更简单
        )

    def forward(self, global_volume, local_volumes):
        # Global path
        global_tokens = self.encoder(global_volume)
        global_compressed = self.global_adapter(global_tokens)

        # Local path (per region)
        local_compressed = {}
        for region_name, vol in local_volumes.items():
            tokens = self.encoder(vol)
            local_compressed[region_name] = self.local_adapter(tokens)

        return global_compressed, local_compressed
```

#### 方案 B: 完全分离 (Encoder + Adapter)
```python
class FullySeparateModel(nn.Module):
    """Encoder 和 Adapter 都分离"""

    def __init__(self):
        # Global path: 更大的感受野，更粗的特征
        self.global_encoder = ViT3D(
            image_patch_size=64,   # 更大的 patch
            frame_patch_size=8,
            depth=8,               # 更浅
        )
        self.global_adapter = PerceiverResampler(num_latents=32, depth=4)

        # Local path: 更小的感受野，更细的特征
        self.local_encoder = ViT3D(
            image_patch_size=32,   # 更小的 patch (更细节)
            frame_patch_size=4,
            depth=12,              # 更深 (提取更多细节)
        )
        self.local_adapter = PerceiverResampler(num_latents=8, depth=6)
```

#### 方案 C: 共享 Encoder + Scale-Conditioned Adapter
```python
class ScaleConditionedAdapter(nn.Module):
    """同一个 Adapter，但根据 scale 调整行为"""

    def __init__(self, dim=768, num_latents=32):
        super().__init__()
        self.base_adapter = PerceiverResampler(dim=dim, num_latents=num_latents)

        # Scale embedding
        self.scale_embed = nn.Embedding(2, dim)  # 0=global, 1=local

    def forward(self, tokens, is_global=True):
        # 添加 scale 信息
        scale_idx = 0 if is_global else 1
        scale_emb = self.scale_embed(torch.tensor([scale_idx], device=tokens.device))
        scale_emb = scale_emb.expand(tokens.shape[0], 1, -1)

        # Concat scale embedding
        tokens_with_scale = torch.cat([scale_emb, tokens], dim=1)

        return self.base_adapter(tokens_with_scale)
```

**实验设计**:
```
对比实验:
  A: Shared Encoder + Shared Adapter (现有 baseline)
  B: Shared Encoder + Separate Adapters
  C: Separate Encoders + Separate Adapters
  D: Shared Encoder + Scale-Conditioned Adapter

评估:
  - Global reconstruction cos
  - Local reconstruction cos
  - 参数量对比
  - 训练/推理速度
```

**预期结果**:
| Config | Global Cos | Local Cos | 参数量 |
|--------|------------|-----------|--------|
| A (Shared) | 0.65 | 0.74 | 1x |
| B (Sep Adapter) | 0.68? | 0.78? | 1.3x |
| C (Full Sep) | 0.70? | 0.80? | 2x |
| D (Conditioned) | 0.67? | 0.76? | 1.05x |

---

## 🧠 异常中心设计 (Anomaly-Centric Design)

### 核心思想

**第一性原理思考**:
```
传统方法: Image → Features → LLM → "漂亮的报告"
问题:     LLM 可能学会"编造"符合医学模板的文本，但不真正理解异常

第一性原理:
  1. 医学报告的核心是什么？→ 描述异常 (发现、定位、性质)
  2. AI 最重要的能力是什么？→ 检测并理解异常
  3. 正常结构有什么用？→ 作为对比基准

新方法: Image → "这里和正常不同" → "异常描述" → Report
```

**核心洞察**:
> "我觉得最重要的是 AI 可以理解这个图片的异常！而不是我要生成多好看的报告"

### 两阶段设计

```
┌─────────────────────────────────────────────────────────────────┐
│  阶段 1: Region-Level (当前可做)                                 │
│  ────────────────────────────────────────                       │
│  Region Features → Normal Template Bank → Deviation Detection   │
│                                                                 │
│  优点: 使用现有的 region masks                                   │
│  限制: 粒度较粗，无法定位具体病灶                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  阶段 2: Lesion-Level (需要 SAM-Med3D)                          │
│  ─────────────────────────────────────                          │
│  SAM-Med3D → Lesion Masks → Per-Lesion Features → Report        │
│                                                                 │
│  优点: 精确到病灶级别，更临床相关                                 │
│  依赖: SAM-Med3D 预训练模型 + 微调                               │
└─────────────────────────────────────────────────────────────────┘
```

---

### Experiment 10: Anomaly Score Prediction

**状态**: 📋 计划中
**优先级**: 🥇 高
**论文卖点**: "Anomaly-Aware Medical VLM: Understanding Before Generating"

**研究问题**:
- 模型能否预测每个 region 的异常程度？
- 异常分数和报告复杂度是否相关？

**核心假设**:
```
假设: 异常 regions 的 visual features 应该和 report 内容高度相关
     正常 regions 的 report 可以用模板生成

验证:
  1. 有异常的 region → 高异常分数 → 需要详细描述
  2. 正常的 region → 低异常分数 → 简单模板即可
```

**实现方案**:

```python
class AnomalyScorePredictor(nn.Module):
    """预测每个 region 的异常程度 (0-1)"""

    def __init__(self, dim=768):
        super().__init__()

        # 异常分数预测头
        self.score_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),  # 输出 0-1 分数
        )

        # 可选: 异常类型分类
        self.type_head = nn.Linear(dim, 5)  # 5种常见异常类型

    def forward(self, region_features):
        """
        Args:
            region_features: [B, num_tokens, dim] - 来自 adapter 的压缩特征
        Returns:
            anomaly_score: [B, 1] - 0=正常, 1=严重异常
            anomaly_type: [B, 5] - 异常类型 logits (可选)
        """
        # Pool tokens to single vector
        pooled = region_features.mean(dim=1)  # [B, dim]

        # Predict anomaly score
        score = self.score_head(pooled)  # [B, 1]

        # Predict anomaly type (optional)
        type_logits = self.type_head(pooled)  # [B, 5]

        return score, type_logits
```

**训练标签获取**:
```
方法 1: 从 report 文本推断
  - "未见明显异常" → score = 0.0
  - "轻度xxx" → score = 0.3
  - "中度xxx" → score = 0.6
  - "严重xxx" → score = 1.0

方法 2: 使用 GPT-4 标注
  prompt = "Based on this radiology report, rate the anomaly severity (0-1)..."

方法 3: 自监督 (无标签)
  - 正常样本: score 应该接近 0
  - 有 findings 的样本: score 应该 > 0
```

**评估指标**:
- Anomaly Score vs Report Length Correlation (异常分数和报告长度的相关性)
- ROC-AUC for Normal/Abnormal Classification
- Per-region Accuracy

---

### Experiment 11: Normal Template Learning

**状态**: 📋 计划中
**优先级**: 🥇 高
**论文卖点**: "Learning What's Normal: Template-Based Anomaly Detection for Medical VLMs"

**研究问题**:
- 如何学习每个 region 的"正常"特征模板？
- 偏离模板多少意味着异常？

**核心思想**:
```
┌─────────────────────────────────────────────────────┐
│  正常模板学习                                        │
│  ─────────────                                      │
│  1. 收集"正常"样本的 region features                 │
│  2. 学习每个 region 的 prototype/template            │
│  3. 新样本: 计算和 template 的偏离                   │
│                                                     │
│  [正常肺] ──→ Template ──→ 新样本偏离多少？→ 异常程度 │
└─────────────────────────────────────────────────────┘
```

**实现方案**:

```python
class NormalTemplateBank(nn.Module):
    """学习每个 region 的"正常"模板"""

    def __init__(self, num_regions=10, dim=768, num_latents=32):
        super().__init__()

        # 每个 region 一个可学习的"正常"模板
        # 形状: (num_latents, dim) - 和 adapter 输出一致
        self.normal_templates = nn.ParameterDict({
            region: nn.Parameter(torch.randn(num_latents, dim) * 0.02)
            for region in [
                "abdomen", "bone", "breast", "esophagus", "heart",
                "lung", "mediastinum", "pleura", "thyroid", "trachea"
            ]
        })

        # 偏离度编码器
        self.deviation_encoder = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def get_template(self, region_name):
        """获取指定 region 的正常模板"""
        return self.normal_templates[region_name]

    def compute_deviation(self, region_features, region_name):
        """
        计算当前特征和正常模板的偏离

        Args:
            region_features: [B, num_latents, dim]
            region_name: str
        Returns:
            deviation_features: [B, num_latents, dim] - 编码偏离信息
            deviation_score: [B] - 整体偏离程度
        """
        # 获取正常模板并扩展到 batch 维度
        template = self.normal_templates[region_name]  # [num_latents, dim]
        template = template.unsqueeze(0).expand(region_features.shape[0], -1, -1)

        # 计算差异
        diff = region_features - template  # [B, num_latents, dim]

        # 拼接原始特征和差异
        concat = torch.cat([region_features, diff], dim=-1)  # [B, num_latents, dim*2]

        # 编码偏离信息
        deviation_features = self.deviation_encoder(concat)  # [B, num_latents, dim]

        # 计算整体偏离分数 (L2 distance)
        deviation_score = torch.norm(diff, dim=-1).mean(dim=-1)  # [B]

        return deviation_features, deviation_score

    @torch.no_grad()
    def update_template_ema(self, normal_features, region_name, momentum=0.99):
        """
        使用 EMA 更新正常模板 (仅用正常样本)

        Args:
            normal_features: [B, num_latents, dim] - 来自正常样本
            momentum: EMA 动量 (0.99 = 慢更新)
        """
        current = self.normal_templates[region_name]
        new_mean = normal_features.mean(dim=0)  # [num_latents, dim]
        updated = momentum * current + (1 - momentum) * new_mean
        self.normal_templates[region_name].data = updated
```

**训练策略**:
```
阶段 1: 模板初始化 (使用正常样本)
  1. 筛选 report 中标记为"正常"的样本
  2. 提取 region features
  3. 使用 EMA 更新 templates

阶段 2: 偏离检测训练
  Loss = MSE(predicted_deviation_score, labeled_anomaly_score)
       + Contrastive(normal_features, abnormal_features)

阶段 3: 联合训练
  Loss = Generation_Loss + λ * Deviation_Consistency_Loss
```

---

### Experiment 12: Anomaly-Aware Report Generation

**状态**: 📋 计划中
**优先级**: 🥈 中高
**论文卖点**: "Deviation-Guided Report Generation: Focusing on What Matters"

**研究问题**:
- 如何让 LLM "关注"偏离信息？
- 异常区域应该得到更详细的描述

**实现方案**:

```python
class AnomalyAwareVLM(nn.Module):
    """异常感知的视觉语言模型"""

    def __init__(self, base_vlm, dim=768):
        super().__init__()

        self.encoder = base_vlm.encoder
        self.adapter = base_vlm.adapter
        self.llm = base_vlm.llm

        # 新增组件
        self.template_bank = NormalTemplateBank(dim=dim)
        self.anomaly_scorer = AnomalyScorePredictor(dim=dim)

        # 异常 token 权重调节器
        self.attention_weight_adjuster = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, images, region_masks, region_names):
        """
        异常感知的前向传播
        """
        all_visual_tokens = []
        all_anomaly_info = []

        for region_name in region_names:
            # 1. 提取 region features
            region_vol = extract_region(images, region_masks[region_name])
            tokens = self.encoder(region_vol)
            compressed = self.adapter(tokens)  # [B, num_latents, dim]

            # 2. 计算偏离
            deviation_features, deviation_score = self.template_bank.compute_deviation(
                compressed, region_name
            )

            # 3. 预测异常分数
            anomaly_score, _ = self.anomaly_scorer(compressed)

            # 4. 融合: 原始特征 + 偏离特征 (异常区域得到更多偏离信息)
            alpha = self.attention_weight_adjuster(anomaly_score)  # [B, 1]
            alpha = alpha.unsqueeze(1)  # [B, 1, 1]
            fused = compressed + alpha * deviation_features

            all_visual_tokens.append(fused)
            all_anomaly_info.append({
                "region": region_name,
                "score": anomaly_score,
                "deviation": deviation_score,
            })

        # 5. 拼接所有 region tokens
        visual_input = torch.cat(all_visual_tokens, dim=1)

        # 6. 添加异常提示 (可选)
        anomaly_prompt = self.build_anomaly_prompt(all_anomaly_info)

        # 7. LLM 生成
        report = self.llm(visual_input, anomaly_prompt)

        return report, all_anomaly_info

    def build_anomaly_prompt(self, anomaly_info):
        """
        根据异常信息构建提示

        Example output:
        "Focus on the following regions with detected anomalies:
         - Lung (anomaly score: 0.8): Pay extra attention
         - Heart (anomaly score: 0.2): Likely normal
         ..."
        """
        lines = ["Detected anomaly levels:"]
        for info in anomaly_info:
            score = info["score"].mean().item()
            if score > 0.5:
                lines.append(f"- {info['region']}: HIGH ({score:.2f}) - describe in detail")
            elif score > 0.2:
                lines.append(f"- {info['region']}: MODERATE ({score:.2f})")
            else:
                lines.append(f"- {info['region']}: LOW ({score:.2f}) - likely normal")
        return "\n".join(lines)
```

**训练损失**:
```python
def anomaly_aware_loss(
    generated_report,
    target_report,
    anomaly_scores,
    labeled_anomalies,
    deviation_scores,
):
    """
    综合损失函数
    """
    # 1. 报告生成损失 (标准 cross-entropy)
    gen_loss = F.cross_entropy(generated_report, target_report)

    # 2. 异常分数预测损失
    anomaly_loss = F.binary_cross_entropy(anomaly_scores, labeled_anomalies)

    # 3. 偏离一致性损失 (偏离分数应该和异常分数正相关)
    consistency_loss = -torch.corrcoef(
        torch.stack([deviation_scores.flatten(), labeled_anomalies.flatten()])
    )[0, 1]

    # 4. 可选: 对比损失 (正常 vs 异常 features)
    # contrastive_loss = ...

    return gen_loss + 0.5 * anomaly_loss + 0.1 * consistency_loss
```

---

### Experiment 13: Medical LLM Comparison

**状态**: 📋 计划中
**优先级**: 🥈 中
**论文卖点**: "Domain-Specific vs General LLMs for Radiology Report Generation"

**研究问题**:
- 医学 LLM (如 MedLlama, BioMedLM) 是否比通用 LLM 更适合？
- 医学知识对报告生成有多大帮助？

**对比实验**:
```
┌────────────────────────────────────────────────────────────────┐
│  LLM 对比实验                                                   │
│  ─────────────                                                 │
│                                                                │
│  A: Llama-2-7B (通用)                                          │
│     ↓                                                          │
│  B: MedLlama-13B (医学微调)                                    │
│     ↓                                                          │
│  C: BioMedLM-2.7B (生物医学预训练)                             │
│     ↓                                                          │
│  D: RadBERT-RoBERTa (放射学专用)                               │
│                                                                │
│  统一视觉编码器 + Adapter，只换 LLM backbone                    │
└────────────────────────────────────────────────────────────────┘
```

**评估维度**:
```
1. 医学术语准确性
   - 术语使用正确率
   - 解剖位置准确率

2. 报告质量
   - BLEU, ROUGE (和参考报告比较)
   - Clinical F1 (临床发现检出)

3. 效率
   - 推理速度
   - 所需训练数据量

4. 安全性
   - 幻觉率 (描述不存在的异常)
   - 遗漏率 (漏报异常)
```

**候选医学 LLM**:
| Model | Size | 特点 |
|-------|------|------|
| MedLlama-13B | 13B | Llama 在医学文献上微调 |
| BioMedLM | 2.7B | 从头在 PubMed 训练 |
| PMC-LLaMA | 7B | Llama 在 PMC 文章微调 |
| Clinical-T5 | 770M | 临床文本专用 |
| RadBERT | 110M | 放射学报告专用 (encoder-only) |

---

## 🔧 Resolution Problem 解决方案 (Garbage In, Garbage Out)

### 核心问题分析

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  信息丢失发生在 Resize 阶段                                                  │
│                                                                             │
│  Large Region (e.g., Lung 400×400×200)                                      │
│         ↓                                                                   │
│  Resize to (256×256×64)  ← 压缩比 4.88x, 高频细节丢失！                       │
│         ↓                                                                   │
│  ViT 看到的是"模糊"的图像 → Perceiver → LLM                                  │
│                                                                             │
│  问题: 小结节、边缘细节等可能在 resize 时就丢了！                              │
│  原则: Garbage In, Garbage Out - 后面的模型再厉害也救不回来                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Experiment 14: Multi-Crop Encoding

**状态**: 📋 计划中
**优先级**: 🥇 高 (从根源解决)
**论文卖点**: "Multi-Resolution Encoding for Variable-Size Anatomical Regions"

**研究问题**:
- 大 region 切成多块，每块保持原分辨率，是否能保留更多细节？

**核心思想**:
```
传统: Large Region → Resize(压缩) → ViT → 模糊特征

Multi-Crop:
  Large Region → Split into N crops (保持原分辨率！)
       ↓
  [Crop1] [Crop2] [Crop3] [Crop4]  (每块 256×256×64)
       ↓       ↓       ↓       ↓
     ViT     ViT     ViT     ViT
       ↓       ↓       ↓       ↓
  [Tokens1][Tokens2][Tokens3][Tokens4]
       ↓
  Crop Aggregator (Transformer)
       ↓
  Perceiver → LLM
```

**实现方案**:
```python
class MultiCropEncoder(nn.Module):
    def __init__(self, vit, perceiver, crop_size=(256, 256, 64), max_crops=4):
        self.vit = vit
        self.perceiver = perceiver
        self.crop_size = crop_size
        self.max_crops = max_crops
        self.crop_aggregator = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8), num_layers=2
        )

    def forward(self, region_volume, compression_ratio):
        if compression_ratio > 1.5:  # 需要压缩的大 region
            crops = self.split_into_crops(region_volume)
            crop_tokens = [self.vit(crop) for crop in crops]
            all_tokens = torch.cat(crop_tokens, dim=1)
            aggregated = self.crop_aggregator(all_tokens)
            return self.perceiver(aggregated)
        else:
            return self.perceiver(self.vit(region_volume))
```

**预期结果**:
- 大 region 的重建 cosine 提升 5-10%
- 小结节检出率提升

---

### Experiment 15: Diffusion Enhancement

**状态**: 📋 计划中
**优先级**: 🥇 高 (创新度最高)
**论文卖点**: "Diffusion-Enhanced Medical Vision Encoding: Recovering Lost Details"

**研究问题**:
- 能否用 Diffusion Model 恢复 resize 丢失的高频细节？

**核心思想**:
```
传统: Resized Image → ViT (看到模糊图像)

Diffusion Enhancement:
  Resized Image → Diffusion Enhancer → Enhanced Image → ViT
                        ↑
              学习 "压缩→原始" 的映射
              (用原始高分辨率数据训练！)
```

**关键洞察**:
```
普通超分: 只有低分辨率，需要"猜"细节 (可能幻觉)
你的场景: 原始高分辨率存在！可以学习真实的恢复映射

训练数据:
  Input:  resize(original, 256×256×64)  ← 压缩版
  Target: crop(original, 256×256×64)    ← 原始分辨率对应区域
```

**实现方案**:
```python
class DiffusionEnhancer(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        self.unet = UNet3D(in_channels * 2, in_channels, base_channels)
        self.ratio_embed = nn.Sequential(
            nn.Linear(1, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, base_channels),
        )
        self.num_steps = 1000

    def forward(self, x_resized, compression_ratio, num_inference_steps=50):
        x_t = torch.randn_like(x_resized)
        for t in reversed(range(0, self.num_steps, self.num_steps // num_inference_steps)):
            ratio_emb = self.ratio_embed(compression_ratio)
            condition = torch.cat([x_resized, x_t], dim=1)
            noise_pred = self.unet(condition, t, ratio_emb)
            x_t = self._ddim_step(x_t, noise_pred, t)
        return x_t
```

**医学安全性验证**:
```
必须验证: Diffusion 恢复的是"真实细节"还是"幻觉"？
评估方法:
  1. 对比 原始高分辨率crop 的 ViT features
  2. 对比 Diffusion恢复后 的 ViT features
  3. 对比 直接resize 的 ViT features

如果 恢复后features 更接近 原始features → 方案可行！
```

---

### Experiment 16: Lightweight Detail Enhancer

**状态**: 📋 计划中
**优先级**: 🥈 中高 (Diffusion的轻量替代)
**论文卖点**: "Residual Detail Recovery for Compressed Medical Images"

**研究问题**:
- 不用 Diffusion，能否用轻量级网络恢复细节？

**核心思想**:
```
残差学习: 只学习丢失的高频成分
  Enhanced = Resized + DetailNet(Resized)
                            ↑
                    学习 Δ = Original - Resized
```

**实现方案**:
```python
class LightweightEnhancer(nn.Module):
    def __init__(self, in_channels=1):
        self.detail_net = nn.Sequential(
            nn.Conv3d(in_channels + 1, 32, 3, padding=1),  # +1 for ratio
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, in_channels, 3, padding=1),
        )

    def forward(self, x_resized, compression_ratio):
        B, C, H, W, D = x_resized.shape
        ratio_map = compression_ratio.view(B, 1, 1, 1, 1).expand(B, 1, H, W, D)
        x_input = torch.cat([x_resized, ratio_map], dim=1)
        detail = self.detail_net(x_input)
        return x_resized + detail  # 残差连接
```

**优势**: 比 Diffusion 快 10-100x，适合验证想法

---

### Experiment 17: Resolution-Conditioned Perceiver

**状态**: 📋 计划中
**优先级**: 🥈 中高 (最简单的改进)
**论文卖点**: "Compression-Aware Visual Token Compression"

**研究问题**:
- 让 Perceiver 知道输入被压缩了多少，能否自适应调整压缩策略？

**核心思想**:
```
传统 Perceiver: 不知道输入的压缩程度

Resolution-Conditioned:
  [Compression_Ratio_Token] + [ViT_Tokens] → Perceiver
          ↑
    告诉模型"这个region被压缩了3.5倍"
    模型可以学习: 高压缩 → 保留更多全局信息
                 低压缩 → 保留更多细节信息
```

**实现方案**:
```python
class ResolutionConditionedPerceiver(nn.Module):
    def __init__(self, dim=768, num_latents=32):
        self.perceiver = PerceiverResampler(dim=dim, num_latents=num_latents)
        self.ratio_encoder = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
        )

    def forward(self, tokens, compression_ratio):
        ratio_embed = self.ratio_encoder(compression_ratio.unsqueeze(-1))
        tokens_with_ratio = torch.cat([ratio_embed, tokens], dim=1)
        return self.perceiver(tokens_with_ratio)
```

**优势**: 实现最简单，快速验证，无额外计算开销

---

### Experiment 18: Anti-Aliased Downsampling

**状态**: 📋 计划中
**优先级**: 🥉 中 (最简单的数据增强)

**研究问题**:
- resize 前加 Gaussian blur 能否减少混叠伪影？

**实现方案**:
```python
def adaptive_resize(volume, target_size, compression_ratio):
    if compression_ratio > 2.0:
        sigma = min(2.0, compression_ratio / 2)
        volume = gaussian_filter(volume, sigma=sigma)
    return F.interpolate(volume, size=target_size, mode='trilinear')
```

**优势**: 只改数据预处理，无需改模型

---

## 🧠 更多 Brainstorm 新点子

### Experiment 19: Contrastive Resolution Alignment

**状态**: 📋 计划中
**论文卖点**: "Resolution-Invariant Feature Learning via Contrastive Alignment"

**核心思想**:
```
让压缩后的特征尽量接近原始特征

Loss = MSE(ViT(resized), ViT(original_crop))
     + ContrastiveLoss(resized_features, original_features)

效果: ViT 学会从压缩图像中提取和原始图像相似的特征
```

```python
class ResolutionContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        self.temperature = temperature

    def forward(self, resized_features, original_features):
        # Normalize
        resized_norm = F.normalize(resized_features.mean(1), dim=-1)
        original_norm = F.normalize(original_features.mean(1), dim=-1)

        # Positive pairs: same region
        pos_sim = (resized_norm * original_norm).sum(dim=-1) / self.temperature

        # Negative pairs: different regions in batch
        neg_sim = torch.matmul(resized_norm, original_norm.T) / self.temperature

        # InfoNCE loss
        labels = torch.arange(len(resized_norm), device=resized_norm.device)
        return F.cross_entropy(neg_sim, labels)
```

---

### Experiment 20: Uncertainty-Aware Encoding

**状态**: 📋 计划中
**论文卖点**: "Uncertainty Quantification for Compressed Medical Visual Features"

**核心思想**:
```
预测每个 token 的压缩不确定性

高压缩 region 的 tokens → 高不确定性 → LLM 应该更谨慎生成
低压缩 region 的 tokens → 低不确定性 → LLM 可以更自信生成
```

```python
class UncertaintyAwareEncoder(nn.Module):
    def __init__(self, dim=768):
        self.uncertainty_head = nn.Sequential(
            nn.Linear(dim + 1, dim // 2),  # +1 for compression ratio
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),  # 0-1 uncertainty
        )

    def forward(self, tokens, compression_ratio):
        # Predict per-token uncertainty
        ratio_expanded = compression_ratio.unsqueeze(1).expand(-1, tokens.shape[1], -1)
        uncertainty = self.uncertainty_head(torch.cat([tokens, ratio_expanded], dim=-1))

        # Attach uncertainty to tokens (LLM can attend to it)
        return tokens, uncertainty
```

---

### Experiment 21: Anatomy-Prior Enhancement

**状态**: 📋 计划中
**论文卖点**: "Anatomy-Guided Detail Recovery for Medical Image Compression"

**核心思想**:
```
利用解剖学先验指导细节恢复

例如:
  - Lung 应该有什么纹理？ → 用正常肺的统计信息指导恢复
  - Heart 的边界应该是什么样？ → 用解剖学 atlas 指导
```

```python
class AnatomyPriorEnhancer(nn.Module):
    def __init__(self, num_regions=10, dim=768):
        # 每个 region 的解剖学先验 (可学习)
        self.anatomy_priors = nn.ParameterDict({
            region: nn.Parameter(torch.randn(1, dim))
            for region in REGIONS
        })

        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, features, region_name):
        prior = self.anatomy_priors[region_name].expand(features.shape[0], -1)
        fused = self.fusion(torch.cat([features, prior], dim=-1))
        return fused
```

---

### Experiment 22: Masked Region Modeling

**状态**: 📋 计划中
**论文卖点**: "Self-Supervised Pre-training for Compression-Robust Medical Features"

**核心思想**:
```
类似 MAE，但 mask 的是"被压缩丢失的信息"

预训练任务:
  Input:  Resized region (部分信息丢失)
  Target: 预测丢失的高频细节

效果: ViT 学会"脑补"被压缩丢失的细节
```

```python
class MaskedRegionModeling(nn.Module):
    def __init__(self, vit, decoder, mask_ratio=0.75):
        self.vit = vit
        self.decoder = decoder
        self.mask_ratio = mask_ratio

    def forward(self, original_region, resized_region):
        # Encode resized (compressed) version
        encoded = self.vit(resized_region)

        # Mask some tokens
        masked_encoded, mask = self.random_mask(encoded, self.mask_ratio)

        # Decode to predict original resolution features
        decoded = self.decoder(masked_encoded)

        # Target: features from original resolution
        with torch.no_grad():
            target = self.vit(original_region)

        # Loss: reconstruct original features
        loss = F.mse_loss(decoded[mask], target[mask])
        return loss
```

---

### Experiment 23: Knowledge Distillation (High-Res Teacher)

**状态**: 📋 计划中
**论文卖点**: "Resolution-Agnostic Medical Vision via Knowledge Distillation"

**核心思想**:
```
用高分辨率模型指导低分辨率模型

Teacher: 处理原始高分辨率 crop (慢但准确)
Student: 处理 resized 版本 (快但模糊)

训练: 让 Student 的特征接近 Teacher
推理: 只用 Student (接受 resized 输入，产出高质量特征)
```

```python
class ResolutionDistillation:
    def __init__(self, teacher_vit, student_vit, temperature=4.0):
        self.teacher = teacher_vit
        self.student = student_vit
        self.temperature = temperature

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

    def distill_loss(self, original_crop, resized_region):
        # Teacher processes original (high-res)
        with torch.no_grad():
            teacher_features = self.teacher(original_crop)
            teacher_soft = F.softmax(teacher_features / self.temperature, dim=-1)

        # Student processes resized (low-res)
        student_features = self.student(resized_region)
        student_soft = F.log_softmax(student_features / self.temperature, dim=-1)

        # KL divergence
        return F.kl_div(student_soft, teacher_soft, reduction='batchmean')
```

---

## 🧪 Advanced Resolution Ideas (Exp 24-28)

### Experiment 24: Frequency-Decomposed Encoding

**状态**: 📋 计划中
**优先级**: 🥈 中高
**论文卖点**: "Frequency-Aware Medical Image Compression for VLMs"

**核心思想**:
```
用 FFT 分离高频和低频分量，分别处理

原始图像
    ↓ FFT
┌─────────────┬─────────────┐
│  低频分量    │  高频分量    │
│  (结构信息)  │  (边缘细节)  │
└─────────────┴─────────────┘
      ↓               ↓
  可以 resize      需要保护！
      ↓               ↓
  ViT(low_freq) + HighFreqEncoder(high_freq)
      ↓               ↓
      └───────┬───────┘
              ↓
         Perceiver
              ↓
            LLM
```

**为什么有效**:
- 低频 = 整体结构 → resize 损失小
- 高频 = 边缘、小结节 → resize 损失大，需要单独处理

**实现方案**:
```python
class FrequencyDecomposedEncoder(nn.Module):
    def __init__(self, vit, high_freq_encoder, perceiver):
        super().__init__()
        self.vit = vit
        self.high_freq_encoder = high_freq_encoder  # 轻量级网络
        self.perceiver = perceiver
        self.freq_fusion = nn.Linear(768 * 2, 768)

    def forward(self, volume):
        # 1. FFT decomposition
        fft = torch.fft.fftn(volume, dim=(-3, -2, -1))
        fft_shift = torch.fft.fftshift(fft)

        # 2. Separate low/high frequency
        center_mask = self.create_low_pass_mask(volume.shape, cutoff=0.3)
        low_freq = torch.fft.ifftn(torch.fft.ifftshift(fft_shift * center_mask)).real
        high_freq = torch.fft.ifftn(torch.fft.ifftshift(fft_shift * (1 - center_mask))).real

        # 3. Process separately
        low_tokens = self.vit(F.interpolate(low_freq, size=(256, 256, 64)))  # Can resize
        high_tokens = self.high_freq_encoder(high_freq)  # Preserve at original resolution

        # 4. Fuse
        fused = self.freq_fusion(torch.cat([low_tokens, high_tokens], dim=-1))
        return self.perceiver(fused)
```

**医学意义**:
- 小结节 (~2-5mm) 主要在高频
- 解剖结构在低频
- 分离处理可以同时保留两者

---

### Experiment 25: Adaptive Patch Size

**状态**: 📋 计划中
**优先级**: 🥈 中高
**论文卖点**: "Compression-Aware Patch Selection for Medical Vision Transformers"

**核心思想**:
```
高压缩 region (如 Lung, ratio=4x)
    ↓
使用更小的 patch size (16×16×2)
    ↓
保留更多细节 tokens

低压缩 region (如 Thyroid, ratio=0.2x)
    ↓
使用更大的 patch size (64×64×8)
    ↓
减少冗余 tokens
```

**为什么有效**:
- 高压缩区域 = 已经丢失很多信息 → 用更细粒度的 patch 补偿
- 低压缩区域 = 信息充足 → 用粗粒度的 patch 提高效率

**实现方案**:
```python
class AdaptivePatchViT(nn.Module):
    def __init__(self, base_vit):
        super().__init__()
        self.base_vit = base_vit

        # Multiple patch embedding layers
        self.patch_embed_small = PatchEmbed3D(patch_size=(16, 16, 2))   # 细粒度
        self.patch_embed_medium = PatchEmbed3D(patch_size=(32, 32, 4)) # 标准
        self.patch_embed_large = PatchEmbed3D(patch_size=(64, 64, 8))  # 粗粒度

    def forward(self, volume, compression_ratio):
        # Select patch size based on compression ratio
        if compression_ratio > 3.0:
            patches = self.patch_embed_small(volume)   # 高压缩用小patch
        elif compression_ratio > 1.0:
            patches = self.patch_embed_medium(volume)  # 中压缩用标准patch
        else:
            patches = self.patch_embed_large(volume)   # 低压缩用大patch

        return self.base_vit.forward_from_patches(patches)
```

**挑战**:
- 不同 patch size 产生不同数量的 tokens
- 需要 Perceiver 统一压缩到固定长度

---

### Experiment 26: Cascaded Resolution Enhancement

**状态**: 📋 计划中
**优先级**: 🥇 高
**论文卖点**: "Progressive Detail Recovery for Medical Image Compression"

**核心思想**:
```
不是一次性恢复所有细节，而是渐进式多阶段恢复

Resized (256³)
    ↓ Stage 1 (粗恢复)
Enhanced_1 (256³, +整体结构)
    ↓ Stage 2 (中恢复)
Enhanced_2 (256³, +中等细节)
    ↓ Stage 3 (细恢复)
Enhanced_3 (256³, +精细边缘)
    ↓
ViT
```

**为什么有效**:
- 单次大跨度恢复困难
- 渐进式更容易学习
- 类似 Progressive GAN 的思想

**实现方案**:
```python
class CascadedEnhancer(nn.Module):
    def __init__(self, num_stages=3):
        super().__init__()
        self.stages = nn.ModuleList([
            ResidualEnhanceBlock(in_ch=1, out_ch=1, focus='structure'),  # 阶段1: 结构
            ResidualEnhanceBlock(in_ch=1, out_ch=1, focus='texture'),    # 阶段2: 纹理
            ResidualEnhanceBlock(in_ch=1, out_ch=1, focus='edge'),       # 阶段3: 边缘
        ])

    def forward(self, x_resized, compression_ratio):
        enhanced = x_resized
        intermediates = []

        # 压缩程度越高，用越多阶段
        num_active_stages = min(3, max(1, int(compression_ratio)))

        for i in range(num_active_stages):
            residual = self.stages[i](enhanced, compression_ratio)
            enhanced = enhanced + residual
            intermediates.append(enhanced)

        return enhanced, intermediates  # 可以用中间结果做辅助监督
```

**训练策略**:
```
Stage 1: 只训练 stage 1，目标是恢复整体结构
Stage 2: 固定 stage 1，训练 stage 2，恢复纹理
Stage 3: 固定 stage 1-2，训练 stage 3，恢复边缘
最后: 端到端微调所有阶段
```

---

### Experiment 27: Region-Specific Compression Strategy

**状态**: 📋 计划中
**优先级**: 🥇 高
**论文卖点**: "Anatomy-Aware Adaptive Compression for Medical VLMs"

**核心思想**:
```
不同器官有不同的信息密度和重要细节

┌──────────────┬─────────────────┬─────────────────┐
│    器官       │    关键细节      │    推荐策略      │
├──────────────┼─────────────────┼─────────────────┤
│    Lung      │ 小结节、纹理     │ Multi-Crop      │
│    Heart     │ 边界、钙化       │ Detail Enhancer │
│    Thyroid   │ 结节形态         │ 标准 resize      │
│    Bone      │ 骨折线、密度     │ High-freq 保护   │
│    Esophagus │ 管壁厚度         │ 标准 resize      │
└──────────────┴─────────────────┴─────────────────┘
```

**为什么有效**:
- 不同器官的诊断重点不同
- "一刀切"的压缩策略是次优的
- 可以根据器官特点定制最优策略

**实现方案**:
```python
class RegionSpecificEncoder(nn.Module):
    def __init__(self, base_encoder):
        super().__init__()
        self.base_encoder = base_encoder

        # Region-specific strategies
        self.strategies = nn.ModuleDict({
            'lung': MultiCropEncoder(num_crops=4),           # 肺：切块保细节
            'heart': DetailEnhancerEncoder(residual=True),   # 心：增强边缘
            'bone': FrequencyProtectedEncoder(high_freq_weight=2.0),  # 骨：保护高频
            'thyroid': nn.Identity(),                        # 甲状腺：标准处理
            'esophagus': nn.Identity(),                      # 食道：标准处理
            'default': nn.Identity(),                        # 其他：标准处理
        })

    def forward(self, volume, region_name, compression_ratio):
        # Get region-specific strategy
        strategy = self.strategies.get(region_name, self.strategies['default'])

        # Apply pre-processing based on strategy
        if region_name == 'lung' and compression_ratio > 2.0:
            enhanced_volume = strategy(volume)
        elif region_name == 'heart':
            enhanced_volume = strategy(volume, focus='boundary')
        elif region_name == 'bone':
            enhanced_volume = strategy(volume, protect_high_freq=True)
        else:
            enhanced_volume = volume

        return self.base_encoder(enhanced_volume)
```

**医学知识整合**:
- 肺: 结节检测最重要，需要保留最多细节
- 心脏: 边界清晰度影响心影评估
- 骨: 骨折线是高频信息
- 甲状腺/食道: 结构简单，标准处理即可

---

### Experiment 28: Compression-Aware Attention

**状态**: 📋 计划中
**优先级**: 🥈 中高
**论文卖点**: "Compression-Modulated Self-Attention for Medical Vision Transformers"

**核心思想**:
```
在 ViT 的 self-attention 中加入 compression-aware bias

标准 attention:
  Attention(Q, K, V) = softmax(QK^T / √d) V

压缩感知 attention:
  Attention(Q, K, V) = softmax(QK^T / √d + Bias(compression_ratio)) V
                                              ↑
                            根据压缩程度调整 attention 分布
```

**为什么有效**:
- 高压缩区域的 tokens 可能包含"错误"信息
- 让模型知道哪些区域被压缩过
- 可以学会给低压缩区域更多权重

**实现方案**:
```python
class CompressionAwareAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = nn.Linear(dim_head * heads, dim)

        # Compression bias network
        self.compression_bias = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, heads),  # One bias per head
        )

    def forward(self, x, compression_ratio):
        b, n, d = x.shape

        # Standard QKV
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Add compression-aware bias
        # 高压缩 → 负偏置 (减少权重)
        # 低压缩 → 正偏置 (增加权重)
        bias = self.compression_bias(compression_ratio.view(-1, 1))  # (B, heads)
        bias = bias.view(b, self.heads, 1, 1)  # (B, heads, 1, 1)

        # Apply bias (broadcast to all token pairs)
        dots = dots + bias

        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)
```

**扩展思路**:
- Token-level bias: 不同位置的 token 可以有不同的 bias
- Learned uncertainty: bias 可以表示"不确定性"
- Cross-region attention: 让低压缩 region 帮助理解高压缩 region

---

### 🔬 Lesion-Centric Architecture (进阶)

**依赖**: SAM-Med3D 或类似的 3D 医学分割模型

**核心思想**:
```
当前: Region-Level (器官级别)
     [Lung Region] → 32 tokens → Report

进阶: Lesion-Level (病灶级别)
     [Lung Region] → SAM-Med3D → [Lesion 1][Lesion 2]... → Report

优势:
  - 精确到每个病灶
  - 可以描述病灶的大小、位置、形态
  - 更接近临床实践
```

**SAM-Med3D 集成**:

```python
class LesionCentricVLM(nn.Module):
    """病灶级别的视觉语言模型"""

    def __init__(self, base_vlm):
        super().__init__()

        # 预训练的 3D 医学分割模型
        self.sam_med3d = load_sam_med3d("sam_med3d_vit_b.pth")
        self.sam_med3d.eval()  # 冻结

        # 病灶特征编码器 (可训练)
        self.lesion_encoder = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Linear(768, 768),
        )

        # 其他组件继承自 base_vlm
        self.region_encoder = base_vlm.encoder
        self.adapter = base_vlm.adapter
        self.llm = base_vlm.llm

    def forward(self, ct_volume, region_masks):
        """
        病灶级别前向传播
        """
        all_lesion_tokens = []
        lesion_metadata = []

        # 1. 使用 SAM-Med3D 检测病灶
        with torch.no_grad():
            lesion_masks = self.sam_med3d.segment_lesions(ct_volume)
            # lesion_masks: List of [H, W, D] binary masks

        # 2. 对每个病灶提取特征
        for i, lesion_mask in enumerate(lesion_masks):
            # 提取病灶区域
            lesion_volume = ct_volume * lesion_mask

            # 计算病灶元数据
            centroid = compute_centroid(lesion_mask)
            size_mm3 = compute_volume_mm3(lesion_mask, voxel_spacing)

            # 编码病灶特征
            lesion_features = self.region_encoder(lesion_volume)
            lesion_tokens = self.lesion_encoder(lesion_features.mean(dim=1))

            all_lesion_tokens.append(lesion_tokens)
            lesion_metadata.append({
                "id": i,
                "centroid": centroid,
                "size_mm3": size_mm3,
                "region": find_containing_region(centroid, region_masks),
            })

        # 3. 构建病灶描述提示
        lesion_prompt = self.build_lesion_prompt(lesion_metadata)

        # 4. 拼接 region tokens + lesion tokens
        region_tokens = self.encode_regions(ct_volume, region_masks)
        all_tokens = torch.cat([region_tokens] + all_lesion_tokens, dim=1)

        # 5. LLM 生成
        report = self.llm(all_tokens, lesion_prompt)

        return report, lesion_metadata

    def build_lesion_prompt(self, lesion_metadata):
        """
        构建病灶描述提示

        Example:
        "Detected lesions:
         1. Lesion in right lung, size 15.2mm³, at position (x, y, z)
         2. Lesion in liver, size 8.7mm³, at position (x, y, z)
         Please describe each lesion in detail."
        """
        lines = [f"Detected {len(lesion_metadata)} lesions:"]
        for meta in lesion_metadata:
            lines.append(
                f"  {meta['id']+1}. {meta['region']}, "
                f"size {meta['size_mm3']:.1f}mm³, "
                f"location {meta['centroid']}"
            )
        lines.append("Describe each lesion's characteristics and clinical significance.")
        return "\n".join(lines)
```

**SAM-Med3D 资源**:
| 模型 | 链接 | 说明 |
|------|------|------|
| SAM-Med3D | https://github.com/uni-medical/SAM-Med3D | 3D医学图像通用分割 |
| MedSAM | https://github.com/bowang-lab/MedSAM | 2D/3D医学SAM |
| SegVol | https://github.com/BAAI-DCAI/SegVol | 3D CT分割基础模型 |

---

## 🛠️ ToolMed: 新架构范式 (Exp 30-34)

> **详细设计文档**: `docs/ToolMed_Architecture_Proposal.md`

### 核心创新

```
传统 VLM:    Image → [Black Box] → Report
                      ???

ToolMed:     Image → [Internal Tools] → [Fusion Hub] → [LLM] → Report
                      ↓ Interpretable    ↓ Translates   ↓ Reasons
                      outputs            languages      over findings
```

### 两层工具系统

```
┌─────────────────────────────────────────────────────────────┐
│  Level 2: EXTERNAL TOOLS (Agent-Controlled)                 │
│  ├─ SAM-Med3D (Segmentation)                                │
│  ├─ TotalSegmentator (Organ Masks)                          │
│  └─ Specialist Models (On-demand, via MCP)                  │
├─────────────────────────────────────────────────────────────┤
│  Level 1: INTERNAL TOOLS (Built-in, Differentiable)         │
│  ├─ OrganRouter (Attention to regions)                      │
│  ├─ AnomalyDetector (Anomaly scoring)                       │
│  ├─ SizeEstimator (Measurement in mm)                       │
│  ├─ TextureAnalyzer (Margin, density)                       │
│  └─ UncertaintyEstimator (When to call external)            │
└─────────────────────────────────────────────────────────────┘
```

---

### Experiment 30: Internal Tools Layer

**状态**: 📋 计划中
**优先级**: 🥇 高 (ToolMed 核心)
**论文卖点**: "Decomposing Medical VLMs into Interpretable Tool Modules"

**研究问题**:
- 如何将隐式的 ViT 表示分解为显式的可解释工具？
- 每个工具应该预测什么？

**实现方案**:

```python
@dataclass
class ToolOutput:
    """所有工具的标准化输出"""
    embedding: Tensor           # Always 256-dim
    structured: Optional[Dict]  # type, size, location, etc.
    text: Optional[str]         # Template-generated description
    confidence: float           # 0.0 to 1.0
    tool_type: ToolType         # SEGMENTOR, DETECTOR, etc.

class OrganRouter(nn.Module):
    """内部工具示例：器官路由"""
    def __init__(self, vit_dim=768, num_organs=10):
        self.organ_queries = nn.Parameter(torch.randn(num_organs, 4, vit_dim))
        self.cross_attention = nn.MultiheadAttention(vit_dim, 8)
        self.projector = nn.Linear(vit_dim, 256)  # To standard dim

    def forward(self, vit_features):
        # Cross-attend to extract organ-specific features
        # Return ToolOutput with organ attention and presence
        pass

class AnomalyDetector(nn.Module):
    """内部工具：异常检测"""
    def __init__(self, vit_dim=768):
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(vit_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 0-1 anomaly score
        )
        self.finding_classifier = nn.Linear(256, 10)  # Finding types

    def forward(self, vit_features):
        # Return per-token anomaly scores + finding classification
        pass
```

**评估指标**:
- OrganRouter: Dice vs TotalSegmentator (pseudo-labels)
- AnomalyDetector: AUROC for abnormal detection
- SizeEstimator: MAE in mm

---

### Experiment 31: Fusion Hub

**状态**: 📋 计划中
**优先级**: 🥇 高 (ToolMed 核心)
**论文卖点**: "Universal Tool Fusion for Medical Vision-Language Models"

**研究问题**:
- 如何融合多个工具的异构输出？
- Type embeddings 是否帮助理解工具类型？

**核心设计**:

```python
class FusionHub(nn.Module):
    """使用 Type Embeddings 的融合枢纽"""

    def __init__(self, num_types=6, embed_dim=256, num_queries=32):
        # Type embeddings - 告诉模型每个输入来自什么工具
        self.type_embed = nn.Embedding(num_types, embed_dim)

        # Learnable queries - 固定数量的输出 tokens
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim))

        # Cross-attention to fuse all tool outputs
        self.cross_attn = nn.MultiheadAttention(embed_dim, 8)

        # To LLM dimension
        self.to_llm = nn.Linear(embed_dim, 4096)

    def forward(self, tool_outputs: List[ToolOutput]):
        all_embeddings = []

        for output in tool_outputs:
            # Add type information
            type_emb = self.type_embed(output.tool_type.value)
            combined = output.embedding + type_emb
            all_embeddings.append(combined)

        # Concatenate all tool outputs
        all_kv = torch.cat(all_embeddings, dim=1)

        # Cross-attend with learnable queries
        fused, _ = self.cross_attn(self.queries, all_kv, all_kv)

        return self.to_llm(fused)
```

**消融实验**:
- With vs without type embeddings
- Different fusion architectures (concat, attention, perceiver)
- Number of output queries

---

### Experiment 32: Adapter + Reconstruction (Modularity)

**状态**: 📋 计划中
**优先级**: 🥇 高 (ToolMed 模块化关键)
**论文卖点**: "Zero-Shot Tool Integration via Adapter Reconstruction"

**核心问题**:
```
To UNDERSTAND something → Need TRAINING
To be MODULAR          → Don't want to RETRAIN

These are in conflict!

Solution: Each tool learns to speak Fusion Hub's language
          via a lightweight adapter + reconstruction loss
```

**实现方案**:

```python
def train_adapter_for_new_tool(tool, fusion_hub, dataloader, epochs=10):
    """训练新工具的适配器"""

    # 只需两层小网络
    adapter = nn.Sequential(
        nn.Linear(tool.output_dim, 256),
        nn.LayerNorm(256),
        nn.GELU(),
    )

    decoder = nn.Sequential(
        nn.Linear(256, tool.output_dim),
        nn.LayerNorm(tool.output_dim),
    )

    # Freeze everything except adapter and decoder
    tool.eval()
    fusion_hub.eval()

    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(decoder.parameters()),
        lr=1e-3
    )

    for epoch in range(epochs):
        for batch in dataloader:
            with torch.no_grad():
                tool_output = tool(batch)

            hub_input = adapter(tool_output)

            with torch.no_grad():
                hub_output = fusion_hub.encode(hub_input)

            reconstructed = decoder(hub_output)

            # Reconstruction loss
            loss = F.mse_loss(reconstructed, tool_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return adapter  # Decoder discarded after training
```

**优势**:
1. **Simple**: 只需 2 层小网络
2. **Fast**: 分钟级训练，不是小时级
3. **Modular**: 添加新工具不改 Fusion Hub
4. **Self-supervised**: 无需额外标签

---

### Experiment 33: External Tools Integration

**状态**: 📋 计划中
**优先级**: 🥈 中高
**论文卖点**: "Agent-Controlled External Tools for Medical VLMs"

**研究问题**:
- 何时调用外部工具？(Uncertainty threshold)
- 如何整合外部工具输出？

**实现方案**:

```python
class ToolMedWithExternalTools(nn.Module):
    def __init__(self, base_model, external_tools, uncertainty_threshold=0.5):
        self.base = base_model
        self.external_tools = external_tools  # Dict of pretrained models
        self.threshold = uncertainty_threshold

    def forward(self, image, max_external_calls=3):
        # Step 1: Run internal tools
        tool_outputs = self.base.run_internal_tools(image)
        fused = self.base.fusion_hub(tool_outputs)

        # Step 2: Check uncertainty
        uncertainty = self.base.uncertainty_estimator(tool_outputs)

        external_calls = 0
        while uncertainty > self.threshold and external_calls < max_external_calls:
            # LLM decides which external tool to call
            tool_to_call = self.base.llm.decide_tool(fused, self.external_tools)

            if tool_to_call:
                # Call external tool (e.g., SAM-Med3D)
                external_output = self.external_tools[tool_to_call](image)

                # Update fused representation
                fused = self.base.fusion_hub.update(fused, external_output)

            external_calls += 1
            uncertainty = self.base.get_uncertainty(tool_outputs)

        # Step 3: Generate report
        return self.base.llm.generate(fused)
```

**External Tools 候选**:
| Tool | Use Case | When to Call |
|------|----------|--------------|
| SAM-Med3D | Detailed segmentation | Uncertainty > 0.7 for localization |
| TotalSegmentator | Organ masks | Initial ROI finding |
| Nodule Detector | Lung nodule analysis | High anomaly score in lung |
| Prior Comparison | Temporal change | When prior study available |

---

### Experiment 34: Component Protocol

**状态**: 📋 计划中
**优先级**: 🥈 中
**论文卖点**: "OpenMedVL: An Open Protocol for Composable Medical VLMs"

**研究问题**:
- 如何定义标准化的工具接口？
- 如何让社区贡献可互换的组件？

**协议定义**:

```python
class ToolProtocol:
    """Every tool must follow this interface."""

    # Identity
    name: str                    # "OrganRouter", "AnomalyDetector"
    tool_type: ToolType          # ENCODER, SEGMENTOR, DETECTOR, etc.

    # Dimensions
    input_type: str              # "image", "features", "region"
    output_dim: int              # Must project to 256

    # Capabilities
    provides: List[str]          # ["organ_segmentation", "anomaly_score"]
    requires: List[str]          # ["vit_features"] or ["image"]

    # Methods
    def forward(self, x) -> ToolOutput:
        """Process input, return standardized output."""
        pass

    def to_text(self, output) -> Optional[str]:
        """Convert output to text (if available)."""
        pass
```

**Composition Engine**:

```python
# User specifies components
config = {
    "encoder": "CT-CLIP",
    "internal_tools": ["OrganRouter", "AnomalyDetector", "TextureNet"],
    "fusion_hub": "CrossAttentionHub",
    "reasoner": "LLaMA-3",
    "external_tools": ["SAM-Med3D", "TotalSegmentator"],
}

# Engine automatically builds the model
model = CompositionEngine.build(config, registry)
```

---

### ToolMed 训练策略

```
Phase 1: Train Core System (一次性)
─────────────────────────────────────
- Train encoder (或使用 pretrained)
- Train internal tools with auxiliary supervision
- Train fusion hub to combine tool outputs
- Train/finetune LLM for report generation

Duration: Days to weeks
Do once, then freeze.


Phase 2: Add New Tools (按需)
─────────────────────────────────────
For each new tool:
- Train adapter using reconstruction loss
- Only adapter trained, everything else frozen

Duration: Minutes to hours per tool
Repeat as needed.


Phase 3: End-to-End Finetuning (可选)
─────────────────────────────────────
- Unfreeze adapters
- Finetune on downstream task
- Keep fusion hub and LLM frozen (or LoRA)

Duration: Hours
For performance boost.
```

---

### ToolMed 论文发表策略

```
Paper 1: "ToolMed" Architecture (CVPR/ICCV/MICCAI)
──────────────────────────────────────────────────
Key contributions:
- Internal + External tool architecture
- Adapter + reconstruction for modularity
- Fusion hub for multi-tool integration
- Full interpretability by design

Strength: ⭐⭐⭐⭐⭐ (核心创新)


Paper 2: "OpenMedVL" Protocol (NeurIPS/Nature Methods)
──────────────────────────────────────────────────
Key contributions:
- Protocol specification for medical VLM components
- Component library with interchangeable parts
- Composition engine for automatic model building

Strength: ⭐⭐⭐ (需要社区采用)


Paper 3: Clinical Interpretability (Radiology/Nature Medicine)
──────────────────────────────────────────────────
Key contributions:
- Clinical interpretability framework
- User study with radiologists
- Trust and adoption metrics

Strength: ⭐⭐⭐⭐ (高临床影响)
```

---

## 📈 关键发现记录

### 发现 1: Region > Global Reconstruction
**日期**: 2025-12-26
```
Global cos ≈ 0.65
Region cos ≈ 0.74-0.85

反直觉！更小的 focused region 重建更好
→ 启发了 "Smaller is Better" 假设
→ 启发了 Adaptive Token Allocation (Exp 8)
```

### 发现 2: Adapter 解冻带来显著提升
**日期**: 2025-12-24
```
Frozen:   cos ≈ 0.80
Unfrozen: cos ≈ 0.85 (+5 pts)

→ 预训练 Adapter 不是最优
→ 启发了 Minimal-Capacity (Exp 5) 和 Separate Weights (Exp 9)
```

### 发现 3: 正常样本获取方法 (for Anomaly-Centric)
**日期**: 2025-12-26
```
RadGenome-ChestCT 提供 case-level 标签:
  - train_case_disorders.csv
  - validation_case_disorders.csv

正常病例 = disorders 列 == "no findings" (严格匹配)

⚠️ 注意事项:
  1. 用完全匹配，不用 contains (避免 "no findings except...")
  2. 统一 normalize: s.strip().lower() == "no findings"
  3. 空值/NaN 不能当正常 (可能是缺标)
```

**代码示例**:
```python
# 方法 1: 使用 🤗 datasets 流式读取
from datasets import load_dataset

ds = load_dataset(
    "RadGenome/RadGenome-ChestCT",
    "case-level vqa",
    split="train",
    streaming=True
)

def is_normal(x):
    d = x.get("disorders")
    if d is None:
        return False  # 空值不当正常
    if isinstance(d, str):
        s = d.strip().lower()
        return s == "no findings"
    return False

normal_cases = ds.filter(is_normal)
first_normal = next(iter(normal_cases))

# 方法 2: 直接读 CSV (更快)
import pandas as pd

df = pd.read_csv("train_case_disorders.csv")
normal_df = df[df["disorders"].str.strip().str.lower() == "no findings"]
normal_case_ids = normal_df["case_id"].tolist()

print(f"Found {len(normal_case_ids)} normal cases")
```

---

## 🎯 下一步行动

### 短期 (本周)
1. [ ] 完成 Exp 5a/5b 训练，分析结果
2. [ ] 分析 `region_metrics.csv`，验证 "Smaller is Better"
3. [ ] 根据 Exp 6 结果，决定 Exp 8 优先级
4. [ ] **筛选正常样本**: 从 `train_case_disorders.csv` 提取 "no findings" 病例

### 中期 (下周)
1. [ ] 实现 Exp 8 (Adaptive Token Allocation)
2. [ ] 实现 Exp 9 (Separate Weights) 方案 B
3. [ ] 如果有 alignment 问题，实现 Exp 7 (Contrastive)
4. [ ] **Exp 10**: 实现 `AnomalyScorePredictor`，验证异常检测假设
5. [ ] **Exp 11**: 使用正常样本初始化 `NormalTemplateBank`

### 长期 (月度)
1. [ ] **Exp 12**: 整合异常感知到报告生成 pipeline
2. [ ] **Exp 13**: 对比 Medical LLM vs General LLM
3. [ ] **Lesion-Level**: 探索 SAM-Med3D 集成

### 论文方向
- **方向 1**: "Smaller is Better" + Adaptive Allocation
  - 核心: Region size 和 reconstruction 的反直觉关系
  - Exp 6 + Exp 8

- **方向 2**: Scale-Specific Compression
  - 核心: Global/Local 需要不同的压缩策略
  - Exp 9

- **方向 3**: Contrastive Alignment for Medical VLMs
  - 核心: 解决 85% reconstruction → 25% F1 的 gap
  - Exp 7

- **方向 4 (推荐!)**: Anomaly-Centric Medical VLM 🌟
  - 核心: "理解异常比生成漂亮报告更重要"
  - Exp 10 + 11 + 12
  - 论文标题候选:
    - "Understanding Before Generating: Anomaly-Aware Radiology Report Generation"
    - "Normal Template Learning for Anomaly Detection in Medical VLMs"
    - "Deviation-Guided Report Generation: Focusing on What Matters"

- **方向 5 (NEW! 🔥)**: Resolution-Aware Medical VLM
  - 核心: "Garbage In, Garbage Out - 从根源解决信息丢失"
  - Exp 14-23
  - 论文标题候选:
    - "Multi-Resolution Encoding for Variable-Size Anatomical Regions"
    - "Diffusion-Enhanced Medical Vision: Recovering Lost Details"
    - "Compression-Aware Visual Token Learning for Medical VLMs"
    - "Resolution-Invariant Features via Contrastive Alignment"

- **方向 6 (NEW! 🔥🔥)**: ToolMed - Tool-Augmented Medical VLM
  - 核心: "分解黑箱为可解释工具，镜像放射科医生工作流"
  - Exp 30-34
  - **详细设计**: `docs/ToolMed_Architecture_Proposal.md`
  - 论文标题候选:
    - "ToolMed: Tool-Augmented Vision-Language Models for Interpretable Medical Image Analysis"
    - "OpenMedVL: An Open Protocol for Composable Medical Vision-Language Systems"
    - "From Black Box to Glass Box: Explainable AI for Radiology through Tool Decomposition"
  - 创新点:
    - Internal + External 两层工具系统
    - Adapter + Reconstruction 实现模块化
    - Fusion Hub 统一异构工具输出
    - 完全可解释 (by design, not afterthought)

---

## 📁 文件索引

| 文件 | 用途 |
|------|------|
| `src/lit_recon_probe.py` | LIT Probe 主训练脚本 |
| `src/Model/one_layer_adapter.py` | 1-layer Adapter (Exp5) |
| `src/Model/adapter_utils.py` | Adapter 工具函数 |
| `docs/Exp5_Implementation_Status.md` | Exp5 实现文档 |
| `docs/Region_Level_Statistics.md` | Region 统计功能文档 |
| `docs/Experiment_Master_Plan.md` | 本文档 |

### 计划中的新文件 (Anomaly-Centric)
| 文件 | 用途 |
|------|------|
| `src/Model/anomaly_detector.py` | 异常分数预测 (Exp10) |
| `src/Model/normal_template_bank.py` | 正常模板学习 (Exp11) |
| `src/Model/anomaly_aware_vlm.py` | 异常感知 VLM (Exp12) |
| `scripts/extract_normal_samples.py` | 正常样本筛选脚本 |

### 计划中的新文件 (ToolMed)
| 文件 | 用途 |
|------|------|
| `docs/ToolMed_Architecture_Proposal.md` | ToolMed 详细设计文档 ✅ |
| `src/Model/toolmed/tool_output.py` | ToolOutput 数据结构 (Exp30) |
| `src/Model/toolmed/internal_tools.py` | 内部工具实现 (Exp30) |
| `src/Model/toolmed/fusion_hub.py` | Fusion Hub (Exp31) |
| `src/Model/toolmed/tool_adapter.py` | 工具适配器 (Exp32) |
| `src/Model/toolmed/external_tools.py` | 外部工具接口 (Exp33) |
| `src/Model/toolmed/protocol.py` | 组件协议定义 (Exp34) |

---

## 📊 实验路线图

```
Timeline:
─────────────────────────────────────────────────────────────────────

[已完成] Exp 1,2,5
    │
    ├──→ [本周] Exp 5a/5b 训练 + Exp 6 分析
    │         └─→ 验证 "Smaller is Better" 假设
    │
    ├──→ [下周] Exp 7,8,9 选择性实现
    │         └─→ 根据 Exp 6 结果决定优先级
    │
    ├──→ [下周] Exp 10 异常分数预测
    │         └─→ 筛选正常样本，训练 AnomalyScorePredictor
    │
    ├──→ [月内] Exp 11 正常模板学习
    │         └─→ 使用正常样本训练 NormalTemplateBank
    │
    ├──→ [月内] Exp 12 异常感知报告生成
    │         └─→ 整合到完整 VLM pipeline
    │
    └──→ [进阶] Lesion-Level Architecture
              └─→ 集成 SAM-Med3D，病灶级别分析

┌─────────────────────────────────────────────────────────────────┐
│  研究方向决策树                                                  │
│  ─────────────                                                  │
│                                                                 │
│  if Exp6 confirms "Smaller is Better":                          │
│      → 优先 Exp 8 (Adaptive Token Allocation)                   │
│                                                                 │
│  if 85% recon → 25% F1 gap persists:                            │
│      → 优先 Exp 7 (Contrastive) + Exp 10-12 (Anomaly-Centric)   │
│                                                                 │
│  if Global/Local 差异显著:                                       │
│      → 优先 Exp 9 (Separate Weights)                            │
└─────────────────────────────────────────────────────────────────┘
```

---

**文档版本**: v5.0
**创建日期**: 2025-12-26
**更新日期**: 2025-12-27
**主要更新**:
- 添加 Resolution Problem 解决方案 (Exp 14-18)
- 添加 Brainstorm 新点子 (Exp 19-23)
- 添加 Advanced Resolution Ideas (Exp 24-28): 频率分解、自适应Patch、渐进增强、区域特定策略、压缩感知注意力
- 新增论文方向 5: Resolution-Aware Medical VLM
