# Saliency-Guided Compression: Implementation Guide

**Date**: 2025-12-25
**Objective**: 让异常区域分配更多tokens，正常区域分配更少tokens

---

## 🎯 核心思想

```python
# 当前（均匀分配）：
所有区域 → 32 tokens (每个区域平等对待)

# 目标（显著性引导）：
正常肺组织 (90%体积，显著性0.1) → 5 tokens
小结节区域 (1%体积，显著性0.9)   → 15 tokens
心脏 (9%体积，显著性0.3)         → 8 tokens
```

**关键问题**：如何知道哪些区域是"异常"的（显著的）？

---

## 🔍 方法1: 基于注意力的显著性检测（最简单，推荐先试）

### 核心思路

**利用ViT encoder的注意力图来识别显著性**

```python
# 观察：
ViT在编码时，注意力会自然聚焦在"不寻常"的区域
- 正常组织：注意力分散（低方差）
- 异常组织：注意力聚焦（高方差）

# 利用这个特性计算显著性
```

### 实现步骤

#### Step 1: 提取ViT的注意力图

```python
class ViT_with_Attention(nn.Module):
    """修改原始ViT，返回注意力权重"""

    def __init__(self, original_vit):
        super().__init__()
        self.vit = original_vit

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (B, C, H, W, D) - CT volume
            return_attention: 是否返回注意力图

        Returns:
            tokens: (B, N, 768) - 编码后的tokens
            attention_maps: (B, num_layers, num_heads, N, N) - 注意力权重
        """
        B = x.shape[0]

        # Patch embedding
        x = self.vit.to_patch_embedding(x)  # (B, N, 768)
        N = x.shape[1]

        attention_maps = []

        # Forward through transformer layers
        for layer in self.vit.transformer.layers:
            # Multi-head attention
            attn_module = layer[0]  # Attention block

            if return_attention:
                # 修改attention模块，返回权重
                # 这需要修改原始ViT代码
                x, attn_weights = attn_module(x, return_attention=True)
                attention_maps.append(attn_weights)  # (B, heads, N, N)
            else:
                x = layer(x)

        if return_attention:
            return x, torch.stack(attention_maps, dim=1)  # (B, layers, heads, N, N)
        return x

# 使用
encoder = ViT_with_Attention(pretrained_vit)
tokens, attn_maps = encoder(CT_scan, return_attention=True)
```

#### Step 2: 从注意力图计算显著性分数

```python
def compute_saliency_from_attention(attn_maps):
    """
    从注意力图计算每个token的显著性分数

    Args:
        attn_maps: (B, num_layers, num_heads, N, N)

    Returns:
        saliency: (B, N) - 每个token的显著性分数 [0, 1]
    """
    B, L, H, N, _ = attn_maps.shape

    # 方法1: 使用最后一层的注意力聚焦度
    last_layer_attn = attn_maps[:, -1, :, :, :]  # (B, H, N, N)

    # 计算每个token被其他token关注的程度
    # 平均所有head
    avg_attn = last_layer_attn.mean(dim=1)  # (B, N, N)

    # 计算每个token作为"query"时的注意力集中度
    # 高集中度 = 这个token在关注特定区域（可能是异常）
    attn_entropy = -torch.sum(avg_attn * torch.log(avg_attn + 1e-8), dim=-1)  # (B, N)

    # 归一化到[0, 1]
    saliency = 1 - (attn_entropy - attn_entropy.min()) / (attn_entropy.max() - attn_entropy.min() + 1e-8)

    return saliency  # (B, N)

# 使用
saliency_scores = compute_saliency_from_attention(attn_maps)
# saliency_scores: (B, 1024), 每个值在[0,1]，高值=异常
```

#### Step 3: 动态分配Perceiver的queries

**核心修改：让Perceiver根据显著性生成不同的latent tokens**

```python
class SaliencyGuidedPerceiver(nn.Module):
    """显著性引导的Perceiver Resampler"""

    def __init__(
        self,
        dim=768,
        num_latents=32,  # 总的token budget
        depth=6,
        heads=8,
        saliency_threshold=0.5,  # 显著性阈值
        min_latents_per_region=2,  # 每个区域至少2个tokens
    ):
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents
        self.saliency_threshold = saliency_threshold
        self.min_latents = min_latents_per_region

        # Perceiver blocks（标准的）
        self.layers = nn.ModuleList([
            PerceiverBlock(dim, heads) for _ in range(depth)
        ])

        # 可学习的query生成器（根据显著性调整）
        self.query_generator = nn.Sequential(
            nn.Linear(dim + 1, dim),  # +1 for saliency score
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # 基础queries（fallback）
        self.base_queries = nn.Parameter(torch.randn(1, num_latents, dim))

    def forward(self, x, saliency=None):
        """
        Args:
            x: (B, N, dim) - ViT输出的tokens
            saliency: (B, N) - 每个token的显著性分数（可选）

        Returns:
            latents: (B, num_latents, dim) - 压缩后的表示
        """
        B, N, dim = x.shape

        if saliency is None:
            # 没有显著性信息，使用标准Perceiver
            latents = self.base_queries.expand(B, -1, -1)
        else:
            # 根据显著性分配queries
            latents = self._allocate_queries_by_saliency(x, saliency)

        # Perceiver cross-attention + self-attention
        for layer in self.layers:
            latents = layer(latents, x)

        return latents

    def _allocate_queries_by_saliency(self, x, saliency):
        """
        根据显著性分数动态分配query tokens

        核心思路：
        1. 将输入tokens分为高显著性和低显著性两组
        2. 高显著性组分配更多latent tokens
        3. 低显著性组分配更少latent tokens
        """
        B, N, dim = x.shape

        # 识别高显著性的tokens
        high_saliency_mask = saliency > self.saliency_threshold  # (B, N)

        # 统计每个batch中高显著性token的数量
        num_salient = high_saliency_mask.sum(dim=1)  # (B,)

        # 动态分配token budget
        # 高显著性区域：60%的tokens
        # 低显著性区域：40%的tokens
        latents_for_salient = int(self.num_latents * 0.6)
        latents_for_normal = self.num_latents - latents_for_salient

        all_latents = []

        for b in range(B):
            # 提取这个batch的高显著性和低显著性tokens
            salient_tokens = x[b, high_saliency_mask[b]]  # (M, dim), M <= N
            normal_tokens = x[b, ~high_saliency_mask[b]]   # (N-M, dim)

            if salient_tokens.shape[0] > 0:
                # 为高显著性tokens生成更多queries
                salient_queries = self._generate_queries(
                    salient_tokens,
                    latents_for_salient,
                    saliency[b, high_saliency_mask[b]]
                )
            else:
                salient_queries = torch.empty(0, dim, device=x.device)

            if normal_tokens.shape[0] > 0:
                # 为低显著性tokens生成较少queries
                normal_queries = self._generate_queries(
                    normal_tokens,
                    latents_for_normal,
                    saliency[b, ~high_saliency_mask[b]]
                )
            else:
                normal_queries = torch.empty(0, dim, device=x.device)

            # 拼接（高显著性的queries在前）
            batch_latents = torch.cat([salient_queries, normal_queries], dim=0)

            # 如果总数不够，补齐
            if batch_latents.shape[0] < self.num_latents:
                padding = self.base_queries[0, :self.num_latents - batch_latents.shape[0]]
                batch_latents = torch.cat([batch_latents, padding], dim=0)

            all_latents.append(batch_latents[:self.num_latents])  # 确保正好num_latents个

        return torch.stack(all_latents, dim=0)  # (B, num_latents, dim)

    def _generate_queries(self, tokens, num_queries, saliency_scores):
        """
        为一组tokens生成指定数量的queries

        Args:
            tokens: (M, dim) - 这组tokens
            num_queries: int - 要生成的query数量
            saliency_scores: (M,) - 这些tokens的显著性分数
        """
        M, dim = tokens.shape

        if M == 0 or num_queries == 0:
            return torch.empty(0, dim, device=tokens.device)

        # 根据显著性加权采样tokens
        # 显著性高的token更可能被选为query的"种子"
        weights = saliency_scores / (saliency_scores.sum() + 1e-8)

        # 采样（可重复）
        indices = torch.multinomial(weights, num_queries, replacement=True)
        selected_tokens = tokens[indices]  # (num_queries, dim)

        # 将显著性分数作为条件信息
        selected_saliency = saliency_scores[indices].unsqueeze(-1)  # (num_queries, 1)

        # 生成queries（结合token embedding和显著性）
        query_input = torch.cat([selected_tokens, selected_saliency], dim=-1)  # (num_queries, dim+1)
        queries = self.query_generator(query_input)  # (num_queries, dim)

        return queries
```

#### Step 4: 集成到训练流程

```python
# 修改LITProbeModel
class LITProbeModel(nn.Module):
    def __init__(self, ...):
        super().__init__()

        # 使用支持注意力输出的encoder
        self.vision_encoder = ViT_with_Attention(pretrained_vit)

        # 使用显著性引导的adapter
        self.adapter = SaliencyGuidedPerceiver(
            dim=768,
            num_latents=32,
            saliency_threshold=0.5
        )

        # ... 其他组件

    def encode_tokens(self, x):
        """
        编码输入CT，返回压缩表示
        """
        B, num_media, num_frames, C, H, W, D = x.shape

        # Reshape
        x = x.reshape(B * num_media * num_frames, C, H, W, D)

        # ViT encoding (with attention)
        vision_tokens, attn_maps = self.vision_encoder(x, return_attention=True)
        # vision_tokens: (B, N, 768)
        # attn_maps: (B, layers, heads, N, N)

        # 计算显著性
        saliency = compute_saliency_from_attention(attn_maps)  # (B, N)

        # 显著性引导压缩
        compressed = self.adapter(vision_tokens, saliency=saliency)  # (B, 32, 768)

        # ... 后续处理
        return compressed, grid
```

---

## 🔍 方法2: 训练一个专门的显著性检测器（更准确）

### 核心思路

**训练一个轻量级网络来预测"异常概率"**

```python
class AbnormalityDetector(nn.Module):
    """预测每个ViT token对应区域的异常概率"""

    def __init__(self, dim=768, hidden_dim=256):
        super().__init__()

        self.detector = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出[0,1]概率
        )

    def forward(self, tokens):
        """
        Args:
            tokens: (B, N, dim) - ViT tokens

        Returns:
            abnormality_prob: (B, N) - 每个token的异常概率
        """
        return self.detector(tokens).squeeze(-1)  # (B, N)
```

### 如何训练这个检测器？

**选项A: 使用弱监督（报告中的关键词）**

```python
# 从报告中提取"异常"的信号
def get_abnormality_labels_from_report(report, region_name):
    """
    从报告文本推断该region是否异常

    Args:
        report: str - 报告文本
        region_name: str - 区域名称（如"lung"）

    Returns:
        is_abnormal: bool
    """
    # 异常关键词
    abnormal_keywords = [
        "nodule", "mass", "lesion", "opacity",
        "infiltrate", "effusion", "pneumonia",
        "enlargement", "abnormal", "suspicious"
    ]

    # 正常关键词
    normal_keywords = ["normal", "clear", "unremarkable"]

    report_lower = report.lower()

    # 检查region相关的描述
    if region_name.lower() in report_lower:
        # 提取相关句子
        sentences = report_lower.split('.')
        region_sentences = [s for s in sentences if region_name.lower() in s]

        for sent in region_sentences:
            if any(kw in sent for kw in abnormal_keywords):
                return 1.0  # 异常
            if any(kw in sent for kw in normal_keywords):
                return 0.0  # 正常

    return 0.5  # 不确定

# 训练循环
abnormality_detector = AbnormalityDetector()
optimizer = torch.optim.Adam(abnormality_detector.parameters(), lr=1e-4)

for epoch in epochs:
    for batch in train_loader:
        CT_scan, report, region_masks = batch

        # 编码
        with torch.no_grad():
            vision_tokens = encoder(CT_scan)  # (B, N, 768)

        # 预测异常概率
        pred_abnormality = abnormality_detector(vision_tokens)  # (B, N)

        # 构建伪标签（从报告推断）
        labels = []
        for i in range(B):
            # 对每个token，检查它对应的空间位置属于哪个region
            # 然后使用该region的报告标签
            token_labels = []
            for token_idx in range(N):
                # 映射token位置到3D空间
                spatial_pos = token_idx_to_spatial_position(token_idx)

                # 找到这个位置属于哪个region
                region = find_region_at_position(spatial_pos, region_masks[i])

                # 从报告获取这个region的标签
                label = get_abnormality_labels_from_report(report[i], region)
                token_labels.append(label)

            labels.append(torch.tensor(token_labels))

        labels = torch.stack(labels).to(device)  # (B, N)

        # BCE loss
        loss = F.binary_cross_entropy(pred_abnormality, labels)

        loss.backward()
        optimizer.step()
```

**选项B: 使用3D region masks作为监督**

```python
# 如果你有region masks，可以直接监督

def create_abnormality_labels_from_masks(vision_tokens_grid, region_masks):
    """
    从region masks创建token级别的异常标签

    Args:
        vision_tokens_grid: 对应ViT tokens的3D网格位置
        region_masks: (B, 8, H, W, D) - 8个region的mask

    Returns:
        labels: (B, N) - 每个token是否在某个region内（简化：region内=可能异常）
    """
    # 这里简化处理：
    # 如果token对应的空间位置在任何region mask内，标记为1
    # 否则标记为0（背景）

    # 具体实现需要根据你的patch grid和mask分辨率匹配
    pass
```

---

## 🔍 方法3: 对比学习的显著性（最先进）

### 核心思路

**训练一个对比学习模型，让异常patch和正常patch的表示分离**

```python
class ContrastiveSaliencyDetector(nn.Module):
    """通过对比学习识别显著性"""

    def __init__(self, dim=768):
        super().__init__()

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 128)  # 投影到低维空间
        )

    def forward(self, tokens):
        """
        Args:
            tokens: (B, N, dim)

        Returns:
            projections: (B, N, 128)
        """
        return self.projector(tokens)

# 训练：对比损失
def contrastive_loss(embeddings, labels):
    """
    Args:
        embeddings: (B, N, 128) - token embeddings
        labels: (B, N) - 0=normal, 1=abnormal

    Returns:
        loss: 让abnormal tokens聚集，与normal tokens分离
    """
    # 简化版的对比损失
    # 完整实现需要考虑positive/negative pairs

    normal_tokens = embeddings[labels == 0]  # 正常tokens
    abnormal_tokens = embeddings[labels == 1]  # 异常tokens

    # 正常tokens应该相似（聚集）
    normal_similarity = F.cosine_similarity(
        normal_tokens.unsqueeze(1),
        normal_tokens.unsqueeze(0),
        dim=-1
    )
    normal_loss = -normal_similarity.mean()

    # 异常tokens应该相似（聚集）
    abnormal_similarity = F.cosine_similarity(
        abnormal_tokens.unsqueeze(1),
        abnormal_tokens.unsqueeze(0),
        dim=-1
    )
    abnormal_loss = -abnormal_similarity.mean()

    # 正常和异常应该不相似（分离）
    cross_similarity = F.cosine_similarity(
        normal_tokens.unsqueeze(1),
        abnormal_tokens.unsqueeze(0),
        dim=-1
    )
    separation_loss = cross_similarity.mean()

    return normal_loss + abnormal_loss + separation_loss

# 训练后，显著性 = 与abnormal cluster的距离
def compute_saliency_from_contrastive(embeddings, abnormal_prototype):
    """
    Args:
        embeddings: (B, N, 128) - token embeddings
        abnormal_prototype: (128,) - 异常tokens的中心

    Returns:
        saliency: (B, N) - 与异常中心的相似度
    """
    similarity = F.cosine_similarity(
        embeddings,  # (B, N, 128)
        abnormal_prototype.unsqueeze(0).unsqueeze(0),  # (1, 1, 128)
        dim=-1
    )

    # 归一化到[0, 1]
    saliency = (similarity + 1) / 2  # cosine ∈ [-1,1] → [0,1]

    return saliency
```

---

## 📊 方法对比

| 方法 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **方法1: 注意力图** | ✅ 无需额外训练<br>✅ 实现简单<br>✅ 可解释性强 | ⚠️ 可能不够准确<br>⚠️ 依赖ViT质量 | ⭐⭐⭐⭐⭐ **先试这个** |
| **方法2: 显著性检测器** | ✅ 更准确<br>✅ 可以融合多种信号 | ⚠️ 需要额外训练<br>⚠️ 需要标签（弱监督） | ⭐⭐⭐⭐ 如果方法1不够好 |
| **方法3: 对比学习** | ✅ 最先进<br>✅ 不需要显式标签 | ⚠️ 训练复杂<br>⚠️ 需要大量数据 | ⭐⭐⭐ 研究项目 |

---

## 🚀 实施计划

### Week 1: 方法1（最简单）

```bash
# 1. 修改ViT，输出attention maps
# 2. 实现compute_saliency_from_attention()
# 3. 修改Perceiver为SaliencyGuidedPerceiver
# 4. 训练并对比baseline
```

**预期时间**: 3-5天

**评估**:
```python
# 看是否改善了小病灶的重建
if val/reg_cos (small organs) 提升:
    "成功！显著性引导有效"
else:
    "方法1不够，尝试方法2"
```

### Week 2-3: 方法2（如果方法1不够）

```bash
# 1. 实现AbnormalityDetector
# 2. 从报告提取弱监督标签
# 3. 训练检测器
# 4. 集成到SaliencyGuidedPerceiver
```

**预期时间**: 1-2周

---

## 🧪 验证显著性的质量

### 可视化显著性图

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_saliency(CT_scan, saliency_scores, save_path):
    """
    可视化显著性分数覆盖在CT上

    Args:
        CT_scan: (H, W, D) - 原始CT
        saliency_scores: (N,) - 每个patch的显著性
        save_path: 保存路径
    """
    # 将patch-level saliency映射回3D空间
    # (这需要知道patch grid的对应关系)

    saliency_3d = reconstruct_saliency_to_3d(saliency_scores)  # (H, W, D)

    # 选择一个代表性的slice
    mid_slice = D // 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 原始CT
    axes[0].imshow(CT_scan[:, :, mid_slice], cmap='gray')
    axes[0].set_title('Original CT')

    # 显著性热图
    im = axes[1].imshow(CT_scan[:, :, mid_slice], cmap='gray')
    overlay = axes[1].imshow(
        saliency_3d[:, :, mid_slice],
        cmap='hot',
        alpha=0.5,  # 半透明叠加
        vmin=0,
        vmax=1
    )
    axes[1].set_title('Saliency Map (red=abnormal)')

    plt.colorbar(overlay, ax=axes[1])
    plt.savefig(save_path)
    plt.close()

# 使用
visualize_saliency(CT_scan, saliency, 'saliency_map.png')
```

### 定量评估

```python
def evaluate_saliency_quality(saliency, ground_truth_lesion_mask):
    """
    评估显著性图是否正确识别了病灶

    Args:
        saliency: (N,) - 显著性分数
        ground_truth_lesion_mask: (N,) - 真实的病灶位置 (0/1)

    Returns:
        metrics: dict
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    # 二分类评估：病灶 vs 正常
    auc = roc_auc_score(ground_truth_lesion_mask, saliency)
    ap = average_precision_score(ground_truth_lesion_mask, saliency)

    # Top-k precision: 显著性最高的k个token中，有多少是真实病灶？
    k = int(0.1 * len(saliency))  # Top 10%
    top_k_indices = saliency.argsort()[-k:]
    precision_at_k = ground_truth_lesion_mask[top_k_indices].mean()

    return {
        'auc': auc,
        'average_precision': ap,
        'precision@10%': precision_at_k
    }
```

---

## 💡 调试技巧

### 如果显著性分数全是均匀的（没有区分度）

**可能原因**:
```python
1. 注意力图太平滑（所有token注意力相似）
   → 尝试使用更深层的attention maps

2. 熵计算有问题
   → 检查compute_saliency_from_attention的实现

3. ViT没有学到异常特征
   → 可能需要方法2（训练专门的检测器）
```

### 如果显著性太激进（所有token都是高显著性）

**解决方案**:
```python
# 调整阈值
saliency_threshold = 0.7  # 提高阈值（原来是0.5）

# 或者使用百分位数
threshold = np.percentile(saliency, 80)  # Top 20%才算显著
high_saliency_mask = saliency > threshold
```

---

## ✅ 总结

**最简单的开始**：

1. **实现方法1**（基于注意力，无需训练）
2. **可视化几个样本的显著性图**（检查是否合理）
3. **训练Exp2B**（用SaliencyGuidedPerceiver替换原Perceiver）
4. **对比Exp2 vs Exp2B**：
   - 小器官的重建质量是否提升？
   - 最终VLM的F1是否提升？

**时间成本**:
- 方法1实现：3-5天
- 完整训练+评估：1-2周

**期望收益**:
- 小器官(thyroid, esophagus) reg_cos: 0.45 → 0.65+
- 整体 val/reg_cos: 0.85 → 0.88+
- VLM F1 score: 0.25 → 0.30+ (如果假设成立)

---

**Last Updated**: 2025-12-25
