# Spec: Context-Aware Adapter

## Overview
Use anatomical neighbors to provide context for hard-to-learn small regions.

## Anatomical Neighborhood Graph

```
Anatomical Context Graph:
─────────────────────────

     ┌─────────┐
     │ Trachea │ ◄───────────────────────┐
     └────┬────┘                         │
          │                              │
          ▼                              │
     ┌─────────┐      ┌─────────────┐    │
     │ Thyroid │ ◄────│  Esophagus  │────┘
     └─────────┘      └──────┬──────┘
                             │
                             ▼
                      ┌─────────────┐
          ┌───────────│ Mediastinum │───────────┐
          │           └─────────────┘           │
          ▼                 │                   ▼
     ┌─────────┐           │            ┌─────────┐
     │  Heart  │ ◄─────────┘            │  Lung   │
     └────┬────┘                        └────┬────┘
          │                                  │
          ▼                                  ▼
     ┌─────────┐                        ┌─────────┐
     │ Pleura  │                        │ Abdomen │
     └─────────┘                        └─────────┘
```

## Neighborhood Definition

```python
ANATOMICAL_NEIGHBORS = {
    # Hard small regions -> use context from easy neighbors
    'thyroid': ['trachea', 'esophagus'],      # Primary context: trachea (cos=0.70!)
    'esophagus': ['trachea', 'mediastinum'],

    # Medium regions -> optional context
    'heart': ['lung', 'mediastinum', 'pleura'],
    'mediastinum': ['heart', 'lung', 'trachea'],

    # Large regions -> no context needed (self-sufficient)
    'lung': [],
    'pleura': [],
    'abdomen': [],
    'bone': [],
    'breast': [],
    'trachea': [],  # Easy, doesn't need context
}

# Which regions should use context-aware adapter
CONTEXT_AWARE_REGIONS = ['thyroid', 'esophagus']  # Small + hard
```

## Core Algorithm

```python
class ContextAwareAdapter(nn.Module):
    """
    Adapter that uses anatomical neighbors as context.

    Key idea: Thyroid is hard alone, but trachea is easy.
    Use trachea features to help localize and understand thyroid.
    """

    def __init__(self, dim: int = 768, num_heads: int = 8):
        super().__init__()

        # Standard perceiver for compression
        self.perceiver = PerceiverResampler(dim=dim, num_latents=8, depth=4)

        # Cross-attention: target attends to context
        self.context_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Layer norm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Learnable mixing weight
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        target_features: torch.Tensor,
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            target_features: [B, N_target, dim] - e.g., thyroid features
            context_features: [B, N_context, dim] - e.g., trachea features

        Returns:
            enhanced_features: [B, num_latents, dim]
        """
        # Normalize
        target_normed = self.norm1(target_features)
        context_normed = self.norm2(context_features)

        # Cross-attention: target queries context
        # Q = target, K = context, V = context
        context_info, attn_weights = self.context_attention(
            query=target_normed,
            key=context_normed,
            value=context_normed,
        )

        # Residual with learnable alpha
        alpha = torch.sigmoid(self.alpha)  # Ensure [0, 1]
        enhanced = target_features + alpha * context_info

        # Compress with perceiver
        compressed = self.perceiver(enhanced.unsqueeze(1).unsqueeze(1))
        return compressed.squeeze(1), attn_weights


class RegionSpecificAdapterRouter(nn.Module):
    """
    Route regions to appropriate adapter type.
    """

    def __init__(self, dim: int = 768):
        super().__init__()

        # Standard adapter for most regions
        self.standard_adapter = PerceiverResampler(dim=dim, num_latents=8, depth=6)

        # Context-aware adapter for hard small regions
        self.context_adapter = ContextAwareAdapter(dim=dim)

    def forward(
        self,
        region_name: str,
        region_features: torch.Tensor,
        all_region_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Route to appropriate adapter based on region.
        """
        if region_name in CONTEXT_AWARE_REGIONS:
            # Get context from neighbors
            neighbors = ANATOMICAL_NEIGHBORS[region_name]
            context_list = []
            for neighbor in neighbors:
                if neighbor in all_region_features:
                    context_list.append(all_region_features[neighbor])

            if context_list:
                # Concatenate all neighbor features
                context = torch.cat(context_list, dim=1)
                return self.context_adapter(region_features, context)

        # Default: standard adapter
        return self.standard_adapter(region_features), None
```

## Training Strategy

```python
def train_with_context(model, batch):
    """
    Training with context-aware regions.
    """
    # First pass: encode ALL regions (needed for context)
    all_features = {}
    for region in REGIONS:
        if region in batch['region_volumes']:
            features = model.encoder(batch['region_volumes'][region])
            all_features[region] = features

    # Second pass: apply adapters with context
    total_loss = 0
    for region in REGIONS:
        if region not in all_features:
            continue

        features = all_features[region]

        # Route to adapter
        compressed, attn_weights = model.adapter_router(
            region_name=region,
            region_features=features,
            all_region_features=all_features,
        )

        # Decode and compute loss
        reconstructed = model.decoder(compressed)
        loss = reconstruction_loss(reconstructed, batch['targets'][region])

        # Optional: attention regularization
        # Encourage thyroid to attend to trachea more than esophagus
        if attn_weights is not None and region == 'thyroid':
            # trachea should get higher attention
            trachea_attn = attn_weights[:, :, :len(all_features['trachea'])]
            attention_reg = -trachea_attn.mean()  # Maximize attention to trachea
            loss += 0.01 * attention_reg

        total_loss += loss

    return total_loss
```

## Expected Results

```
Without Context (baseline):
  Thyroid cos: 0.33
  Esophagus cos: 0.57

With Context:
  Thyroid cos: 0.50+ (using trachea as anchor)
  Esophagus cos: 0.65+ (using trachea and mediastinum)

Attention Pattern (for thyroid):
  - High attention to trachea boundary (localization cue)
  - Medium attention to trachea center
  - Lower attention to esophagus (less relevant)
```

## Visualization

```
Attention Map: Thyroid -> Trachea

Thyroid Tokens    Trachea Tokens
      │                 │
      ▼                 ▼
    ┌───┐           ┌───────────┐
    │ T1│ ──0.8──►  │ boundary  │ (high attention)
    │ T2│ ──0.6──►  │  center   │ (medium)
    │ T3│ ──0.3──►  │  edge     │ (lower)
    └───┘           └───────────┘

Interpretation: Thyroid learns to use trachea boundary
                for localization reference
```
