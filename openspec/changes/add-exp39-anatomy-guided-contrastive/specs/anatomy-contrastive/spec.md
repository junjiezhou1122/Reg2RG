# Spec: Anatomy-Guided Contrastive Learning

## Overview
Use anatomical adjacency to guide representation learning through contrastive loss.

## Anatomical Adjacency Graph

```python
# Anatomical adjacency with edge weights
# Weight represents how "related" two regions are (0-1)
ANATOMICAL_ADJACENCY = {
    # Strong connections (weight > 0.8)
    ('trachea', 'thyroid'): 0.9,      # Thyroid wraps around trachea
    ('trachea', 'esophagus'): 0.85,   # Both in neck/mediastinum
    ('lung', 'pleura'): 0.95,         # Pleura surrounds lung
    ('heart', 'mediastinum'): 0.9,    # Heart is in mediastinum

    # Medium connections (weight 0.5-0.8)
    ('lung', 'mediastinum'): 0.7,
    ('lung', 'heart'): 0.6,
    ('heart', 'pleura'): 0.65,
    ('esophagus', 'mediastinum'): 0.6,
    ('thyroid', 'esophagus'): 0.5,

    # Weak connections (weight < 0.5)
    ('abdomen', 'lung'): 0.3,         # Diaphragm separates
    ('bone', 'lung'): 0.2,            # Ribs surround but different tissue
}

def get_adjacency_weight(region1: str, region2: str) -> float:
    """Get adjacency weight, 0 if not adjacent."""
    key = tuple(sorted([region1, region2]))
    return ANATOMICAL_ADJACENCY.get(key, 0.0)
```

## Core Algorithm

```python
class AnatomyContrastiveLoss(nn.Module):
    """
    Contrastive loss guided by anatomical adjacency.

    Key idea: Adjacent regions from the same patient should have
    similar representations (positive pairs).

    This encourages:
    1. Spatial consistency (adjacent = related)
    2. Knowledge transfer (trachea helps thyroid)
    3. Anatomically meaningful features
    """

    def __init__(self, temperature: float = 0.07, lambda_weight: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.lambda_weight = lambda_weight

        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(
        self,
        region_features: Dict[str, torch.Tensor],  # {region: [B, dim]}
    ) -> torch.Tensor:
        """
        Compute anatomy-guided contrastive loss.

        Args:
            region_features: Dictionary mapping region name to pooled features

        Returns:
            Contrastive loss scalar
        """
        total_loss = 0.0
        num_pairs = 0

        # Project all features
        projected = {
            region: F.normalize(self.projector(feat), dim=-1)
            for region, feat in region_features.items()
        }

        regions = list(projected.keys())
        batch_size = projected[regions[0]].shape[0]

        # For each pair of adjacent regions
        for i, region1 in enumerate(regions):
            for region2 in regions[i+1:]:
                weight = get_adjacency_weight(region1, region2)

                if weight > 0:  # Only adjacent regions
                    # Positive: same patient, adjacent regions
                    # Shape: [B, 128] and [B, 128]
                    z1 = projected[region1]
                    z2 = projected[region2]

                    # Positive similarity (diagonal of similarity matrix)
                    pos_sim = (z1 * z2).sum(dim=-1) / self.temperature  # [B]

                    # Negative: different patients, same region
                    # Cross-patient similarity matrix
                    neg_sim_1 = torch.mm(z1, z1.T) / self.temperature  # [B, B]
                    neg_sim_2 = torch.mm(z2, z2.T) / self.temperature  # [B, B]

                    # Mask out self-similarity
                    mask = torch.eye(batch_size, device=z1.device).bool()
                    neg_sim_1 = neg_sim_1.masked_fill(mask, float('-inf'))
                    neg_sim_2 = neg_sim_2.masked_fill(mask, float('-inf'))

                    # InfoNCE loss for region1 -> region2
                    logits_1 = torch.cat([
                        pos_sim.unsqueeze(1),
                        neg_sim_1
                    ], dim=1)  # [B, 1+B]
                    labels = torch.zeros(batch_size, dtype=torch.long, device=z1.device)
                    loss_1 = F.cross_entropy(logits_1, labels)

                    # InfoNCE loss for region2 -> region1
                    logits_2 = torch.cat([
                        pos_sim.unsqueeze(1),
                        neg_sim_2
                    ], dim=1)
                    loss_2 = F.cross_entropy(logits_2, labels)

                    # Weighted by anatomical adjacency
                    pair_loss = weight * (loss_1 + loss_2) / 2
                    total_loss += pair_loss
                    num_pairs += 1

        if num_pairs > 0:
            total_loss /= num_pairs

        return self.lambda_weight * total_loss
```

## Simplified Version (Soft Targets)

```python
class SimpleAnatomyContrastiveLoss(nn.Module):
    """
    Simplified version using MSE on feature similarity.

    Instead of InfoNCE, directly encourage:
    - Adjacent regions: high cosine similarity
    - Non-adjacent: don't care (no penalty)
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, region_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = 0.0
        count = 0

        regions = list(region_features.keys())

        for i, region1 in enumerate(regions):
            for region2 in regions[i+1:]:
                weight = get_adjacency_weight(region1, region2)

                if weight > 0:
                    feat1 = F.normalize(region_features[region1].mean(dim=1), dim=-1)
                    feat2 = F.normalize(region_features[region2].mean(dim=1), dim=-1)

                    # Cosine similarity
                    cos_sim = (feat1 * feat2).sum(dim=-1).mean()

                    # Target similarity based on adjacency weight
                    target_sim = weight * self.margin

                    # Hinge loss: penalize if similarity < target
                    pair_loss = F.relu(target_sim - cos_sim)
                    loss += pair_loss
                    count += 1

        return loss / max(count, 1)
```

## Training Integration

```python
# In lit_recon_probe.py

anatomy_loss_fn = AnatomyContrastiveLoss(
    temperature=0.07,
    lambda_weight=0.1,
)

for batch in dataloader:
    # Get features for all regions
    region_features = {}
    recon_loss_total = 0

    for region in present_regions:
        features = encoder(batch[region])
        compressed = adapter(features)

        # Store for contrastive loss
        region_features[region] = compressed.mean(dim=1)  # Pool to [B, dim]

        # Reconstruction loss
        reconstructed = decoder(compressed)
        recon_loss_total += reconstruction_loss(reconstructed, batch[f'{region}_target'])

    # Anatomy contrastive loss
    contrastive_loss = anatomy_loss_fn(region_features)

    # Total loss
    total_loss = recon_loss_total + contrastive_loss

    total_loss.backward()
```

## Expected Effects

```
Before Training (random init):
  sim(trachea, thyroid) = 0.02  (random)
  sim(lung, pleura) = 0.03      (random)
  cos(thyroid) = 0.33           (hard to learn alone)

After Training with Anatomy Contrastive:
  sim(trachea, thyroid) = 0.45  (adjacent regions aligned!)
  sim(lung, pleura) = 0.60      (strongly adjacent)
  cos(thyroid) = 0.42+          (benefits from trachea guidance)

Mechanism:
  1. Trachea features are easy to learn (cos=0.70)
  2. Contrastive loss pulls thyroid features toward trachea
  3. Thyroid inherits some of trachea's good properties
  4. Result: thyroid cos improves
```

## Visualization

```
Feature Space Before:                Feature Space After:

  ★ trachea                            ★ trachea
                                       ↓ pulled closer
                    ★ mediastinum      ★ thyroid
                                           ↘
  ★ thyroid                                ★ mediastinum
                                               ↓
      ★ lung                               ★ lung ←→ ★ pleura
          ★ pleura                              (adjacent, aligned)

Interpretation:
- Adjacent regions cluster together
- Hard regions (thyroid) benefit from easy neighbors (trachea)
- Anatomical structure is preserved in feature space
```

## Ablation Studies

1. **Temperature**: 0.05, 0.07, 0.1, 0.2
2. **Lambda weight**: 0.01, 0.1, 0.5, 1.0
3. **Adjacency threshold**: Include only weight > 0.3, 0.5, 0.7
4. **Projector depth**: 1, 2, 3 layers
5. **With/without**: Verify contrastive loss helps
