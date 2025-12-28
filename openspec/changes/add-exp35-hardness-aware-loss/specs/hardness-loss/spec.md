# Spec: Hardness-Aware Loss Weighting

## Overview
Dynamic loss weighting based on per-region learning difficulty (hardness).

## Core Algorithm

```python
class HardnessAwareLoss(nn.Module):
    """
    Dynamically weight region losses based on learning difficulty.

    Regions with lower cos (harder to learn) get higher loss weights,
    forcing the model to pay more attention to difficult regions.
    """

    def __init__(
        self,
        regions: List[str],
        ema_momentum: float = 0.99,
        min_weight: float = 1.0,
        max_weight: float = 2.0,
    ):
        super().__init__()
        self.regions = regions
        self.ema_momentum = ema_momentum
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Running average cos per region (initialized at 0.5)
        self.register_buffer(
            'region_cos_ema',
            torch.ones(len(regions)) * 0.5
        )
        self.region_to_idx = {r: i for i, r in enumerate(regions)}

    def update_ema(self, region: str, cos: float):
        """Update running average cos for a region."""
        idx = self.region_to_idx[region]
        self.region_cos_ema[idx] = (
            self.ema_momentum * self.region_cos_ema[idx] +
            (1 - self.ema_momentum) * cos
        )

    def get_weight(self, region: str) -> float:
        """
        Get loss weight for a region.

        hardness = 1 - cos (range [0, 1])
        weight = min_weight + hardness * (max_weight - min_weight)

        Example:
          cos = 0.33 (thyroid) -> hardness = 0.67 -> weight = 1.67
          cos = 0.70 (trachea) -> hardness = 0.30 -> weight = 1.30
        """
        idx = self.region_to_idx[region]
        cos = self.region_cos_ema[idx].item()
        hardness = 1.0 - cos
        weight = self.min_weight + hardness * (self.max_weight - self.min_weight)
        return weight

    def forward(
        self,
        losses_per_region: Dict[str, torch.Tensor],
        cos_per_region: Dict[str, float],
    ) -> torch.Tensor:
        """
        Compute weighted loss.

        Args:
            losses_per_region: {region_name: loss_tensor}
            cos_per_region: {region_name: cos_value} for EMA update

        Returns:
            Weighted total loss
        """
        weighted_loss = 0.0
        total_weight = 0.0
        weight_log = {}

        for region, loss in losses_per_region.items():
            # Update EMA
            if region in cos_per_region:
                self.update_ema(region, cos_per_region[region])

            # Get weight
            weight = self.get_weight(region)
            weight_log[region] = weight

            weighted_loss += weight * loss
            total_weight += weight

        # Return normalized weighted loss
        return weighted_loss / total_weight, weight_log
```

## Integration Example

```python
# In lit_recon_probe.py

from Model.hardness_aware_loss import HardnessAwareLoss

# Initialize
hardness_loss = HardnessAwareLoss(
    regions=REGIONS,
    ema_momentum=0.99,
    min_weight=1.0,
    max_weight=2.0,
)

# In training loop
losses_per_region = {}
cos_per_region = {}

for region in present_regions:
    loss_r, mse_r, cos_r, top1_r = recon_loss(x_hat_r, x_r)
    losses_per_region[region] = loss_r
    cos_per_region[region] = cos_r.item()

# Compute weighted loss
total_loss, weight_log = hardness_loss(losses_per_region, cos_per_region)

# Log weights
if wandb_run:
    for region, weight in weight_log.items():
        wandb_run.log({f"weight/{region}": weight}, step=global_step)
```

## Expected Weight Evolution

```
Epoch 1:  All weights ~1.5 (initial cos ~0.5)
Epoch 5:  thyroid: 1.7, trachea: 1.3 (differentiation starts)
Epoch 20: thyroid: 1.67, trachea: 1.30 (stabilized)

Weight Timeline:
────────────────────────────────────────
Region      Initial   Epoch 5   Epoch 20
────────────────────────────────────────
thyroid     1.50      1.65      1.67
lung        1.50      1.55      1.56
trachea     1.50      1.35      1.30
────────────────────────────────────────
```

## Ablation Study

Compare these configurations:
1. Baseline: No weighting (uniform weight = 1.0)
2. Hardness-aware: weight = 1.0 + (1 - cos)
3. Aggressive: weight = 1.0 + 2 * (1 - cos) (max_weight = 3.0)
4. Volume-based: weight = 1 / sqrt(volume) (for comparison)
