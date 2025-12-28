# Spec: Curriculum Learning by Region Difficulty

## Overview
Train regions in order of difficulty - easy regions first, hard regions later.

## Difficulty Ranking (from actual training data)

```python
# Ordered by cos (high = easy, low = hard)
DIFFICULTY_ORDER = [
    ('trachea', 0.70),      # 1. Easiest
    ('mediastinum', 0.63),  # 2.
    ('breast', 0.57),       # 3.
    ('esophagus', 0.57),    # 4.
    ('heart', 0.55),        # 5.
    ('abdomen', 0.49),      # 6.
    ('bone', 0.46),         # 7.
    ('pleura', 0.45),       # 8.
    ('lung', 0.44),         # 9.
    ('thyroid', 0.33),      # 10. Hardest
]
```

## Core Algorithm

```python
class CurriculumScheduler:
    """
    Controls which regions are active during training.

    Philosophy: Learn to walk before you run.
    - Easy regions build good foundational representations
    - These representations help bootstrap hard regions
    """

    def __init__(
        self,
        difficulty_order: List[Tuple[str, float]],
        total_epochs: int = 20,
        warmup_epochs: int = 2,
        unlock_every: int = 2,
    ):
        """
        Args:
            difficulty_order: [(region, cos)] sorted by cos descending
            total_epochs: Total training epochs
            warmup_epochs: Train only easiest regions for this many epochs
            unlock_every: Add one new region every N epochs
        """
        self.regions = [r for r, _ in difficulty_order]
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.unlock_every = unlock_every

        # Start with top 3 easiest
        self.initial_regions = 3

    def get_active_regions(self, epoch: int) -> List[str]:
        """
        Get list of active regions for current epoch.

        Schedule (default settings):
          Epoch 0-1:  [trachea, mediastinum, breast]
          Epoch 2-3:  + esophagus
          Epoch 4-5:  + heart
          Epoch 6-7:  + abdomen
          Epoch 8-9:  + bone
          Epoch 10-11: + pleura
          Epoch 12-13: + lung
          Epoch 14+:   + thyroid (all active)
        """
        if epoch < self.warmup_epochs:
            # Warmup: only initial easy regions
            num_active = self.initial_regions
        else:
            # Gradually unlock
            unlocks = (epoch - self.warmup_epochs) // self.unlock_every
            num_active = min(
                self.initial_regions + unlocks + 1,
                len(self.regions)
            )

        return self.regions[:num_active]

    def is_region_active(self, region: str, epoch: int) -> bool:
        """Check if a specific region is active."""
        return region in self.get_active_regions(epoch)

    def get_schedule_summary(self) -> str:
        """Print full schedule for visualization."""
        lines = ["Curriculum Schedule:"]
        lines.append("=" * 50)

        for epoch in range(self.total_epochs):
            active = self.get_active_regions(epoch)
            newly_added = None
            if epoch > 0:
                prev_active = self.get_active_regions(epoch - 1)
                new = set(active) - set(prev_active)
                if new:
                    newly_added = list(new)[0]

            if newly_added:
                lines.append(f"Epoch {epoch:2d}: {len(active)} regions (+{newly_added})")
            else:
                lines.append(f"Epoch {epoch:2d}: {len(active)} regions")

        return "\n".join(lines)
```

## Integration

```python
# In lit_recon_probe.py

scheduler = CurriculumScheduler(
    difficulty_order=DIFFICULTY_ORDER,
    total_epochs=20,
    warmup_epochs=2,
    unlock_every=2,
)

print(scheduler.get_schedule_summary())

for epoch in range(total_epochs):
    active_regions = scheduler.get_active_regions(epoch)
    print(f"Epoch {epoch}: Training on {active_regions}")

    for batch in dataloader:
        total_loss = 0

        for region in REGIONS:
            if not scheduler.is_region_active(region, epoch):
                continue  # Skip inactive regions

            # Normal training for active regions
            features = encoder(batch[region])
            compressed = adapter(features)
            reconstructed = decoder(compressed)
            loss = reconstruction_loss(reconstructed, batch[f'{region}_target'])

            total_loss += loss

        total_loss.backward()
        optimizer.step()
```

## Alternative: Soft Curriculum (Loss Weighting)

```python
class SoftCurriculumScheduler:
    """
    Instead of hard on/off, gradually increase loss weight.

    Hard regions start with low weight, gradually increase.
    """

    def __init__(self, difficulty_order, total_epochs=20):
        self.regions = {r: cos for r, cos in difficulty_order}
        self.total_epochs = total_epochs

    def get_loss_weight(self, region: str, epoch: int) -> float:
        """
        Get loss weight for region at epoch.

        Easy regions: weight = 1.0 from start
        Hard regions: weight ramps up over epochs

        weight = min(1.0, base_weight + epoch * ramp_rate)
        """
        cos = self.regions[region]
        base_weight = cos  # Start proportional to difficulty

        # Ramp rate: hard regions (low cos) ramp up faster
        ramp_rate = (1 - cos) / (self.total_epochs / 2)

        weight = min(1.0, base_weight + epoch * ramp_rate)
        return weight

    def get_all_weights(self, epoch: int) -> Dict[str, float]:
        return {r: self.get_loss_weight(r, epoch) for r in self.regions}
```

## Expected Schedule Visualization

```
Epoch    Active Regions
─────────────────────────────────────────────────────────────────
  0      [trac ████] [medi ████] [brea ████]
  1      [trac ████] [medi ████] [brea ████]
  2      [trac ████] [medi ████] [brea ████] [esop ████]
  3      [trac ████] [medi ████] [brea ████] [esop ████]
  4      [trac ████] [medi ████] [brea ████] [esop ████] [hear ████]
  ...
  14     [ALL 10 REGIONS ACTIVE]
  ...
  20     [ALL 10 REGIONS ACTIVE] - Final training

Timeline:
─────────────────────────────────────────────────────────────────
        Warmup      Gradual Unlock              Full Training
       ├──────┤├─────────────────────────┤├──────────────────────┤
Epoch: 0      2                         14                      20
       Easy only   Add 1 region/2 epochs     All regions
```

## Expected Benefits

```
Without Curriculum:
  - All regions compete from epoch 0
  - Easy regions learn fast, dominate gradients
  - Hard regions (thyroid) never catch up
  - Final thyroid cos: 0.33

With Curriculum:
  - Easy regions build strong foundations first
  - When thyroid is added (epoch 14), other regions are stable
  - Thyroid can leverage learned representations
  - Final thyroid cos: 0.40+ (expected)

Knowledge Transfer Mechanism:
  1. Trachea learns good "tubular structure" representation
  2. This representation is useful for nearby thyroid
  3. When thyroid training starts, it bootstraps from trachea
```

## Ablation Studies

1. **Warmup duration**: 1, 2, 3, 4 epochs
2. **Unlock frequency**: 1, 2, 3 epochs per region
3. **Initial regions**: 2, 3, 4 easiest
4. **Hard vs Soft curriculum**: on/off vs weighted
