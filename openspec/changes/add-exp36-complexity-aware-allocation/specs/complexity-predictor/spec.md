# Spec: Complexity-Aware Token Allocation

## Overview
Learn visual complexity from features, not volume, for better token allocation.

## Core Insight

```
Why volume fails as complexity proxy:

           Volume    Cos    True Complexity
           ------    ---    ---------------
Thyroid    2K        0.33   HIGH (hard boundaries, low contrast)
Trachea    3K        0.70   LOW  (simple tube, high contrast)

Volume says: Thyroid ~ Trachea (both ~3K)
Reality:     Thyroid >> Trachea in difficulty

Solution: Learn complexity from visual features
```

## Core Algorithm

```python
class ComplexityPredictor(nn.Module):
    """
    Predict learning complexity from visual features.

    Complexity = how hard is this region to reconstruct?
    High complexity -> needs more tokens
    Low complexity -> can use fewer tokens
    """

    def __init__(self, dim: int = 768, num_factors: int = 4):
        super().__init__()

        # Multi-factor complexity estimation
        # Factors: structure, boundary, contrast, consistency
        self.factor_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.ReLU(),
                nn.Linear(dim // 4, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_factors)
        ])

        # Learned factor weights
        self.factor_weights = nn.Parameter(torch.ones(num_factors) / num_factors)

    def forward(self, region_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            region_features: [B, num_tokens, dim]

        Returns:
            complexity: [B, 1] in range [0, 1]
        """
        # Pool to single vector
        pooled = region_features.mean(dim=1)  # [B, dim]

        # Predict each factor
        factors = []
        for head in self.factor_heads:
            factor = head(pooled)  # [B, 1]
            factors.append(factor)

        factors = torch.cat(factors, dim=-1)  # [B, num_factors]

        # Weighted combination
        weights = F.softmax(self.factor_weights, dim=0)
        complexity = (factors * weights).sum(dim=-1, keepdim=True)

        return complexity

    def get_factor_breakdown(self, region_features: torch.Tensor) -> Dict[str, float]:
        """Get interpretable factor scores."""
        pooled = region_features.mean(dim=1)
        factor_names = ['structure', 'boundary', 'contrast', 'consistency']
        breakdown = {}

        for name, head in zip(factor_names, self.factor_heads):
            score = head(pooled).mean().item()
            breakdown[name] = score

        return breakdown
```

## Complexity-Guided Token Allocation

```python
class ComplexityGuidedPerceiver(nn.Module):
    """
    Allocate tokens based on predicted complexity, not volume.
    """

    def __init__(self, dim=768, token_options=[4, 8, 16, 32]):
        super().__init__()
        self.token_options = token_options
        self.complexity_predictor = ComplexityPredictor(dim)

        # Perceiver for each token count
        self.perceivers = nn.ModuleDict({
            str(n): PerceiverResampler(dim=dim, num_latents=n, depth=4)
            for n in token_options
        })

    def complexity_to_tokens(self, complexity: float) -> int:
        """
        Map complexity [0, 1] to token count.

        complexity < 0.25  -> 4 tokens  (easy, e.g., trachea)
        complexity < 0.50  -> 8 tokens  (medium)
        complexity < 0.75  -> 16 tokens (hard)
        complexity >= 0.75 -> 32 tokens (very hard, e.g., thyroid!)
        """
        thresholds = [0.25, 0.50, 0.75]
        for i, thresh in enumerate(thresholds):
            if complexity < thresh:
                return self.token_options[i]
        return self.token_options[-1]

    def forward(self, region_features: torch.Tensor, return_info=False):
        # Predict complexity
        complexity = self.complexity_predictor(region_features)  # [B, 1]

        # Select perceiver based on complexity
        # (simplified: use batch mean complexity)
        avg_complexity = complexity.mean().item()
        num_tokens = self.complexity_to_tokens(avg_complexity)

        # Run selected perceiver
        output = self.perceivers[str(num_tokens)](region_features)

        if return_info:
            return output, {
                'complexity': avg_complexity,
                'num_tokens': num_tokens,
            }
        return output
```

## Training with Pseudo-Labels

```python
def train_complexity_predictor(model, dataloader, epochs=5):
    """
    Train complexity predictor using cos as pseudo-label.

    Label: complexity = 1 - cos
      cos = 0.33 (thyroid) -> complexity = 0.67
      cos = 0.70 (trachea) -> complexity = 0.30
    """
    optimizer = torch.optim.Adam(model.complexity_predictor.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for batch in dataloader:
            region_features = model.encoder(batch['region_volume'])

            # Get actual reconstruction quality
            with torch.no_grad():
                reconstructed = model.decoder(model.adapter(region_features))
                actual_cos = F.cosine_similarity(
                    reconstructed.flatten(1),
                    batch['target'].flatten(1),
                    dim=1
                ).mean()

            # Predict complexity
            predicted_complexity = model.complexity_predictor(region_features)

            # Pseudo-label: complexity = 1 - cos
            target_complexity = 1 - actual_cos

            # MSE loss
            loss = F.mse_loss(predicted_complexity.mean(), target_complexity)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## Expected Token Allocation

```
After training complexity predictor:

Region      Volume    Old Tokens    New Tokens    Reason
                      (volume-based) (complexity-based)
────────────────────────────────────────────────────────
Thyroid     2K        4             32 (+28!)     Very hard
Trachea     3K        4             4  (same)     Very easy
Esophagus   3K        4             8  (+4)       Medium
Lung        392K      32            16 (-16)      Hard but not hardest
Pleura      392K      32            16 (-16)      Hard but not hardest
Mediastinum 40K       8             8  (same)     Medium-easy
────────────────────────────────────────────────────────
```

## Comparison with Exp 8

| Aspect | Exp 8 (Volume-Based) | Exp 36 (Complexity-Based) |
|--------|---------------------|---------------------------|
| Proxy for complexity | Volume | Learned from features |
| Thyroid tokens | 4 (wrong!) | 32 (correct!) |
| Trachea tokens | 4 (correct) | 4 (correct) |
| Requires training | No | Yes (complexity predictor) |
| Interpretability | Low | High (factor breakdown) |
