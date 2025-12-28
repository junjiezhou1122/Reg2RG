## ADDED Requirements

### Requirement: Resolution Conditioning
The system SHALL condition the Perceiver adapter on the input resolution to enable resolution-aware compression.

#### Scenario: Resolution embedding generation
- **WHEN** input volume has known original resolution (H, W, D)
- **THEN** resolution embedding is computed from the resolution values
- **AND** embedding captures relative scale compared to encoder's native resolution

#### Scenario: Continuous resolution encoding
- **WHEN** using continuous resolution mode
- **THEN** Fourier positional encoding transforms resolution values to embedding
- **AND** nearby resolutions have similar embeddings

#### Scenario: Discrete resolution encoding
- **WHEN** using discrete resolution mode
- **THEN** resolution is categorized (small/medium/large)
- **AND** learned embeddings represent each category

### Requirement: Resolution-Aware Perceiver
The system SHALL use resolution information to adapt compression behavior.

#### Scenario: Resolution token prepending
- **WHEN** resolution embedding is computed
- **THEN** it is prepended as the first token to ViT features
- **AND** Perceiver attends to resolution information

#### Scenario: Resolution-adaptive compression
- **WHEN** processing high-resolution inputs (downsampled)
- **THEN** Perceiver learns to preserve more detail information
- **AND** low-resolution inputs receive appropriate handling
