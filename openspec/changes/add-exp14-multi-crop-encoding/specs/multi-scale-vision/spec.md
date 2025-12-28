## ADDED Requirements

### Requirement: Multi-Crop Visual Encoding
The system SHALL encode large volumes using multiple overlapping crops to preserve fine details.

#### Scenario: Crop generation
- **WHEN** input volume exceeds target resolution (e.g., 512³ vs 256³ target)
- **THEN** the volume is split into overlapping crops of size 256³
- **AND** overlap ratio is configurable (default 50%)

#### Scenario: Per-crop encoding
- **WHEN** crops are generated
- **THEN** each crop is encoded independently by the ViT encoder
- **AND** crop position encoding is added to distinguish spatial location

#### Scenario: Feature merging
- **WHEN** all crops are encoded
- **THEN** features are merged using the configured strategy (mean, concat, attention)
- **AND** the output has the same shape as single-crop encoding

### Requirement: Attention-Based Crop Merging
The system SHALL support learnable attention-based merging of multi-crop features.

#### Scenario: Attention merger initialization
- **WHEN** using attention-based merging
- **THEN** learnable query tokens aggregate information across crops
- **AND** cross-attention attends to all crop features

#### Scenario: Position-aware attention
- **WHEN** computing attention weights
- **THEN** crop position encodings influence attention
- **AND** spatially adjacent crops can share more information
