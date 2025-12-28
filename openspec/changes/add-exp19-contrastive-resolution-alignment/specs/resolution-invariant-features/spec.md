## ADDED Requirements

### Requirement: Resolution Contrastive Learning
The system SHALL learn resolution-invariant features by contrasting compressed and original features.

#### Scenario: Positive pair alignment
- **WHEN** given features from resized volume and corresponding original crop
- **THEN** contrastive loss encourages high similarity between them
- **AND** the model learns to extract consistent features regardless of resolution

#### Scenario: Negative pair separation
- **WHEN** computing contrastive loss
- **THEN** features from different regions are pushed apart
- **AND** the model learns discriminative region-specific features

### Requirement: Resolution-Invariant Embeddings
The system SHALL produce embeddings that are robust to input resolution changes.

#### Scenario: Embedding consistency
- **WHEN** the same anatomical region is encoded at different resolutions
- **THEN** the resulting embeddings have high cosine similarity (>0.9)
- **AND** downstream tasks perform consistently across resolutions
