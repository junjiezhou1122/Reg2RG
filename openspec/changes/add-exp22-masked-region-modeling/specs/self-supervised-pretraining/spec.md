## ADDED Requirements

### Requirement: Masked Region Pre-training
The system SHALL support self-supervised pre-training via masked region modeling.

#### Scenario: Region masking
- **WHEN** pre-training with masked region modeling
- **THEN** 25-50% of regions are randomly masked per sample
- **AND** mask tokens replace the hidden region features

#### Scenario: Masked feature prediction
- **WHEN** visible regions are encoded
- **THEN** the model predicts features for masked regions
- **AND** loss is computed between predicted and actual features

### Requirement: Transfer Learning
The system SHALL support transferring pretrained representations to downstream tasks.

#### Scenario: Adapter initialization
- **WHEN** fine-tuning on report generation
- **THEN** adapter weights are initialized from MRM pre-training
- **AND** pre-trained representations improve downstream performance

#### Scenario: Missing region robustness
- **WHEN** some regions are unavailable at inference
- **THEN** the model can still generate reasonable outputs
- **AND** pre-training has taught the model to handle partial inputs
