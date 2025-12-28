## ADDED Requirements

### Requirement: Compression Uncertainty Estimation
The system SHALL estimate and encode uncertainty due to input compression.

#### Scenario: Uncertainty from compression ratio
- **WHEN** a region is encoded after resizing
- **THEN** compression ratio (original/encoded size) is computed
- **AND** uncertainty is estimated as an increasing function of compression ratio

#### Scenario: Uncertainty embedding
- **WHEN** uncertainty is estimated
- **THEN** it is encoded as a continuous embedding via MLP
- **AND** this embedding is fused with visual features

### Requirement: Uncertainty-Aware Downstream Processing
The system SHALL propagate uncertainty information to downstream modules.

#### Scenario: LLM receives uncertainty
- **WHEN** generating reports
- **THEN** LLM has access to per-region uncertainty estimates
- **AND** can express appropriate confidence levels in descriptions

#### Scenario: Uncertainty-calibrated outputs
- **WHEN** producing final outputs
- **THEN** high-uncertainty regions receive hedged language
- **AND** low-uncertainty regions receive confident descriptions
