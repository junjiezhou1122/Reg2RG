## ADDED Requirements

### Requirement: Compression Quality Estimation
The system SHALL estimate the quality of compressed features.

#### Scenario: Quality score computation
- **WHEN** region features are produced
- **THEN** a quality score (0-1) is computed
- **AND** score is based on compression ratio and estimated information loss

#### Scenario: Quality from reconstruction
- **WHEN** reconstruction probe is available
- **THEN** quality is estimated from reconstruction accuracy
- **AND** high reconstruction → high quality score

### Requirement: Quality-Modulated Attention
The system SHALL modulate attention weights based on feature quality.

#### Scenario: Attention weight adjustment
- **WHEN** LLM attends to visual features
- **THEN** attention weights are modulated by quality scores
- **AND** high-quality features receive higher attention

#### Scenario: Report generation bias
- **WHEN** generating report text
- **THEN** descriptions are weighted toward high-quality regions
- **AND** low-quality regions receive more hedged language
