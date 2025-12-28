## ADDED Requirements

### Requirement: Anomaly-Aware Feature Fusion
The system SHALL fuse original visual features with deviation features, weighted by predicted anomaly scores.

#### Scenario: Anomaly-weighted fusion
- **WHEN** processing region features
- **THEN** deviation features are computed from NormalTemplateBank
- **AND** fusion weight alpha is determined by anomaly score
- **AND** output = original_features + alpha * deviation_features

#### Scenario: Attention weight adjustment
- **WHEN** regions have different anomaly scores
- **THEN** high-anomaly regions receive higher attention weights in LLM
- **AND** low-anomaly regions receive proportionally lower weights

### Requirement: Anomaly Prompt Generation
The system SHALL generate text prompts describing anomaly levels to guide LLM generation.

#### Scenario: Build anomaly summary prompt
- **WHEN** all region anomaly scores are computed
- **THEN** generate prompt like: "Detected anomaly levels: Lung (HIGH 0.8), Heart (LOW 0.2), ..."
- **AND** high-anomaly regions include instruction "describe in detail"
- **AND** low-anomaly regions include "likely normal"

#### Scenario: Prompt integration with LLM
- **WHEN** generating report
- **THEN** anomaly prompt is prepended to visual tokens
- **AND** LLM conditions on both visual features and anomaly context

### Requirement: Anomaly-Aware Training
The system SHALL train with combined losses that encourage anomaly-aware behavior.

#### Scenario: Combined loss computation
- **WHEN** computing training loss
- **THEN** total_loss = generation_loss + λ₁ * anomaly_loss + λ₂ * consistency_loss
- **AND** consistency_loss encourages correlation between deviation_score and anomaly_score

#### Scenario: Deviation-anomaly consistency
- **WHEN** computing consistency loss
- **THEN** regions with high deviation from normal template should have high anomaly scores
- **AND** negative correlation is penalized
