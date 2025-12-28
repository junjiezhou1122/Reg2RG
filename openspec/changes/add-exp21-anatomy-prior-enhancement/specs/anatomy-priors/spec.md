## ADDED Requirements

### Requirement: Anatomy Prior Templates
The system SHALL maintain learned anatomical prior templates for each region.

#### Scenario: Prior initialization
- **WHEN** training begins
- **THEN** priors are initialized from averaged features of normal samples
- **AND** each region has a distinct prior template

#### Scenario: Prior-visual fusion
- **WHEN** processing visual features
- **THEN** anatomical priors are combined with visual features
- **AND** fusion provides context about expected anatomy

### Requirement: Deviation-Based Anomaly Detection
The system SHALL use deviation from priors for anomaly identification.

#### Scenario: Deviation computation
- **WHEN** visual features deviate significantly from prior
- **THEN** a high deviation score is computed
- **AND** this indicates potential clinical finding

#### Scenario: Prior-guided attention
- **WHEN** visual features are uncertain (high compression)
- **THEN** priors receive higher weight in fusion
- **AND** expected anatomy guides the model's understanding
