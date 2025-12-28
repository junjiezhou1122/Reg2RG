## ADDED Requirements

### Requirement: Anomaly Score Prediction
The system SHALL predict an anomaly severity score (0-1) for each anatomical region based on visual features.

#### Scenario: Per-region anomaly scoring
- **WHEN** region visual features are extracted from the adapter
- **THEN** the AnomalyScorePredictor outputs a score between 0 and 1
- **AND** score 0 indicates normal, score 1 indicates severe abnormality

#### Scenario: Anomaly type classification
- **WHEN** anomaly type classification is enabled
- **THEN** the system predicts the type of abnormality (nodule, mass, consolidation, effusion, etc.)
- **AND** returns probability distribution over finding types

### Requirement: Anomaly Label Extraction
The system SHALL derive anomaly labels from report text for training supervision.

#### Scenario: Text-based label extraction
- **WHEN** processing a report with phrases like "未见明显异常"
- **THEN** the system assigns anomaly score 0.0 to that region
- **AND** reports with severity indicators receive proportional scores

#### Scenario: Normal sample identification
- **WHEN** case-level disorder label is "no findings"
- **THEN** all regions for that case receive anomaly score 0.0
- **AND** these samples are used for normal template learning

### Requirement: Anomaly-Weighted Attention
The system SHALL use anomaly scores to weight region attention during report generation.

#### Scenario: High anomaly regions get more attention
- **WHEN** a region has high anomaly score (>0.5)
- **THEN** that region receives higher attention weight in the LLM
- **AND** the generated report contains more detailed description of that region
