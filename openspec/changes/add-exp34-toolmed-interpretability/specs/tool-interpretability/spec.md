## ADDED Requirements

### Requirement: Audit Trail Generation
The system SHALL produce complete audit trails for every analysis.

#### Scenario: Audit entry creation
- **WHEN** an analysis is completed
- **THEN** a comprehensive audit entry is created
- **AND** includes: tools executed, outputs, confidences, attention maps

#### Scenario: Export formats
- **WHEN** audit is exported
- **THEN** JSON and YAML formats are supported
- **AND** DICOM Structured Report format is available for clinical integration

### Requirement: Attention Visualization
The system SHALL visualize attention patterns for interpretability.

#### Scenario: Organ attention overlay
- **WHEN** visualizing OrganRouter output
- **THEN** attention weights are overlaid on CT slices
- **AND** organ focus areas are clearly highlighted

#### Scenario: Anomaly highlighting
- **WHEN** visualizing AnomalyDetector output
- **THEN** suspected abnormal regions are highlighted
- **AND** anomaly scores are color-coded

### Requirement: Finding Attribution
The system SHALL attribute each finding to specific tools.

#### Scenario: Tool-to-finding mapping
- **WHEN** a finding is reported
- **THEN** the tool that detected it is identified
- **AND** the evidence (attention, score) is linked

#### Scenario: Report-tool linking
- **WHEN** a report sentence is generated
- **THEN** the source tool outputs are traceable
- **AND** users can verify the evidence

### Requirement: DICOM Integration
The system SHALL support clinical DICOM standards.

#### Scenario: Structured Report export
- **WHEN** exporting to DICOM
- **THEN** findings are coded with RadLex terms
- **AND** measurements include units and confidence
- **AND** tool outputs are embedded as coded entries
