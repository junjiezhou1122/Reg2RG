## ADDED Requirements

### Requirement: Internal Tool Architecture
The system SHALL decompose the VLM into explicit, interpretable internal tools.

#### Scenario: Tool protocol compliance
- **WHEN** a new tool is implemented
- **THEN** it follows the ToolProtocol interface
- **AND** produces ToolOutput with embedding, structured data, and text

#### Scenario: OrganRouter functionality
- **WHEN** ViT features are processed by OrganRouter
- **THEN** organ-specific attention maps are produced
- **AND** per-organ features are extracted
- **AND** organ presence confidence is output

#### Scenario: AnomalyDetector functionality
- **WHEN** ViT features are processed by AnomalyDetector
- **THEN** per-token anomaly scores are computed
- **AND** finding type is classified
- **AND** anomaly location is identified

#### Scenario: SizeEstimator functionality
- **WHEN** region features are processed by SizeEstimator
- **THEN** physical dimensions are predicted in mm
- **AND** volume is estimated
- **AND** size category is assigned (small/medium/large)

#### Scenario: TextureAnalyzer functionality
- **WHEN** region features are processed by TextureAnalyzer
- **THEN** margin type is classified (smooth/lobulated/spiculated/irregular)
- **AND** density is classified (solid/part-solid/ground-glass/cystic)
- **AND** internal pattern is classified

### Requirement: Fusion Hub
The system SHALL combine outputs from multiple tools via cross-attention.

#### Scenario: Tool output fusion
- **WHEN** all internal tools have produced outputs
- **THEN** FusionHub combines embeddings via cross-attention
- **AND** tool type embeddings distinguish different tool outputs
- **AND** learnable queries aggregate information

#### Scenario: LLM-compatible output
- **WHEN** fusion is complete
- **THEN** output is projected to LLM dimension (4096)
- **AND** can be directly consumed by LLM decoder
