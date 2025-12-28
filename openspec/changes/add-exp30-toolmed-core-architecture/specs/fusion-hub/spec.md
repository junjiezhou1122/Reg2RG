## ADDED Requirements

### Requirement: Cross-Attention Fusion
The system SHALL use cross-attention to aggregate information from multiple tools.

#### Scenario: Query-based aggregation
- **WHEN** fusion hub processes tool outputs
- **THEN** learnable query tokens attend to all tool outputs
- **AND** queries extract relevant information across tools

#### Scenario: Type-aware fusion
- **WHEN** fusing tool outputs
- **THEN** tool type embeddings are added to distinguish sources
- **AND** fusion hub knows which tool produced which output

### Requirement: Text Context Integration
The system SHALL integrate text descriptions from tools with visual features.

#### Scenario: Text aggregation
- **WHEN** tools produce text outputs
- **THEN** texts are collected and formatted
- **AND** context is prepended to LLM input

#### Scenario: Multi-modal input to LLM
- **WHEN** LLM generates report
- **THEN** it receives both fused visual tokens and text context
- **AND** can attend to both modalities
