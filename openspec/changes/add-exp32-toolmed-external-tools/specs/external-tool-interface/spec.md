## ADDED Requirements

### Requirement: External Tool Interface
The system SHALL support calling external pretrained models on demand.

#### Scenario: External tool abstraction
- **WHEN** implementing an external tool
- **THEN** it follows the ExternalTool interface
- **AND** accepts (image, context) and returns ToolOutput

#### Scenario: Supported external tools
- **WHEN** external tools are enabled
- **THEN** system can call: SAM-Med3D, TotalSegmentator, NoduleDetector, PriorComparison
- **AND** tools are loaded lazily to save memory

### Requirement: On-Demand Tool Calling
The system SHALL decide when to call external tools based on uncertainty.

#### Scenario: Uncertainty-based calling
- **WHEN** internal tool uncertainty exceeds threshold (0.5)
- **THEN** appropriate external tool is called
- **AND** external output is integrated into analysis

#### Scenario: Call limit enforcement
- **WHEN** processing a single image
- **THEN** maximum external tool calls is enforced (default 3)
- **AND** most informative tools are prioritized

### Requirement: External Result Integration
The system SHALL integrate external tool outputs into the analysis.

#### Scenario: Fusion hub update
- **WHEN** external tool produces output
- **THEN** output is projected via adapter
- **AND** fused representation is updated with new information

#### Scenario: Text context augmentation
- **WHEN** external tool produces text description
- **THEN** text is added to LLM context
- **AND** final report can reference external tool findings
