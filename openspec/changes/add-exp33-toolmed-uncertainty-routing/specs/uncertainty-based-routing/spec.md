## ADDED Requirements

### Requirement: Adaptive Analysis Routing
The system SHALL route cases to appropriate analysis depth based on estimated complexity.

#### Scenario: Initial complexity estimation
- **WHEN** an image is first processed
- **THEN** quick scan estimates case complexity
- **AND** appropriate analysis level is selected

#### Scenario: Level 1 routing (fast path)
- **WHEN** case is estimated as simple
- **THEN** only basic internal tools are used
- **AND** latency is minimized (<0.5s)

#### Scenario: Level 3 routing (comprehensive)
- **WHEN** case is estimated as complex
- **THEN** all internal tools + external tools are used
- **AND** thorough analysis is performed

### Requirement: Dynamic Level Escalation
The system SHALL escalate analysis level when initial assessment is uncertain.

#### Scenario: Tool disagreement escalation
- **WHEN** internal tools produce conflicting outputs
- **THEN** analysis level is escalated
- **AND** more sophisticated analysis is triggered

#### Scenario: Uncertainty threshold escalation
- **WHEN** uncertainty score exceeds threshold
- **THEN** analysis level is escalated
- **AND** external tools may be called
