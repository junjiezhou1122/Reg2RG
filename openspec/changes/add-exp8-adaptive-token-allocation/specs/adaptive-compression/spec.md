## ADDED Requirements

### Requirement: Adaptive Token Allocation
The system SHALL dynamically allocate different numbers of visual tokens to each anatomical region based on predicted complexity.

#### Scenario: Complexity prediction
- **WHEN** a region's ViT features are processed
- **THEN** the complexity predictor outputs a probability distribution over token options [4, 8, 16, 32]
- **AND** complex regions (lung) receive higher token counts than simple regions (thyroid)

#### Scenario: Differentiable selection during training
- **WHEN** training with adaptive token allocation
- **THEN** Gumbel-Softmax is used for differentiable token count selection
- **AND** gradients flow through the selection mechanism

#### Scenario: Hard selection during inference
- **WHEN** performing inference
- **THEN** argmax is used to select the single best token count
- **AND** only one Perceiver variant is executed (no weighted combination)

### Requirement: Token Efficiency Optimization
The system SHALL encourage minimal token usage while maintaining reconstruction quality.

#### Scenario: Efficiency loss computation
- **WHEN** computing training loss
- **THEN** token efficiency loss is added: `λ * mean(selected_tokens) / max_tokens`
- **AND** the system balances reconstruction quality with token efficiency

#### Scenario: Variable-length output handling
- **WHEN** different regions produce different token counts
- **THEN** outputs are padded to maximum length for batching
- **AND** attention masks indicate valid vs padded positions
