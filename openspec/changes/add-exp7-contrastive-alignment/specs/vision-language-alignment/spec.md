## ADDED Requirements

### Requirement: Contrastive Vision-Language Alignment
The system SHALL align visual tokens with corresponding text tokens using contrastive learning to ensure visual features contain report-relevant information.

#### Scenario: Global contrastive alignment
- **WHEN** a batch of (image, report) pairs is processed
- **THEN** the system computes bidirectional contrastive loss between pooled visual and text embeddings
- **AND** positive pairs (same sample) have higher similarity than negative pairs (different samples)

#### Scenario: Region-level contrastive alignment
- **WHEN** region visual tokens and corresponding region report text are available
- **THEN** the system aligns each region's visual embedding with its specific report segment
- **AND** lung visual tokens are most similar to lung report text (not heart or liver text)

#### Scenario: Two-stage training
- **WHEN** contrastive pre-alignment is enabled
- **THEN** Stage 1 trains only projection heads with contrastive loss (LLM frozen)
- **AND** Stage 2 performs joint training with combined generation + contrastive loss

### Requirement: Retrieval Evaluation
The system SHALL evaluate vision-language alignment using retrieval accuracy metrics.

#### Scenario: Visual-to-text retrieval
- **WHEN** given a visual embedding
- **THEN** the system retrieves the top-K most similar text embeddings
- **AND** reports Recall@1 and Recall@5 for correct report matching

#### Scenario: Text-to-visual retrieval
- **WHEN** given a text embedding
- **THEN** the system retrieves the top-K most similar visual embeddings
- **AND** reports Recall@1 and Recall@5 for correct image matching
