## ADDED Requirements

### Requirement: Modular LLM Backend
The system SHALL support multiple LLM backends through a unified interface.

#### Scenario: LLM backend selection
- **WHEN** configuring the model
- **THEN** the user can select from: llama2, medllama, biomedlm, pmc-llama
- **AND** the appropriate backend is loaded with correct tokenizer

#### Scenario: Consistent interface
- **WHEN** using any LLM backend
- **THEN** the forward signature is: `forward(visual_tokens, prompt) -> logits`
- **AND** all backends support LoRA fine-tuning

### Requirement: Medical Terminology Evaluation
The system SHALL evaluate medical terminology accuracy in generated reports.

#### Scenario: Entity extraction
- **WHEN** evaluating a generated report
- **THEN** medical entities are extracted using RadGraph or similar
- **AND** entities include: findings, anatomy, severity, laterality

#### Scenario: Terminology validation
- **WHEN** validating extracted entities
- **THEN** each term is checked against UMLS/RadLex medical ontologies
- **AND** accuracy is computed as: correct_terms / total_terms

### Requirement: Safety Metrics
The system SHALL measure safety-related metrics for clinical deployment.

#### Scenario: Hallucination detection
- **WHEN** comparing generated report to ground truth
- **THEN** findings mentioned in generated but not in ground truth are flagged as hallucinations
- **AND** hallucination rate is computed

#### Scenario: Miss rate computation
- **WHEN** comparing generated report to ground truth
- **THEN** findings in ground truth but missing from generated are flagged as misses
- **AND** miss rate is computed separately for critical vs non-critical findings
