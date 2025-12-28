## ADDED Requirements

### Requirement: Normal Template Bank
The system SHALL maintain learnable "normal" template prototypes for each anatomical region.

#### Scenario: Template initialization
- **WHEN** training begins with normal template learning
- **THEN** each region has a learnable template parameter of shape [num_latents, dim]
- **AND** templates are initialized from aggregated normal sample features

#### Scenario: EMA template update
- **WHEN** processing a batch of normal samples
- **THEN** templates are updated using exponential moving average: `new = 0.99 * old + 0.01 * batch_mean`
- **AND** only verified normal samples contribute to template updates

### Requirement: Deviation Computation
The system SHALL compute deviation between current features and normal templates.

#### Scenario: Feature-template deviation
- **WHEN** given region features and the corresponding normal template
- **THEN** the system computes element-wise difference: `deviation = features - template`
- **AND** returns both deviation features and scalar deviation score (L2 norm)

#### Scenario: Deviation encoding
- **WHEN** deviation is computed
- **THEN** the DeviationEncoder transforms deviation into a representation usable by downstream modules
- **AND** the encoding captures both magnitude and direction of deviation

### Requirement: Normal Sample Identification
The system SHALL correctly identify and use normal samples for template learning.

#### Scenario: Case-level normal filtering
- **WHEN** reading from RadGenome-ChestCT dataset
- **THEN** cases with `disorders == "no findings"` (exact, case-insensitive) are marked as normal
- **AND** empty or null disorder values are NOT treated as normal

#### Scenario: Strict normal validation
- **WHEN** using normal samples for template update
- **THEN** samples with any ambiguous findings are excluded
- **AND** only definitively normal samples contribute to templates
