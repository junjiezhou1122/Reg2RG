## ADDED Requirements

### Requirement: Region-Specific Compression
The system SHALL apply different compression strategies to different anatomical regions.

#### Scenario: Anatomy-based strategy selection
- **WHEN** processing a region
- **THEN** compression strategy is selected based on region type
- **AND** complex regions (lung) get more tokens
- **AND** simple regions (bone) get fewer tokens

#### Scenario: Configurable compression parameters
- **WHEN** defining a compression strategy
- **THEN** parameters include: token count, adapter depth, downsample factor
- **AND** strategies can be expert-defined or learned

### Requirement: Efficient Anatomical Encoding
The system SHALL optimize total token budget across regions.

#### Scenario: Token budget allocation
- **WHEN** encoding all regions
- **THEN** total tokens are allocated based on region complexity
- **AND** budget is used efficiently without waste

#### Scenario: Quality-per-region optimization
- **WHEN** training with region-specific strategies
- **THEN** each region achieves appropriate reconstruction quality
- **AND** complex regions are not under-represented
