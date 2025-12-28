## ADDED Requirements

### Requirement: Scale-Specific Compression
The system SHALL support different compression strategies for global (whole-image) and local (region-level) visual features.

#### Scenario: Separate adapters for global/local
- **WHEN** processing global volume and local region volumes
- **THEN** global features pass through GlobalAdapter (optimized for layout)
- **AND** local features pass through LocalAdapter (optimized for details)

#### Scenario: Scale-conditioned single adapter
- **WHEN** using scale-conditioned mode
- **THEN** a scale embedding (global=0, local=1) is prepended to input tokens
- **AND** the adapter learns to adjust behavior based on scale context

#### Scenario: Fully separate encoders
- **WHEN** using fully separate mode
- **THEN** GlobalEncoder uses larger patches (64×64×8) for coarse features
- **AND** LocalEncoder uses smaller patches (32×32×4) for fine details

### Requirement: Scale-Specific Optimization
The system SHALL allow independent optimization of global and local compression pathways.

#### Scenario: Global pathway optimization
- **WHEN** training the global pathway
- **THEN** the loss emphasizes overall structure preservation
- **AND** fine details are deprioritized

#### Scenario: Local pathway optimization
- **WHEN** training the local pathway
- **THEN** the loss emphasizes local detail preservation
- **AND** global context is deprioritized
