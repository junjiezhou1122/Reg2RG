## ADDED Requirements

### Requirement: Anti-Aliased Downsampling
The system SHALL use anti-aliased downsampling to preserve more information during resolution reduction.

#### Scenario: Gaussian blur before subsample
- **WHEN** downsampling a 3D volume
- **THEN** Gaussian blur is applied before subsampling
- **AND** blur sigma is configurable (default 1.0)

#### Scenario: Per-axis blur for 3D
- **WHEN** processing 3D volumes
- **THEN** Gaussian blur is applied separately along each axis
- **AND** anisotropic sigma values are supported (different for xy vs z)

#### Scenario: Frequency preservation
- **WHEN** anti-aliased downsampling is applied
- **THEN** low-frequency content is preserved with higher fidelity than standard downsampling
- **AND** aliasing artifacts are minimized

### Requirement: Learnable Downsample Kernel
The system SHALL optionally learn the optimal blur kernel for downsampling.

#### Scenario: Kernel initialization
- **WHEN** using learnable kernel mode
- **THEN** kernel is initialized as Gaussian
- **AND** kernel parameters are optimized during training

#### Scenario: Kernel constraints
- **WHEN** learning the kernel
- **THEN** kernel is constrained to be symmetric
- **AND** kernel values sum to 1 (normalized)
