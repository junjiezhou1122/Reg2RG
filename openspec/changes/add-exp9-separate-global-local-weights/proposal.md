# Change: Add Separate Global/Local Weights (Exp 9)

## Why
Current design shares the same encoder and adapter for both global (512³) and local (256³) inputs. However, global needs to preserve overall layout while local needs fine details. Shared weights represent a suboptimal compromise.

## What Changes
- Implement separate adapters for global vs local paths
- Option A: Shared Encoder + Separate Adapters
- Option B: Fully Separate (Encoder + Adapter)
- Option C: Scale-Conditioned Adapter (single adapter with scale embedding)
- Compare all approaches

## Impact
- Affected specs: scale-specific-compression (new)
- Affected code:
  - `src/Model/separate_adapter.py` (new)
  - `src/Model/scale_conditioned_adapter.py` (new)
  - `src/Model/my_embedding_layer.py` (integration)
- Priority: High
- Paper potential: "Scale-Specific Compression for Hierarchical Medical VLMs"
