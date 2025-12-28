# Change: Add ToolMed Uncertainty-Based Routing (Exp 33)

## Why
Not all cases need the same level of analysis. Simple cases (obviously normal) can use fast internal tools only. Complex cases (subtle findings, disagreeing tools) need external tools. Route adaptively based on estimated difficulty.

## What Changes
- Add `UncertaintyRouter` that decides analysis depth
- Level 1: Internal tools only (fast, simple cases)
- Level 2: + uncertainty estimation (medium cases)
- Level 3: + external tools (complex cases)
- Routing based on initial scan confidence

## Impact
- Affected specs: uncertainty-based-routing (new)
- Affected code:
  - `src/Model/uncertainty_router.py` (new)
  - `src/Model/ToolMed.py` (routing integration)
- Priority: Medium
- Paper potential: "Adaptive Analysis Depth for Efficient Medical VLMs"
