# Change: Add ToolMed External Tools (Exp 32)

## Why
Internal tools are fast but limited. External tools (SAM-Med3D, TotalSegmentator, specialized detectors) are slower but more capable. Use external tools on-demand for complex cases that internal tools can't handle confidently.

## What Changes
- Add external tool interface (MCP-style API)
- Support: SAM-Med3D, TotalSegmentator, NoduleDetector, PriorComparison
- LLM decides when to call external tools based on uncertainty
- External tool outputs update fusion hub representation

## Impact
- Affected specs: external-tool-interface (new)
- Affected code:
  - `src/Model/external_tools/interface.py` (new)
  - `src/Model/external_tools/sam_med3d.py` (new)
  - `src/Model/external_tools/total_segmentator.py` (new)
  - `src/Model/ToolMed.py` (integration)
- Priority: Medium-High
- Paper potential: "Agent-Based External Tool Calling for Medical VLMs"
