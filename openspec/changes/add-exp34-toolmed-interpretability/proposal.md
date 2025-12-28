# Change: Add ToolMed Interpretability Framework (Exp 34)

## Why
ToolMed's key advantage is interpretability by design. Each finding is traceable to specific tool outputs. Need to build the infrastructure to expose and visualize this interpretability for clinical trust.

## What Changes
- Add `InterpretabilityExporter` for audit trails
- Export tool outputs, attention maps, confidence scores
- Generate human-readable explanations
- Support DICOM Structured Report export
- Add visualization tools for attention and findings

## Impact
- Affected specs: tool-interpretability (new)
- Affected code:
  - `src/Model/interpretability/exporter.py` (new)
  - `src/Model/interpretability/visualizer.py` (new)
  - `src/Model/interpretability/dicom_sr.py` (new)
  - `evaluation/interpretability_metrics.py` (new)
- Priority: High (Key differentiator from black-box VLMs)
- Paper potential: "From Black Box to Glass Box: Interpretable Medical VLMs"
