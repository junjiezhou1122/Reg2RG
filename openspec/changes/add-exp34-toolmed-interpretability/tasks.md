# Tasks: Exp 34 - ToolMed Interpretability Framework

## 1. Audit Trail Export
- [ ] 1.1 Implement `AuditEntry` dataclass
  - timestamp, input_hash, model_version
  - tools_executed, tool_outputs, tool_confidences
  - external_tools_called, external_tool_reasons
  - report_text, overall_confidence
  - attention_maps, decision_explanation
- [ ] 1.2 Implement `InterpretabilityExporter` class
- [ ] 1.3 Export to JSON, YAML, and DICOM SR formats

## 2. Attention Visualization
- [ ] 2.1 Implement `AttentionVisualizer` class
- [ ] 2.2 Overlay organ attention on CT slices
- [ ] 2.3 Highlight anomaly detection regions
- [ ] 2.4 Show cross-attention weights between tools and LLM

## 3. Finding Attribution
- [ ] 3.1 Track which tool found which finding
- [ ] 3.2 Compute tool attribution scores
- [ ] 3.3 Generate per-finding explanations
- [ ] 3.4 Link report sentences to tool outputs

## 4. DICOM Integration
- [ ] 4.1 Implement `DICOMStructuredReport` exporter
- [ ] 4.2 Add RadLex coded entries for findings
- [ ] 4.3 Include measurements and confidence scores
- [ ] 4.4 Embed tool outputs as coded entries

## 5. Interpretability Metrics
- [ ] 5.1 Tool attribution accuracy: correct finding→tool mapping
- [ ] 5.2 Attention-GT alignment: IoU with radiologist annotations
- [ ] 5.3 Explanation quality: human evaluation (1-5 scale)
- [ ] 5.4 Failure traceability: can identify failing tool

## 6. Human Evaluation Study
- [ ] 6.1 Design user study with radiologists
- [ ] 6.2 Compare trust levels: ToolMed vs black-box baseline
- [ ] 6.3 Measure explanation usefulness
- [ ] 6.4 Gather feedback for improvement
