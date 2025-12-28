# Tasks: Exp 32 - ToolMed External Tools

## 1. External Tool Interface
- [ ] 1.1 Define `ExternalTool` abstract base class
- [ ] 1.2 Interface: `__call__(image, context) -> ToolOutput`
- [ ] 1.3 Support async/batch execution
- [ ] 1.4 Handle timeouts and failures gracefully

## 2. Tool Implementations
- [ ] 2.1 Implement `SAMMed3DTool` wrapper
  - Load SAM-Med3D model
  - Run on demand for detailed segmentation
- [ ] 2.2 Implement `TotalSegmentatorTool` wrapper
  - Run for robust organ segmentation
- [ ] 2.3 Implement `NoduleDetectorTool` (if available)
- [ ] 2.4 Implement `PriorComparisonTool` (if prior scan available)

## 3. Tool Calling Logic
- [ ] 3.1 Implement `should_call_external()` decision function
- [ ] 3.2 Based on: uncertainty > threshold (default 0.5)
- [ ] 3.3 LLM can override with explicit tool request
- [ ] 3.4 Limit max external calls per inference (default 3)

## 4. Result Integration
- [ ] 4.1 External tool output → Adapter → Fusion Hub
- [ ] 4.2 Update fused representation with external info
- [ ] 4.3 Add external tool text to context

## 5. Evaluation
- [ ] 5.1 Measure external tool call frequency
- [ ] 5.2 Impact on report quality
- [ ] 5.3 Latency impact
