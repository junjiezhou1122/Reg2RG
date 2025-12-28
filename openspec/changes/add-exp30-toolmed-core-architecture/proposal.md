# Change: Add ToolMed Core Architecture (Exp 30)

## Why
Current VLMs are black boxes. Radiologists use explicit "mental tools" (segment organs, detect abnormalities, measure, characterize). We should make these tools EXPLICIT in the architecture for interpretability, modularity, and clinical trust.

**Key Innovation:**
```
Traditional VLM:     Image → [Black Box] → Report
ToolMed:             Image → [Internal Tools] → [Fusion Hub] → [LLM] → Report
                              ↓ Interpretable    ↓ Translates   ↓ Reasons
                              outputs            languages      over findings
```

## What Changes
- Add internal tool modules: OrganRouter, AnomalyDetector, SizeEstimator, TextureAnalyzer
- Add Fusion Hub to combine tool outputs via cross-attention
- Each tool produces: embedding + structured output + text description
- Tools are differentiable, fast, always-on

## Impact
- Affected specs: internal-tools, fusion-hub (new)
- Affected code:
  - `src/Model/tools/organ_router.py` (new)
  - `src/Model/tools/anomaly_detector.py` (new)
  - `src/Model/tools/size_estimator.py` (new)
  - `src/Model/tools/texture_analyzer.py` (new)
  - `src/Model/fusion_hub.py` (new)
  - `src/Model/ToolMed.py` (new main model)
- Priority: High (Major architecture change)
- Paper potential: "ToolMed: Tool-Augmented VLMs for Interpretable Medical Image Analysis"
