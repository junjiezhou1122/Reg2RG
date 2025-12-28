# Tasks: Exp 30 - ToolMed Core Architecture

## 1. Tool Protocol Definition
- [ ] 1.1 Define `ToolProtocol` interface (name, type, input/output dims)
- [ ] 1.2 Define `ToolOutput` dataclass (embedding, structured, text, confidence)
- [ ] 1.3 Define `ToolType` enum (ENCODER, SEGMENTOR, DETECTOR, CLASSIFIER, REGRESSOR)

## 2. Internal Tool Implementation
- [ ] 2.1 Implement `OrganRouter`: Learn to focus on anatomical regions
  - Cross-attention with organ queries
  - Output: per-organ attention weights and features
- [ ] 2.2 Implement `AnomalyDetector`: Detect abnormal regions
  - Per-token anomaly scoring
  - Output: anomaly score, location, finding type
- [ ] 2.3 Implement `SizeEstimator`: Estimate physical measurements
  - Regression head for dimensions
  - Output: size in mm, volume, category
- [ ] 2.4 Implement `TextureAnalyzer`: Classify texture patterns
  - Classification heads for margin, density, pattern
  - Output: texture categories with confidence

## 3. Uncertainty Estimator
- [ ] 3.1 Implement `UncertaintyEstimator` tool
- [ ] 3.2 Input: outputs from all other tools
- [ ] 3.3 Output: overall uncertainty, tool agreement, external tool recommendation

## 4. Fusion Hub
- [ ] 4.1 Implement `FusionHub` with cross-attention
- [ ] 4.2 Add tool type embeddings (learned)
- [ ] 4.3 Add learnable query tokens for aggregation
- [ ] 4.4 Project to LLM dimension (256 → 4096)

## 5. Text Aggregation
- [ ] 5.1 Collect text outputs from all tools
- [ ] 5.2 Format as context for LLM
- [ ] 5.3 Prepend to visual tokens

## 6. Main Model Integration
- [ ] 6.1 Create `ToolMed` main model class
- [ ] 6.2 Wire: Encoder → Tools → Fusion Hub → LLM
- [ ] 6.3 Support configurable tool list

## 7. Training
- [ ] 7.1 End-to-end training with generation loss
- [ ] 7.2 Add auxiliary losses for tools:
  - Organ segmentation loss (pseudo-labels from TotalSegmentator)
  - Anomaly detection loss (from report NLP extraction)
  - Size estimation loss (from report extraction)
  - Texture classification loss (from report keywords)

## 8. Evaluation
- [ ] 8.1 Report generation quality
- [ ] 8.2 Per-tool accuracy metrics
- [ ] 8.3 Interpretability assessment
