# Tasks: Exp 12 - Anomaly-Aware Report Generation

## 1. Integration
- [ ] 1.1 Import AnomalyScorePredictor from Exp 10
- [ ] 1.2 Import NormalTemplateBank from Exp 11
- [ ] 1.3 Create `AnomalyAwareVLM` wrapper class
- [ ] 1.4 Wire components in correct order

## 2. Feature Fusion
- [ ] 2.1 Implement attention weight adjuster based on anomaly score
- [ ] 2.2 Fuse: `output = original + alpha * deviation_features`
- [ ] 2.3 Alpha controlled by anomaly score (high anomaly → more deviation info)
- [ ] 2.4 Concatenate all region tokens with anomaly-weighted features

## 3. Anomaly Prompt Construction
- [ ] 3.1 Build text prompt from anomaly scores per region
- [ ] 3.2 Format: "Lung (HIGH 0.8): describe in detail; Heart (LOW 0.2): likely normal"
- [ ] 3.3 Prepend anomaly prompt to LLM input
- [ ] 3.4 Make prompt format configurable

## 4. Training
- [ ] 4.1 Combined loss: Generation + Anomaly + Deviation Consistency
- [ ] 4.2 Deviation consistency: correlation(deviation_score, anomaly_score) should be positive
- [ ] 4.3 Add loss weights: λ_anomaly, λ_consistency

## 5. Evaluation
- [ ] 5.1 Compare report quality: baseline vs anomaly-aware
- [ ] 5.2 Measure per-region description accuracy
- [ ] 5.3 Human evaluation: do high-anomaly regions get better descriptions?
- [ ] 5.4 Measure hallucination rate (describing non-existent findings)

## 6. Analysis
- [ ] 6.1 Attention visualization: does model attend to high-anomaly regions?
- [ ] 6.2 Compare generated text length per region vs anomaly score
