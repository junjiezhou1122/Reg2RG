# Tasks: Exp 10 - Anomaly Score Prediction

## 1. Label Extraction
- [ ] 1.1 Parse report text to identify anomaly severity per region
- [ ] 1.2 Create severity mapping: "未见异常"→0, "轻度"→0.3, "中度"→0.6, "重度"→1.0
- [ ] 1.3 Optional: Use GPT-4 to annotate ambiguous cases
- [ ] 1.4 Create `anomaly_labels.csv` with (case_id, region, score) triples

## 2. Implementation
- [ ] 2.1 Implement `AnomalyScorePredictor` with MLP head
- [ ] 2.2 Add optional `AnomalyTypeClassifier` (5-10 finding types)
- [ ] 2.3 Integrate into MyEmbedding layer
- [ ] 2.4 Add BCE loss for anomaly score supervision

## 3. Training
- [ ] 3.1 Add anomaly prediction as auxiliary task
- [ ] 3.2 Weight anomaly loss (λ=0.1 to 0.5)
- [ ] 3.3 Optional: Curriculum (easy anomalies first)

## 4. Evaluation
- [ ] 4.1 Compute ROC-AUC for normal/abnormal classification
- [ ] 4.2 Compute correlation: anomaly score vs report length
- [ ] 4.3 Compute per-region accuracy
- [ ] 4.4 Visualize anomaly score distribution

## 5. Analysis
- [ ] 5.1 Compare regions with high vs low anomaly scores
- [ ] 5.2 Check if model learns clinically meaningful patterns
- [ ] 5.3 Identify failure cases (false positives/negatives)
