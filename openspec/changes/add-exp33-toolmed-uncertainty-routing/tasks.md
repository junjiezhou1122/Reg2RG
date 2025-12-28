# Tasks: Exp 33 - ToolMed Uncertainty-Based Routing

## 1. Routing Logic
- [ ] 1.1 Implement `UncertaintyRouter` module
- [ ] 1.2 Input: Initial ViT features (fast scan)
- [ ] 1.3 Output: Recommended analysis level (1/2/3)
- [ ] 1.4 Trainable routing network

## 2. Analysis Levels
- [ ] 2.1 Level 1: OrganRouter + AnomalyDetector (fast path)
  - For obviously normal or clearly abnormal cases
  - Latency target: <0.5s
- [ ] 2.2 Level 2: All internal tools + Uncertainty estimation
  - For ambiguous cases
  - Latency target: <1s
- [ ] 2.3 Level 3: + External tools
  - For complex cases requiring expert analysis
  - Latency target: <5s

## 3. Routing Criteria
- [ ] 3.1 Initial anomaly score: Low → Level 1
- [ ] 3.2 Tool agreement: High → stay at current level
- [ ] 3.3 Tool disagreement: escalate level
- [ ] 3.4 Explicit uncertainty > threshold: escalate

## 4. Training
- [ ] 4.1 Train router to predict optimal level
- [ ] 4.2 Ground truth: level that achieves good accuracy
- [ ] 4.3 Balance efficiency vs accuracy

## 5. Evaluation
- [ ] 5.1 Routing accuracy: correct level selection
- [ ] 5.2 Efficiency: average analysis time
- [ ] 5.3 Report quality per level
