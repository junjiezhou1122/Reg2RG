# Change: Add Anomaly Score Prediction (Exp 10)

## Why
Medical reports fundamentally describe **anomalies**. Instead of blindly generating text, the model should first understand what is abnormal. Anomaly detection allows the model to focus attention on clinically relevant regions.

## What Changes
- Add `AnomalyScorePredictor` module predicting per-region anomaly score (0-1)
- Add optional anomaly type classification head (nodule, mass, effusion, etc.)
- Derive training labels from report text (NLP extraction or GPT-4 annotation)
- Use anomaly scores to weight region attention

## Impact
- Affected specs: anomaly-detection (new)
- Affected code:
  - `src/Model/anomaly_detector.py` (new)
  - `src/Dataset/radgenome_dataset_train.py` (add anomaly labels)
  - `scripts/extract_anomaly_labels.py` (new)
- Priority: High
- Part of: Anomaly-Centric Design (Exp 10-12)
- Paper potential: "Anomaly-Aware Medical VLM: Understanding Before Generating"
