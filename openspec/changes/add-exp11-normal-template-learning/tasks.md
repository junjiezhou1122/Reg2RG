# Tasks: Exp 11 - Normal Template Learning

## 1. Normal Sample Extraction
- [ ] 1.1 Load `train_case_disorders.csv` from RadGenome dataset
- [ ] 1.2 Filter cases with `disorders == "no findings"` (exact match, normalized)
- [ ] 1.3 Create `normal_case_ids.txt` list
- [ ] 1.4 Validate: ensure no false negatives (cases with subtle findings)

## 2. Implementation
- [ ] 2.1 Implement `NormalTemplateBank` with ParameterDict (10 regions)
- [ ] 2.2 Initialize templates as learnable parameters [num_latents, dim]
- [ ] 2.3 Implement `compute_deviation(features, region_name)` method
- [ ] 2.4 Implement `DeviationEncoder` to transform deviation into features
- [ ] 2.5 Add `update_template_ema(normal_features, region, momentum=0.99)` method

## 3. Training Strategy
- [ ] 3.1 Phase 1: Template initialization using normal samples only
- [ ] 3.2 Phase 2: Deviation detection training
- [ ] 3.3 Add deviation-anomaly consistency loss
- [ ] 3.4 Implement template freezing after initialization

## 4. Evaluation
- [ ] 4.1 Compute deviation score distribution for normal vs abnormal samples
- [ ] 4.2 Compute ROC-AUC using deviation score as anomaly predictor
- [ ] 4.3 Visualize learned templates (t-SNE of template embeddings)
- [ ] 4.4 Correlation: deviation score vs manual anomaly score (Exp 10)

## 5. Analysis
- [ ] 5.1 Per-region template quality assessment
- [ ] 5.2 Identify regions where template learning fails
- [ ] 5.3 Compare EMA-updated vs fixed templates
