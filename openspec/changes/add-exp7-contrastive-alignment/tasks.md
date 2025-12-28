# Tasks: Exp 7 - Contrastive Vision-Language Alignment

## 1. Implementation
- [ ] 1.1 Implement `ContrastiveAlignmentLoss` with temperature parameter
- [ ] 1.2 Add visual and text projection heads (768→512, 4096→512)
- [ ] 1.3 Implement bidirectional InfoNCE loss (V→T and T→V)
- [ ] 1.4 Implement `RegionContrastiveLoss` for per-region alignment
- [ ] 1.5 Add region-report text encoder (reuse LLM embeddings)

## 2. Training Pipeline
- [ ] 2.1 Add Stage 1: Contrastive pre-alignment (freeze LLM)
- [ ] 2.2 Add Stage 2: Joint training with weighted loss
- [ ] 2.3 Add `contrastive_weight` hyperparameter (λ)
- [ ] 2.4 Implement learning rate scheduling for contrastive heads

## 3. Evaluation
- [ ] 3.1 Add V→T retrieval accuracy metric (Recall@1, @5)
- [ ] 3.2 Add T→V retrieval accuracy metric
- [ ] 3.3 Compare report generation F1 before/after alignment
- [ ] 3.4 Visualize embedding space with t-SNE

## 4. Experiments
- [ ] 4.1 Ablation: Global-only vs Region-level contrastive
- [ ] 4.2 Ablation: Different temperature values (0.05, 0.07, 0.1)
- [ ] 4.3 Ablation: Contrastive weight λ (0.1, 0.5, 1.0)
- [ ] 4.4 Compare with baseline (no contrastive)

## 5. Documentation
- [ ] 5.1 Update experiment log
- [ ] 5.2 Record best hyperparameters
- [ ] 5.3 Document findings for paper
