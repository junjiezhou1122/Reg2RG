# Tasks: Exp 13 - Medical LLM Comparison

## 1. LLM Backend Implementation
- [ ] 1.1 Create `LLMBackend` abstract base class
- [ ] 1.2 Implement `LlaMA2Backend` (baseline)
- [ ] 1.3 Implement `MedLlamaBackend` (13B medical-finetuned)
- [ ] 1.4 Implement `BioMedLMBackend` (2.7B biomedical pretrained)
- [ ] 1.5 Implement `PMCLlamaBackend` (7B PMC-finetuned)

## 2. Unified Interface
- [ ] 2.1 Define common interface: `forward(visual_tokens, prompt) -> logits`
- [ ] 2.2 Implement LoRA wrapper for each backend
- [ ] 2.3 Handle different tokenizers consistently
- [ ] 2.4 Add model selection in training args

## 3. Medical Accuracy Evaluation
- [ ] 3.1 Extract medical entities from generated reports (RadGraph)
- [ ] 3.2 Compute terminology correctness (UMLS/RadLex lookup)
- [ ] 3.3 Measure anatomical position accuracy
- [ ] 3.4 Compute hallucination rate (findings not in image)

## 4. Experiments
- [ ] 4.1 Train with LLaMA-2-7B (baseline)
- [ ] 4.2 Train with MedLlama-13B
- [ ] 4.3 Train with BioMedLM-2.7B
- [ ] 4.4 Train with PMC-LLaMA-7B
- [ ] 4.5 Keep vision encoder and adapter FIXED across all

## 5. Evaluation Dimensions
- [ ] 5.1 Report quality: BLEU, ROUGE, F1
- [ ] 5.2 Medical accuracy: Terminology, anatomy
- [ ] 5.3 Safety: Hallucination rate, miss rate
- [ ] 5.4 Efficiency: Inference speed, memory usage

## 6. Analysis
- [ ] 6.1 Cost-benefit analysis per LLM
- [ ] 6.2 Identify best trade-off
- [ ] 6.3 Document findings for paper
