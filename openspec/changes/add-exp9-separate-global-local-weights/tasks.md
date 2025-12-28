# Tasks: Exp 9 - Separate Global/Local Weights

## 1. Implementation - Option A (Shared Encoder + Separate Adapters)
- [ ] 1.1 Create `GlobalAdapter` with 32 latents, depth 6
- [ ] 1.2 Create `LocalAdapter` with 8 latents, depth 4
- [ ] 1.3 Route inputs through correct adapter based on type
- [ ] 1.4 Maintain shared ViT encoder

## 2. Implementation - Option B (Fully Separate)
- [ ] 2.1 Create `GlobalEncoder` with larger patch size (64)
- [ ] 2.2 Create `LocalEncoder` with smaller patch size (32)
- [ ] 2.3 Implement separate parameter loading

## 3. Implementation - Option C (Scale-Conditioned)
- [ ] 3.1 Add `ScaleEmbedding` (2 types: global/local)
- [ ] 3.2 Concatenate scale embedding with input tokens
- [ ] 3.3 Train single adapter to be scale-aware

## 4. Evaluation
- [ ] 4.1 Measure Global reconstruction cosine for each option
- [ ] 4.2 Measure Local reconstruction cosine for each option
- [ ] 4.3 Compare parameter counts
- [ ] 4.4 Compare training/inference speed

## 5. Experiments
- [ ] 5.1 Config A: Shared + Shared (baseline)
- [ ] 5.2 Config B: Shared Encoder + Separate Adapters
- [ ] 5.3 Config C: Separate Encoders + Separate Adapters
- [ ] 5.4 Config D: Shared Encoder + Scale-Conditioned Adapter

## 6. Analysis
- [ ] 6.1 Determine if scale-specific weights justify extra parameters
- [ ] 6.2 Identify best cost/performance trade-off
