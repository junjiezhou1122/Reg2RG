# Tasks: Exp 23 - Knowledge Distillation from High-Res

## 1. Implementation
- [ ] 1.1 Implement teacher model (high-res encoder, frozen)
- [ ] 1.2 Implement student model (low-res encoder, trainable)
- [ ] 1.3 Add feature alignment layers between teacher and student
- [ ] 1.4 Implement distillation loss: KL-div + feature MSE

## 2. Teacher Preparation
- [ ] 2.1 Train teacher on high-resolution crops
- [ ] 2.2 Cache teacher's features for efficiency
- [ ] 2.3 Verify teacher quality before distillation

## 3. Distillation Training
- [ ] 3.1 Combined loss: Task loss + λ * Distillation loss
- [ ] 3.2 Distillation at multiple layers (early + late)
- [ ] 3.3 Temperature scheduling for soft labels

## 4. Evaluation
- [ ] 4.1 Compare: Student alone vs Distilled student
- [ ] 4.2 Efficiency: Student FLOPs vs Teacher FLOPs
- [ ] 4.3 Performance gap: Student vs Teacher
