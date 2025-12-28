# Tasks: Exp 31 - ToolMed Adapter Reconstruction

## 1. Adapter Architecture
- [ ] 1.1 Implement `ToolAdapter` module
  - Linear(tool_dim → 256) + LayerNorm + GELU
- [ ] 1.2 Implement `AdapterDecoder` module (for training only)
  - Linear(256 → tool_dim) + LayerNorm
- [ ] 1.3 Support variable input dimensions

## 2. Training Framework
- [ ] 2.1 Create `train_adapter_for_new_tool()` function
- [ ] 2.2 Freeze tool and fusion hub during adapter training
- [ ] 2.3 Train only adapter + decoder
- [ ] 2.4 Loss = MSE(original_output, reconstructed_output)

## 3. Training Script
- [ ] 3.1 Create `scripts/train_tool_adapter.py`
- [ ] 3.2 Accept tool name and dataloader as input
- [ ] 3.3 Support early stopping on reconstruction loss
- [ ] 3.4 Save adapter weights only (discard decoder)

## 4. Integration
- [ ] 4.1 Load adapter weights into main model
- [ ] 4.2 Apply adapter before fusion hub
- [ ] 4.3 Support multiple adapters (one per tool)

## 5. Evaluation
- [ ] 5.1 Measure reconstruction quality per tool
- [ ] 5.2 Time to train new adapter (target: <1 hour)
- [ ] 5.3 Performance retention after adding new tool
