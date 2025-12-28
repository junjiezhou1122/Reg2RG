# Tasks: Exp 15 - Diffusion-Based Super-Resolution

## 1. Research & Selection
- [ ] 1.1 Survey medical image diffusion models (MedDiff, DiffuseVAE, etc.)
- [ ] 1.2 Identify 3D CT compatible models
- [ ] 1.3 Evaluate pretrained checkpoints availability
- [ ] 1.4 Select best candidate model

## 2. Implementation
- [ ] 2.1 Implement `DiffusionSRModule` wrapper
- [ ] 2.2 Add conditioning interface (use ViT features as condition)
- [ ] 2.3 Implement single-step and multi-step inference
- [ ] 2.4 Add noise schedule configuration

## 3. Integration
- [ ] 3.1 Create optional enhancement path in MyEmbedding
- [ ] 3.2 Mode A: Enhance input before ViT (upscale LR → HR)
- [ ] 3.3 Mode B: Enhance ViT output features
- [ ] 3.4 Make enhancement toggleable

## 4. Training (if fine-tuning needed)
- [ ] 4.1 Prepare paired LR-HR dataset from RadGenome
- [ ] 4.2 Fine-tune diffusion model on CT data
- [ ] 4.3 Add perceptual loss (LPIPS, VGG)

## 5. Evaluation
- [ ] 5.1 Visual quality assessment (PSNR, SSIM)
- [ ] 5.2 Downstream report generation impact
- [ ] 5.3 Hallucination analysis (do enhanced details match ground truth?)
- [ ] 5.4 Compute cost analysis

## 6. Safety Analysis
- [ ] 6.1 Identify cases where diffusion hallucinates incorrect anatomy
- [ ] 6.2 Measure clinical safety impact
- [ ] 6.3 Document risks and mitigations
