# Tasks: Exp 18 - Anti-Aliased Downsampling

## 1. Implementation
- [ ] 1.1 Implement `AntiAliasedDownsample3D` module
- [ ] 1.2 Gaussian blur kernel: configurable sigma (0.5, 1.0, 1.5)
- [ ] 1.3 Implement blur → subsample pipeline
- [ ] 1.4 Support 3D volumes (separate blur per axis)

## 2. Learnable Version (Optional)
- [ ] 2.1 Implement learnable blur kernel
- [ ] 2.2 Initialize with Gaussian, allow learning
- [ ] 2.3 Constrain kernel to be symmetric

## 3. Integration
- [ ] 3.1 Option A: Apply in data preprocessing (MONAI transform)
- [ ] 3.2 Option B: Apply in model (differentiable)
- [ ] 3.3 Ensure compatibility with existing cache

## 4. Experiments
- [ ] 4.1 Baseline: Standard bilinear downsample
- [ ] 4.2 Config A: Gaussian sigma=0.5
- [ ] 4.3 Config B: Gaussian sigma=1.0
- [ ] 4.4 Config C: Gaussian sigma=1.5
- [ ] 4.5 Config D: Learned kernel

## 5. Evaluation
- [ ] 5.1 Frequency spectrum analysis (how much HF preserved?)
- [ ] 5.2 Reconstruction cosine comparison
- [ ] 5.3 Report generation metrics
- [ ] 5.4 Visual inspection of downsampled volumes

## 6. Analysis
- [ ] 6.1 Identify optimal sigma for CT data
- [ ] 6.2 Check if learnable kernel diverges from Gaussian
- [ ] 6.3 Document best configuration
