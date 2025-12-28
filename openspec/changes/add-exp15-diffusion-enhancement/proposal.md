# Change: Add Diffusion-Based Super-Resolution (Exp 15)

## Why
Downsampling destroys high-frequency details. Diffusion models excel at hallucinating plausible details. Use pretrained medical diffusion model to enhance low-res reconstructions with high-frequency content.

## What Changes
- Integrate pretrained medical image diffusion model (MedDiff, etc.)
- Add conditional super-resolution pipeline: LowRes → Diffusion → HighRes
- Use ViT features as conditioning for diffusion
- Add perceptual loss for detail preservation

## Impact
- Affected specs: super-resolution (new)
- Affected code:
  - `src/Model/diffusion_sr.py` (new)
  - `src/Model/my_embedding_layer.py` (optional enhancement path)
- Priority: Low-Medium (requires external pretrained model)
- Risk: May hallucinate incorrect details (clinical safety concern)
- Paper potential: "Diffusion-Enhanced Visual Encoding for Medical VLMs"
