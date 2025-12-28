# Change: Add Knowledge Distillation from High-Res (Exp 23)

## Why
A teacher model processing high-resolution input "knows" what details are important. Distill this knowledge to a student model processing low-resolution input. The student learns what to preserve even when it can't see the details directly.

## What Changes
- Add `ResolutionDistillation` training framework
- Teacher: Process full-resolution volume (expensive but accurate)
- Student: Process resized volume (efficient but lossy)
- Distill teacher's knowledge to student via feature matching

## Impact
- Affected specs: resolution-distillation (new)
- Affected code:
  - `src/Model/resolution_distillation.py` (new)
  - `src/train_distillation.py` (new training script)
- Priority: Medium
- Paper potential: "Resolution-Agnostic Medical VLMs via Knowledge Distillation"
