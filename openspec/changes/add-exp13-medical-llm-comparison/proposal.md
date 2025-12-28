# Change: Add Medical LLM Comparison (Exp 13)

## Why
Currently using LLaMA-2-7B (general-purpose). Medical-specialized LLMs may provide better domain knowledge, terminology accuracy, and reduced hallucinations. Need systematic comparison to determine if domain-specific LLM is worth the effort.

## What Changes
- Support multiple LLM backends: LLaMA-2, MedLlama, BioMedLM, PMC-LLaMA
- Create unified LLM interface for swappable backbones
- Implement evaluation metrics for medical terminology accuracy
- Compare across multiple dimensions

## Impact
- Affected specs: llm-backbone (new)
- Affected code:
  - `src/Model/llm_backends/` (new directory)
  - `src/Model/Reg2RG.py` (modular LLM loading)
  - `evaluation/medical_accuracy.py` (new)
- Priority: Medium
- Paper potential: "Domain-Specific vs General LLMs for Radiology Report Generation"
