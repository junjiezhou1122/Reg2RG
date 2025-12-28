# Change: Add ToolMed Adapter Reconstruction (Exp 31)

## Why
The core dilemma: To understand tools → need training. To be modular → don't want to retrain. Solution: Each new tool learns to speak Fusion Hub's language via a lightweight adapter. Reconstruction loss ensures information is preserved.

## What Changes
- Add adapter training framework for new tools
- Each tool gets a small adapter: tool_output → hub_language
- Train via reconstruction: hub_language → decoder → reconstructed_output
- After training: discard decoder, keep adapter

## Impact
- Affected specs: tool-adapters (new)
- Affected code:
  - `src/Model/tool_adapter.py` (new)
  - `scripts/train_tool_adapter.py` (new)
- Priority: High (Enables modularity)
- Paper potential: "Modular Tool Integration via Adapter Reconstruction"
