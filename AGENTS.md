# Repository Guidelines

## Project Structure & Module Organization
- `src/Model/`: Core model code (e.g., `Reg2RG.py`, `vit_3d.py`, `transformer_decoder.py`, `cross_attention.py`, helpers).
- `src/Dataset/`: Data loaders and dataset utilities for RadGenome.
- `src/args/`: Experiment argument presets and host-specific launch configs.
- `evaluation/`: Evaluation scripts (NLG metrics, classification, region parsing).
- `scripts/`: Convenience launchers (e.g., `train_radgenome.sh`, `test_radgenome.sh`) and W&B run artifacts.
- `configs/`, `ds_configs/`: Shell and DeepSpeed configs for runs.
- `results/`: Generated reports, metrics, and CSV outputs (not source of truth).

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Train: `python src/train_radgenome.py --help` or `bash scripts/train_radgenome.sh`
- Test/Inference: `python src/test_radgenome.py --help` or `bash scripts/test_radgenome.sh`
- Evaluate (example): `python evaluation/hf_nlg_evaluation_region.py --help`

## Coding Style & Naming Conventions
- Python 3.x, 4-space indentation, PEP 8, f-strings.
- Names: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.
- Keep new modules under existing domains (`Model`, `Dataset`, `evaluation`). Avoid new top-level dirs.
- Prefer type hints and docstrings for public functions. Keep lines ≤ 100 chars.

## Testing Guidelines
- Existing script-level tests live in `src/test_radgenome.py` and `evaluation/` tools.
- For new work, prefer `pytest`; place under `tests/` as `tests/test_*.py`.
- Strive for coverage on utilities and data transforms; set seeds for determinism.
- Run: `pytest -q` (if added) or use the test/eval scripts above.

## Commit & Pull Request Guidelines
- Use Conventional Commits (observed in history): `feat`, `fix`, `docs`, `refactor(scope)`, `chore`.
- Commit message: short imperative summary; optional body for rationale/impacts; link issues.
- PRs should include: purpose, configuration used (`configs/`, `ds_configs/`), steps to reproduce, and result artifacts (e.g., paths in `results/Reg2RG_*/*.csv`).
- Structure PR description for clarity: Problem → Intuition (2–3 sentences) → Changes → Results (numbers) → Limits.

## Security & Configuration Tips
- Do not commit PHI or raw datasets. Large outputs belong in `results/`, not tracked.
- Keep secrets (e.g., `WANDB_API_KEY`) in environment variables, not files.
- Validate paths/filenames; avoid hard-coding machine-specific directories in committed code.

## Learning-Focused Communication
- Start with the problem before the solution; make it concrete with numbers when relevant.
- Add brief intuition (2–3 sentences), then actionable steps or commands.
- Present key insights as concise bullets (3–6), ordered by importance; include exact metrics when possible.
- Show code/config examples with file references and optional line numbers (e.g., `src/train_radgenome.py:42`, `ds_configs/stage2.json`). Prefer “Without vs With” side-by-side when explaining changes.
- For math/metrics, show intermediate steps and units to make reasoning verifiable.
- Call out limitations and comparisons briefly so reviewers understand trade-offs.

## Agent-Specific Notes
- Minimize changes; follow existing structure and naming. Update `README.md` when user-facing behavior changes. Do not move files unless agreed.
