# Repository Guidelines

## Project Structure & Module Organization
Keep CLI inference and demos in `scripts/`, with training entry points under `train/src/train/`. Dataset tooling lives in `train/src/train/data.py`, and YAML configs sit in `train/train/config/`. Place media prompts and LoRA checkpoints inside `assets/`, hydrated parquet shards in `train/parquet/`, and training outputs under `train/runs/`. Documentation and figures belong in `docs/` and `docs/images/` to keep the root clean.

## Build, Test, and Development Commands
Create a Python 3.10 virtualenv, then install dependencies via `pip install -r requirements.txt`. Run training extras with `pip install -r train/requirements.txt` when needed. Execute a single edit using `python scripts/inference.py --image assets/girl.png --instruction "..."`. Launch the baseline Gradio UI through `python scripts/gradio_demo.py` or switch to the MoE variant with `python scripts/gradio_demo_moe.py`. Prepare datasets using `bash train/parquet/prepare.sh`, and kick off training through `bash train/train/script/train.sh` (or `train_moe.sh` for mixtures).

## Coding Style & Naming Conventions
Write Python following PEP 8 with four-space indentation, snake_case for functions and variables, and PascalCase for classes. Surface seed and device configuration near file tops. Mirror existing docstrings: `"""Short summary."""`. Favor composition-friendly modules over monolith scripts. Stick to YAML configs with lowercase-hyphenated keys, and include concise comments only where logic is non-obvious.

## Testing Guidelines
No formal unit suite exists yet. Validate inference changes with the CLI against reference assets and capture before/after imagery. For training tweaks, run short validation sweeps by tuning `max_steps` and `eval_every_n` in YAML, then document metrics from the console logger and summarize the relevant `train/runs/` output directory.

## Commit & Pull Request Guidelines
Use present-tense, ≤72-character commit subjects (e.g., `update moe training`). PRs should explain intent, outline reproduction steps, attach images or loss curves, and link related issues. Confirm CI or manual smoke tests before requesting review, and note any regressions or instabilities observed during validation.
