## CSE 158 Assignment 1

This repository contains clean, modular solutions for the Goodreads recommendation tasks. The code adheres to the assignment specification included in `Assignment 1.pdf` and can regenerate every required prediction file.

### Environment

```bash
uv venv
source .venv/bin/activate
uv pip install -r pyproject.toml
```

### Usage

All commands accept `--data-dir` (default: `assignment1/`) and `--output-dir` (default: repo root).

```bash
uv run python assignment1.py rating
uv run python assignment1.py read
uv run python assignment1.py category
```

Each command writes `predictions_Rating.csv`, `predictions_Read.csv`, and `predictions_Category.csv` in the output directory using the required schema.

### Tests

```bash
uv run pytest
```

### Notes

- Models are intentionally simple (biased matrix factorization, feature-based logistic regression, TF-IDF + logistic regression) for reliability and speed.
- Configuration is driven by environment variables `GOOB_AI_DATA_DIR` and `GOOB_AI_OUTPUT_DIR`.
- See `writeup.txt` (to be supplied by the student) for task-specific rationale when submitting to Gradescope.

