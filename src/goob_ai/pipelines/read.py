"""Pipeline for the read prediction task."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from goob_ai.config import Settings
from goob_ai.data import load_interactions, load_pairs
from goob_ai.models import ReadClassifier

LOGGER = logging.getLogger(__name__)


def run_read_pipeline(
    settings: Settings,
    output_path: Path | None = None,
    negative_ratio: int = 3,
    max_samples: int | None = 120_000,
    decision_threshold: float = 0.5,
) -> Path:
    """Train the read classifier and emit predictions."""
    train_path = settings.data_dir / "train_Interactions.csv.gz"
    pairs_path = settings.data_dir / "pairs_Read.csv"
    output = output_path or settings.output_dir / "predictions_Read.csv"

    LOGGER.info("Loading interaction data for read prediction.")
    interactions = load_interactions(train_path)
    model = ReadClassifier(
        negative_ratio=negative_ratio,
        max_samples=max_samples,
        decision_threshold=decision_threshold,
    )
    LOGGER.info("Training read classifier with %s samples.", len(interactions))
    model.fit(interactions)

    pairs = load_pairs(pairs_path)
    LOGGER.info("Scoring %s read pairs.", len(pairs))
    predictions = model.predict(pairs)
    result = pd.DataFrame(
        {
            "userID": pairs["userID"],
            "bookID": pairs["bookID"],
            "prediction": predictions,
        }
    )
    result.to_csv(output, index=False)
    LOGGER.info("Saved read predictions to %s", output)
    return output


__all__ = ["run_read_pipeline"]

