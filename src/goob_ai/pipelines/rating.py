"""Pipeline for the rating prediction task."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from goob_ai.config import Settings
from goob_ai.data import load_interactions, load_pairs
from goob_ai.models import BiasedMatrixFactorization

LOGGER = logging.getLogger(__name__)


def run_rating_pipeline(
    settings: Settings,
    output_path: Path | None = None,
    n_factors: int = 60,
    n_epochs: int = 20,
    learning_rate: float = 0.01,
    reg: float = 0.05,
) -> Path:
    """Train the rating model and write predictions to disk."""
    train_path = settings.data_dir / "train_Interactions.csv.gz"
    pairs_path = settings.data_dir / "pairs_Rating.csv"
    output = output_path or settings.output_dir / "predictions_Rating.csv"

    LOGGER.info("Loading interactions from %s", train_path)
    interactions = load_interactions(train_path)
    model = BiasedMatrixFactorization(
        n_factors=n_factors,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        reg=reg,
    )
    LOGGER.info("Training matrix factorization model.")
    model.fit(interactions)

    LOGGER.info("Generating predictions for rating pairs.")
    pairs = load_pairs(pairs_path)
    predictions = model.predict(pairs["userID"].astype(str), pairs["bookID"].astype(str))

    result = pd.DataFrame(
        {
            "userID": pairs["userID"],
            "bookID": pairs["bookID"],
            "prediction": predictions,
        }
    )
    result.to_csv(output, index=False)
    LOGGER.info("Saved rating predictions to %s", output)
    return output


__all__ = ["run_rating_pipeline"]

