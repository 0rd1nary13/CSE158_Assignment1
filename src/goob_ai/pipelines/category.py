"""Pipeline for the category prediction task."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from goob_ai.config import Settings
from goob_ai.data import load_category_test, load_category_training
from goob_ai.models import CategoryClassifier

LOGGER = logging.getLogger(__name__)


def run_category_pipeline(
    settings: Settings,
    output_path: Path | None = None,
    max_features: int = 60_000,
    min_df: int = 3,
) -> Path:
    """Train the text classifier and output predictions."""
    train_path = settings.data_dir / "train_Category.json.gz"
    test_path = settings.data_dir / "test_Category.json.gz"
    output = output_path or settings.output_dir / "predictions_Category.csv"

    LOGGER.info("Loading category training data.")
    train_df = load_category_training(train_path)
    LOGGER.info("Training dataset contains %s reviews.", len(train_df))
    model = CategoryClassifier(max_features=max_features, min_df=min_df)
    model.fit(train_df)

    LOGGER.info("Scoring category test data.")
    test_df = load_category_test(test_path)
    predictions = model.predict(test_df)
    result = pd.DataFrame(
        {
            "userID": test_df["user_id"],
            "reviewID": test_df["review_id"],
            "prediction": predictions,
        }
    )
    result.to_csv(output, index=False)
    LOGGER.info("Saved category predictions to %s", output)
    return output


__all__ = ["run_category_pipeline"]

