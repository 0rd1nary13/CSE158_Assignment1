"""Tests for the category classifier."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from goob_ai.models import CategoryClassifier

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


def test_category_classifier_identifies_keywords() -> None:
    """Classifier should distinguish simple genre cues."""
    train_df = pd.DataFrame(
        {
            "review_text": [
                "A whimsical fantasy tale with dragons",
                "A gritty mystery thriller with detectives",
                "Romantic love story set in summer",
            ],
            "genreID": [2, 3, 4],
        }
    )
    test_df = pd.DataFrame(
        {
            "review_text": [
                "Detectives solve a mystery",
                "Young dragons in fantasy lands",
            ]
        }
    )
    classifier = CategoryClassifier(max_features=1000, min_df=1)
    classifier.fit(train_df)
    probabilities = classifier.predict_proba(test_df)
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape[0] == len(test_df)
    predictions = classifier.predict(test_df)
    assert predictions.shape == (2,)

