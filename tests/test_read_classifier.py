"""Tests for the read classifier."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from goob_ai.models import ReadClassifier

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


def test_read_classifier_predicts_probabilities() -> None:
    """Read classifier should output calibrated probabilities."""
    interactions = pd.DataFrame(
        {
            "userID": ["u1", "u1", "u2", "u3"],
            "bookID": ["b1", "b2", "b1", "b3"],
            "rating": [5, 4, 2, 5],
        }
    )
    pairs = pd.DataFrame(
        {
            "userID": ["u1", "u2", "u4"],
            "bookID": ["b1", "b3", "b2"],
        }
    )
    classifier = ReadClassifier(negative_ratio=1, max_samples=None, decision_threshold=0.6)
    classifier.fit(interactions)
    probabilities = classifier.predict_proba(pairs)
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (3,)
    labels = classifier.predict(pairs)
    assert labels.shape == (3,)
    assert set(labels).issubset({0, 1})

