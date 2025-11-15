"""Tests for the rating prediction model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from goob_ai.models import BiasedMatrixFactorization

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


def test_biased_mf_produces_reasonable_scores() -> None:
    """Matrix factorization should learn from a tiny synthetic dataset."""
    interactions = pd.DataFrame(
        {
            "userID": ["u1", "u1", "u2", "u2"],
            "bookID": ["b1", "b2", "b1", "b2"],
            "rating": [5, 1, 4, 2],
        }
    )
    model = BiasedMatrixFactorization(n_factors=5, n_epochs=50, learning_rate=0.05, reg=0.01)
    model.fit(interactions)
    prediction = model.predict_single("u1", "b1")
    assert 0 <= prediction <= 5
    batch = model.predict(["u1", "u2"], ["b2", "b1"])
    assert isinstance(batch, np.ndarray)
    assert batch.shape == (2,)

