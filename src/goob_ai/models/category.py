"""Text classification for predicting review categories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class CategoryClassifier:
    """TF-IDF + logistic regression classifier for review genres."""

    def __init__(
        self,
        max_features: int = 50_000,
        min_df: int = 3,
        random_state: int = 42,
    ) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.random_state = random_state
        self._pipeline: Pipeline | None = None
        self._label_lookup: Dict[int, str] = {}

    def fit(self, training_frame: pd.DataFrame) -> None:
        """Train the classifier using the labeled review text."""
        required = {"review_text", "genreID"}
        if missing := required - set(training_frame.columns):
            raise ValueError(f"training_frame missing required columns: {missing}")
        texts = training_frame["review_text"].astype(str).to_list()
        labels = training_frame["genreID"].astype(int).to_numpy()
        vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=(1, 2),
        )
        classifier = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        )
        pipeline = Pipeline([("tfidf", vectorizer), ("clf", classifier)])
        pipeline.fit(texts, labels)
        self._pipeline = pipeline

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict category IDs for each review."""
        probabilities = self.predict_proba(frame)
        return probabilities.argmax(axis=1)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return probability distribution over genres for each review."""
        if self._pipeline is None:
            raise RuntimeError("CategoryClassifier must be fit before prediction.")
        texts = frame["review_text"].astype(str).to_list()
        probabilities = self._pipeline.predict_proba(texts)
        return probabilities


__all__ = ["CategoryClassifier"]

