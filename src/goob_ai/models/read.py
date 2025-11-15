"""Binary classification model for the read prediction task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class InteractionStats:
    """Aggregate statistics derived from the interactions table."""

    global_mean: float
    user_avg: Dict[str, float]
    user_count: Dict[str, int]
    item_avg: Dict[str, float]
    item_count: Dict[str, int]
    user_books: Dict[str, set[str]]
    books: List[str]
    max_user_count: int
    max_item_count: int


class ReadClassifier:
    """Logistic regression model that uses interaction statistics as features."""

    def __init__(
        self,
        negative_ratio: int = 2,
        max_samples: int | None = 150_000,
        random_state: int = 42,
        decision_threshold: float = 0.5,
    ) -> None:
        self.negative_ratio = negative_ratio
        self.max_samples = max_samples
        self.random_state = random_state
        self.decision_threshold = decision_threshold
        self._rng = np.random.default_rng(random_state)
        self._stats: InteractionStats | None = None
        self._pipeline: Pipeline | None = None

    def fit(self, interactions: pd.DataFrame) -> None:
        """Fit the classifier using sampled positives and negatives."""
        stats = self._build_stats(interactions)
        training_frame = self._build_training_frame(interactions, stats)
        labels = training_frame.pop("label")

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "model",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=500,
                        random_state=self.random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        pipeline.fit(training_frame.to_numpy(), labels.to_numpy())
        self._stats = stats
        self._pipeline = pipeline

    def predict_proba(self, pairs: pd.DataFrame) -> np.ndarray:
        """Return the probability of a 'read' outcome for each pair."""
        if self._pipeline is None or self._stats is None:
            raise RuntimeError("ReadClassifier must be fit before calling predict_proba.")
        features = self._build_feature_frame(pairs, self._stats)
        probabilities = self._pipeline.predict_proba(features.to_numpy())[:, 1]
        return probabilities

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        """Predict binary read labels for each pair."""
        probabilities = self.predict_proba(pairs)
        return (probabilities >= self.decision_threshold).astype(int)

    def _build_stats(self, interactions: pd.DataFrame) -> InteractionStats:
        """Compute aggregate statistics required for feature generation."""
        required = {"userID", "bookID", "rating"}
        if missing := required - set(interactions.columns):
            raise ValueError(f"interactions missing required columns: {missing}")

        user_groups = interactions.groupby("userID")["rating"]
        item_groups = interactions.groupby("bookID")["rating"]
        user_avg = user_groups.mean().to_dict()
        user_count = user_groups.count().to_dict()
        item_avg = item_groups.mean().to_dict()
        item_count = item_groups.count().to_dict()
        user_books: Dict[str, set[str]] = (
            interactions.groupby("userID")["bookID"].apply(lambda x: set(x)).to_dict()
        )
        books = sorted(item_count.keys())
        stats = InteractionStats(
            global_mean=float(interactions["rating"].mean()),
            user_avg=user_avg,
            user_count=user_count,
            item_avg=item_avg,
            item_count=item_count,
            user_books=user_books,
            books=books,
            max_user_count=max(user_count.values(), default=1),
            max_item_count=max(item_count.values(), default=1),
        )
        return stats

    def _build_training_frame(
        self,
        interactions: pd.DataFrame,
        stats: InteractionStats,
    ) -> pd.DataFrame:
        """Construct a dataframe with engineered features and labels."""
        sample_df = interactions
        if self.max_samples is not None and len(interactions) > self.max_samples:
            sample_df = interactions.sample(self.max_samples, random_state=self.random_state)

        positive_pairs = sample_df[["userID", "bookID"]].copy()
        positive_pairs["label"] = 1

        negative_records: List[Tuple[str, str, int]] = []
        for _, row in positive_pairs.iterrows():
            user_id = row["userID"]
            for _ in range(self.negative_ratio):
                book_id = self._sample_negative_book(user_id, stats)
                negative_records.append((user_id, book_id, 0))

        negative_pairs = pd.DataFrame(negative_records, columns=["userID", "bookID", "label"])
        combined = pd.concat([positive_pairs, negative_pairs], ignore_index=True)
        feature_frame = self._build_feature_frame(combined.drop(columns="label"), stats)
        feature_frame["label"] = combined["label"].to_numpy()
        return feature_frame

    def _sample_negative_book(self, user_id: str, stats: InteractionStats) -> str:
        """Sample a book the user has not interacted with."""
        owned = stats.user_books.get(user_id, set())
        attempts = 0
        while True:
            book_id = stats.books[self._rng.integers(0, len(stats.books))]
            if book_id not in owned:
                return book_id
            attempts += 1
            if attempts > 50:
                return book_id

    def _build_feature_frame(self, pairs: pd.DataFrame, stats: InteractionStats) -> pd.DataFrame:
        """Create the numeric features for each (user, book) pair."""
        records = []
        for _, row in pairs.iterrows():
            user_id = row["userID"]
            book_id = row["bookID"]
            user_avg = stats.user_avg.get(user_id, stats.global_mean)
            user_count = stats.user_count.get(user_id, 0)
            book_avg = stats.item_avg.get(book_id, stats.global_mean)
            book_count = stats.item_count.get(book_id, 0)
            records.append(
                {
                    "user_avg": user_avg,
                    "user_count": user_count,
                    "book_avg": book_avg,
                    "book_count": book_count,
                    "user_centered": user_avg - stats.global_mean,
                    "book_centered": book_avg - stats.global_mean,
                    "user_popularity": user_count / max(stats.max_user_count, 1),
                    "book_popularity": book_count / max(stats.max_item_count, 1),
                }
            )
        return pd.DataFrame.from_records(records)


__all__ = ["ReadClassifier"]

