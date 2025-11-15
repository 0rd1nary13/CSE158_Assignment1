"""Rating prediction via biased matrix factorization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class MFState:
    """Holds the trained matrices for serialization or reuse."""

    user_bias: np.ndarray
    item_bias: np.ndarray
    user_factors: np.ndarray
    item_factors: np.ndarray
    global_mean: float
    user_index: Dict[str, int]
    item_index: Dict[str, int]


class BiasedMatrixFactorization:
    """Simple biased matrix factorization optimized with SGD."""

    def __init__(
        self,
        n_factors: int = 40,
        n_epochs: int = 15,
        learning_rate: float = 0.01,
        reg: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.reg = reg
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self._state: MFState | None = None

    def fit(self, ratings: pd.DataFrame) -> None:
        """Fit the model to the provided ratings dataframe."""
        required_columns = {"userID", "bookID", "rating"}
        missing = required_columns - set(ratings.columns)
        if missing:
            raise ValueError(f"ratings missing required columns: {missing}")

        user_ids = ratings["userID"].astype(str)
        item_ids = ratings["bookID"].astype(str)
        values = ratings["rating"].astype(float).to_numpy()

        user_index = {user_id: idx for idx, user_id in enumerate(user_ids.unique())}
        item_index = {item_id: idx for idx, item_id in enumerate(item_ids.unique())}

        user_idx = user_ids.map(user_index).to_numpy()
        item_idx = item_ids.map(item_index).to_numpy()

        n_users = len(user_index)
        n_items = len(item_index)

        user_bias = np.zeros(n_users)
        item_bias = np.zeros(n_items)
        user_factors = self._rng.normal(scale=0.1, size=(n_users, self.n_factors))
        item_factors = self._rng.normal(scale=0.1, size=(n_items, self.n_factors))
        global_mean = float(values.mean())

        for epoch in range(self.n_epochs):
            permutation = self._rng.permutation(len(values))
            for idx in permutation:
                u = user_idx[idx]
                i = item_idx[idx]
                rating = values[idx]
                pred = (
                    global_mean
                    + user_bias[u]
                    + item_bias[i]
                    + user_factors[u] @ item_factors[i]
                )
                err = rating - pred
                user_bias[u] += self.learning_rate * (err - self.reg * user_bias[u])
                item_bias[i] += self.learning_rate * (err - self.reg * item_bias[i])

                user_factor = user_factors[u].copy()
                item_factor = item_factors[i].copy()
                user_factors[u] += self.learning_rate * (
                    err * item_factor - self.reg * user_factor
                )
                item_factors[i] += self.learning_rate * (
                    err * user_factor - self.reg * item_factor
                )

        self._state = MFState(
            user_bias=user_bias,
            item_bias=item_bias,
            user_factors=user_factors,
            item_factors=item_factors,
            global_mean=global_mean,
            user_index=user_index,
            item_index=item_index,
        )

    def predict_single(self, user_id: str, book_id: str) -> float:
        """Predict a single rating."""
        if self._state is None:
            raise RuntimeError("Model must be fit before calling predict_single.")
        user_bias = 0.0
        item_bias = 0.0
        user_vector = np.zeros(self.n_factors)
        item_vector = np.zeros(self.n_factors)

        if user_id in self._state.user_index:
            u_idx = self._state.user_index[user_id]
            user_bias = self._state.user_bias[u_idx]
            user_vector = self._state.user_factors[u_idx]

        if book_id in self._state.item_index:
            i_idx = self._state.item_index[book_id]
            item_bias = self._state.item_bias[i_idx]
            item_vector = self._state.item_factors[i_idx]

        return float(
            self._state.global_mean + user_bias + item_bias + user_vector @ item_vector
        )

    def predict(self, user_ids: Sequence[str], book_ids: Sequence[str]) -> np.ndarray:
        """Predict ratings for aligned sequences of users and books."""
        if len(user_ids) != len(book_ids):
            raise ValueError("user_ids and book_ids must be the same length.")
        return np.array(
            [self.predict_single(u, b) for u, b in zip(user_ids, book_ids, strict=True)]
        )


__all__ = ["BiasedMatrixFactorization"]

