"""Data loading helpers for the Goodreads assignment."""

from __future__ import annotations

import ast
import gzip
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

import pandas as pd


def load_interactions(path: Path) -> pd.DataFrame:
    """Load the ratings interactions CSV into a DataFrame."""
    return pd.read_csv(path)


def load_pairs(path: Path) -> pd.DataFrame:
    """Load a CSV containing (userID, bookID) pairs."""
    return pd.read_csv(path)


def _read_category_records(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield dictionary records from the gzip-compressed category file."""
    with gzip.open(path, "rt") as handle:
        for raw_line in handle:
            yield ast.literal_eval(raw_line)


def load_category_training(path: Path) -> pd.DataFrame:
    """Load the category training dataset."""
    records = list(_read_category_records(path))
    return pd.DataFrame.from_records(records)


def load_category_test(path: Path) -> pd.DataFrame:
    """Load the category test dataset."""
    records = list(_read_category_records(path))
    return pd.DataFrame.from_records(records)


__all__ = [
    "load_interactions",
    "load_pairs",
    "load_category_training",
    "load_category_test",
]

