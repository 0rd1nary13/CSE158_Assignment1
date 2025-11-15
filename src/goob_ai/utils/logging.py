"""Logging helpers."""

from __future__ import annotations

import logging
from typing import Literal


def configure_logging(level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO") -> None:
    """Configure basic logging for CLI usage."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


__all__ = ["configure_logging"]

