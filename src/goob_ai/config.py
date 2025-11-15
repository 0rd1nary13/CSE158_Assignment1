"""Runtime configuration utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Application settings resolved from environment variables."""

    data_dir: Path
    output_dir: Path

    @staticmethod
    def from_env() -> "Settings":
        """Create a settings instance using environment variables."""
        project_root = Path(__file__).resolve().parents[2]
        data_dir = Path(os.getenv("GOOB_AI_DATA_DIR", project_root / "assignment1")).resolve()
        output_dir = Path(os.getenv("GOOB_AI_OUTPUT_DIR", project_root)).resolve()
        return Settings(data_dir=data_dir, output_dir=output_dir)


DEFAULT_SETTINGS = Settings.from_env()

__all__ = ["Settings", "DEFAULT_SETTINGS"]

