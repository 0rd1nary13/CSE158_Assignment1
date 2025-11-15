"""Command-line interface for running assignment pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from goob_ai.config import DEFAULT_SETTINGS, Settings
from goob_ai.pipelines.category import run_category_pipeline
from goob_ai.pipelines.rating import run_rating_pipeline
from goob_ai.pipelines.read import run_read_pipeline
from goob_ai.utils.logging import configure_logging

app = typer.Typer(add_completion=False, help="Pipelines for the Goodreads assignment.")


@app.callback()
def main(
    ctx: typer.Context,
    data_dir: Path = typer.Option(
        DEFAULT_SETTINGS.data_dir,
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Directory containing the assignment data files.",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_SETTINGS.output_dir,
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Directory where predictions will be written.",
    ),
    log_level: str = typer.Option(
        "INFO",
        help="Log level (DEBUG, INFO, WARNING, ERROR).",
    ),
) -> None:
    """Configure shared CLI context."""
    configure_logging(log_level.upper())
    ctx.obj = Settings(data_dir=data_dir, output_dir=output_dir)


@app.command("rating")
def rating_command(
    ctx: typer.Context,
    n_factors: int = typer.Option(80, help="Number of latent factors for MF."),
    n_epochs: int = typer.Option(25, help="Number of SGD epochs."),
    learning_rate: float = typer.Option(0.01, help="Learning rate for SGD."),
    reg: float = typer.Option(0.05, help="L2 regularization strength."),
) -> None:
    """Generate predictions for the rating task."""
    settings: Settings = ctx.obj
    output = run_rating_pipeline(
        settings=settings,
        n_factors=n_factors,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        reg=reg,
    )
    typer.echo(f"Wrote rating predictions to {output}")


@app.command("read")
def read_command(
    ctx: typer.Context,
    negative_ratio: int = typer.Option(3, help="Number of negatives per positive."),
    max_samples: Optional[int] = typer.Option(
        150_000,
        help="Maximum number of positive interactions sampled for training.",
    ),
    threshold: float = typer.Option(0.5, help="Probability threshold for classification."),
) -> None:
    """Generate predictions for the read task."""
    settings: Settings = ctx.obj
    output = run_read_pipeline(
        settings=settings,
        negative_ratio=negative_ratio,
        max_samples=max_samples,
        decision_threshold=threshold,
    )
    typer.echo(f"Wrote read predictions to {output}")


@app.command("category")
def category_command(
    ctx: typer.Context,
    max_features: int = typer.Option(70_000, help="Maximum vocabulary size."),
    min_df: int = typer.Option(3, help="Minimum document frequency."),
) -> None:
    """Generate predictions for the category task."""
    settings: Settings = ctx.obj
    output = run_category_pipeline(
        settings=settings,
        max_features=max_features,
        min_df=min_df,
    )
    typer.echo(f"Wrote category predictions to {output}")


def run() -> None:
    """Entry point for `python -m goob_ai.cli`."""
    app()


__all__ = ["app", "run"]

