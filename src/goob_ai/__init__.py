"""Top-level package for the Goodreads assignment solution."""

from importlib.metadata import version, PackageNotFoundError


def get_version() -> str:
    """Return the installed package version."""
    try:
        return version("goob-ai")
    except PackageNotFoundError:
        return "0.0.0"


__all__ = ["get_version"]

