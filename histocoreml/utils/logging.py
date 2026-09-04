"""Logging setup."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: Path | str | None = None,
    force: bool = False,
) -> None:
    """Configure root logging with the HistoCoreML format.

    Args:
        level:    Logging level name ('DEBUG' | 'INFO' | 'WARNING' | 'ERROR').
        log_file: If given, also append records to this file. Parent directories
                  are created automatically.
        force:    Replace handlers already installed on the root logger. Needed
                  when a library (or an earlier call) has configured logging.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=force,
    )
