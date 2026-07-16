"""Logging helpers shared by command-line dataset runs."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(output_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Configure console and file logging and return the package logger."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("protein_conformer_dataset")
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(destination / "run_log.txt", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
