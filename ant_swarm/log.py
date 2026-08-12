"""Loguru setup shared by all entry scripts.

    from ant_swarm import setup_logging
    setup_logging()                 # console only
    setup_logging(run_dir)          # console + <run_dir>/train.log

Console stays compact; the file sink keeps everything at DEBUG for post-mortems.
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_FMT = ("<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <7}</level> | {message}")


def setup_logging(run_dir: str | Path | None = None, level: str = "INFO"):
    """Reset loguru sinks: stderr console (+ optional per-run log file)."""
    logger.remove()
    logger.add(sys.stderr, level=level, format=_FMT, colorize=True)
    if run_dir is not None:
        logger.add(Path(run_dir) / "train.log", level="DEBUG", format=_FMT,
                   colorize=False, enqueue=True)
    return logger
