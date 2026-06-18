"""
Centralized logging configuration using loguru.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_dir: str | Path | None = None, level: str = "INFO") -> None:
    """
    Configure loguru logger with console and optional file output.

    Args:
        log_dir: Directory for log files. If None, logs to console only.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    logger.remove()  # Remove default handler

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    logger.add(sys.stderr, format=fmt, level=level, colorize=True)

    if log_dir is not None:
        log_path = Path(log_dir) / "run_{time:YYYYMMDD_HHmmss}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            format=fmt,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )


def get_logger(name: str):
    """
    Return a loguru logger bound to a module name.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        Bound loguru logger.
    """
    return logger.bind(module=name)