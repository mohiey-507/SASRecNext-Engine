from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


_GLOBAL_LOG_FILE: Path | None = None


def set_global_log_file(log_dir: Path, filename: str = "train.log") -> None:
    """Set a global log file and attach it to all existing and future loggers."""
    global _GLOBAL_LOG_FILE
    log_dir.mkdir(parents=True, exist_ok=True)
    _GLOBAL_LOG_FILE = log_dir / filename

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(_GLOBAL_LOG_FILE)
    file_handler.setFormatter(formatter)

    # Attach to already created loggers
    for _, logger_obj in logging.root.manager.loggerDict.items():
        if (
            isinstance(logger_obj, logging.Logger)
            and logger_obj.handlers
            and not any(isinstance(h, logging.FileHandler) for h in logger_obj.handlers)
        ):
            logger_obj.addHandler(file_handler)

    # Silence noisy torch.compile debug logs
    for noisy_logger in ("torch", "torch._inductor", "torch.__trace"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(
    name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create or retrieve a named logger with console (and optional file) output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if _GLOBAL_LOG_FILE is not None:
        file_handler = logging.FileHandler(_GLOBAL_LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
