"""Single-place logger setup so module logs are consistent."""
from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
