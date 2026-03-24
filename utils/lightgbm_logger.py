"""LightGBM log interceptor to suppress noisy split-gain warnings."""

from __future__ import annotations

import threading
from typing import Optional


class LightGBMSplitWarningCounter:
    """Count and suppress LightGBM 'no further splits with positive gain' warnings."""

    _LOCK = threading.Lock()
    _count: int = 0
    _installed: bool = False
    _pattern = "no further splits with positive gain"

    @classmethod
    def reset(cls) -> None:
        with cls._LOCK:
            cls._count = 0

    @classmethod
    def get_count(cls) -> int:
        with cls._LOCK:
            return cls._count

    @classmethod
    def _logger(cls, msg: str) -> None:
        if cls._pattern in msg.lower():
            with cls._LOCK:
                cls._count += 1
            return
        # For all other LightGBM messages, do nothing to avoid duplicate stdout.

    @classmethod
    def install(cls) -> None:
        """Register the LightGBM logger if available."""
        if cls._installed:
            return
        try:
            import lightgbm as lgb
        except Exception:
            return
        lgb.register_logger(cls._logger)
        cls._installed = True


def install_lightgbm_warning_counter() -> None:
    LightGBMSplitWarningCounter.install()


def reset_lightgbm_warning_counter() -> None:
    LightGBMSplitWarningCounter.reset()


def get_lightgbm_warning_count() -> int:
    return LightGBMSplitWarningCounter.get_count()
