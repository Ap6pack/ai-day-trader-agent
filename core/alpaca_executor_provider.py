#!/usr/bin/env python3
"""Process-scoped AlpacaExecutor provider."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from typing import Tuple

from core.alpaca_executor import AlpacaExecutor


def _secret_fingerprint(value: str | None) -> str:
    """Return a stable non-secret cache key fragment."""
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _configured_executor_cache_key() -> Tuple[str, str, str]:
    """Return a cache key that changes when Alpaca credentials/config changes."""
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET")
    base_url = (
        os.getenv("ALPACA_TRADING_BASE_URL")
        or os.getenv("ALPACA_BASE_URL")
        or "https://paper-api.alpaca.markets/v2"
    )
    return (
        _secret_fingerprint(api_key),
        _secret_fingerprint(secret_key),
        base_url.rstrip("/"),
    )


@lru_cache(maxsize=4)
def _get_cached_alpaca_executor(_cache_key: Tuple[str, str, str]) -> AlpacaExecutor:
    """Create one Alpaca executor per credentials/base-url configuration."""
    return AlpacaExecutor()


def get_alpaca_executor() -> AlpacaExecutor:
    """Return the process-scoped Alpaca paper-trading executor."""
    return _get_cached_alpaca_executor(_configured_executor_cache_key())


def clear_alpaca_executor_cache() -> None:
    """Clear cached Alpaca executors after test or configuration changes."""
    _get_cached_alpaca_executor.cache_clear()
