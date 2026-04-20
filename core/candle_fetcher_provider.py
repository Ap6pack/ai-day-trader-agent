#!/usr/bin/env python3
"""Process-scoped candlestick data fetcher provider."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from typing import Tuple

from core.candle_fetcher import CandlestickDataFetcher


def _secret_fingerprint(value: str | None) -> str:
    """Return a stable non-secret cache key fragment."""
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _configured_fetcher_cache_key() -> Tuple[str, ...]:
    """Return a cache key that changes when market-data config changes."""
    alpaca_key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY_ID")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET")
    return (
        os.getenv("MARKET_DATA_PROVIDERS", ""),
        _secret_fingerprint(alpaca_key),
        _secret_fingerprint(alpaca_secret),
        os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets").rstrip("/"),
        os.getenv("ALPACA_DATA_FEED", "iex"),
        _secret_fingerprint(os.getenv("TWELVE_DATA_API_KEY")),
        _secret_fingerprint(os.getenv("ALPHA_VANTAGE_API_KEY")),
    )


@lru_cache(maxsize=4)
def _get_cached_candlestick_fetcher(_cache_key: Tuple[str, ...]) -> CandlestickDataFetcher:
    """Create one market-data fetcher per provider/credential configuration."""
    return CandlestickDataFetcher()


def get_candlestick_fetcher() -> CandlestickDataFetcher:
    """Return the process-scoped market-data fetcher."""
    return _get_cached_candlestick_fetcher(_configured_fetcher_cache_key())


def clear_candlestick_fetcher_cache() -> None:
    """Clear cached market-data fetchers after test or configuration changes."""
    _get_cached_candlestick_fetcher.cache_clear()
