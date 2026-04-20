#!/usr/bin/env python3
"""Dividend provider configuration helpers."""

from __future__ import annotations

import os
from typing import Dict, List


DIVIDEND_PROVIDER_ALIASES = {
    "twelve_data": "twelve_data",
    "twelvedata": "twelve_data",
    "alpha_vantage": "alpha_vantage",
    "alphavantage": "alpha_vantage",
    "alpha": "alpha_vantage",
    "yahoo_finance": "yahoo_finance",
    "yahoo": "yahoo_finance",
}


def dividend_strategy_enabled() -> bool:
    """Return whether dividend capture analysis should run."""
    return os.getenv("DIVIDEND_STRATEGY_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def configured_dividend_provider_order() -> List[str]:
    """Return configured dividend provider order."""
    raw_order = os.getenv("DIVIDEND_DATA_PROVIDERS", "twelve_data,alpha_vantage")
    providers: List[str] = []
    for provider in raw_order.split(","):
        normalized = DIVIDEND_PROVIDER_ALIASES.get(provider.strip().lower())
        if normalized and normalized not in providers:
            providers.append(normalized)
    return providers


def configured_dividend_providers(api_keys: Dict[str, str | None] | None = None) -> Dict[str, bool]:
    """Return provider configuration status without exposing secrets."""
    api_keys = api_keys or {}
    return {
        "twelve_data": bool(
            api_keys.get("TWELVE_DATA_API_KEY") or os.getenv("TWELVE_DATA_API_KEY")
        ),
        "alpha_vantage": bool(
            api_keys.get("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY")
        ),
        "yahoo_finance": True,
    }


def active_dividend_providers(api_keys: Dict[str, str | None] | None = None) -> List[str]:
    """Return enabled providers that are currently callable."""
    configured = configured_dividend_providers(api_keys)
    return [
        provider
        for provider in configured_dividend_provider_order()
        if configured.get(provider, False)
    ]


def dividend_strategy_available(api_keys: Dict[str, str | None] | None = None) -> bool:
    """Return whether dividend analysis can run with current configuration."""
    return dividend_strategy_enabled() and bool(active_dividend_providers(api_keys))
