#!/usr/bin/env python3
"""Process-scoped PortfolioManager provider."""

from __future__ import annotations

from functools import lru_cache
import os

from core.portfolio_manager import PortfolioManager


def configured_portfolio_db_path() -> str:
    """Return a stable cache key for the configured portfolio database."""
    db_path = os.getenv("PORTFOLIO_DB_PATH", "data/portfolios.db")
    return os.path.abspath(os.path.expanduser(db_path))


@lru_cache(maxsize=8)
def get_cached_portfolio_manager(db_path: str) -> PortfolioManager:
    """Create one portfolio manager per database path for this process."""
    return PortfolioManager(db_path)


def get_portfolio_manager() -> PortfolioManager:
    """Return the process-scoped portfolio manager for the configured database."""
    return get_cached_portfolio_manager(configured_portfolio_db_path())


def clear_portfolio_manager_cache() -> None:
    """Clear cached portfolio managers after test or configuration changes."""
    get_cached_portfolio_manager.cache_clear()
