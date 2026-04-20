#!/usr/bin/env python3
"""Shared FastAPI dependency providers."""

from __future__ import annotations

from core.portfolio_manager_provider import (
    clear_portfolio_manager_cache,
    get_portfolio_manager,
)

__all__ = ["clear_portfolio_manager_cache", "get_portfolio_manager"]
