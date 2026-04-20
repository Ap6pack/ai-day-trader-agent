from __future__ import annotations

from core.alpaca_executor_provider import (
    clear_alpaca_executor_cache,
    get_alpaca_executor,
)


def _set_alpaca_env(monkeypatch, api_key: str, secret_key: str) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", api_key)
    monkeypatch.setenv("ALPACA_SECRET_KEY", secret_key)
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets/v2")


def test_alpaca_executor_provider_reuses_executor_for_same_config(monkeypatch):
    clear_alpaca_executor_cache()
    _set_alpaca_env(monkeypatch, "paper-key", "paper-secret")

    first = get_alpaca_executor()
    second = get_alpaca_executor()

    assert first is second

    clear_alpaca_executor_cache()


def test_alpaca_executor_provider_uses_new_executor_when_config_changes(monkeypatch):
    clear_alpaca_executor_cache()
    _set_alpaca_env(monkeypatch, "paper-key-1", "paper-secret")
    first = get_alpaca_executor()

    _set_alpaca_env(monkeypatch, "paper-key-2", "paper-secret")
    second = get_alpaca_executor()

    assert first is not second

    clear_alpaca_executor_cache()
