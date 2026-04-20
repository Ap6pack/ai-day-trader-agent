from __future__ import annotations

import pytest

from core.alpaca_executor import AlpacaExecutor


def test_alpaca_executor_uses_paper_v2_endpoint(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets/v2")

    captured = {}

    class FakeResponse:
        ok = True
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "order-123",
                "side": "buy",
                "qty": "2",
                "symbol": "AAPL",
                "status": "accepted",
            }

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("core.alpaca_executor.requests.post", fake_post)
    monkeypatch.setattr(AlpacaExecutor, "is_market_open", lambda self: True)

    executor = AlpacaExecutor()
    order = executor.execute_signal({"symbol": "AAPL", "recommendation": "BUY", "quantity": 2})

    assert captured["url"] == "https://paper-api.alpaca.markets/v2/orders"
    assert captured["headers"]["APCA-API-KEY-ID"] == "key"
    assert captured["headers"]["APCA-API-SECRET-KEY"] == "secret"
    assert captured["json"] == {
        "symbol": "AAPL",
        "qty": "2",
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
    }
    assert order["id"] == "order-123"


def test_alpaca_executor_rejects_non_paper_endpoint(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets/v2")

    with pytest.raises(ValueError, match="Paper trading requires"):
        AlpacaExecutor()


def test_alpaca_executor_accepts_legacy_root_paper_url(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets")

    executor = AlpacaExecutor()

    assert executor.base_url == "https://paper-api.alpaca.markets/v2"
