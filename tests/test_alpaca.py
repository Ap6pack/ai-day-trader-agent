from __future__ import annotations

import pytest

from trader import alpaca


@pytest.fixture
def no_http(monkeypatch):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return {"id": "o1"}

    monkeypatch.setattr(alpaca, "_request", fake_request)
    return calls


def test_default_base_url_is_paper(monkeypatch):
    monkeypatch.delenv("ALPACA_TRADING_BASE_URL", raising=False)
    assert alpaca.trading_base_url() == "https://paper-api.alpaca.markets"
    assert alpaca.is_paper()


def test_trailing_v2_is_stripped(monkeypatch):
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets/v2/")
    assert alpaca.trading_base_url() == "https://paper-api.alpaca.markets"
    assert alpaca.is_paper()


@pytest.mark.parametrize("url", [
    "https://api.alpaca.markets",
    "http://paper-api.alpaca.markets",
    "https://paper-api.alpaca.markets.evil.example",
    "https://evil.example/paper-api.alpaca.markets",
])
def test_every_order_function_refuses_non_paper(monkeypatch, no_http, url):
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", url)
    assert not alpaca.is_paper()
    payload = alpaca.bracket_order_payload("AAPL", 1, 100.0, 2.0, 1.0)
    with pytest.raises(alpaca.NotPaperError):
        alpaca.submit_order(payload)
    with pytest.raises(alpaca.NotPaperError):
        alpaca.close_all_positions()
    with pytest.raises(alpaca.NotPaperError):
        alpaca.positions()
    assert no_http == []


def test_order_functions_call_paper_endpoints(monkeypatch, no_http):
    monkeypatch.delenv("ALPACA_TRADING_BASE_URL", raising=False)
    alpaca.submit_order({"symbol": "AAPL"})
    alpaca.close_all_positions()
    assert no_http[0][:2] == ("POST", "https://paper-api.alpaca.markets/v2/orders")
    assert no_http[1][:2] == ("DELETE", "https://paper-api.alpaca.markets/v2/positions")
    assert no_http[1][2]["params"] == {"cancel_orders": "true"}


def test_bracket_payload_prices():
    payload = alpaca.bracket_order_payload("aapl", 4, 123.456, 2.0, 1.0)
    assert payload["symbol"] == "AAPL"
    assert payload["qty"] == "4"
    assert payload["side"] == "buy"
    assert payload["type"] == "market"
    assert payload["time_in_force"] == "day"
    assert payload["order_class"] == "bracket"
    assert payload["take_profit"] == {"limit_price": "125.93"}  # 123.456 * 1.02, 2 decimals
    assert payload["stop_loss"] == {"stop_price": "122.22"}     # 123.456 * 0.99
    assert payload["client_order_id"].startswith("newsbot-")


def test_client_order_id_is_deterministic_per_key():
    a = alpaca.bracket_order_payload("AAPL", 1, 100.0, 2.0, 1.0, key="123:AAPL")
    b = alpaca.bracket_order_payload("AAPL", 1, 100.0, 2.0, 1.0, key="123:AAPL")
    c = alpaca.bracket_order_payload("AAPL", 1, 100.0, 2.0, 1.0, key="124:AAPL")
    assert a["client_order_id"] == b["client_order_id"] != c["client_order_id"]
    assert len(a["client_order_id"]) <= 48


@pytest.mark.parametrize("qty", [0, 1.5])
def test_bracket_needs_whole_shares(qty):
    with pytest.raises(ValueError):
        alpaca.bracket_order_payload("AAPL", qty, 100.0, 2.0, 1.0)


def test_latest_price_uses_configured_feed(monkeypatch):
    seen = {}

    def fake_request(method, url, **kwargs):
        seen.update(url=url, params=kwargs.get("params"))
        return {"symbol": "AAPL", "trade": {"p": 201.25}}

    monkeypatch.setattr(alpaca, "_request", fake_request)
    monkeypatch.setenv("ALPACA_DATA_FEED", "sip")
    assert alpaca.latest_price("AAPL") == 201.25
    assert seen["url"].endswith("/v2/stocks/AAPL/trades/latest")
    assert seen["params"] == {"feed": "sip"}
