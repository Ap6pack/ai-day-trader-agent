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
    monkeypatch.delenv("ALPACA_LIVE_TRADING", raising=False)
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


# --- live (real money) ----------------------------------------------------------------


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets")
    monkeypatch.setenv("ALPACA_LIVE_TRADING", "true")
    for name in ("ALPACA_LIVE_MAX_ORDER_USD", "ALPACA_LIVE_MAX_ORDERS_PER_DAY", "ALPACA_LIVE_MAX_DAILY_LOSS_USD"):
        monkeypatch.delenv(name, raising=False)


def test_live_needs_both_the_host_and_the_opt_in(monkeypatch):
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets")
    monkeypatch.delenv("ALPACA_LIVE_TRADING", raising=False)
    with pytest.raises(alpaca.AccountError, match="ALPACA_LIVE_TRADING"):
        alpaca.account_mode()
    monkeypatch.setenv("ALPACA_LIVE_TRADING", "true")
    assert alpaca.account_mode() == "live" and alpaca.is_live() and not alpaca.is_paper()
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets")
    assert alpaca.account_mode() == "paper"  # the flag alone never makes paper live
    with pytest.raises(alpaca.AccountError, match="set to trade LIVE"):
        alpaca.require_mode("live")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets:8443")
    with pytest.raises(alpaca.AccountError):
        alpaca.account_mode()


def fake_live_api(monkeypatch, price=100.0, equity=10000.0, last_equity=10000.0, todays=()):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if url.endswith("/trades/latest"):
            return {"trade": {"p": price}}
        if url.endswith("/v2/account"):
            return {"equity": str(equity), "last_equity": str(last_equity)}
        if method == "GET" and url.endswith("/v2/orders"):
            return list(todays)
        return {"id": "o1", "status": "accepted"}

    monkeypatch.setattr(alpaca, "_request", fake_request)
    return calls


def posted(calls):
    return [c for c in calls if c[0] == "POST"]


def test_live_buy_within_limits_is_sent(live, monkeypatch):
    calls = fake_live_api(monkeypatch)
    alpaca.submit_order(alpaca.bracket_order_payload("AAPL", 4, 100.0, 2.0, 1.0))
    [(_, url, _)] = posted(calls)
    assert url == "https://api.alpaca.markets/v2/orders"


@pytest.mark.parametrize("kwargs, env, match", [
    ({"price": 200.0}, {}, "ALPACA_LIVE_MAX_ORDER_USD"),                       # 4 x $200 = $800 > $500
    ({"equity": 9790.0}, {}, "ALPACA_LIVE_MAX_DAILY_LOSS_USD"),                 # down $210 today
    ({"todays": [{"side": "buy", "status": "filled"}] * 3}, {}, "daily buy limit"),
    ({"todays": [{"side": "buy", "status": "filled"}]}, {"ALPACA_LIVE_MAX_ORDERS_PER_DAY": "1"}, "daily buy limit"),
])
def test_live_buy_limits(live, monkeypatch, kwargs, env, match):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    calls = fake_live_api(monkeypatch, **kwargs)
    with pytest.raises(alpaca.LiveLimitError, match=match):
        alpaca.submit_order(alpaca.bracket_order_payload("AAPL", 4, 100.0, 2.0, 1.0))
    assert posted(calls) == []


def test_live_sells_always_pass_even_after_the_loss_limit(live, monkeypatch):
    calls = fake_live_api(monkeypatch, equity=5000.0, todays=[{"side": "buy", "status": "rejected"}] * 5)
    alpaca.submit_order(alpaca.market_order_payload("AAPL", "sell", 50))
    assert len(posted(calls)) == 1


def test_close_all_positions_is_paper_only(live, monkeypatch):
    calls = fake_live_api(monkeypatch)
    with pytest.raises(alpaca.AccountError, match="paper only"):
        alpaca.close_all_positions()
    assert calls == []


def test_owned_positions_count_only_the_agents_shares(live, monkeypatch):
    todays = [
        {"id": "p1", "symbol": "NVDA", "side": "buy", "filled_qty": "5", "status": "filled",
         "client_order_id": "newsbot-a",
         "legs": [{"id": "tp1", "side": "sell", "filled_qty": "0", "status": "new"},
                  {"id": "sl1", "side": "sell", "filled_qty": "0", "status": "held"}]},
        {"id": "p2", "symbol": "AMD", "side": "buy", "filled_qty": "3", "status": "filled",
         "client_order_id": "newsbot-b",
         "legs": [{"id": "tp2", "side": "sell", "filled_qty": "3", "status": "filled"},
                  {"id": "sl2", "side": "sell", "filled_qty": "0", "status": "canceled"}]},
        {"id": "m1", "symbol": "NVDA", "side": "buy", "filled_qty": "100", "status": "filled",
         "client_order_id": "desk-mine"},
    ]
    fake_live_api(monkeypatch, todays=todays)
    owned = alpaca.owned_positions("newsbot-")
    assert owned == {"NVDA": {"qty": 5.0, "open_order_ids": ["tp1", "sl1"]}}
    assert alpaca.owned_qty("newsbot-", "nvda") == 5 and alpaca.owned_qty("newsbot-", "AMD") == 0


def test_flatten_owned_cancels_legs_then_sells_only_the_agents_shares(live, monkeypatch):
    todays = [{"id": "p1", "symbol": "NVDA", "side": "buy", "filled_qty": "5", "status": "filled",
               "client_order_id": "newsbot-a",
               "legs": [{"id": "tp1", "side": "sell", "filled_qty": "0", "status": "new"}]}]
    calls = fake_live_api(monkeypatch, todays=todays)
    alpaca.flatten_owned("newsbot-")
    deletes = [c[1] for c in calls if c[0] == "DELETE"]
    [(_, url, kwargs)] = posted(calls)
    assert deletes == ["https://api.alpaca.markets/v2/orders/tp1"]
    assert kwargs["json"]["side"] == "sell" and kwargs["json"]["qty"] == "5"
    assert kwargs["json"]["client_order_id"].startswith("newsbot-flat-")
