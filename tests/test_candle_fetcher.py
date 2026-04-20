from __future__ import annotations


def test_env_loader_does_not_require_legacy_market_data_keys(monkeypatch):
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("TWELVE_DATA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    from config import env_loader

    env_vars = env_loader.load_env_variables()

    assert "ALPHA_VANTAGE_API_KEY" in env_vars
    assert "TWELVE_DATA_API_KEY" in env_vars
    assert env_loader.validate_environment() is True


def test_source_priority_defaults_to_alpaca_then_yahoo(monkeypatch):
    monkeypatch.delenv("MARKET_DATA_PROVIDERS", raising=False)

    import core.candle_fetcher as candle_fetcher

    fetcher = candle_fetcher.CandlestickDataFetcher()

    assert fetcher.source_priority[:2] == [
        candle_fetcher.DataSource.ALPACA,
        candle_fetcher.DataSource.YAHOO_FINANCE,
    ]


def test_source_priority_can_be_configured(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_PROVIDERS", "alpaca,twelve_data,alpha_vantage,yahoo")

    import core.candle_fetcher as candle_fetcher

    fetcher = candle_fetcher.CandlestickDataFetcher()

    assert fetcher.source_priority == [
        candle_fetcher.DataSource.ALPACA,
        candle_fetcher.DataSource.TWELVE_DATA,
        candle_fetcher.DataSource.ALPHA_VANTAGE,
        candle_fetcher.DataSource.YAHOO_FINANCE,
    ]


def test_fetch_with_failover_skips_unconfigured_paid_sources(monkeypatch):
    import core.candle_fetcher as candle_fetcher

    fetcher = candle_fetcher.CandlestickDataFetcher()
    fetcher.alpaca_api_key = None
    fetcher.alpaca_secret_key = None
    fetcher.twelve_data_key = None
    fetcher.alpha_vantage_key = None
    fetcher.source_priority = [
        candle_fetcher.DataSource.ALPACA,
        candle_fetcher.DataSource.TWELVE_DATA,
        candle_fetcher.DataSource.ALPHA_VANTAGE,
        candle_fetcher.DataSource.YAHOO_FINANCE,
    ]

    candles = [
        {
            "datetime": f"2026-04-17 10:{minute:02d}:00",
            "open": "100",
            "high": "101",
            "low": "99",
            "close": str(100 + minute),
            "volume": "1000",
        }
        for minute in range(10)
    ]
    monkeypatch.setattr(
        fetcher,
        "_fetch_yahoo_candles",
        lambda symbol, intervals, outputsize: {"1min": candles},
    )

    result = fetcher.fetch_with_failover("AAPL", ("1min",), 10)

    assert result["success"] is True
    assert result["source"] == candle_fetcher.DataSource.YAHOO_FINANCE
    assert result["data"]["yahoo_finance"]["1min"] == candles
    assert "alpaca: Not configured" in result["errors"]
    assert "twelve_data: Not configured" in result["errors"]
    assert "alpha_vantage: Not configured" in result["errors"]


def test_alpaca_candles_use_official_market_data_endpoint(monkeypatch):
    import core.candle_fetcher as candle_fetcher

    captured = {}

    class FakeResponse:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "bars": [
                    {"t": f"2026-04-17T10:{minute:02d}:00Z", "o": 100, "h": 101, "l": 99, "c": 100 + minute, "v": 1000}
                    for minute in range(10)
                ]
            }

    def fake_get(url, headers, params, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["params"] = params
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(candle_fetcher.requests, "get", fake_get)

    fetcher = candle_fetcher.CandlestickDataFetcher()
    fetcher.alpaca_api_key = "key"
    fetcher.alpaca_secret_key = "secret"
    fetcher.alpaca_data_base_url = "https://data.alpaca.markets"
    fetcher.alpaca_data_feed = "iex"

    result = fetcher._fetch_alpaca_candles("AAPL", ("1min",), 10)

    assert captured["url"] == "https://data.alpaca.markets/v2/stocks/AAPL/bars"
    assert captured["headers"] == {
        "APCA-API-KEY-ID": "key",
        "APCA-API-SECRET-KEY": "secret",
    }
    assert captured["params"]["timeframe"] == "1Min"
    assert captured["params"]["feed"] == "iex"
    assert captured["params"]["sort"] == "desc"
    assert len(result["1min"]) == 10


def test_rate_limited_provider_fails_fast_without_sleep(monkeypatch):
    import core.candle_fetcher as candle_fetcher

    fetcher = candle_fetcher.CandlestickDataFetcher()
    fetcher.alpaca_api_key = "key"
    fetcher.alpaca_secret_key = "secret"
    fetcher.source_priority = [
        candle_fetcher.DataSource.ALPACA,
        candle_fetcher.DataSource.YAHOO_FINANCE,
    ]

    rate_limiter = fetcher.rate_limiters[candle_fetcher.DataSource.ALPACA]
    rate_limiter.record_rate_limit(60)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Rate-limited provider should not make an HTTP request")

    candles = [
        {
            "datetime": f"2026-04-17 10:{minute:02d}:00",
            "open": "100",
            "high": "101",
            "low": "99",
            "close": str(100 + minute),
            "volume": "1000",
        }
        for minute in range(10)
    ]

    monkeypatch.setattr(candle_fetcher.requests, "get", fail_if_called)
    monkeypatch.setattr(
        fetcher,
        "_fetch_yahoo_candles",
        lambda symbol, intervals, outputsize: {"1min": candles},
    )

    result = fetcher.fetch_with_failover("AAPL", ("1min",), 10)

    assert result["success"] is True
    assert result["source"] == candle_fetcher.DataSource.YAHOO_FINANCE
    assert any("alpaca:" in error and "Rate limited" in error for error in result["errors"])


def test_candlestick_fetcher_provider_reuses_fetcher(monkeypatch):
    from core.candle_fetcher_provider import (
        clear_candlestick_fetcher_cache,
        get_candlestick_fetcher,
    )

    clear_candlestick_fetcher_cache()
    monkeypatch.setenv("MARKET_DATA_PROVIDERS", "alpaca,yahoo_finance")
    monkeypatch.setenv("ALPACA_API_KEY", "paper-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "paper-secret")

    first = get_candlestick_fetcher()
    second = get_candlestick_fetcher()

    assert first is second

    clear_candlestick_fetcher_cache()


def test_candlestick_fetcher_provider_refreshes_when_config_changes(monkeypatch):
    from core.candle_fetcher_provider import (
        clear_candlestick_fetcher_cache,
        get_candlestick_fetcher,
    )

    clear_candlestick_fetcher_cache()
    monkeypatch.setenv("MARKET_DATA_PROVIDERS", "alpaca,yahoo_finance")
    monkeypatch.setenv("ALPACA_API_KEY", "paper-key-1")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "paper-secret")
    first = get_candlestick_fetcher()

    monkeypatch.setenv("ALPACA_API_KEY", "paper-key-2")
    second = get_candlestick_fetcher()

    assert first is not second

    clear_candlestick_fetcher_cache()


def test_get_candlestick_data_uses_cached_provider(monkeypatch):
    import core.candle_fetcher as candle_fetcher
    from core.candle_fetcher_provider import clear_candlestick_fetcher_cache

    class FakeFetcher:
        def fetch_with_failover(self, symbol, intervals, outputsize):
            return {
                "success": True,
                "source": candle_fetcher.DataSource.ALPACA,
                "data": {"alpaca": {"1min": [{"close": 100}]}},
                "errors": [],
            }

    clear_candlestick_fetcher_cache()
    monkeypatch.setattr(
        "core.candle_fetcher_provider.get_candlestick_fetcher",
        lambda: FakeFetcher(),
    )

    result = candle_fetcher.get_candlestick_data("AAPL", ("1min",), 1)

    assert result == {"alpaca": {"1min": [{"close": 100}]}}
