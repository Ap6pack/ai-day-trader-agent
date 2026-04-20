from __future__ import annotations

from core.dividend_provider_config import (
    active_dividend_providers,
    configured_dividend_provider_order,
    configured_dividend_providers,
    dividend_strategy_available,
    dividend_strategy_enabled,
)
from core.pipeline import EnhancedTradingPipeline


def test_dividend_provider_status_respects_env(monkeypatch):
    monkeypatch.setenv("DIVIDEND_STRATEGY_ENABLED", "true")
    monkeypatch.setenv("DIVIDEND_DATA_PROVIDERS", "alpha_vantage,yahoo_finance")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "alpha-key")
    monkeypatch.delenv("TWELVE_DATA_API_KEY", raising=False)

    assert dividend_strategy_enabled() is True
    assert configured_dividend_provider_order() == ["alpha_vantage", "yahoo_finance"]
    assert configured_dividend_providers()["alpha_vantage"] is True
    assert configured_dividend_providers()["twelve_data"] is False
    assert active_dividend_providers() == ["alpha_vantage", "yahoo_finance"]
    assert dividend_strategy_available() is True


def test_dividend_strategy_can_be_disabled(monkeypatch):
    monkeypatch.setenv("DIVIDEND_STRATEGY_ENABLED", "false")
    monkeypatch.setenv("DIVIDEND_DATA_PROVIDERS", "alpha_vantage")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "alpha-key")

    assert dividend_strategy_enabled() is False
    assert dividend_strategy_available() is False


def test_dividend_analysis_skips_fetcher_when_disabled(monkeypatch):
    monkeypatch.setenv("DIVIDEND_STRATEGY_ENABLED", "false")

    def fail_if_fetcher_is_constructed(*args, **kwargs):
        raise AssertionError("DividendDataFetcher should not be constructed")

    monkeypatch.setattr("core.pipeline.DividendDataFetcher", fail_if_fetcher_is_constructed)

    pipeline = EnhancedTradingPipeline.__new__(EnhancedTradingPipeline)
    pipeline.symbol = "APAM"

    signal = pipeline._run_dividend_analysis({}, {}, {})

    assert signal["signal"] == "HOLD"
    assert signal["reason"] == "Dividend strategy disabled by configuration"


def test_dividend_analysis_skips_fetcher_when_no_providers(monkeypatch):
    monkeypatch.setenv("DIVIDEND_STRATEGY_ENABLED", "true")
    monkeypatch.setenv("DIVIDEND_DATA_PROVIDERS", "alpha_vantage,twelve_data")
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("TWELVE_DATA_API_KEY", raising=False)

    def fail_if_fetcher_is_constructed(*args, **kwargs):
        raise AssertionError("DividendDataFetcher should not be constructed")

    monkeypatch.setattr("core.pipeline.DividendDataFetcher", fail_if_fetcher_is_constructed)

    pipeline = EnhancedTradingPipeline.__new__(EnhancedTradingPipeline)
    pipeline.symbol = "APAM"

    signal = pipeline._run_dividend_analysis({}, {}, {})

    assert signal["signal"] == "HOLD"
    assert signal["reason"] == "Dividend strategy skipped: no dividend data providers configured"
