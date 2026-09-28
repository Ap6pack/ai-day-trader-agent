from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.indicator_engine import compute_indicators, sort_candlesticks
from core.portfolio_manager_provider import clear_portfolio_manager_cache

BARS = 200
START = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)


def _uptrend_closes():
    # Steady climb from ~100 to ~200 with a small zigzag so RSI/ATR are well defined.
    return [100 + i * 0.5 + (0.3 if i % 2 else -0.3) for i in range(BARS)]


def _alpaca_style_candles():
    """Hourly candles shaped like Alpaca's response with sort=desc: newest first."""
    closes = _uptrend_closes()
    candles = []
    for i, close in enumerate(closes):
        open_ = closes[i - 1] if i else close
        candles.append({
            "datetime": (START + timedelta(hours=i)).isoformat().replace("+00:00", "Z"),
            "open": open_,
            "high": max(open_, close) + 0.2,
            "low": min(open_, close) - 0.2,
            "close": close,
            "volume": 1000 + i,
        })
    candles.reverse()
    return candles


def test_sort_candlesticks_orders_newest_first_input_oldest_first():
    data = {
        "close": [3.0, 2.0, 1.0],
        "open": [3.0, 2.0, 1.0],
        "datetime": ["2026-01-03 10:00:00", "2026-01-02 10:00:00", "2026-01-01 10:00:00"],
    }

    result = sort_candlesticks(data)

    assert result["close"] == [1.0, 2.0, 3.0]
    assert result["datetime"][0] == "2026-01-01 10:00:00"
    assert data["close"] == [3.0, 2.0, 1.0]  # input is not mutated


def test_sort_candlesticks_keeps_order_when_datetimes_are_unusable():
    data = {"close": [3.0, 2.0, 1.0], "datetime": ["2026-01-03", "not a date", "2026-01-01"]}

    assert sort_candlesticks(data)["close"] == [3.0, 2.0, 1.0]


def test_compute_indicators_uses_most_recent_bar_regardless_of_order():
    candles = _alpaca_style_candles()
    newest_first = {
        field: [c[field] for c in candles]
        for field in ("open", "high", "low", "close", "volume", "datetime")
    }
    oldest_first = {field: list(reversed(values)) for field, values in newest_first.items()}

    from_newest_first = compute_indicators({"candlesticks": newest_first})
    from_oldest_first = compute_indicators({"candlesticks": oldest_first})

    assert from_newest_first == pytest.approx(from_oldest_first)
    assert from_newest_first["rsi"] > 70  # uptrend, not the mirrored downtrend
    assert from_newest_first["sma_20"] > 190


def test_pipeline_analysis_is_based_on_most_recent_bar(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "ordering.db"))
    monkeypatch.setenv("DIVIDEND_STRATEGY_ENABLED", "false")
    monkeypatch.delenv("DESK_DEMO_MODE", raising=False)
    clear_portfolio_manager_cache()
    candles = _alpaca_style_candles()
    monkeypatch.setattr(
        "core.pipeline.get_candlestick_data",
        lambda symbol: {"alpaca": {"1min": [], "15min": [], "1h": candles}},
    )
    monkeypatch.setattr("core.pipeline.get_news_articles", lambda symbol: [])
    from core.pipeline import EnhancedTradingPipeline

    result = EnhancedTradingPipeline("AAPL").run_analysis({})
    clear_portfolio_manager_cache()

    newest_close = _uptrend_closes()[-1]
    technical = result["all_signals"]["technical"]
    assert technical["current_price"] == pytest.approx(newest_close)
    assert technical["indicators"]["rsi"] > 70

    risk = result["risk_parameters"]
    assert risk["stop_loss"] < newest_close < risk["take_profit"]
    # Stops are a few ATRs from the latest price, not anchored ~100 points away at the oldest bar.
    assert newest_close - risk["stop_loss"] < 5
    assert risk["take_profit"] - newest_close < 5
