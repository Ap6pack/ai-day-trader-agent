from __future__ import annotations

import math
from datetime import date

import pytest

from trader import analysis, dividends, technicals


def bars_from(closes, spread=1.0):
    return [{"time": i, "open": c, "high": c + spread, "low": c - spread, "close": c, "volume": 1000}
            for i, c in enumerate(closes)]


# --- indicators ------------------------------------------------------------------


def test_sma_ema_basics():
    assert technicals.sma([1, 2, 3, 4], 2) == 3.5
    assert technicals.sma([1], 2) is None
    assert technicals.ema([5] * 30, 10) == pytest.approx(5)
    assert technicals.ema([1, 2], 5) is None


def test_rsi_extremes_and_midpoint():
    assert technicals.rsi(list(range(1, 40))) == 100.0
    assert technicals.rsi(list(range(40, 1, -1))) == pytest.approx(0.0)
    zigzag = [10 + (1 if i % 2 else -1) for i in range(60)]
    assert technicals.rsi(zigzag) == pytest.approx(50, abs=5)
    assert technicals.rsi([1, 2, 3]) is None


def test_macd_sign_follows_trend():
    up = technicals.macd([float(i) for i in range(100)])
    down = technicals.macd([float(100 - i) for i in range(100)])
    assert up["macd"] > 0 and down["macd"] < 0
    assert technicals.macd([1.0] * 10)["macd"] is None


def test_atr_constant_range():
    assert technicals.atr(bars_from([100.0] * 30, spread=1.0)) == pytest.approx(2.0)


# --- technical signal ---------------------------------------------------------------


def test_oversold_downtrend_bounce_is_buy():
    closes = [100 - i * 0.8 for i in range(60)] + [52.5, 53.5, 54.8]
    sig = technicals.technical_signal(bars_from(closes))
    assert sig["signal"] == "BUY", sig["reason"]


ROLLOVER = [50 + i * 0.5 for i in range(50)] + [75 - i * 0.6 for i in range(7)]


def test_rally_rolling_over_is_sell():
    sig = technicals.technical_signal(bars_from(ROLLOVER))
    assert sig["signal"] == "SELL", sig["reason"]
    assert "MACD below signal" in sig["reason"]


def test_overbought_but_trending_up_holds():
    # Overbought RSI (-2) is offset by MACD and price above the averages (+2): the original rules hold.
    closes = [50 + i * 0.1 for i in range(40)] + [55 + i * 1.5 for i in range(20)]
    sig = technicals.technical_signal(bars_from(closes))
    assert sig["indicators"]["rsi"] > 70 and sig["signal"] == "HOLD"


def test_not_enough_history_is_hold():
    assert technicals.technical_signal(bars_from([10, 11]))["signal"] == "HOLD"


# --- dividends ---------------------------------------------------------------------------

TODAY = date(2026, 9, 28)


def div(ex, rate, special=False):
    return {"ex_date": ex, "rate": rate, "special": special}


@pytest.fixture
def div_env(monkeypatch):
    monkeypatch.setenv("DIVIDEND_CAPTURE_WINDOW_DAYS", "7")
    monkeypatch.setenv("MIN_DIVIDEND_YIELD", "6")
    monkeypatch.delenv("DIVIDEND_STRATEGY_ENABLED", raising=False)


def test_upcoming_high_yield_ex_date_is_capture_buy(div_env):
    rows = [div("2025-12-15", 0.5), div("2026-03-15", 0.5), div("2026-06-15", 0.5), div("2026-09-01", 0.5),
            div("2026-10-01", 0.5)]
    sig = dividends.dividend_signal("KO", 30.0, today=TODAY, rows=rows)
    assert sig["signal"] == "BUY" and sig["days_to_ex_dividend"] == 3
    assert sig["yield_pct"] == pytest.approx(6.67, abs=0.01)
    assert 0.3 <= sig["strength"] <= 0.7


def test_low_yield_or_far_ex_date_is_hold(div_env):
    rows = [div("2026-09-01", 0.25), div("2026-10-01", 0.25)]
    assert dividends.dividend_signal("AAPL", 200.0, today=TODAY, rows=rows)["signal"] == "HOLD"
    far = [div("2026-06-01", 3.0), div("2026-11-20", 3.0)]
    sig = dividends.dividend_signal("X", 50.0, today=TODAY, rows=far)
    assert sig["signal"] == "HOLD" and "window" in sig["reason"]


def test_special_dividends_excluded_and_no_dividend(div_env):
    rows = [div("2026-06-01", 10.0, special=True), div("2026-10-01", 0.1)]
    assert dividends.dividend_signal("X", 10.0, today=TODAY, rows=rows)["yield_pct"] == 0
    assert "no cash dividend" in dividends.dividend_signal("NVDA", 100.0, today=TODAY, rows=[])["reason"]


def test_dividend_can_be_switched_off(div_env, monkeypatch):
    monkeypatch.setenv("DIVIDEND_STRATEGY_ENABLED", "false")
    assert "off" in dividends.dividend_signal("KO", 30.0, today=TODAY, rows=[div("2026-10-01", 5)])["reason"]


# --- fusion, sizing, risk ------------------------------------------------------------------


def s(signal, strength, reason="r"):
    return {"signal": signal, "strength": strength, "reason": reason}


def test_fusion_priorities_and_confidence():
    fused = analysis.fuse({"technical": s("BUY", 0.6), "sentiment": s("BUY", 0.4), "dividend": s("HOLD", 0)})
    assert fused["signal"] == "BUY" and fused["primary_strategy"] == "technical"
    assert fused["confirming"] == 1 and fused["confidence"] == pytest.approx(0.7)

    fused = analysis.fuse({"technical": s("BUY", 0.6), "sentiment": s("SELL", 0.8), "dividend": s("HOLD", 0)})
    assert fused["primary_strategy"] == "technical" and fused["conflicting"] == 1
    assert fused["confidence"] == pytest.approx(0.45)

    fused = analysis.fuse({"technical": s("SELL", 0.9), "sentiment": s("HOLD", 0), "dividend": s("BUY", 0.5)})
    assert fused["primary_strategy"] == "dividend" and fused["signal"] == "BUY"

    fused = analysis.fuse({"technical": s("HOLD", 0.1), "sentiment": s("HOLD", 0), "dividend": s("HOLD", 0)})
    assert fused["signal"] == "HOLD"


def test_position_size_scales_and_is_long_only(monkeypatch):
    monkeypatch.setenv("MIN_POSITION_PERCENTAGE", "0.02")
    monkeypatch.setenv("MAX_POSITION_PERCENTAGE", "0.10")
    assert analysis.position_size("BUY", 0.8, 100.0, 10000, None, 900) == 8     # 10% * 0.8
    assert analysis.position_size("BUY", 0.55, 100.0, 10000, None, 900) == 3    # 6% * 0.55
    assert analysis.position_size("BUY", 0.3, 100.0, 10000, None, 900) == 1     # $60 >= half a share
    assert analysis.position_size("SELL", 0.9, 100.0, 10000, None, 900, held=0) == 0
    assert analysis.position_size("SELL", 0.9, 100.0, 10000, None, 900, held=7) == 7
    assert analysis.position_size("HOLD", 0.9, 100.0, 10000, None, 900) == 0
    calm = analysis.position_size("BUY", 0.8, 100.0, 10000, 0.0005, 900)
    wild = analysis.position_size("BUY", 0.8, 100.0, 10000, 0.02, 900)
    assert calm > wild >= 4


def test_risk_parameters_atr_and_fallback(monkeypatch):
    r = analysis.risk_parameters("BUY", 10, 100.0, 2.0, 10000)
    assert (r["stop_loss"], r["take_profit"], r["risk_reward_ratio"]) == (96.0, 106.0, 1.5)
    assert r["position_value"] == 1000.0 and r["total_risk"] == 40.0 and r["risk_percentage"] == 0.4
    monkeypatch.setenv("STOP_LOSS_PCT", "3")
    r = analysis.risk_parameters("SELL", 5, 100.0, None, 10000)
    assert r["stop_loss"] == 103.0 and r["take_profit"] == 95.5 and r["method"] == "STOP_LOSS_PCT"


# --- end to end ---------------------------------------------------------------------------------


def test_analyze_combines_all_three(div_env, monkeypatch):
    monkeypatch.setenv("SENTIMENT_THRESHOLD", "0.3")
    closes = [100 - i * 0.8 for i in range(60)] + [52.5, 53.5, 54.8]

    def judge(symbol, articles, max_articles):
        return {"score": 0.5, "evidence_weight": 0.4, "rationale": "product: big contract", "articles": [{}, {}]}

    result = analysis.analyze("acme", capital=10000, bars=bars_from(closes),
                              articles=[{"title": "x"}], dividend_rows=[], judge=judge)
    assert result["symbol"] == "ACME" and result["recommendation"] == "BUY"
    assert result["all_signals"]["sentiment"]["signal"] == "BUY"
    assert result["confirming_strategies"] == 1
    assert result["quantity"] > 0 and result["risk_parameters"]["stop_loss"] < result["current_price"]
    assert result["technical_indicators"]["rsi"] is not None
    assert math.isclose(result["risk_parameters"]["position_value"], result["quantity"] * result["current_price"],
                        rel_tol=1e-6)


def test_analyze_sell_without_holding_is_zero_quantity(div_env):
    def judge(symbol, articles, max_articles):
        return {"score": 0.0, "articles": []}

    result = analysis.analyze("ACME", capital=10000, bars=bars_from(ROLLOVER), articles=[{"title": "x"}],
                              dividend_rows=[], judge=judge)
    assert result["recommendation"] == "SELL" and result["quantity"] == 0
    assert "nothing held" in result["reason"]
    held = analysis.analyze("ACME", capital=10000, held=6, bars=bars_from(ROLLOVER), articles=[{"title": "x"}],
                            dividend_rows=[], judge=judge)
    assert held["quantity"] == 6 and held["risk_parameters"]["stop_loss"] > held["current_price"]


def test_sentiment_without_jev_or_news(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert "not configured" in analysis.sentiment_signal("X")["reason"]
    assert analysis.sentiment_signal("X", articles=[], judge=lambda *a, **k: {})["reason"].endswith("no recent headlines")
