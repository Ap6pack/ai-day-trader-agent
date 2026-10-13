"""Regressions for data provenance (D), flatten accounting (H) and stored state.

Each test asserts the INTENDED SAFE BEHAVIOUR; unfixed findings carry
`xfail(strict=True)` naming the finding id.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from trader import analysis, market_feed, newsbot
from trader.autotrader import AutoTrader, Limits, RunConfig
from trader.portfolios import PortfolioError
from .conftest import NOW, FakeBroker, decision

LIMITS = Limits(min_confidence=0.5, max_trades_per_day=5, cooldown_minutes=0,
                min_price=5, entry_cutoff_minutes=15, flatten_minutes=10)


# --- D: simulated or stale prices must not reach a broker order ------------


@pytest.mark.xfail(
    strict=True,
    reason="FINDING-D1a: analyze() does `get_bars(...).get('bars')`, dropping the "
           "'source' key, so nothing downstream can tell an Alpaca bar from a "
           "DESK_DEMO_MODE random walk. Remove when the result carries provenance.",
)
def test_d1a_analysis_result_carries_its_market_data_provenance(monkeypatch):
    monkeypatch.setenv("DESK_DEMO_MODE", "true")
    monkeypatch.setattr(analysis.jev_news, "is_configured", lambda: False)
    monkeypatch.setattr(analysis.dividends, "enabled", lambda: False)

    result = analysis.analyze("NVDA", capital=10_000)

    provenance = result.get("price_source") or result.get("source")
    assert provenance == "demo", (
        "analyze() reports no market-data provenance, so a caller about to send a "
        f"real order cannot tell simulated data from live. Got: {provenance!r}"
    )


@pytest.mark.xfail(
    strict=True,
    reason="FINDING-D1b: with DESK_DEMO_MODE=true and a live-configured account, a "
           "fabricated random-walk price is used to build and send a real bracket "
           "order. Remove when broker execution modes refuse demo-sourced data.",
)
def test_d1b_broker_modes_refuse_demo_sourced_prices(journal, portfolios, monkeypatch):
    monkeypatch.setenv("DESK_DEMO_MODE", "true")
    broker = FakeBroker(mode="paper")

    def demo_analyze(symbol, capital, held, capital_source=None):
        bars = market_feed.get_bars(symbol, "15Min", 50)
        assert bars["source"] == "demo"
        return decision(symbol=symbol, price=bars["bars"][-1]["close"])

    pilot = AutoTrader(journal, portfolios, analyze=demo_analyze, broker=broker,
                       limits=LIMITS, now=lambda: NOW,
                       quotes=lambda syms: {}, universe=lambda: ["NVDA"])
    pilot.run_cycle(RunConfig(["NVDA"], 60, "paper", "default"))

    assert broker.orders == [], (
        "a broker order was built from simulated data: "
        f"{broker.orders[0] if broker.orders else None}"
    )


@pytest.mark.xfail(
    strict=True,
    reason="FINDING-D1c: DemoMarket stamps simulated bars with the current wall clock, "
           "so a bar-age freshness check passes on fabricated data. Any fix for D1 must "
           "key off provenance as well as timestamp. Remove when demo bars are "
           "distinguishable by something other than their timestamp.",
)
def test_d1c_demo_bars_are_not_disguised_as_fresh_real_data(monkeypatch):
    import time

    monkeypatch.setenv("DESK_DEMO_MODE", "true")
    bars = market_feed.get_bars("NVDA", "1Min", 20)
    newest_age = time.time() - bars["bars"][-1]["time"]

    assert newest_age > 120, (
        f"the newest simulated bar is {newest_age:.0f}s old, so it is indistinguishable "
        "from live data by timestamp alone"
    )


# --- H2: a failed flatten must not be recorded as a success ----------------


@pytest.mark.xfail(
    strict=True,
    reason="FINDING-H2: Alpaca answers DELETE /v2/positions with HTTP 207 and a "
           "per-symbol status. close_all_positions returns every element and "
           "maybe_flatten reports len(closed) as 'closed N positions', then sets "
           "_flattened_for, permanently suppressing retries for the session. A "
           "position stays open overnight with its day-order bracket legs cancelled. "
           "Remove when per-element status is inspected and failures are retried.",
)
def test_h2_a_partially_failed_flatten_is_not_counted_as_closed(journal):
    class PartialFailBroker(FakeBroker):
        def close_all_positions(self):
            self.flatten_calls += 1
            return [{"symbol": "AAPL", "status": 200},
                    {"symbol": "TSLA", "status": 500},
                    {"symbol": "NVDA", "status": 403}]

    from datetime import datetime, timezone

    # 19:52Z is 15:52 ET: eight minutes before the 16:00 close, so the flatten
    # window (NEWSBOT_FLATTEN_MINUTES, default 10) is open.
    pre_close = datetime(2026, 9, 28, 19, 52, tzinfo=timezone.utc)
    broker = PartialFailBroker(mode="paper")
    settings = newsbot.load_settings({"NEWSBOT_EXECUTION": "paper"})
    bot = newsbot.NewsBot(settings, journal=journal, broker=broker,
                          judge=lambda s, a: None, now=lambda: pre_close)

    first = bot.maybe_flatten()
    assert first is not None, "the flatten window never opened; the test proves nothing"
    assert first is not None and first["closed"] == 1, (
        f"reported {first['closed'] if first else None} positions closed; only AAPL "
        "actually closed (TSLA 500, NVDA 403)"
    )

    bot.maybe_flatten()
    assert broker.flatten_calls >= 2, "the failed symbols were never retried"


# --- portfolio ledger integrity --------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="FINDING-P1: Portfolios.record_fill SELECTs cash and holdings before "
           "SQLite's deferred transaction takes a write lock, so two concurrent fills "
           "read the same pre-state and the later UPDATE wins. Reproduced 25/25: two "
           "$1,000 buys both accepted against $1,000 of cash. Remove when the "
           "transaction opens with BEGIN IMMEDIATE.",
)
def test_p1_concurrent_fills_cannot_overspend_the_cash_balance(portfolios):
    portfolios.create("race", 1000.0)
    start = threading.Barrier(2)
    accepted, rejected = [], []

    def buy():
        start.wait(timeout=5)
        try:
            portfolios.record_fill("race", "AAPL", "buy", 10, 100.0)  # $1,000 each
            accepted.append(1)
        except PortfolioError:
            rejected.append(1)

    threads = [threading.Thread(target=buy) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    view = portfolios.get("race")
    held = view["holdings"][0]["qty"] if view["holdings"] else 0
    assert len(accepted) == 1 and len(rejected) == 1, (
        f"{len(accepted)} of 2 $1,000 buys were accepted against $1,000 of cash"
    )
    assert view["cash"] == 0.0 and held == 10, (
        f"ledger disagrees with itself: cash={view['cash']}, held={held}, "
        f"accepted={len(accepted)}"
    )


def test_portfolio_rejects_an_oversized_buy_when_uncontended(portfolios):
    """The sequential path is correct; pinned so a concurrency fix cannot
    regress it."""
    portfolios.create("solo", 1000.0)
    portfolios.record_fill("solo", "AAPL", "buy", 10, 100.0)
    with pytest.raises(PortfolioError, match="Not enough cash"):
        portfolios.record_fill("solo", "AAPL", "buy", 10, 100.0)
    assert portfolios.get("solo")["cash"] == 0.0


# --- H1: the scoring window ------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="FINDING-H1: `journal pending` filters ts < today-midnight-ET, but the "
           "cron runs `autopilot score` at 16:20 ET, twenty minutes AFTER the close. "
           "Every decision made during the session it was meant to score is excluded "
           "and waits a day. Low severity: nothing is lost, the query has no lower "
           "bound. Remove when pending covers decisions whose trading day has closed.",
)
def test_h1_same_day_decisions_are_scorable_after_the_close(journal):
    from datetime import datetime, timezone

    from trader.journal import MARKET_TZ

    ten_am_et = datetime(2026, 9, 28, 14, 0, tzinfo=timezone.utc)   # 10:00 ET
    journal.add_decision("NVDA", "buy", 100.0, thesis="intraday", now=ten_am_et)

    score_time = datetime(2026, 9, 28, 20, 20, tzinfo=timezone.utc)  # 16:20 ET
    midnight = score_time.astimezone(MARKET_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    pending = journal.unscored_decisions(older_than=midnight)

    assert [r["symbol"] for r in pending] == ["NVDA"], (
        "the decision made at 10:00 ET is not in `pending` at 16:20 ET, so the "
        "after-close scoring run skips the session it was meant to score"
    )
