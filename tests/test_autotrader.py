from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from trader import alpaca
from trader.autotrader import ORDER_EVENT, AutoTrader, Limits, RunConfig
from trader.journal import Journal
from trader.portfolios import Portfolios

NOW = datetime(2026, 9, 28, 15, 0, tzinfo=timezone.utc)  # 11:00 ET
OPEN = {"is_open": True, "next_close": "2026-09-28T16:00:00-04:00"}
LIMITS = Limits(min_confidence=0.5, max_trades_per_day=2, cooldown_minutes=30, min_price=5, entry_cutoff_minutes=15)


def decision(symbol="NVDA", rec="BUY", qty=5, conf=0.7, price=100.0):
    return {
        "symbol": symbol, "recommendation": rec, "quantity": qty, "confidence": conf, "current_price": price,
        "primary_strategy": "technical", "reason": "Technical: +2 net indicators",
        "all_signals": {"technical": {"signal": rec, "strength": conf, "reason": "t"},
                        "sentiment": {"signal": "HOLD", "strength": 0, "reason": "s"},
                        "dividend": {"signal": "HOLD", "strength": 0, "reason": "d"}},
        "risk_parameters": {"stop_loss": 96.0, "take_profit": 106.0},
    }


class Broker:
    def __init__(self, clock=OPEN):
        self.clock = clock
        self.orders = []
        self.paper = True

    def is_paper(self):
        return self.paper

    def market_clock(self):
        return self.clock

    def account(self):
        return {"equity": "20000"}

    def positions(self):
        return [{"symbol": "AMD", "qty": "3", "side": "long"}]

    bracket_order_payload = staticmethod(alpaca.bracket_order_payload)
    market_order_payload = staticmethod(alpaca.market_order_payload)

    def submit_order(self, payload):
        self.orders.append(payload)
        return {"id": f"o{len(self.orders)}", "status": "accepted"}


@pytest.fixture
def setup(tmp_path):
    journal = Journal(tmp_path / "j.db")
    portfolios = Portfolios(tmp_path / "j.db")
    portfolios.create("default", 10000)
    events = []
    broker = Broker()
    calls = []

    def analyze(symbol, capital, held, capital_source):
        calls.append((symbol, capital, held, capital_source))
        return results.get(symbol, decision(symbol))

    results = {}
    trader = AutoTrader(journal, portfolios, publish=events.append, analyze=analyze, broker=broker,
                        quotes=lambda syms: {}, limits=LIMITS, now=lambda: NOW)
    return trader, journal, portfolios, broker, events, results, calls


def test_config_validation():
    assert RunConfig(["nvda", " amd "], 300, "paper").validate().symbols == ["NVDA", "AMD"]
    for bad in (RunConfig([], 300), RunConfig(["NVDA"], 30), RunConfig(["NVDA"], 300, "live"),
                RunConfig(["NV DA1"], 300)):
        with pytest.raises(ValueError):
            bad.validate()


def test_signals_mode_analyzes_and_journals_but_never_trades(setup):
    trader, journal, portfolios, broker, events, results, calls = setup
    trader.run_cycle(RunConfig(["NVDA", "AMD"], 300, "signals"))
    assert broker.orders == [] and portfolios.get("default")["holdings"] == []
    assert [c[0] for c in calls] == ["NVDA", "AMD"] and calls[0][3] == "portfolio default"
    types = [e["type"] for e in events]
    assert types.count("analysis_started") == 2 and types.count("decision") == 2
    assert types.count("strategy_signal") == 6 and "order_skipped" not in types
    decisions = journal.decisions()
    assert {d["mode"] for d in decisions} == {"autopilot-signals"}
    assert all(d["thesis"].startswith("[autopilot]") for d in decisions)


def test_record_mode_trades_the_portfolio(setup):
    trader, journal, portfolios, broker, events, results, calls = setup
    trader.run_cycle(RunConfig(["NVDA"], 300, "record", "default"))
    p = portfolios.get("default")
    assert p["holdings"][0]["symbol"] == "NVDA" and p["holdings"][0]["qty"] == 5
    assert p["cash"] == 9500 and broker.orders == []
    [order] = [e for e in journal.events(10) if e["kind"] == ORDER_EVENT]
    assert order["quantity"] == 5 and "recorded BUY 5 NVDA" in order["detail"]
    results["NVDA"] = decision(rec="SELL", qty=5, price=110.0)
    trader.limits = Limits(cooldown_minutes=0, max_trades_per_day=5)
    trader.run_cycle(RunConfig(["NVDA"], 300, "record", "default"))
    assert portfolios.get("default")["holdings"] == [] and portfolios.get("default")["realized_pl"] == 50


def test_paper_mode_bracket_buy_and_market_sell(setup):
    trader, journal, portfolios, broker, events, results, calls = setup
    results["AMD"] = decision("AMD", rec="SELL", qty=3)
    trader.run_cycle(RunConfig(["NVDA", "AMD"], 300, "paper"))
    buy, sell = broker.orders
    assert buy["order_class"] == "bracket" and buy["qty"] == "5"
    assert buy["stop_loss"] == {"stop_price": "96.0"} and buy["take_profit"] == {"limit_price": "106.0"}
    assert sell["side"] == "sell" and sell["type"] == "market" and sell["qty"] == "3"
    assert all(o["client_order_id"].startswith("autopilot-") for o in broker.orders)
    assert calls[0][1] == 20000 and calls[1][2] == 3 and calls[1][3] == "alpaca paper equity"


@pytest.mark.parametrize("result, clock, reason", [
    (decision(conf=0.4), OPEN, "confidence"),
    (decision(price=3.0), OPEN, "price"),
    (decision(qty=0), OPEN, "quantity 0"),
    (decision(), {"is_open": False}, "market closed"),
    (decision(), {"is_open": True, "next_close": "2026-09-28T11:10:00-04:00"}, "of the close"),
])
def test_limits_block_with_a_reason(setup, result, clock, reason):
    trader, journal, portfolios, broker, events, results, calls = setup
    broker.clock = clock
    results["NVDA"] = result
    [outcome] = trader.run_cycle(RunConfig(["NVDA"], 300, "paper"))
    assert outcome["action"] == "none" and reason in outcome["reason"]
    assert broker.orders == []
    assert any(e["type"] == "order_skipped" and reason in e["message"] for e in events)


def test_daily_cap_and_cooldown(setup):
    trader, journal, portfolios, broker, events, results, calls = setup
    trader.run_cycle(RunConfig(["NVDA", "AMD", "TSLA"], 300, "paper"))
    assert len(broker.orders) == 2  # TSLA blocked by the cap of 2
    trader.limits = Limits(max_trades_per_day=10, cooldown_minutes=30)
    [outcome] = trader.run_cycle(RunConfig(["NVDA"], 300, "paper"))
    assert "cooldown" in outcome["reason"]


def test_failed_order_is_reported_not_raised(setup):
    trader, journal, portfolios, broker, events, results, calls = setup

    def reject(payload):
        raise RuntimeError("insufficient buying power")

    broker.submit_order = reject
    [outcome] = trader.run_cycle(RunConfig(["NVDA"], 300, "paper"))
    assert outcome["action"] == "failed"
    assert any(e["type"] == "order_failed" for e in events)


def test_analysis_errors_skip_the_symbol(setup):
    trader, journal, portfolios, broker, events, results, calls = setup

    def boom(*a, **k):
        raise RuntimeError("no bars")

    trader.analyze = boom
    assert trader.run_cycle(RunConfig(["NVDA"], 300, "paper")) == []
    assert any(e["type"] == "analysis_error" for e in events)


def test_start_stop_and_paper_guard(setup):
    trader, journal, portfolios, broker, events, results, calls = setup

    async def run():
        status = await trader.start(RunConfig(["NVDA"], 60, "signals"))
        assert status["running"] and status["config"]["symbols"] == ["NVDA"]
        await asyncio.sleep(0.05)
        status = await trader.stop()
        assert not status["running"] and status["cycles"] >= 1
        broker.paper = False
        with pytest.raises(ValueError, match="paper host"):
            await trader.start(RunConfig(["NVDA"], 60, "paper"))
        with pytest.raises(Exception):
            await trader.start(RunConfig(["NVDA"], 60, "record", "missing"))
        assert not trader.running

    asyncio.run(run())
    messages = [e["message"] for e in events if e["type"] == "autopilot"]
    assert messages[0].startswith("Autopilot ON") and "Autopilot OFF" in messages
