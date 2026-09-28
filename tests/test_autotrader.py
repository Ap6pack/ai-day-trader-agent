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
    def __init__(self, clock=OPEN, mode="paper"):
        self.clock = clock
        self.orders = []
        self.mode = mode
        self.owned = {}
        self.max_live = 1000
        self.flattened = 0

    def require_mode(self, expected, who="x"):
        if self.mode != expected:
            raise alpaca.AccountError(f"{who} is set to trade {expected.upper()} but the account is {self.mode.upper()}")

    def owned_positions(self, prefix):
        return {s: {"qty": q, "open_order_ids": []} for s, q in self.owned.items()}

    def owned_qty(self, prefix, symbol):
        assert prefix == "autopilot-"
        return self.owned.get(symbol, 0.0)

    def max_live_qty(self, price):
        return self.max_live

    def flatten_owned(self, prefix):
        assert prefix == "autopilot-"
        self.flattened += 1
        return [{"symbol": s} for s in self.owned]

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
    assert RunConfig(["NVDA"], 300, "live").validate().mode == "live"
    for bad in (RunConfig([], 300), RunConfig(["NVDA"], 30), RunConfig(["NVDA"], 300, "real"),
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
        broker.mode = "live"
        with pytest.raises(ValueError, match="set to trade PAPER"):
            await trader.start(RunConfig(["NVDA"], 60, "paper"))
        broker.mode = "paper"
        with pytest.raises(ValueError, match="set to trade LIVE"):
            await trader.start(RunConfig(["NVDA"], 60, "live"))
        with pytest.raises(Exception):
            await trader.start(RunConfig(["NVDA"], 60, "record", "missing"))
        assert not trader.running

    asyncio.run(run())
    messages = [e["message"] for e in events if e["type"] == "autopilot"]
    assert messages[0].startswith("Autopilot ON") and "Autopilot OFF" in messages


# --- live (real money) ------------------------------------------------------------------


def test_live_mode_sizes_on_live_equity_and_only_its_own_shares(setup):
    trader, journal, portfolios, broker, events, results, calls = setup
    broker.mode = "live"
    broker.owned = {"AMD": 2.0}  # the account also holds 3 AMD of your own (Broker.positions)
    results["AMD"] = decision("AMD", rec="SELL", qty=2)
    trader.run_cycle(RunConfig(["NVDA", "AMD"], 300, "live"))
    assert calls[0][1] == 20000 and calls[0][3] == "alpaca LIVE equity"
    assert calls[1][2] == 2.0  # held = autopilot-owned shares, not the account's 3
    buy, sell = broker.orders
    assert buy["order_class"] == "bracket" and sell["side"] == "sell" and sell["qty"] == "2"
    assert all(o["client_order_id"].startswith("autopilot-") for o in broker.orders)
    details = [e["detail"] for e in journal.events(10) if e["kind"] == ORDER_EVENT]
    assert all(d.startswith("LIVE ") for d in details)


def test_live_buy_is_capped_by_the_live_order_limit(setup):
    trader, journal, portfolios, broker, events, results, calls = setup
    broker.mode, broker.max_live = "live", 2
    trader.run_cycle(RunConfig(["NVDA"], 300, "live"))
    assert broker.orders[0]["qty"] == "2"  # analysis said 5
    broker.max_live = 0
    trader.limits = Limits(cooldown_minutes=0)
    [outcome] = trader.run_cycle(RunConfig(["NVDA"], 300, "live"))
    assert outcome["action"] == "none" and "ALPACA_LIVE_MAX_ORDER_USD" in outcome["reason"]
    assert len(broker.orders) == 1


def test_pre_close_flatten_runs_once_for_broker_modes_only(setup):
    trader, journal, portfolios, broker, events, results, calls = setup
    broker.clock = {"is_open": True, "next_close": "2026-09-28T11:08:00-04:00"}  # 8 min away
    assert trader.maybe_flatten(RunConfig(["NVDA"], 300, "record")) is None
    broker.owned = {"NVDA": 5.0}
    assert trader.maybe_flatten(RunConfig(["NVDA"], 300, "live")) == 1
    assert trader.maybe_flatten(RunConfig(["NVDA"], 300, "live")) is None  # once per session
    assert broker.flattened == 1
    assert any(e["type"] == "flatten" for e in events)
    early = Broker(clock={"is_open": True, "next_close": "2026-09-28T16:00:00-04:00"})
    trader.broker = early
    assert trader.maybe_flatten(RunConfig(["NVDA"], 300, "paper")) is None and early.flattened == 0


# --- AUTO: today's in-play stocks -------------------------------------------------------


def test_auto_config_validation():
    assert RunConfig(["auto"], 300).validate().auto
    assert not RunConfig(["NVDA"], 300).validate().auto
    with pytest.raises(ValueError, match="AUTO on its own"):
        RunConfig(["AUTO", "NVDA"], 300).validate()


def test_auto_analyzes_the_in_play_list_capped_plus_what_it_holds(setup):
    trader, journal, portfolios, broker, events, results, calls = setup
    lists = [["nvda", "tsla", "amd", "pltr"]]
    trader.universe = lambda: lists[-1]
    trader.limits = Limits(max_symbols=3)
    for sym in ("NVDA", "TSLA", "AMD", "PLTR", "KO"):
        results[sym] = decision(sym, rec="HOLD", qty=0)
    trader.run_cycle(RunConfig(["AUTO"], 300, "signals"))
    assert [c[0] for c in calls] == ["NVDA", "TSLA", "AMD"]
    assert any("universe: 3 in-play" in e["message"] for e in events)
    assert trader.status()["universe"] == ["NVDA", "TSLA", "AMD"]

    # A failed refresh keeps the last list; paper mode also re-checks what the autopilot holds.
    calls.clear()

    def down():
        raise RuntimeError("screener down")

    trader.universe = down
    broker.owned = {"KO": 4.0}
    trader.run_cycle(RunConfig(["AUTO"], 300, "paper"))
    assert [c[0] for c in calls] == ["KO", "NVDA", "TSLA", "AMD"]
    assert any("universe refresh failed" in e["message"] for e in events)
