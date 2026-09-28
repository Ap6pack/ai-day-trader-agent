from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from trader import journal as journal_module
from trader.journal import Journal

# 2026-09-28 is a Monday; 03:30 UTC is still Sunday 23:30 in New York.
LATE_SUNDAY_ET = datetime(2026, 9, 28, 3, 30, tzinfo=timezone.utc)
MONDAY_OPEN_ET = datetime(2026, 9, 28, 13, 30, tzinfo=timezone.utc)


@pytest.fixture
def journal(tmp_path):
    return Journal(tmp_path / "journal.db")


def test_orders_counted_per_eastern_trading_day(journal):
    journal.log_event("order_allowed", symbol="AAPL", now=LATE_SUNDAY_ET)
    journal.log_event("order_blocked", symbol="AAPL", now=MONDAY_OPEN_ET)
    journal.log_event("order_allowed", symbol="AAPL", now=MONDAY_OPEN_ET)
    assert journal.orders_allowed_today(now=MONDAY_OPEN_ET + timedelta(hours=1)) == 1


def test_latest_review_matches_symbol_side_and_window(journal):
    journal.log_event("review", symbol="aapl", side="BUY", now=MONDAY_OPEN_ET)
    later = MONDAY_OPEN_ET + timedelta(minutes=10)
    assert journal.latest_review("AAPL", "buy", 15, now=later) is not None
    assert journal.latest_review("AAPL", "sell", 15, now=later) is None
    assert journal.latest_review("AAPL", "buy", 5, now=later) is None


def test_decisions_outcomes_and_summary(journal):
    buy = journal.add_decision("AAPL", "buy", 200.0, confidence=0.6, thesis="t", mode="review",
                               now=LATE_SUNDAY_ET)
    sell = journal.add_decision("MSFT", "sell", 400.0, mode="review", now=LATE_SUNDAY_ET)
    journal.add_decision("NVDA", "pass", 100.0, mode="review", now=LATE_SUNDAY_ET)

    pending = journal.unscored_decisions(older_than=MONDAY_OPEN_ET)
    assert [r["symbol"] for r in pending] == ["AAPL", "MSFT", "NVDA"]

    journal.set_outcome(buy, 210.0, 200.0)   # buy, price up 5%: hit
    journal.set_outcome(sell, 420.0, 400.0)  # sell, price up 5%: miss
    summary = journal.summary()
    assert summary["decisions"] == {"buy": 1, "sell": 1, "pass": 1}
    assert summary["scored"] == 2
    assert summary["hit_rate"] == 0.5
    assert summary["avg_signed_return"] == 0.0


@pytest.mark.parametrize("kwargs", [
    {"action": "hold"}, {"price": 0}, {"confidence": 1.5},
])
def test_invalid_decisions_rejected(journal, kwargs):
    args = {"symbol": "AAPL", "action": "buy", "price": 10.0, "mode": "review", **kwargs}
    with pytest.raises(ValueError):
        journal.add_decision(**args)


def test_cli_decide_and_outcome(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(journal_module.config, "load_env", lambda *a, **k: None)
    monkeypatch.setenv("TRADER_JOURNAL_PATH", str(tmp_path / "cli.db"))
    monkeypatch.setenv("TRADER_MODE", "review")

    assert journal_module.main(["decide", "--symbol", "aapl", "--action", "buy", "--price", "200",
                                "--thesis", "breakout"]) == 0
    decision_id = json.loads(capsys.readouterr().out)["recorded_decision_id"]

    assert journal_module.main(["outcome", "--id", str(decision_id), "--price", "204"]) == 0
    row = json.loads(capsys.readouterr().out)
    assert row["symbol"] == "AAPL" and row["mode"] == "review"
    assert row["outcome_return"] == pytest.approx(0.02)
