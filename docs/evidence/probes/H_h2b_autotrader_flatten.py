"""H2b: does trader/autotrader.py maybe_flatten() have the same no-retry-on-partial-failure hole?"""
import os
from datetime import datetime, timedelta, timezone

SCRATCH = "docs/evidence/probes"
DB = os.path.join(SCRATCH, "H_h2b_journal.db")
if os.path.exists(DB):
    os.remove(DB)
os.environ["TRADER_JOURNAL_PATH"] = DB
os.environ["TRADER_MODE"] = "review"

from trader import autotrader
from trader.journal import Journal
from trader.portfolios import Portfolios

CLOSE = datetime(2026, 10, 13, 16, 0, tzinfo=timezone.utc)
NOW = {"t": CLOSE - timedelta(minutes=9)}
EVENTS = []

cfg = autotrader.RunConfig(symbols=["AAPL"], mode="paper").validate()


def make(broker):
    at = autotrader.AutoTrader(Journal(), Portfolios(), publish=lambda e: EVENTS.append(e),
                               broker=broker, now=lambda: NOW["t"])
    at.limits = autotrader.Limits()
    return at


class Base:
    def __init__(self):
        self.calls = 0

    def market_clock(self):
        return {"is_open": True, "next_close": CLOSE.isoformat()}


print("=== A: flatten_owned RAISES (total failure) ===")


class Raising(Base):
    def flatten_owned(self, prefix):
        self.calls += 1
        raise RuntimeError("502 Bad Gateway from Alpaca")


b = Raising()
at = make(b)
for i in range(3):
    print(f"  check {i+1}: maybe_flatten() -> {at.maybe_flatten(cfg)}  _flattened_for={at._flattened_for!r}")
    NOW["t"] += timedelta(seconds=30)
print("  flatten_owned attempts:", b.calls, "-> RETRIED (the exception keeps _flattened_for unset)")
print("  last event:", EVENTS[-1]["message"])

print("\n=== B: flatten_owned RETURNS, but the sell orders it submitted are later rejected ===")
NOW["t"] = CLOSE - timedelta(minutes=9)
EVENTS.clear()


class Rejected(Base):
    """flatten_owned returns the submitted orders; it never waits for a fill."""
    def flatten_owned(self, prefix):
        self.calls += 1
        return [{"id": "o1", "symbol": "AAPL", "side": "sell", "qty": "10", "status": "accepted"},
                {"id": "o2", "symbol": "TSLA", "side": "sell", "qty": "7", "status": "rejected",
                 "failed_at": "2026-10-13T15:51:02Z"}]


b2 = Rejected()
at2 = make(b2)
for i in range(3):
    print(f"  check {i+1}: maybe_flatten() -> {at2.maybe_flatten(cfg)}  _flattened_for={at2._flattened_for!r}")
    NOW["t"] += timedelta(seconds=30)
print("  flatten_owned attempts:", b2.calls)
print("  event:", EVENTS[0]["message"])

print("\n=== C: for comparison, newsbot's paper path from the same angle ===")
print("  newsbot paper  -> close_all_positions() returns a 207 array; failures are DATA, not exceptions")
print("  newsbot live / autotrader -> flatten_owned() raises on failure, so the retry guard still works")
