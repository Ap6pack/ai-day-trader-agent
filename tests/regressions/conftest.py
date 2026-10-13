"""Shared harness for the finding regression tests.

Every test in this package is deterministic and offline. Two things enforce
that rather than merely asserting it:

* `no_network` is autouse and replaces every requests entry point with a raiser,
  so a test that accidentally reaches a real host fails loudly instead of
  sending a request. Several tests exercise LIVE-mode code paths on purpose
  (that is where the money limits live); the kill switch is what makes doing so
  safe, not the fact that the broker happens to be a fake.
* No test reads a credential, a real account, or the operator's .env. The
  `clean_env` fixture strips every trader variable so a developer's shell
  cannot change an outcome.

See FINDINGS.md for the finding-id -> test map.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest
import requests

from trader import alpaca
from trader.journal import Journal
from trader.portfolios import Portfolios

# 2026-09-28 15:00Z is 11:00 ET: mid-session on a Monday, well clear of both the
# open and the entry cutoff, so no test depends on when it is run.
NOW = datetime(2026, 9, 28, 15, 0, tzinfo=timezone.utc)
MARKET_OPEN = {"is_open": True, "next_close": "2026-09-28T16:00:00-04:00"}

TRADER_ENV_PREFIXES = ("ALPACA_", "NEWSBOT_", "TRADER_", "DESK_", "AUTOPILOT_",
                       "DIVIDEND_", "TYPESAFE_", "NEWS_API", "ANALYSIS_",
                       "MIN_", "MAX_", "STOP_LOSS_PCT", "TRADING_CAPITAL",
                       "SENTIMENT_")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Make any real HTTP call a hard failure, in every test in this package."""

    def forbidden(*args, **kwargs):
        raise AssertionError(
            f"regression tests are offline: blocked {args[:2] or kwargs}. "
            "Use the fake broker, or stub alpaca._request."
        )

    for name in ("request", "get", "post", "put", "patch", "delete", "head"):
        monkeypatch.setattr(requests, name, forbidden)
    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(alpaca, "_request", forbidden)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No credential, no operator setting, no .env reaches a regression test."""
    import os

    for key in list(os.environ):
        if key.startswith(TRADER_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    # config.load_env() would re-read a real .env from the project root.
    monkeypatch.setattr("trader.config.load_env", lambda path=None: None)


@pytest.fixture
def journal(tmp_path):
    return Journal(tmp_path / "journal.db")


@pytest.fixture
def portfolios(tmp_path):
    store = Portfolios(tmp_path / "journal.db")
    store.ensure_default()
    return store


def decision(symbol="NVDA", rec="BUY", qty=5, conf=0.7, price=100.0):
    """An analysis result in the shape trader.analysis.analyze returns."""
    return {
        "symbol": symbol, "recommendation": rec, "signal": rec, "quantity": qty,
        "confidence": conf, "current_price": price, "primary_strategy": "technical",
        "reason": "Technical: +2 net indicators", "timeframe": "15Min",
        "all_signals": {"technical": {"signal": rec, "strength": conf, "reason": "t"},
                        "sentiment": {"signal": "HOLD", "strength": 0, "reason": "s"},
                        "dividend": {"signal": "HOLD", "strength": 0, "reason": "d"}},
        "risk_parameters": {"stop_loss": 96.0, "take_profit": 106.0},
    }


class FakeBroker:
    """An in-process stand-in for trader.alpaca that records what was sent.

    Deliberately permissive: it accepts whatever the agent asks for and records
    it, so a test can assert on what the agent *tried* to do. Rules that the
    real Alpaca would enforce (held_for_orders, the shorting gate) are opt-in
    via `strict`, because some findings only show up once those apply.
    """

    def __init__(self, clock=MARKET_OPEN, mode="paper", price=100.0, strict=False):
        self.clock = dict(clock)
        self.mode = mode
        self.price = price
        self.strict = strict
        self.orders: list[dict] = []
        self.cancelled: list[str] = []
        self.owned: dict[str, dict] = {}
        self.positions_held: dict[str, float] = {}
        self.max_live = 10_000
        self.flatten_calls = 0
        self.submitted = threading.Event()
        self.equity = 100_000.0

    # --- account / mode ----------------------------------------------------
    def require_mode(self, expected, who="agent"):
        if self.mode != expected:
            raise alpaca.AccountError(
                f"{who} is set to trade {expected.upper()} but the account is {self.mode.upper()}")

    def account(self):
        return {"equity": str(self.equity), "last_equity": str(self.equity)}

    def positions(self):
        return [{"symbol": s, "qty": str(q), "side": "long"}
                for s, q in self.positions_held.items()]

    def market_clock(self):
        return self.clock

    def latest_price(self, symbol):
        return self.price

    def max_live_qty(self, price):
        return self.max_live

    # --- orders ------------------------------------------------------------
    def bracket_order_payload(self, symbol, qty, entry, tp_pct, sl_pct, key=None):
        return alpaca.bracket_order_payload(symbol, qty, entry, tp_pct, sl_pct, key=key)

    def market_order_payload(self, symbol, side, qty):
        return alpaca.market_order_payload(symbol, side, qty)

    def submit_order(self, payload):
        if self.strict and payload.get("side") == "sell":
            held = self.positions_held.get(payload["symbol"], 0.0)
            if float(payload["qty"]) > held:
                raise RuntimeError(
                    f"insufficient qty available for order (held for orders): "
                    f"asked {payload['qty']}, held {held}")
        self.orders.append(dict(payload))
        self.submitted.set()
        return {"id": f"ord-{len(self.orders)}", "status": "accepted", **payload}

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)

    # --- agent-owned positions ---------------------------------------------
    def owned_positions(self, prefix, now=None):
        return {s: dict(v) for s, v in self.owned.items()}

    def owned_qty(self, prefix, symbol):
        return self.owned.get(symbol, {}).get("qty", 0.0)

    def flatten_owned(self, prefix):
        self.flatten_calls += 1
        return [{"symbol": s} for s in self.owned]

    def close_all_positions(self):
        self.flatten_calls += 1
        return [{"symbol": s, "status": 200} for s in self.positions_held]

    # --- helpers for assertions --------------------------------------------
    @property
    def buys(self):
        return [o for o in self.orders if o.get("side") == "buy"]

    @property
    def sells(self):
        return [o for o in self.orders if o.get("side") == "sell"]


@pytest.fixture
def broker():
    return FakeBroker()


def wait_for(predicate, timeout=1.0, interval=0.01):
    """Poll until `predicate` is true. Returns whether it became true.

    Used where the defect is "something happens that should not": the test
    asserts the wait TIMED OUT. Kept short so a fixed build is not slow.
    """
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
