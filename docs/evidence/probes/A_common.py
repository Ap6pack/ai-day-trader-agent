"""Shared fakes for claim-group A probes. No network: broker is a local fake."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from trader import alpaca

T0 = time.monotonic()
LOG = []
_loglock = threading.Lock()


def log(msg):
    with _loglock:
        LOG.append((time.monotonic() - T0, threading.current_thread().name, msg))


def dump(title):
    print(f"==== {title} ====")
    for t, thread, msg in LOG:
        print(f"t=+{t:6.3f}s [{thread:<22}] {msg}")


NOW = datetime(2026, 9, 28, 15, 0, tzinfo=timezone.utc)  # 11:00 ET, market open
OPEN = {"is_open": True, "next_close": "2026-09-28T16:00:00-04:00"}


def decision(symbol="NVDA", rec="BUY", qty=5, conf=0.7, price=100.0):
    return {
        "symbol": symbol, "recommendation": rec, "quantity": qty, "confidence": conf,
        "current_price": price, "primary_strategy": "technical",
        "reason": "Technical: +2 net indicators",
        "all_signals": {"technical": {"signal": rec, "strength": conf, "reason": "t"}},
        "risk_parameters": {"stop_loss": 96.0, "take_profit": 106.0},
    }


class FakeBroker:
    """Stands in for trader.alpaca. NOTHING leaves the process."""

    def __init__(self, mode="paper", clock=None, slow_flatten=0.0):
        self.mode = mode
        self.clock = clock or OPEN
        self.orders = []
        self.owned = {}
        self.flatten_calls = 0
        self.slow_flatten = slow_flatten

    def require_mode(self, expected, who="x"):
        if expected != self.mode:
            raise alpaca.AccountError(f"{who} wants {expected} but account is {self.mode}")

    def market_clock(self):
        return self.clock

    def account(self):
        return {"equity": "20000"}

    def positions(self):
        return []

    def owned_positions(self, prefix):
        return {s: {"qty": q, "open_order_ids": []} for s, q in self.owned.items()}

    def owned_qty(self, prefix, symbol):
        return self.owned.get(symbol, 0.0)

    def max_live_qty(self, price):
        return 1000

    bracket_order_payload = staticmethod(alpaca.bracket_order_payload)
    market_order_payload = staticmethod(alpaca.market_order_payload)

    def flatten_owned(self, prefix):
        self.flatten_calls += 1
        log(f"FLATTEN_OWNED entered (call {self.flatten_calls}), sleeping {self.slow_flatten}s")
        time.sleep(self.slow_flatten)
        out = []
        for s, q in list(self.owned.items()):
            self.orders.append({"symbol": s, "side": "sell", "qty": str(int(q)), "via": "flatten"})
            log(f"*** BROKER ORDER (flatten) SELL {int(q)} {s}")
            out.append({"symbol": s})
        return out

    def submit_order(self, payload):
        self.orders.append(payload)
        log(f"*** BROKER ORDER submit_order {payload['side'].upper()} {payload['qty']} "
            f"{payload['symbol']} client_order_id={payload.get('client_order_id','')[:18]}...")
        return {"id": f"o{len(self.orders)}", "status": "accepted"}
