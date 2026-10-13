"""Shared in-process fake Alpaca transport for the C_* probes.

Replaces trader.alpaca._request. No socket is ever opened to Alpaca.
Simulates realistic round-trip latency so the TOCTOU window is visible.
"""
import threading
import time
from typing import Any, Dict, List

GET_LATENCY = 0.06   # GET /v2/orders round trip
POST_LATENCY = 0.04  # POST /v2/orders round trip


class FakeAlpaca:
    def __init__(self, price: float = 10.0, equity: float = 100_000.0,
                 last_equity: float = 100_000.0, seed_orders: List[Dict[str, Any]] | None = None):
        self.price = price
        self.equity = equity
        self.last_equity = last_equity
        self.orders: List[Dict[str, Any]] = list(seed_orders or [])
        self.log: List[str] = []
        self._lock = threading.Lock()  # protects self.orders / self.log only (it is the *broker*)
        self._n = 0

    def _note(self, msg: str) -> None:
        with self._lock:
            self.log.append(f"[{time.monotonic() - T0:7.3f}s] {msg}")

    def request(self, method: str, url: str, **kw: Any) -> Any:
        if "/trades/latest" in url:
            return {"trade": {"p": self.price}}
        if url.endswith("/v2/account"):
            return {"equity": str(self.equity), "last_equity": str(self.last_equity),
                    "buying_power": "100000", "status": "ACTIVE"}
        if url.endswith("/v2/clock"):
            return {"is_open": True, "next_close": "2026-10-13T16:00:00-04:00"}
        if method == "GET" and url.endswith("/v2/orders"):
            who = threading.current_thread().name
            self._note(f"{who}: GET /v2/orders ->")
            time.sleep(GET_LATENCY)
            with self._lock:
                snap = [dict(o) for o in self.orders]
            self._note(f"{who}: GET /v2/orders <- {len(snap)} orders")
            return snap
        if method == "POST" and url.endswith("/v2/orders"):
            who = threading.current_thread().name
            body = kw.get("json") or {}
            self._note(f"{who}: POST /v2/orders {body.get('side')} {body.get('qty')} {body.get('symbol')} ->")
            time.sleep(POST_LATENCY)
            with self._lock:
                self._n += 1
                order = {"id": f"ord-{self._n}", "symbol": body.get("symbol"), "qty": body.get("qty"),
                         "side": body.get("side"), "status": "accepted", "filled_qty": "0",
                         "client_order_id": body.get("client_order_id")}
                self.orders.append(order)
            self._note(f"{who}: POST /v2/orders <- FILLED REQUEST id={order['id']}  "
                       f"(broker now holds {len(self.orders)} orders)")
            return order
        raise AssertionError(f"unexpected call {method} {url}")


T0 = time.monotonic()
