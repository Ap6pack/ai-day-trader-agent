"""
Adversarial re-test of CLAIM B1 with a broker model that enforces the two Alpaca
rules the peer's ledger did NOT model:
  (1) held_for_orders: shares backing an OPEN sell order cannot be sold again
      -> a second sell for the same shares is rejected 403 "insufficient qty".
  (2) shorting gate: a sell that would take the account position negative is
      only accepted when the account is margin + shortable; otherwise 403.
No network: trader.alpaca._request is replaced wholesale.
"""
import os, sys, time, json, uuid

os.environ["ALPACA_TRADING_BASE_URL"] = "https://api.alpaca.markets"
os.environ["ALPACA_LIVE_TRADING"] = "true"
os.environ["ALPACA_API_KEY"] = "k"
os.environ["ALPACA_SECRET_KEY"] = "s"

import requests
from trader import alpaca


class Broker:
    """Minimal Alpaca model. positions[sym] = net account shares (agent + personal)."""

    def __init__(self, shorting=True):
        self.shorting = shorting
        self.positions = {}        # symbol -> net account qty
        self.orders = {}           # id -> order dict (parents, nested legs inline)
        self.log = []
        self.on_get = None         # hook fired right AFTER a GET /v2/orders returns
        self.cancel_fails_transport = set()

    # --- helpers ----------------------------------------------------------
    def open_sell_hold(self, symbol):
        """Shares held by the broker for still-open sell orders (held_for_orders)."""
        held = 0.0
        for o in self.orders.values():
            if o["symbol"] != symbol:
                continue
            for x in [o] + list(o.get("legs") or []):
                if x.get("side") == "sell" and x.get("status") in alpaca.OPEN_ORDER_STATUSES:
                    held = max(held, float(x["qty"]))   # OCO siblings hold the qty once
        return held

    def fill_leg(self, parent_id, leg_id, price):
        p = self.orders[parent_id]
        for leg in p["legs"]:
            if leg["id"] == leg_id:
                leg["status"] = "filled"; leg["filled_qty"] = leg["qty"]
                self.positions[p["symbol"]] = self.positions.get(p["symbol"], 0) - float(leg["qty"])
            elif leg["status"] in alpaca.OPEN_ORDER_STATUSES:
                leg["status"] = "canceled"          # OCO sibling
        self.log.append(f"   [market] leg {leg_id} FILLED @ {price}; OCO sibling canceled; "
                        f"{p['symbol']} account position now {self.positions[p['symbol']]:+g}")

    # --- the single entry point trader.alpaca uses ------------------------
    def request(self, method, url, **kw):
        path = url.split("alpaca.markets", 1)[-1]
        if method == "GET" and path.startswith("/v2/orders"):
            self.log.append(f"   GET  {path.split('?')[0]}  (snapshot)")
            snap = json.loads(json.dumps(list(self.orders.values())))
            if self.on_get:
                h, self.on_get = self.on_get, None
                h()
            return snap
        if method == "DELETE" and "/v2/orders/" in path:
            oid = path.rsplit("/", 1)[-1]
            if oid in self.cancel_fails_transport:
                self.log.append(f"   DEL  {oid} -> ConnectionError (leg stays LIVE at broker)")
                raise requests.exceptions.ConnectionError("connection reset by peer")
            for p in self.orders.values():
                for x in [p] + list(p.get("legs") or []):
                    if x["id"] == oid:
                        if x["status"] in alpaca.OPEN_ORDER_STATUSES:
                            x["status"] = "canceled"
                            self.log.append(f"   DEL  {oid} -> 204 canceled")
                        else:
                            self.log.append(f"   DEL  {oid} -> 422 (already {x['status']})")
                            r = requests.Response(); r.status_code = 422; r.url = url
                            raise requests.HTTPError("422 order is not cancelable", response=r)
                        return None
            raise requests.HTTPError("404")
        if method == "POST" and path.startswith("/v2/orders"):
            j = kw["json"]; sym = j["symbol"]; qty = float(j["qty"])
            if j["side"] == "sell":
                avail = self.positions.get(sym, 0) - self.open_sell_hold(sym)
                if qty > avail and not self.shorting:
                    self.log.append(f"   POST sell {qty:g} {sym} -> 403 insufficient qty "
                                    f"(position {self.positions.get(sym,0):g}, held_for_orders "
                                    f"{self.open_sell_hold(sym):g}, available {avail:g})")
                    r = requests.Response(); r.status_code = 403; r.url = url
                    raise requests.HTTPError("403 insufficient qty available for order", response=r)
                if qty > avail and self.open_sell_hold(sym) > 0:
                    # shares are locked by a live sell order: Alpaca rejects even on margin
                    self.log.append(f"   POST sell {qty:g} {sym} -> 403 insufficient qty "
                                    f"(held_for_orders {self.open_sell_hold(sym):g})")
                    r = requests.Response(); r.status_code = 403; r.url = url
                    raise requests.HTTPError("403 insufficient qty available for order", response=r)
                self.positions[sym] = self.positions.get(sym, 0) - qty
                self.log.append(f"   POST sell {qty:g} {sym} -> ACCEPTED/FILLED; "
                                f"{sym} account position now {self.positions[sym]:+g}")
            oid = str(uuid.uuid4())
            o = dict(j, id=oid, status="filled", filled_qty=j["qty"])
            return o
        raise AssertionError(f"unexpected {method} {path}")


def bracket(broker, sym, qty, personal=0, agent_filled=True):
    pid = f"p-{sym}"
    broker.orders[pid] = {
        "id": pid, "symbol": sym, "side": "buy", "qty": str(qty),
        "filled_qty": str(qty) if agent_filled else "0",
        "status": "filled", "client_order_id": f"newsbot-{sym}",
        "legs": [
            {"id": f"{sym}-tp", "symbol": sym, "side": "sell", "qty": str(qty),
             "filled_qty": "0", "status": "new"},
            {"id": f"{sym}-sl", "symbol": sym, "side": "sell", "qty": str(qty),
             "filled_qty": "0", "status": "held"},
        ],
    }
    broker.positions[sym] = broker.positions.get(sym, 0) + qty + personal
    return pid


def run(title, setup, shorting=True, wait=5.0):
    print("\n" + "=" * 78); print(title); print("=" * 78)
    b = Broker(shorting=shorting)
    setup(b)
    alpaca._request = b.request
    t0 = time.monotonic()
    try:
        res = alpaca.flatten_owned("newsbot-", wait_seconds=wait)
        err = None
    except Exception as exc:
        res = None; err = f"{type(exc).__name__}: {exc}"
    dt = time.monotonic() - t0
    print("\n".join(b.log))
    print(f"   -> returned {len(res) if res is not None else 'RAISED'}"
          + (f"  ({err})" if err else "") + f"   [{dt:.2f}s]")
    print(f"   FINAL account positions: {dict(b.positions)}")
    return b


# S1: the race, margin + shorting enabled, no personal shares
def s1(b):
    pid = bracket(b, "NVDA", 5)
    b.on_get = lambda: b.fill_leg(pid, "NVDA-sl", 180.55)
run("S1  RACE: stop leg fills right after the snapshot; margin acct, shorting ON", s1, shorting=True)

# S2: the race, but the user ALSO holds 100 NVDA personally
def s2(b):
    pid = bracket(b, "NVDA", 5, personal=100)
    b.on_get = lambda: b.fill_leg(pid, "NVDA-sl", 180.55)
run("S2  RACE + user personally holds 100 NVDA (docstring promise under test)", s2, shorting=True)

# S3: NO race. Cancel fails for a transport reason, legs stay LIVE and hold the shares.
def s3(b):
    bracket(b, "NVDA", 5)
    b.cancel_fails_transport = {"NVDA-tp", "NVDA-sl"}
run("S3  NO RACE, cancel dies on the wire: legs still LIVE -> held_for_orders", s3,
    shorting=True, wait=1.5)

# S4: the race on a cash / non-shortable account -> 403, retry, then raise
def s4(b):
    pid = bracket(b, "AAA", 5)
    bracket(b, "BBB", 5)
    bracket(b, "CCC", 5)
    b.on_get = lambda: b.fill_leg(pid, "AAA-sl", 10.0)
run("S4  RACE on a cash / non-shortable account: 403 -> 5s retry -> raise", s4,
    shorting=False, wait=1.5)

# S5: baseline, no race at all -> correct behaviour
def s5(b):
    bracket(b, "NVDA", 5)
run("S5  BASELINE no race: legs cancel cleanly, sell 5 -> flat", s5, shorting=True)
