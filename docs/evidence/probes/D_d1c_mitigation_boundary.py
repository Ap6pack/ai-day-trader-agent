"""D1c: exactly what alpaca.check_live_buy's NOTIONAL re-pricing does and does not catch
when the price that built the order is demo/stale. HTTP intercepted."""
import json, os, pathlib
from datetime import datetime, timedelta, timezone
SCRATCH = pathlib.Path("docs/evidence/probes")
os.environ.update({
    "ALPACA_TRADING_BASE_URL": "https://api.alpaca.markets", "ALPACA_LIVE_TRADING": "true",
    "ALPACA_API_KEY": "K", "ALPACA_SECRET_KEY": "S", "ALPACA_LIVE_MAX_ORDER_USD": "500",
})
import requests
REAL = {"p": 0.0}; CAPTURED = []
class Resp:
    def __init__(self, p): self._p=p; self.content=b"x"
    def raise_for_status(self): pass
    def json(self): return self._p
def _route(m,u,**k):
    now = datetime.now(timezone.utc)
    if u.endswith("/trades/latest"): return Resp({"trade":{"p":REAL["p"]}})
    if u.endswith("/v2/account"): return Resp({"equity":"100000","last_equity":"100000"})
    if u.endswith("/v2/orders") and m=="GET": return Resp([])
    if u.endswith("/v2/orders") and m=="POST": CAPTURED.append(k.get("json")); return Resp({"id":"X","status":"accepted"})
    raise AssertionError("unrouted "+m+" "+u)
requests.request = lambda m,u,**k: _route(m.upper(),u,**k)
requests.get = lambda u,**k: _route("GET",u,**k)

from trader import alpaca
from trader.autotrader import AutoTrader

print(f"{'stale/demo px':>13} {'real px':>8} {'qty sent':>8} {'notional@real':>14}  outcome")
for stale, real in [(62.33, 58.00), (100.0, 71.40), (121.0, 180.0), (100.0, 130.0), (50.0, 50.0)]:
    REAL["p"] = real
    CAPTURED.clear()
    qty = min(1000, alpaca.max_live_qty(stale))          # same cap autotrader.act() applies
    payload = alpaca.bracket_order_payload("XYZ", qty, stale, 3.0, 2.0)
    try:
        alpaca.submit_order(payload)
        tp = float(CAPTURED[0]["take_profit"]["limit_price"]); sl = float(CAPTURED[0]["stop_loss"]["stop_price"])
        note = (f"SENT   tp={tp} sl={sl} "
                + ("(stop ABOVE market: instant stop-out)" if sl > real else
                   "(tp BELOW market: instant flatten)" if tp < real else "(legs plausible)"))
    except alpaca.LiveLimitError as exc:
        note = f"REFUSED by notional cap: {exc}"
    print(f"{stale:13.2f} {real:8.2f} {qty:8d} {qty*real:14.2f}  {note}")
