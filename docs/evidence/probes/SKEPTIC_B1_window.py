"""How wide is the TOCTOU window really? Count broker round-trips between the
single GET snapshot and each symbol's market sell, at max_trades_per_day=5."""
import os, time, uuid, json
os.environ.update(ALPACA_TRADING_BASE_URL="https://api.alpaca.markets",
                  ALPACA_LIVE_TRADING="true", ALPACA_API_KEY="k", ALPACA_SECRET_KEY="s")
import requests
from trader import alpaca

LATENCY = 0.12  # a realistic Alpaca REST round-trip
orders, t0, marks = {}, None, []

for i, sym in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
    orders[f"p{i}"] = {"id": f"p{i}", "symbol": sym, "side": "buy", "qty": "5",
                       "filled_qty": "5", "status": "filled", "client_order_id": f"newsbot-{i}",
                       "legs": [{"id": f"{sym}-tp", "side": "sell", "qty": "5", "filled_qty": "0", "status": "new"},
                                {"id": f"{sym}-sl", "side": "sell", "qty": "5", "filled_qty": "0", "status": "held"}]}

def req(method, url, **kw):
    global t0
    time.sleep(LATENCY)
    path = url.split("alpaca.markets", 1)[-1]
    if method == "GET":
        t0 = time.monotonic()
        marks.append((0.0, "GET /v2/orders  <-- the ONLY read; every qty below comes from here"))
        return json.loads(json.dumps(list(orders.values())))
    if method == "DELETE":
        marks.append((time.monotonic() - t0, f"DELETE {path.rsplit('/',1)[-1]}"))
        return None
    j = kw["json"]
    marks.append((time.monotonic() - t0, f"POST  sell {j['qty']} {j['symbol']}   <-- qty is "
                                         f"{time.monotonic()-t0:.2f}s stale"))
    return dict(j, id=str(uuid.uuid4()), status="filled")

alpaca._request = req
alpaca.flatten_owned("newsbot-")
for dt, what in marks:
    print(f"  t={dt:5.2f}s  {what}")
print(f"\n  total broker round-trips: {len(marks)}")
print(f"  staleness of the LAST symbol's qty: {marks[-1][0]:.2f}s at {LATENCY*1000:.0f}ms/req")
