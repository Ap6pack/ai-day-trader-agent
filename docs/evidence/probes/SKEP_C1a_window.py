"""How far apart can the two live submitters start and still both get through?
Measures the TOCTOU window width as a function of broker round-trip latency."""
import os, sys, threading, time

os.environ.update({
    "ALPACA_TRADING_BASE_URL": "https://api.alpaca.markets", "ALPACA_LIVE_TRADING": "true",
    "ALPACA_LIVE_MAX_ORDERS_PER_DAY": "1", "ALPACA_LIVE_MAX_ORDER_USD": "500",
    "ALPACA_LIVE_MAX_DAILY_LOSS_USD": "200", "ALPACA_API_KEY": "PLACEHOLDER", "ALPACA_SECRET_KEY": "PLACEHOLDER"})
from trader import alpaca  # noqa: E402

def trial(lat, offset_s):
    orders, lk = [], threading.Lock()
    def req(method, url, **kw):
        if "/trades/latest" in url: time.sleep(lat); return {"trade": {"p": 10.0}}
        if url.endswith("/v2/account"): time.sleep(lat); return {"equity": "100000", "last_equity": "100000"}
        if method == "GET" and url.endswith("/v2/orders"):
            time.sleep(lat)
            with lk: return [dict(o) for o in orders]
        if method == "POST" and url.endswith("/v2/orders"):
            time.sleep(lat)
            with lk:
                o = {"id": f"ord-{len(orders)+1}", "side": "buy", "symbol": (kw.get('json') or {}).get("symbol"),
                     "status": "accepted", "qty": "10"}
                orders.append(o); return o
        raise AssertionError(url)
    alpaca._request = req
    def actor(sym, delay):
        time.sleep(delay)
        try: alpaca.submit_order(alpaca.market_order_payload(sym, "buy", 10))
        except alpaca.LiveLimitError: pass
    ts = [threading.Thread(target=actor, args=("AAPL", 0.0)),
          threading.Thread(target=actor, args=("MSFT", offset_s))]
    for t in ts: t.start()
    for t in ts: t.join()
    return len(orders)

for lat in (0.02, 0.06, 0.12):
    widest = None
    row = []
    for off_ms in (0, 10, 20, 40, 60, 80, 100, 150, 200, 300, 400):
        n = trial(lat, off_ms / 1000.0)
        row.append(f"{off_ms:>4}ms:{'BREACH' if n > 1 else '  held'}")
        if n > 1: widest = off_ms
    print(f"RTT={lat*1000:>4.0f}ms per call  |  " + " ".join(row))
    print(f"    -> widest start-stagger that still breaches the cap: {widest} ms\n")
