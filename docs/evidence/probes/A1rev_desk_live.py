"""
A1 re-check, LIVE (real-money) configuration, end to end through the real desk.
Same as A1rev_desk_e2e.py but:
  ALPACA_TRADING_BASE_URL=https://api.alpaca.markets + ALPACA_LIVE_TRADING=true
  POST /api/desk/autopilot {mode: live, confirm_live: true}
The fake broker enforces what Alpaca cannot: nothing leaves the process, and the
fake /v2/orders listing returns today's accepted buys so trader.alpaca.check_live_buy
(ALPACA_LIVE_MAX_ORDERS_PER_DAY) is actually exercised.
"""
from __future__ import annotations
import os, sys, threading, time, sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRATCH = Path("docs/evidence/probes")
DB = SCRATCH / "A1rev_live.db"
if DB.exists():
    DB.unlink()

os.environ.update({
    "ALPACA_API_KEY": "PLACEHOLDER", "ALPACA_SECRET_KEY": "PLACEHOLDER",
    "ALPACA_TRADING_BASE_URL": "https://api.alpaca.markets",   # REAL MONEY host
    "ALPACA_LIVE_TRADING": "true",                              # opt-in
    "DESK_START_NEWSBOT": "false", "DESK_AUTOPILOT_MODE": "off",
    "DIVIDEND_STRATEGY_ENABLED": "false", "DESK_DEMO_MODE": "false",
})
os.environ.pop("TYPESAFE_API_KEY", None)
os.environ.pop("DESK_TOKEN", None)
sys.path.insert(0, "/home/user/ai-day-trader-agent")

from trader import alpaca, market_feed           # noqa: E402
from trader.journal import Journal               # noqa: E402
from trader import desk                          # noqa: E402

T0 = time.monotonic(); LOG = []; _lk = threading.Lock()
def log(m):
    with _lk: LOG.append((time.monotonic()-T0, threading.current_thread().name, m))

ORDERS = []
NEXT_CLOSE = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()

def fake_request(method, url, **kw):
    if method == "GET" and url.endswith("/v2/clock"):
        return {"is_open": True, "next_close": NEXT_CLOSE, "next_open": NEXT_CLOSE}
    if method == "GET" and url.endswith("/v2/account"):
        return {"equity": "20000", "last_equity": "20000", "cash": "20000"}
    if method == "GET" and url.endswith("/v2/positions"):
        return []
    if method == "GET" and "/trades/latest" in url:
        sym = url.split("/stocks/")[1].split("/")[0]
        return {"trade": {"p": 47.94}}
    if method == "GET" and url.endswith("/v2/orders"):
        # today's orders, as Alpaca would report them (drives check_live_buy's daily cap)
        return [{"id": f"ord{i}", "side": p["side"], "status": "accepted", "symbol": p["symbol"],
                 "client_order_id": p.get("client_order_id"), "filled_qty": "0", "legs": []}
                for i, (_, p) in enumerate(ORDERS)]
    if method == "POST" and url.endswith("/v2/orders"):
        p = kw["json"]
        ORDERS.append((time.monotonic()-T0, p))
        log(f"*** REAL-MONEY ORDER accepted: {p['side'].upper()} {p['qty']} {p['symbol']} "
            f"(${float(p['qty'])*47.94:,.0f}) coid={p.get('client_order_id','')[:18]}")
        return {"id": f"ord{len(ORDERS)}", "status": "accepted", **p}
    raise AssertionError(f"unexpected {method} {url}")

alpaca._request = fake_request

def _closes():
    c = [100.0]
    for _ in range(250): c.append(c[-1]*0.997)
    for _ in range(8):   c.append(c[-1]*1.002)
    return c[-300:]
BARS = {"bars": [{"open": c, "high": c*1.004, "low": c*0.996, "close": c, "volume": 100000} for c in _closes()]}

def fake_get_bars(symbol, timeframe=None, limit=300, **kw):
    log(f"analyze({symbol}): 1.0s of Alpaca data latency")
    time.sleep(1.0)
    return BARS
market_feed.get_bars = fake_get_bars

from starlette.testclient import TestClient      # noqa: E402

app = desk.create_app(journal=Journal(DB), background=False, start_agents=False)
SYMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
with TestClient(app) as client:
    pilot = app.state.autopilot
    log(f"alpaca.account_mode() = {alpaca.account_mode()!r}; live limits = {alpaca.LiveLimits.from_env()}")
    r = client.post("/api/desk/autopilot",
                    json={"symbols": SYMS, "interval_seconds": 60, "mode": "live", "confirm_live": True})
    log(f"POST /api/desk/autopilot mode=live -> {r.status_code} {r.json().get('running')}")
    time.sleep(0.5)
    log("--- operator clicks OFF")
    r = client.delete("/api/desk/autopilot")
    t_off = time.monotonic()-T0
    log(f"DELETE -> {r.status_code} running={r.json()['running']}; orders so far={len(ORDERS)}")
    n_off = len(ORDERS)
    for _ in range(50): time.sleep(0.25)
    log(f"FINAL orders={len(ORDERS)} running={pilot.status()['running']} cycles={pilot.cycles}")

print("==== A1 LIVE re-check ====")
for t, th, m in LOG: print(f"t=+{t:6.3f}s [{th:<22}] {m}")
print()
after = ORDERS[n_off:]
print(f"REAL-MONEY orders accepted AFTER the desk reported OFF: {len(after)}")
for t, p in after:
    print(f"   +{t-t_off:5.2f}s after OFF  {p['side']} {p['qty']} {p['symbol']} notional ${float(p['qty'])*47.94:,.0f}")
con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
rows = con.execute("select ts, kind, detail from events where kind='autopilot_order' order by id").fetchall()
print(f"journal autopilot_order rows: {len(rows)}")
for r in rows: print("   ", r["ts"], r["detail"])
