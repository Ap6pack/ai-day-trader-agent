"""
A1 adversarial re-check, END TO END through the REAL desk HTTP app.

Nothing is faked above the HTTP boundary of the broker:
  * real trader.desk.create_app(), real AutoTrader built by the desk,
  * real trader.analysis.analyze (real technicals/fusion/sizing),
  * real trader.alpaca.submit_order / bracket_order_payload / account_mode,
  * ONLY trader.alpaca._request (the requests.request wrapper) and
    market_feed.get_bars are replaced -> no packet leaves the process.
The autopilot is started and stopped through the SAME HTTP endpoints the
dashboard uses: POST/DELETE /api/desk/autopilot.
"""
from __future__ import annotations

import os, sys, threading, time, json
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRATCH = Path("docs/evidence/probes")
DB = SCRATCH / "A1rev_desk.db"
if DB.exists():
    DB.unlink()

os.environ.update({
    "ALPACA_API_KEY": "PLACEHOLDER", "ALPACA_SECRET_KEY": "PLACEHOLDER",
    # default paper host; NOT the live host
    "ALPACA_TRADING_BASE_URL": "https://paper-api.alpaca.markets",
    "DESK_START_NEWSBOT": "false", "DESK_AUTOPILOT_MODE": "off",
    "DIVIDEND_STRATEGY_ENABLED": "false",
    "DESK_DEMO_MODE": "false",
})
os.environ.pop("TYPESAFE_API_KEY", None)   # Jev off -> sentiment HOLD, no network
os.environ.pop("DESK_TOKEN", None)

sys.path.insert(0, "/home/user/ai-day-trader-agent")

from trader import alpaca, market_feed            # noqa: E402
from trader.journal import Journal                # noqa: E402
from trader import desk                           # noqa: E402

T0 = time.monotonic()
LOG, _lk = [], threading.Lock()
def log(msg):
    with _lk:
        LOG.append((time.monotonic() - T0, threading.current_thread().name, msg))

ORDERS = []            # (t, payload)

# ---- the broker, at the HTTP layer -------------------------------------------------
NEXT_CLOSE = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()

def fake_request(method, url, **kw):
    if method == "GET" and url.endswith("/v2/clock"):
        return {"is_open": True, "next_close": NEXT_CLOSE, "next_open": NEXT_CLOSE, "timestamp": NEXT_CLOSE}
    if method == "GET" and url.endswith("/v2/account"):
        return {"equity": "20000", "last_equity": "20000", "cash": "20000"}
    if method == "GET" and url.endswith("/v2/positions"):
        return []
    if method == "GET" and url.endswith("/v2/orders"):
        return []
    if method == "POST" and url.endswith("/v2/orders"):
        p = kw["json"]
        ORDERS.append((time.monotonic() - T0, p))
        log(f"*** BROKER HTTP POST /v2/orders  {p['side'].upper()} {p['qty']} {p['symbol']} "
            f"coid={p.get('client_order_id','')[:20]}")
        return {"id": f"ord{len(ORDERS)}", "status": "accepted", **p}
    raise AssertionError(f"probe did not expect {method} {url}")

alpaca._request = fake_request

# ---- bars: a dip that technicals score BUY @ ~0.6 confidence -----------------------
def _closes():
    c = [100.0]
    for _ in range(250):
        c.append(c[-1] * 0.997)
    for _ in range(8):
        c.append(c[-1] * 1.002)
    return c[-300:]

BARS = {"bars": [{"open": c, "high": c * 1.004, "low": c * 0.996, "close": c, "volume": 100000,
                  "t": "2026-10-13T14:00:00Z"} for c in _closes()]}

ANALYZE_SECONDS = float(os.getenv("PROBE_ANALYZE_SECONDS", "1.0"))

def fake_get_bars(symbol, timeframe=None, limit=300, **kw):
    log(f"analysis: get_bars({symbol}) -> {ANALYZE_SECONDS}s of simulated Alpaca data latency")
    time.sleep(ANALYZE_SECONDS)
    return BARS

market_feed.get_bars = fake_get_bars

# ---- run the desk -------------------------------------------------------------------
from starlette.testclient import TestClient      # noqa: E402

app = desk.create_app(journal=Journal(DB), background=False, start_agents=False)
SYMBOLS = ["AAA", "BBB", "CCC"]

with TestClient(app) as client:
    pilot = app.state.autopilot
    log(f"limits in force: {pilot.limits}")
    log("POST /api/desk/autopilot  {mode: paper, symbols: AAA,BBB,CCC, interval: 60}")
    r = client.post("/api/desk/autopilot",
                    json={"symbols": SYMBOLS, "interval_seconds": 60, "mode": "paper"})
    log(f"POST returned {r.status_code} running={r.json()['running']}")

    time.sleep(0.5)                       # worker thread is inside analyze(AAA)
    log(f"--- operator clicks the AUTOPILOT toggle OFF; broker orders so far = {len(ORDERS)}")
    t_del = time.monotonic() - T0
    r = client.delete("/api/desk/autopilot")
    t_del_done = time.monotonic() - T0
    body = r.json()
    log(f"DELETE /api/desk/autopilot returned {r.status_code} in {(t_del_done-t_del)*1000:.1f} ms: "
        f"running={body['running']} current_symbol={body['current_symbol']}")
    n_at_off = len(ORDERS)
    log(f"orders at the instant the desk said OFF = {n_at_off}")

    for _ in range(60):
        time.sleep(0.25)
    log(f"FINAL broker orders={len(ORDERS)} status.running={pilot.status()['running']} cycles={pilot.cycles}")

print("==== A1 re-check: real desk app, real analysis, real alpaca client, fake socket ====")
for t, th, m in LOG:
    print(f"t=+{t:6.3f}s [{th:<24}] {m}")
print()
print(f"orders BEFORE the DELETE handler returned: {n_at_off}")
print(f"orders AFTER  the DELETE handler returned: {len(ORDERS) - n_at_off} "
      f"-> {[(p['side'], p['qty'], p['symbol'], p['order_class']) for _, p in ORDERS[n_at_off:]]}")
j = Journal(DB)
rows = [dict(r) for r in j.events(50) if r["event"] == "autopilot_order"]
print(f"journal autopilot_order rows: {len(rows)} -> {[r['detail'] for r in rows]}")
print(f"exposure window after OFF: {ORDERS[-1][0]-t_del_done:.2f}s" if len(ORDERS) > n_at_off else "none")
