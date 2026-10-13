"""A1 re-check: desk SHUTDOWN (Ctrl-C / systemd stop) mid-cycle -- real desk lifespan."""
from __future__ import annotations
import os, sys, threading, time, sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
SCRATCH = Path("docs/evidence/probes")
DB = SCRATCH / "A1rev_shutdown.db"
if DB.exists(): DB.unlink()
os.environ.update({"ALPACA_API_KEY": "d", "ALPACA_SECRET_KEY": "d",
                   "ALPACA_TRADING_BASE_URL": "https://paper-api.alpaca.markets",
                   "DESK_START_NEWSBOT": "false",
                   "DESK_AUTOPILOT_MODE": "paper",          # autopilot ON at boot, as shipped-config allows
                   "DESK_AUTOPILOT_SYMBOLS": "AAA,BBB,CCC",
                   "DESK_AUTOPILOT_INTERVAL_SECONDS": "60",
                   "DIVIDEND_STRATEGY_ENABLED": "false", "DESK_DEMO_MODE": "false"})
os.environ.pop("TYPESAFE_API_KEY", None); os.environ.pop("DESK_TOKEN", None)
sys.path.insert(0, "/home/user/ai-day-trader-agent")
from trader import alpaca, market_feed            # noqa
from trader.journal import Journal                # noqa
from trader import desk                           # noqa
T0=time.monotonic(); LOG=[]; _lk=threading.Lock()
def log(m):
    with _lk: LOG.append((time.monotonic()-T0, threading.current_thread().name, m))
ORDERS=[]; NEXT_CLOSE=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat()
def fake_request(method, url, **kw):
    if url.endswith("/v2/clock"): return {"is_open": True, "next_close": NEXT_CLOSE}
    if url.endswith("/v2/account"): return {"equity": "20000", "last_equity": "20000"}
    if url.endswith("/v2/positions"): return []
    if method=="GET" and url.endswith("/v2/orders"): return []
    if method=="POST" and url.endswith("/v2/orders"):
        p=kw["json"]; ORDERS.append((time.monotonic()-T0,p))
        log(f"*** BROKER HTTP POST /v2/orders {p['side'].upper()} {p['qty']} {p['symbol']}")
        return {"id": f"o{len(ORDERS)}", "status": "accepted", **p}
    raise AssertionError(f"unexpected {method} {url}")
alpaca._request = fake_request
def _c():
    c=[100.0]
    for _ in range(250): c.append(c[-1]*0.997)
    for _ in range(8): c.append(c[-1]*1.002)
    return c[-300:]
BARS={"bars":[{"open":c,"high":c*1.004,"low":c*0.996,"close":c,"volume":1e5} for c in _c()]}
def bars(symbol, tf=None, limit=300, **kw):
    log(f"analyze({symbol}) 1.0s"); time.sleep(1.0); return BARS
market_feed.get_bars = bars
from starlette.testclient import TestClient       # noqa
app = desk.create_app(journal=Journal(DB), background=True, start_agents=True)
log("desk booting (DESK_AUTOPILOT_MODE=paper -> autopilot ON at boot, no clicking)")
with TestClient(app) as client:
    time.sleep(0.6)   # first cycle in flight, inside analyze(AAA)
    log("--- SIGTERM / Ctrl-C: leaving the app context -> lifespan shutdown (await pilot.stop())")
t_shut = time.monotonic()-T0
log(f"lifespan shutdown COMPLETE, event loop closed. orders at that point = {len(ORDERS)}")
time.sleep(4)
log(f"after the loop is gone: orders = {len(ORDERS)}")
print("==== A1 shutdown re-check ====")
for t,th,m in LOG: print(f"t=+{t:6.3f}s [{th:<22}] {m}")
print()
print(f"orders submitted while/after the desk was shutting down: "
      f"{[(p['side'],p['qty'],p['symbol'],round(t,2)) for t,p in ORDERS]}")
con=sqlite3.connect(DB)
print("journal autopilot_order rows:", con.execute("select count(*) from events where kind='autopilot_order'").fetchone()[0])
