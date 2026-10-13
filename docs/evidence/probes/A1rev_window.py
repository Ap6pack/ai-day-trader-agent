"""A1 re-check: how wide is the window? stop() DURING the between-cycle sleep must be clean."""
from __future__ import annotations
import os, sys, threading, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
SCRATCH = Path("docs/evidence/probes")
DB = SCRATCH / "A1rev_window.db"
if DB.exists(): DB.unlink()
os.environ.update({"ALPACA_API_KEY":"d","ALPACA_SECRET_KEY":"d",
  "ALPACA_TRADING_BASE_URL":"https://paper-api.alpaca.markets","DESK_START_NEWSBOT":"false",
  "DESK_AUTOPILOT_MODE":"off","DIVIDEND_STRATEGY_ENABLED":"false",
  "AUTOPILOT_COOLDOWN_MINUTES":"0","AUTOPILOT_MAX_TRADES_PER_DAY":"50"})
os.environ.pop("TYPESAFE_API_KEY", None); os.environ.pop("DESK_TOKEN", None)
sys.path.insert(0,"/home/user/ai-day-trader-agent")
from trader import alpaca, market_feed
from trader.journal import Journal
from trader import desk
T0=time.monotonic(); LOG=[]; L=threading.Lock()
def log(m):
    with L: LOG.append((time.monotonic()-T0, threading.current_thread().name, m))
ORDERS=[]; NC=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat()
def req(method,url,**kw):
    if url.endswith("/v2/clock"): return {"is_open":True,"next_close":NC}
    if url.endswith("/v2/account"): return {"equity":"20000","last_equity":"20000"}
    if url.endswith("/v2/positions"): return []
    if method=="GET" and url.endswith("/v2/orders"): return []
    if method=="POST" and url.endswith("/v2/orders"):
        p=kw["json"]; ORDERS.append(p); log(f"*** ORDER {p['side']} {p['qty']} {p['symbol']}")
        return {"id":f"o{len(ORDERS)}","status":"accepted",**p}
    raise AssertionError(url)
alpaca._request=req
def _c():
    c=[100.0]
    for _ in range(250): c.append(c[-1]*0.997)
    for _ in range(8): c.append(c[-1]*1.002)
    return c[-300:]
B={"bars":[{"open":c,"high":c*1.004,"low":c*0.996,"close":c,"volume":1e5} for c in _c()]}
def bars(sym,tf=None,limit=300,**kw):
    time.sleep(0.3); return B
market_feed.get_bars=bars
from starlette.testclient import TestClient
app=desk.create_app(journal=Journal(DB), background=False, start_agents=False)
with TestClient(app) as c:
    pilot=app.state.autopilot
    c.post("/api/desk/autopilot", json={"symbols":["AAA","BBB"],"interval_seconds":60,"mode":"paper"})
    log("started; cycle of 2 symbols takes ~0.7s, interval 60s")
    time.sleep(3.0)                      # cycle finished; pilot is now in the idle sleep
    n=len(ORDERS); log(f"cycle done, {n} orders; now IDLE between cycles. pressing OFF")
    t=time.monotonic(); r=c.delete("/api/desk/autopilot")
    log(f"DELETE -> {r.status_code} in {(time.monotonic()-t)*1000:.1f}ms")
    time.sleep(4); log(f"orders after OFF-during-idle: {len(ORDERS)-n}")
print("==== A1 window: OFF pressed between cycles ====")
for t,th,m in LOG: print(f"t=+{t:6.3f}s [{th:<20}] {m}")
