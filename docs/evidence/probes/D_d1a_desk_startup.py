"""D1a (second half): does the DESK itself block demo mode from driving the LIVE autopilot?
Boots the real FastAPI app with DESK_DEMO_MODE=true and DESK_AUTOPILOT_MODE=live.
All HTTP intercepted."""
import json, os, pathlib, time
SCRATCH = pathlib.Path("docs/evidence/probes")
DB = SCRATCH / "D_d1a_desk.db"
DB.unlink(missing_ok=True)

os.environ.update({
    "DESK_DEMO_MODE": "true",
    "DESK_START_NEWSBOT": "false",
    "DESK_AUTOPILOT_MODE": "live",
    "DESK_AUTOPILOT_SYMBOLS": "O",
    "DESK_AUTOPILOT_INTERVAL_SECONDS": "3600",
    "ALPACA_TRADING_BASE_URL": "https://api.alpaca.markets",
    "ALPACA_LIVE_TRADING": "true",
    "ALPACA_API_KEY": "PLACEHOLDER", "ALPACA_SECRET_KEY": "PLACEHOLDER",
    "ANALYSIS_TIMEFRAME": "15Min",
    "TRADER_JOURNAL_PATH": str(DB),
    "DESK_TOKEN": "",
})

import requests
from datetime import datetime, timedelta, timezone
CAPTURED = []
class Resp:
    def __init__(self, p): self._p=p; self.content=b"x"; self.status_code=200
    def raise_for_status(self): pass
    def json(self): return self._p
def _route(method, url, **kw):
    if "alpaca.markets" not in url: raise AssertionError("unexpected host " + url)
    now = datetime.now(timezone.utc)
    if url.endswith("/v2/clock"):
        return Resp({"is_open": True, "timestamp": now.isoformat(),
                     "next_open": (now+timedelta(hours=20)).isoformat(),
                     "next_close": (now+timedelta(hours=3)).isoformat()})
    if url.endswith("/v2/account"): return Resp({"equity":"100000","last_equity":"100000"})
    if url.endswith("/trades/latest"): return Resp({"trade":{"p":58.0}})
    if "/v1/corporate-actions" in url: return Resp({"corporate_actions":{"cash_dividends":[]}})
    if url.endswith("/v2/orders") and method=="GET": return Resp([])
    if url.endswith("/v2/orders") and method=="POST":
        CAPTURED.append(kw.get("json")); return Resp({"id":"DESK-ORDER-1","status":"accepted"})
    if url.endswith("/v2/positions"): return Resp([])
    raise AssertionError("unrouted " + method + " " + url)
requests.request = lambda m,u,**k: _route(m.upper(),u,**k)
requests.get = lambda u,**k: _route("GET",u,**k)
requests.post = lambda u,**k: _route("POST",u,**k)

from fastapi.testclient import TestClient
from trader import desk, market_feed
from trader.journal import Journal

print("newsbot_start_blocker() =", repr(desk.newsbot_start_blocker()))
print("autopilot blocker       = (no such function; grep below)")
import subprocess
print(subprocess.run(["grep","-c","autopilot_start_blocker","/home/user/ai-day-trader-agent/trader/desk.py"],
                     capture_output=True, text=True).stdout.strip(), "matches for 'autopilot_start_blocker'")

app = desk.create_app(Journal(DB), background=False, start_agents=True)
with TestClient(app) as client:
    for _ in range(60):
        if CAPTURED: break
        time.sleep(0.25)
    st = client.get("/api/desk/autopilot").json()
    print("autopilot running       =", st["running"], " mode =", st["config"]["mode"], " cycles =", st["cycles"])
    cfg = client.get("/api/desk/config").json()
    print("desk config demo_mode   =", cfg["demo_mode"], " account_mode =", cfg["account_mode"])
print("LIVE orders POSTed during desk startup:", len(CAPTURED))
for p in CAPTURED: print(json.dumps(p, indent=2))
