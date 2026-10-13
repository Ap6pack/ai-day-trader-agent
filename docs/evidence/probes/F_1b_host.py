import sys, json
sys.path.insert(0, "docs/evidence/probes")
import F_common
F_common.arm(live=True)           # LIVE account: ALPACA_LIVE_TRADING=true
app, client = F_common.app_and_client()

print("=== F1b: is there any Host / TrustedHostMiddleware check? ===")
print("  app.user_middleware:", [m.cls.__name__ for m in app.user_middleware] or "NONE")
src = open("/home/user/ai-day-trader-agent/trader/desk.py").read()
print("  'TrustedHost' in desk.py:", "TrustedHost" in src)
print("  'allowed_hosts' in desk.py:", "allowed_hosts" in src)
print("  request.headers['host'] read anywhere in desk.py:",
      any("headers" in l and "host" in l.lower() for l in src.splitlines()))
import subprocess
print("  repo-wide grep for TrustedHost:",
      subprocess.run(["grep","-rn","TrustedHost","/home/user/ai-day-trader-agent/trader",
                      "/home/user/ai-day-trader-agent/tests"],
                     capture_output=True, text=True).stdout.strip() or "(no hits)")

print()
print("=== F1b: requests carrying an attacker Host header (what a rebound page sends) ===")
EVIL = {"Host": "desk.evil.example:8000"}
for path in ("/api/desk/auth", "/api/desk/config", "/api/desk/account", "/api/desk/portfolios"):
    r = client.get(path, headers=EVIL)
    print(f"  GET  {path:28} Host=desk.evil.example -> {r.status_code}  {r.text[:90]}")

print()
print("=== F1b: state-changing endpoints under the attacker Host header (LIVE account armed) ===")
before = len(F_common.CALLS)
r = client.post("/api/desk/orders",
                json={"symbol": "NVDA", "side": "buy", "qty": 500, "confirm_live": True},
                headers=EVIL)
print("  POST /api/desk/orders ->", r.status_code, r.text[:160])
r = client.post("/api/desk/orders/ORD-REAL-123/cancel", headers=EVIL)
print("  POST /api/desk/orders/{id}/cancel ->", r.status_code, r.text[:120])
r = client.post("/api/desk/autopilot",
                json={"symbols": ["SPY"], "interval_seconds": 300, "mode": "live", "confirm_live": True},
                headers=EVIL)
print("  POST /api/desk/autopilot (mode=live) ->", r.status_code, r.text[:200])
r = client.request("DELETE", "/api/desk/autopilot", headers=EVIL)
print("  DELETE /api/desk/autopilot ->", r.status_code, r.text[:120])
r = client.post("/api/desk/newsbot/pause", json={"paused": True}, headers=EVIL)
print("  POST /api/desk/newsbot/pause ->", r.status_code, r.text[:120])
r = client.post("/api/desk/portfolios", json={"name": "evil", "cash": 1000}, headers=EVIL)
print("  POST /api/desk/portfolios ->", r.status_code, r.text[:120])
r = client.request("DELETE", "/api/desk/portfolios/evil", headers=EVIL)
print("  DELETE /api/desk/portfolios/evil ->", r.status_code, r.text[:120])

print()
print("=== F1b: Alpaca HTTP calls the above actually made (recorder) ===")
for c in F_common.CALLS[before:]:
    print("  ", c["method"], c["url"], json.dumps(c["json"]) if c["json"] else "")
print("  account mode during this run:", __import__("trader.alpaca", fromlist=["x"])._mode_or_none())
