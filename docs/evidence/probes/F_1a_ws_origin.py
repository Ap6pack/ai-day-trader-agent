import sys, json
sys.path.insert(0, "docs/evidence/probes")
import F_common
F_common.arm(live=False)
app, client = F_common.app_and_client()

print("=== F1a: static grep for any Origin check in trader/desk.py ===")
src = open("/home/user/ai-day-trader-agent/trader/desk.py").read()
for needle in ("origin", "Origin", "CORS", "TrustedHost", "add_middleware", "allow_origins"):
    hits = [l for l in src.splitlines() if needle in l]
    print(f"  {needle!r:16} -> {len(hits)} line(s)")
print("  middleware stack on the app object:", [m.cls.__name__ for m in app.user_middleware] or "NONE")

print()
print("=== F1a: WebSocket handshake with Origin: https://evil.example, DESK_TOKEN unset ===")
with client.websocket_connect("/ws/desk", headers={"Origin": "https://evil.example"}) as ws:
    msg = ws.receive_json()
    print("  handshake: ACCEPTED (no 4401 close)")
    print("  first frame type:", msg["type"])
    d = msg["data"]
    print("  hello keys:", sorted(d.keys()))
    print("  portfolios leaked:", d["portfolios"])
    print("  autopilot leaked:", json.dumps(d["autopilot"])[:160])
    print("  newsbot leaked:", json.dumps(d["newsbot"])[:200])
    # account data is pushed by the background pump; prove the same socket also
    # accepts attacker-chosen commands:
    ws.send_json({"type": "ping"})
    print("  attacker-sent {'type':'ping'} -> server replied:", ws.receive_json())

print()
print("=== F1a: does the socket stream the Alpaca ACCOUNT? (hub.last_account path) ===")
app2, client2 = F_common.app_and_client()
hub = app2.state.hub
hub.last_account = {"equity": "250000", "account_number": "REAL-ACCT-9999",
                    "positions": [{"symbol": "NVDA", "qty": "500", "unrealized_pl": "12345"}]}
with client2.websocket_connect("/ws/desk?token=anything-at-all",
                               headers={"Origin": "https://evil.example"}) as ws:
    hello = ws.receive_json()
    print("  account in hello frame:", json.dumps(hello["data"]["account"]))

print()
print("=== F1a: control -- with DESK_TOKEN SET, is the bad-Origin socket still accepted? ===")
import os
os.environ["DESK_TOKEN"] = "s3cret"
app3, client3 = F_common.app_and_client()
try:
    with client3.websocket_connect("/ws/desk", headers={"Origin": "https://evil.example"}) as ws:
        ws.receive_json()
        print("  no-token socket: ACCEPTED  <-- would be a token bug")
except Exception as exc:
    print("  no-token socket rejected:", type(exc).__name__, exc)
try:
    with client3.websocket_connect("/ws/desk?token=s3cret",
                                   headers={"Origin": "https://evil.example"}) as ws:
        ws.receive_json()
        print("  correct-token + evil Origin: ACCEPTED (Origin ignored even when a token exists)")
except Exception as exc:
    print("  correct-token + evil Origin rejected:", type(exc).__name__, exc)
os.environ.pop("DESK_TOKEN")
