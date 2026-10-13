import sys, json
sys.path.insert(0, "docs/evidence/probes")
import F_common
alp = F_common.arm(live=True)

# make the live-limit price lookup work so the LIVE order really reaches POST /v2/orders
_orig = alp._request
def fake(method, url, **kw):
    F_common.CALLS.append({"method": method, "url": url, "json": kw.get("json")})
    if "/trades/latest" in url:
        return {"trade": {"p": 1.00}}
    if url.endswith("/v2/account"):
        return {"equity": "250000", "last_equity": "250000", "account_number": "REAL-9999"}
    if url.endswith("/v2/orders") and method == "GET":
        return []
    if url.endswith("/v2/orders") and method == "POST":
        return {"id": "ORD-FAKE-0001", "status": "accepted"}
    if "/v2/orders/" in url and method == "DELETE":
        return None
    return {}
alp._request = fake

app, client = F_common.app_and_client()
BODY = json.dumps({"symbol": "NVDA", "side": "sell", "qty": 500, "confirm_live": True})

def show(label, resp):
    print(f"  {label:58} -> HTTP {resp.status_code}  {resp.text[:110]}")

print("=== F1c-1: POST /api/desk/orders/{id}/cancel  -- the bodyless route ===")
n = len(F_common.CALLS)
show("no Content-Type, no body (pure CORS 'simple request')",
     client.post("/api/desk/orders/ORD-VICTIM-7/cancel",
                 headers={"Origin": "https://evil.example"}))
show("Content-Type: text/plain, body ignored",
     client.post("/api/desk/orders/ORD-VICTIM-8/cancel", content=b"anything",
                 headers={"Content-Type": "text/plain;charset=UTF-8",
                          "Origin": "https://evil.example"}))
show("Content-Type: application/x-www-form-urlencoded (<form> POST)",
     client.post("/api/desk/orders/ORD-VICTIM-9/cancel", content=b"a=b",
                 headers={"Content-Type": "application/x-www-form-urlencoded",
                          "Origin": "https://evil.example"}))
print("  broker calls made:")
for c in F_common.CALLS[n:]:
    print("   ", c["method"], c["url"])

print()
print("=== F1c-2: POST /api/desk/orders -- does FastAPI parse a non-JSON content-type as JSON? ===")
cases = [
    ("application/json           (preflighted, NOT simple)", {"Content-Type": "application/json"}),
    ("text/plain;charset=UTF-8   (SIMPLE: fetch default for a string body)",
     {"Content-Type": "text/plain;charset=UTF-8"}),
    ("application/x-www-form-urlencoded (SIMPLE: classic <form>)",
     {"Content-Type": "application/x-www-form-urlencoded"}),
    ("multipart/form-data        (SIMPLE)", {"Content-Type": "multipart/form-data"}),
    ("<<NO Content-Type header>> (SIMPLE: fetch + new Blob([...],{type:''}))", {}),
]
for label, ct in cases:
    n = len(F_common.CALLS)
    h = dict(ct); h["Origin"] = "https://evil.example"
    r = client.post("/api/desk/orders", content=BODY.encode(), headers=h)
    broker = [f"{c['method']} {c['url']}" for c in F_common.CALLS[n:]
              if c["url"].endswith("/v2/orders") and c["method"] == "POST"]
    print(f"  {label:60} -> HTTP {r.status_code}  broker_POST={broker}")
    print(f"      body: {r.text[:150]}")

print()
print("=== F1c-3: how TestClient actually framed the no-Content-Type request ===")
import httpx
req = client.build_request("POST", "/api/desk/orders", content=BODY.encode())
print("  outgoing headers:", dict(req.headers))

print()
print("=== F1c-4: same no-Content-Type trick on the other state-changing POSTs ===")
for path, body in (("/api/desk/autopilot",
                    {"symbols": ["SPY"], "interval_seconds": 300, "mode": "live", "confirm_live": True}),
                   ("/api/desk/newsbot/pause", {"paused": True}),
                   ("/api/desk/newsbot/process", {"run": False}),
                   ("/api/desk/portfolios", {"name": "pwned", "cash": 1.0})):
    r = client.post(path, content=json.dumps(body).encode(),
                    headers={"Origin": "https://evil.example"})
    print(f"  POST {path:30} (no Content-Type) -> HTTP {r.status_code} {r.text[:90]}")
client.request("DELETE", "/api/desk/autopilot")
client.request("DELETE", "/api/desk/portfolios/pwned")

print()
print("=== F1c-5: GET/bodyless POSTs that need no body at all ===")
for m, p in (("POST", "/api/desk/judge/NVDA"), ("POST", "/api/desk/analyze/NVDA")):
    r = client.request(m, p, headers={"Origin": "https://evil.example"})
    print(f"  {m} {p:28} (no body, no Content-Type) -> HTTP {r.status_code} {r.text[:80]}")

print()
print("=== F1c-6: FastAPI's actual content-type rule (fastapi/routing.py) ===")
import inspect, fastapi.routing, re
src = inspect.getsource(fastapi.routing)
i = src.find("content_type_value = request.headers.get")
print(src[i-420:i+620])
print("  fastapi version:", __import__("fastapi").__version__)
