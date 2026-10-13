import sys
sys.path.insert(0, "docs/evidence/probes")
from SKEPTIC_F1_common import install_recorder
CALLS = install_recorder(live=False)
from trader.desk import create_app, desk_token, token_ok
from fastapi.testclient import TestClient

app = create_app(background=False, start_agents=False)
print("DESK_TOKEN default    :", desk_token())
print("token_ok(None)        :", token_ok(None))
print("user_middleware       :", app.user_middleware)
print("middleware_stack cls  :", [type(m).__name__ for m in app.user_middleware])

c = TestClient(app)
print("GET /api/desk/auth    :", c.get("/api/desk/auth").status_code, c.get("/api/desk/auth").json())
r = c.get("/api/desk/account")
print("GET /api/desk/account :", r.status_code, str(r.json())[:120])
r = c.get("/api/desk/account", headers={"Origin": "https://evil.example", "Host": "evil.example"})
print("  with evil Origin+Host:", r.status_code)
print("  CORS resp hdr ACAO  :", r.headers.get("access-control-allow-origin"))

# cancel: bodyless POST
CALLS.clear()
r = c.post("/api/desk/orders/deadbeef-1111/cancel", headers={"Origin": "https://evil.example"})
print("POST cancel bodyless  :", r.status_code, r.json() if r.status_code<400 else r.text[:80])
print("  broker calls        :", CALLS)

# order endpoint content-type matrix
body = '{"symbol":"AAPL","side":"buy","qty":5,"destination":"alpaca","confirm_live":true}'
for ct in ["application/json", "text/plain;charset=UTF-8",
           "application/x-www-form-urlencoded", "multipart/form-data"]:
    CALLS.clear()
    r = c.post("/api/desk/orders", content=body,
               headers={"Content-Type": ct, "Origin": "https://evil.example"})
    print(f"POST /orders CT={ct:38s} -> {r.status_code}  broker={[x['method']+' '+x['path'] for x in CALLS]}")

# no Content-Type at all (the Blob({type:''}) CSRF trick)
import httpx
CALLS.clear()
req = c.build_request("POST", "/api/desk/orders", content=body)
req.headers.pop("content-type", None)
r = c.send(req)
print("POST /orders NO Content-Type ->", r.status_code, str(r.json())[:140])
print("  broker calls        :", [x['method']+' '+x['path']+' '+str(x['kw'].get('json')) for x in CALLS])
