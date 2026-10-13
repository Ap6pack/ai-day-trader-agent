"""A local stand-in for the Alpaca trading API, on 127.0.0.1.
One shared order book across every client process. Nothing leaves the loopback.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ORDERS = []
LOG = []
LOCK = threading.Lock()
T0 = time.monotonic()
GET_LATENCY = 0.06
POST_LATENCY = 0.04


def note(msg):
    with LOCK:
        LOG.append(f"[{time.monotonic() - T0:7.3f}s] {msg}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        who = self.headers.get("X-Actor", "?")
        if self.path.startswith("/v2/orders"):
            note(f"{who}: GET /v2/orders ->")
            time.sleep(GET_LATENCY)
            with LOCK:
                snap = [dict(o) for o in ORDERS]
            note(f"{who}: GET /v2/orders <- {len(snap)} orders")
            return self._send(snap)
        if self.path.startswith("/v2/account"):
            return self._send({"equity": "100000", "last_equity": "100000"})
        if self.path.startswith("/v2/clock"):
            return self._send({"is_open": True, "next_close": "2026-10-13T16:00:00-04:00"})
        if "/trades/latest" in self.path:
            return self._send({"trade": {"p": 10.0}})
        return self._send({})

    def do_POST(self):
        who = self.headers.get("X-Actor", "?")
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        note(f"{who}: POST /v2/orders {body.get('side')} {body.get('qty')} {body.get('symbol')} ->")
        time.sleep(POST_LATENCY)
        with LOCK:
            order = {"id": f"ord-{len(ORDERS) + 1}", "symbol": body.get("symbol"), "qty": body.get("qty"),
                     "side": body.get("side"), "status": "accepted", "filled_qty": "0",
                     "client_order_id": body.get("client_order_id")}
            ORDERS.append(order)
            total = len(ORDERS)
        note(f"{who}: POST /v2/orders <- ACCEPTED id={order['id']} (book now {total})")
        return self._send(order)


def serve(port=0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
