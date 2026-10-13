"""C1 adversarial re-check driver.

Stands up a local fake Alpaca on 127.0.0.1 (real sockets), then launches the
REAL NewsBot / AutoTrader / desk-ticket code in SEPARATE OS PROCESSES (so no
shared interpreter, no shared lock is even possible) with a controlled start
offset, and reports how many live buys the broker actually received against a
cap of 1.

Sweeping the offset measures how wide the TOCTOU window really is.

usage: C1rev_race.py <roleA> <roleB> [get_ms] [post_ms]
"""
import json
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROLE_A = sys.argv[1] if len(sys.argv) > 1 else "newsbot"
ROLE_B = sys.argv[2] if len(sys.argv) > 2 else "autopilot"
GET_LATENCY = (float(sys.argv[3]) if len(sys.argv) > 3 else 45) / 1000.0
POST_LATENCY = (float(sys.argv[4]) if len(sys.argv) > 4 else 45) / 1000.0
CAP = "1"
PY = ".venv/bin/python"
REPO = "/home/user/ai-day-trader-agent"
HERE = Path(__file__).parent

STATE = {"orders": [], "log": [], "t0": time.monotonic()}
LOCK = threading.Lock()


def note(msg):
    with LOCK:
        STATE["log"].append(f"[{(time.monotonic() - STATE['t0']) * 1000:8.1f} ms] {msg}")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

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
        path = urlparse(self.path).path
        if path.endswith("/trades/latest"):
            return self._send({"trade": {"p": 10.0}})
        if path == "/v2/clock":
            close = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3 * 3600))
            return self._send({"is_open": True, "next_close": close,
                               "next_open": close, "timestamp": close})
        if path == "/v2/account":
            return self._send({"equity": "100000", "last_equity": "100000",
                               "buying_power": "100000", "status": "ACTIVE"})
        if path == "/v2/orders":
            note("GET  /v2/orders  request in")
            time.sleep(GET_LATENCY)
            with LOCK:
                snap = [dict(o) for o in STATE["orders"]]
            note(f"GET  /v2/orders  answered with {len(snap)} order(s)")
            return self._send(snap)
        if path == "/v2/positions":
            return self._send([])
        return self._send({})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        note(f"POST /v2/orders  {body.get('side')} {body.get('qty')} {body.get('symbol')} in")
        time.sleep(POST_LATENCY)
        with LOCK:
            order = {"id": f"ord-{len(STATE['orders']) + 1}", "symbol": body.get("symbol"),
                     "qty": body.get("qty"), "side": body.get("side"), "status": "accepted",
                     "filled_qty": "0", "client_order_id": body.get("client_order_id")}
            STATE["orders"].append(order)
            n = len(STATE["orders"])
        note(f"POST /v2/orders  ACCEPTED {order['id']} (broker now holds {n} buy order(s))")
        return self._send(order)


srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
print(f"local fake broker on 127.0.0.1:{port}  "
      f"(GET {GET_LATENCY * 1000:.0f} ms, POST {POST_LATENCY * 1000:.0f} ms round trip)")
print(f"racing {ROLE_A} vs {ROLE_B}, ALPACA_LIVE_MAX_ORDERS_PER_DAY={CAP}\n")


def one_run(offset_ms, verbose=False):
    with LOCK:
        STATE["orders"], STATE["log"], STATE["t0"] = [], [], time.monotonic()
    tmp = Path(tempfile.mkdtemp(prefix="C1rev_"))
    start = time.time() + 1.2
    procs = []
    for role, off in ((ROLE_A, 0.0), (ROLE_B, offset_ms / 1000.0)):
        procs.append(subprocess.Popen(
            [PY, str(HERE / "C1rev_actor.py"), role, str(port), str(start + off),
             str(tmp / f"{role}.db"), CAP],
            cwd=REPO, env={"PATH": "/usr/bin:/bin", "PYTHONPATH": REPO, "HOME": str(tmp)},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    outs = []
    for p in procs:
        so, se = p.communicate(timeout=120)
        line = [ln for ln in so.strip().splitlines() if ln.startswith("{")]
        outs.append(json.loads(line[-1]) if line else {"stderr": se.strip()[-400:]})
    with LOCK:
        n, log = len(STATE["orders"]), list(STATE["log"])
    if verbose:
        for ln in log:
            print("   ", ln)
    return n, outs


print("offset of the 2nd actor -> live buy orders the broker actually received (cap 1)")
for offset in (0, 10, 25, 50, 75, 100, 150, 200, 300, 500):
    n, outs = one_run(offset)
    verdict = "CAP BREACHED" if n > 1 else "cap held"
    print(f"  +{offset:4d} ms : {n} order(s)  {verdict:13s} "
          f"| {'; '.join(json.dumps(o)[:95] for o in outs)}")

print("\n--- wire trace of a 0 ms-offset run ---")
n, outs = one_run(0, verbose=True)
print(f"\nbroker received {n} live buy order(s) against a cap of {CAP}")
for o in outs:
    print("   ", json.dumps(o))
srv.shutdown()
