"""E1c: end-to-end through the real desk app + uvicorn.

Two PARALLEL `POST /api/desk/newsbot/process {"run": true}` requests from one client.
The NewsbotProcess is injected with a harmless child command (sleep 30); no broker,
no network to Alpaca (ALPACA_TRADING_BASE_URL points at a dead local port).
"""
import os, sys, signal, subprocess, threading, time
from pathlib import Path

SCRATCH = Path("docs/evidence/probes")
os.environ.update({
    "ALPACA_API_KEY": "PLACEHOLDER", "ALPACA_SECRET_KEY": "PLACEHOLDER", "TYPESAFE_API_KEY": "PLACEHOLDER",
    "DESK_DEMO_MODE": "false", "ALPACA_LIVE_TRADING": "false",
    "ALPACA_TRADING_BASE_URL": "http://127.0.0.1:9",
    "TRADER_JOURNAL_PATH": str(SCRATCH / "E_desk_journal.db"),
})

import httpx, uvicorn
from trader.desk import create_app, newsbot_start_blocker
from trader.journal import Journal
from trader.supervisor import NewsbotProcess

CHILD = [sys.executable, "-c", "import time; time.sleep(30)"]
PIDS, L = [], threading.Lock()

def tp(*a, **kw):
    p = subprocess.Popen(*a, **kw)
    with L: PIDS.append(p.pid)
    return p

def alive(pid):
    try: os.kill(pid, 0)
    except OSError: return False
    try: return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0] != "Z"
    except OSError: return False

print("newsbot_start_blocker() =", newsbot_start_blocker())
proc = NewsbotProcess(log_path=SCRATCH / "E_desk_newsbot.log", command=CHILD, popen=tp)
app = create_app(journal=Journal(SCRATCH / "E_desk_journal.db"), background=False,
                 start_agents=False, newsbot_process=proc)
cfg = uvicorn.Config(app, host="127.0.0.1", port=8771, log_level="error")
server = uvicorn.Server(cfg)
t = threading.Thread(target=server.run, daemon=True); t.start()
for _ in range(100):
    if server.started: break
    time.sleep(0.05)

results = {}
def post(i, client):
    r = client.post("http://127.0.0.1:8771/api/desk/newsbot/process", json={"run": True})
    results[i] = (r.status_code, r.json().get("process", {}).get("pid"))

try:
    # two parallel requests, two connections -- what `curl ... & curl ... &` or two
    # dashboard tabs clicking Start at the same moment produce.
    with httpx.Client(timeout=10) as c1, httpx.Client(timeout=10) as c2:
        th = [threading.Thread(target=post, args=(0, c1)), threading.Thread(target=post, args=(1, c2))]
        for x in th: x.start()
        for x in th: x.join()
    print("responses (status, reported pid):", results)
    print("OS processes actually spawned:", PIDS)
    print("supervisor keeps handle on pid:", proc.proc.pid if proc.proc else None)
    print("alive now:", {p: alive(p) for p in PIDS})

    r = httpx.post("http://127.0.0.1:8771/api/desk/newsbot/process", json={"run": False}, timeout=20)
    time.sleep(0.5)
    st = r.json()["process"]
    print('POST {"run": false} ->', {k: st[k] for k in ("running", "pid", "managed", "error")})
    print("alive after STOP:", {p: alive(p) for p in PIDS})
    snap = httpx.get("http://127.0.0.1:8771/api/desk/newsbot", timeout=10).json()["process"]
    print("GET /api/desk/newsbot process:", {k: snap[k] for k in ("running", "pid", "managed", "restart_at", "error")})
finally:
    server.should_exit = True
    t.join(timeout=10)
    for p in PIDS:
        try: os.kill(p, signal.SIGKILL)
        except OSError: pass
    for p in PIDS:
        try: os.waitpid(p, 0)
        except OSError: pass
    print("[cleanup] pids:", PIDS)
