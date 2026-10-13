"""E1d: does the orphan survive the desk's own shutdown (lifespan -> proc.stop())?"""
import os, sys, signal, subprocess, threading, time
from pathlib import Path
SCRATCH = Path("docs/evidence/probes")
os.environ.update({"ALPACA_API_KEY": "d", "ALPACA_SECRET_KEY": "d", "TYPESAFE_API_KEY": "d",
                   "DESK_DEMO_MODE": "false", "ALPACA_TRADING_BASE_URL": "http://127.0.0.1:9",
                   "TRADER_JOURNAL_PATH": str(SCRATCH / "E_desk_journal.db")})
import httpx, uvicorn
from trader.desk import create_app
from trader.journal import Journal
from trader.supervisor import NewsbotProcess

CHILD = [sys.executable, "-c", "import time; time.sleep(60)"]
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

proc = NewsbotProcess(log_path=SCRATCH / "E_desk_newsbot2.log", command=CHILD, popen=tp)
app = create_app(journal=Journal(SCRATCH / "E_desk_journal.db"), background=False,
                 start_agents=False, newsbot_process=proc)
cfg = uvicorn.Config(app, host="127.0.0.1", port=8772, log_level="error")
server = uvicorn.Server(cfg)
t = threading.Thread(target=server.run, daemon=True); t.start()
for _ in range(100):
    if server.started: break
    time.sleep(0.05)
def post(c): c.post("http://127.0.0.1:8772/api/desk/newsbot/process", json={"run": True})
try:
    with httpx.Client() as c1, httpx.Client() as c2:
        th = [threading.Thread(target=post, args=(c1,)), threading.Thread(target=post, args=(c2,))]
        for x in th: x.start()
        for x in th: x.join()
    print("spawned:", PIDS, "handle kept on:", proc.proc.pid)
    print("alive:", {p: alive(p) for p in PIDS})
    print("--- shutting the whole desk down (lifespan runs proc.stop()) ---")
    server.should_exit = True
    t.join(timeout=15)
    time.sleep(1.0)
    print("desk server thread finished:", not t.is_alive())
    print("alive AFTER full desk shutdown:", {p: alive(p) for p in PIDS})
finally:
    for p in PIDS:
        try: os.kill(p, signal.SIGKILL)
        except OSError: pass
    for p in PIDS:
        try: os.waitpid(p, 0)
        except OSError: pass
    print("[cleanup] pids:", PIDS)
