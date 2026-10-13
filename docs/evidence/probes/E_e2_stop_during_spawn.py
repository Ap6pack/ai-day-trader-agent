"""STOP arriving while START is inside _spawn(): wanted=False with a live child."""
import os, sys, subprocess, threading, time, signal, collections
from pathlib import Path
from trader.supervisor import NewsbotProcess
SCRATCH = Path("docs/evidence/probes")
CHILD = [sys.executable, "-c", "import time; time.sleep(20)"]
PIDS, L = [], threading.Lock()
def tp(*a, **kw):
    p = subprocess.Popen(*a, **kw)
    with L: PIDS.append(p.pid)
    return p
def alive(pid):
    try: return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0] != "Z"
    except OSError: return False
tally = collections.Counter()
for i in range(60):
    before = len(PIDS)
    q = NewsbotProcess(log_path=SCRATCH / f"E_sd{i}.log", command=CHILD, popen=tp)
    res = {}
    def later():
        time.sleep(0.0004)      # ~0.4 ms: inside the ~0.6 ms _spawn()
        res["s"] = q.stop()
    t = threading.Thread(target=later)
    t.start(); q.start(); t.join()
    spawned = PIDS[before:]
    live = [p for p in spawned if alive(p)]
    tally[(f"stop_said_running={res['s']['running']}", f"wanted={q.wanted}",
           f"handle={'set' if q.proc else 'None'}", f"live={len(live)}",
           f"status_running={q.status()['running']}", f"check_supervises={q.wanted}")] += 1
    for p in spawned:
        try: os.kill(p, signal.SIGKILL)
        except OSError: pass
for k, v in tally.most_common():
    print(f"{v:3d}x  " + "  ".join(k))
for p in PIDS:
    try: os.kill(p, signal.SIGKILL)
    except OSError: pass
for p in PIDS:
    try: os.waitpid(p, 0)
    except OSError: pass
print("[cleanup] pids:", len(PIDS))
