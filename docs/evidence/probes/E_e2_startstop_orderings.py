"""Which end-state does a concurrent START+STOP leave? Count both orderings."""
import os, sys, signal, subprocess, threading, collections
from pathlib import Path
from trader.supervisor import NewsbotProcess
SCRATCH = Path("docs/evidence/probes")
CHILD = [sys.executable, "-c", "import time; time.sleep(20)"]
PIDS, L = [], threading.Lock()
def tp(*a, **kw):
    p = subprocess.Popen(*a, **kw); 
    with L: PIDS.append(p.pid)
    return p
def alive(pid):
    try: return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0] not in ("Z",)
    except OSError: return False
tally = collections.Counter()
for i in range(60):
    before = len(PIDS)
    q = NewsbotProcess(log_path=SCRATCH / f"E_ord{i}.log", command=CHILD, popen=tp)
    b = threading.Barrier(2); res = {}
    t1 = threading.Thread(target=lambda: (b.wait(), q.start()))
    t2 = threading.Thread(target=lambda: (b.wait(), res.__setitem__("s", q.stop())))
    t1.start(); t2.start(); t1.join(); t2.join()
    spawned = PIDS[before:]
    live = [p for p in spawned if alive(p)]
    key = (f"stop_said_running={res['s']['running']}", f"wanted={q.wanted}",
           f"handle={'set' if q.proc else 'None'}", f"live_children={len(live)}",
           f"status_running={q.status()['running']}")
    tally[key] += 1
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
