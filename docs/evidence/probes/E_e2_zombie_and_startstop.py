"""E2b: (i) _spawn failing on an unusable log path, (ii) check() and a process that
exits while wanted=False, (iii) the concurrent START/STOP interleaving."""
import os, sys, signal, subprocess, threading, time
from pathlib import Path
from trader.supervisor import NewsbotProcess

SCRATCH = Path("docs/evidence/probes")
CHILD = [sys.executable, "-c", "import time; time.sleep(30)"]
SHORT = [sys.executable, "-c", "import time; time.sleep(0.4)"]
PIDS, L = [], threading.Lock()
def tp(*a, **kw):
    p = subprocess.Popen(*a, **kw)
    with L: PIDS.append(p.pid)
    return p
def pstate(pid):
    try: return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]
    except OSError: return "gone"

print("=" * 70, "\n(i) _spawn with an unusable log path (log_path is a directory)\n", "=" * 70)
d = SCRATCH / "E_e2_logdir"; d.mkdir(exist_ok=True)
spawns = []
p = NewsbotProcess(log_path=d, command=["x"], popen=lambda *a, **k: spawns.append(1))
class Dead:
    pid = 1; 
    def poll(self): return 1
p.wanted, p.proc = True, Dead()
p.check(); print("retry scheduled:", p._retry_at is not None)
time.sleep(0.01); p._retry_at = p.now()      # make the retry due
try:
    p.check()
except Exception as exc:
    print("check() raised  :", type(exc).__name__, exc)
print("after failure   : wanted=%s _retry_at=%s error=%r" % (p.wanted, p._retry_at, p.error))
p._retry_at = None
for _ in range(5): p.check()
print("further spawn attempts:", len(spawns))

print("\n" + "=" * 70, "\n(ii) a child that exits while wanted=False: zombie?\n", "=" * 70)
z = NewsbotProcess(log_path=SCRATCH / "E_e2_z.log", command=SHORT, popen=tp)
z.start()
pid = z.proc.pid
z.wanted = False                 # the state a STOP racing a START leaves behind
time.sleep(1.0)                  # the child exits on its own
print(f"child {pid} /proc state after it exited:", pstate(pid))
for _ in range(10): z.check()
print("after 10 check() calls (wanted=False -> early return):", pstate(pid))
print("status()['running'] (this poll() is what finally reaps it):", z.status()["running"])
print("state after status():", pstate(pid))

print("\n" + "=" * 70, "\n(iii) concurrent START + STOP: does STOP leave a live child behind?\n", "=" * 70)
bad = 0
TRIALS = 40
for i in range(TRIALS):
    before = len(PIDS)
    q = NewsbotProcess(log_path=SCRATCH / f"E_e2_ss{i}.log", command=CHILD, popen=tp)
    b = threading.Barrier(2)
    res = {}
    def do_start():
        b.wait(); q.start()
    def do_stop():
        b.wait(); res["stop"] = q.stop()
    ts = [threading.Thread(target=do_start), threading.Thread(target=do_stop)]
    for t in ts: t.start()
    for t in ts: t.join()
    spawned = PIDS[before:]
    leftover = [x for x in spawned if pstate(x) not in ("gone", "Z")]
    if res.get("stop", {}).get("running") is False and leftover:
        bad += 1
        if bad == 1:
            print(f"trial {i}: stop() returned running=False / managed={res['stop']['managed']}, "
                  f"but pid(s) {leftover} are still alive; wanted={q.wanted} "
                  f"supervisor.proc={'set' if q.proc else 'None'}")
            print(f"         status() now: running={q.status()['running']} pid={q.status()['pid']}")
            q.check()
            print(f"         after check() (wanted=False -> returns immediately), pid alive:",
                  [pstate(x) for x in leftover])
    for x in spawned:
        try: os.kill(x, signal.SIGKILL)
        except OSError: pass
print(f"trials where a concurrent STOP reported 'stopped' yet left a live child: {bad}/{TRIALS}")

for x in PIDS:
    try: os.kill(x, signal.SIGKILL)
    except OSError: pass
for x in PIDS:
    try: os.waitpid(x, 0)
    except OSError: pass
print("[cleanup] pids:", len(PIDS))
