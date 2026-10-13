"""E1b: how wide is the unsynchronised window in start()? Stagger the second caller."""
import os, sys, subprocess, threading, time, signal, statistics
from pathlib import Path
from trader.supervisor import NewsbotProcess

SCRATCH = Path("docs/evidence/probes")
CHILD = [sys.executable, "-c", "import time; time.sleep(30)"]
ALL = []
L = threading.Lock()

def tp(*a, **kw):
    p = subprocess.Popen(*a, **kw)
    with L: ALL.append(p.pid)
    return p

def measure_spawn():
    proc = NewsbotProcess(log_path=SCRATCH / "E_win_base.log", command=CHILD, popen=tp)
    t0 = time.perf_counter()
    proc.start()
    dt = (time.perf_counter() - t0) * 1000
    proc.stop()
    return dt

def race(delay_ms, n=8):
    dups = 0
    for i in range(n):
        before = len(ALL)
        proc = NewsbotProcess(log_path=SCRATCH / f"E_win_{delay_ms}_{i}.log", command=CHILD, popen=tp)
        def second():
            time.sleep(delay_ms / 1000.0)
            proc.start()
        t = threading.Thread(target=second)
        t.start(); proc.start(); t.join()
        spawned = ALL[before:]
        if len(spawned) > 1: dups += 1
        for pid in spawned:
            try: os.kill(pid, signal.SIGKILL)
            except OSError: pass
    return dups, n

try:
    ds = [measure_spawn() for _ in range(5)]
    print("single start() wall time (ms):", [f"{d:.1f}" for d in ds], " median=%.1f" % statistics.median(ds))
    for d in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0):
        dup, n = race(d)
        print(f"second start() {d:>5.1f} ms after the first -> duplicate spawn in {dup}/{n} trials")
finally:
    for pid in ALL:
        try: os.kill(pid, signal.SIGKILL)
        except OSError: pass
    for pid in ALL:
        try: os.waitpid(pid, 0)
        except OSError: pass
    print("[cleanup] pids:", len(ALL))
