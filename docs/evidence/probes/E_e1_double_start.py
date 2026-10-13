"""E1: two concurrent start() calls on NewsbotProcess -> how many real child processes?

Uses REAL harmless children (python -c 'import time; time.sleep(30)') via the
`command` constructor arg. No network, no broker. Every pid spawned is killed at exit.
Exercises the exact desk path: starlette.concurrency.run_in_threadpool(proc.start).
"""
import os, sys, subprocess, threading, time, signal
from pathlib import Path

import anyio
from starlette.concurrency import run_in_threadpool
from trader.supervisor import NewsbotProcess

SCRATCH = Path("docs/evidence/probes")
CHILD = [sys.executable, "-c", "import time; time.sleep(30)"]

ALL_PIDS = []          # every pid we ever spawn, for cleanup
LOCK = threading.Lock()


def tracking_popen(*a, **kw):
    p = subprocess.Popen(*a, **kw)
    with LOCK:
        ALL_PIDS.append(p.pid)
    return p


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # distinguish zombie from running
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]
    except OSError:
        return False
    return state != "Z"


def cleanup():
    for pid in ALL_PIDS:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    for pid in ALL_PIDS:
        try:
            os.waitpid(pid, 0)
        except OSError:
            pass


def trial(n):
    """Two threads enter start() together, as two concurrent POSTs would."""
    before = len(ALL_PIDS)
    proc = NewsbotProcess(log_path=SCRATCH / f"E_newsbot_trial{n}.log",
                          command=CHILD, popen=tracking_popen)
    barrier = threading.Barrier(2)

    def call():
        barrier.wait()
        proc.start()

    ts = [threading.Thread(target=call) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    spawned = ALL_PIDS[before:]
    return proc, spawned


def main():
    print("python:", sys.version.split()[0])
    dup_trials, spawn_counts = 0, []
    first_dup = None
    for n in range(40):
        proc, spawned = trial(n)
        spawn_counts.append(len(spawned))
        if len(spawned) > 1 and first_dup is None:
            first_dup = (proc, spawned)
        elif len(spawned) > 1:
            dup_trials += 1
        if len(spawned) > 1:
            dup_trials_total = True
        if first_dup is None:
            # tidy up single-process trials immediately
            for pid in spawned:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        if first_dup is not None and n >= 5:
            break
    dups = sum(1 for c in spawn_counts if c > 1)
    print(f"trials run: {len(spawn_counts)}  processes spawned per trial: {spawn_counts}")
    print(f"trials where TWO OS processes were spawned by two concurrent start()s: {dups}")

    if first_dup is None:
        print("NO DUPLICATE OBSERVED in the natural race")
        return

    proc, spawned = first_dup
    print(f"\n--- duplicate trial: pids spawned = {spawned}")
    print(f"supervisor.proc.pid (the only handle it kept) = {proc.proc.pid}")
    orphan = [p for p in spawned if p != proc.proc.pid]
    print(f"orphaned pid(s) with no handle = {orphan}")
    for pid in spawned:
        print(f"  pid {pid} alive before stop(): {alive(pid)}")
    print(f"status() before stop(): running={proc.status()['running']} pid={proc.status()['pid']}")

    st = proc.stop()
    time.sleep(0.5)
    print(f"\nstatus() after stop(): running={st['running']} pid={st['pid']} managed={st['managed']} error={st['error']}")
    for pid in spawned:
        print(f"  pid {pid} alive AFTER stop(): {alive(pid)}")
    print("\nsupervisor reports STOPPED while an orphan child is still alive:",
          (not st["running"]) and any(alive(p) for p in orphan))

    # what does check() do with the survivor?
    proc.check()
    print("after one check() cycle, orphan still alive:", [alive(p) for p in orphan])


def main_threadpool():
    """Same race through starlette's run_in_threadpool, i.e. the literal desk route."""
    print("\n=== via starlette run_in_threadpool (the desk's own call) ===")
    before = len(ALL_PIDS)
    proc = NewsbotProcess(log_path=SCRATCH / "E_newsbot_tp.log", command=CHILD, popen=tracking_popen)

    async def go():
        async with anyio.create_task_group() as tg:
            for _ in range(2):
                tg.start_soon(run_in_threadpool, proc.start)

    anyio.run(go)
    spawned = ALL_PIDS[before:]
    print("pids spawned by two concurrent run_in_threadpool(proc.start):", spawned,
          "-> kept handle:", proc.proc.pid)
    st = proc.stop()
    time.sleep(0.3)
    print("after stop(): running=", st["running"], " alive:", {p: alive(p) for p in spawned})


if __name__ == "__main__":
    try:
        main()
        main_threadpool()
    finally:
        cleanup()
        print("\n[cleanup] killed/reaped pids:", ALL_PIDS)
