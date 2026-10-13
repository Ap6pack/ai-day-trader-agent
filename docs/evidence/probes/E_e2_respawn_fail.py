"""E2: a failed respawn clears _retry_at before _spawn() runs -> no further retry ever."""
import asyncio, logging, os, stat, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trader.supervisor import NewsbotProcess

SCRATCH = Path("docs/evidence/probes")
logging.basicConfig(level=logging.INFO, format="LOG %(levelname)s %(name)s: %(message)s")

CLOCK = [datetime(2026, 10, 13, 14, 0, 0, tzinfo=timezone.utc)]
def now(): return CLOCK[0]
def advance(sec): CLOCK[0] += timedelta(seconds=sec)


class FakeProc:
    """A child that has already exited with the given code."""
    pid = 4242
    def __init__(self, code=1): self.code = code
    def poll(self): return self.code
    def terminate(self): pass
    def wait(self, timeout=None): return self.code


def section(t): print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


# ---------------------------------------------------------------- scenario 1
section("1. supervised crash -> retry scheduled -> Popen raises OSError at retry")
calls = []
def popen_fail(cmd, **kw):
    calls.append(cmd)
    raise OSError(24, "Too many open files")   # a real, ordinary Popen failure

p = NewsbotProcess(log_path=SCRATCH / "E_e2_a.log", command=["x"], popen=popen_fail, now=now)
p.wanted = True
p.proc = FakeProc(code=1)        # the bot crashed (exit 1, e.g. StreamError)
p.started_at = now() - timedelta(seconds=30)

p.check()                        # notices the exit, schedules the restart
print("after exit noticed : wanted=%s proc=%s _retry_at=%s backoff=%s" % (p.wanted, p.proc, p._retry_at, p.backoff))
print("status()           :", {k: p.status()[k] for k in ("managed", "running", "restart_at", "error", "last_exit")})

advance(10)                      # backoff elapsed
try:
    p.check()                    # <- the retry: _retry_at=None, restarts+=1, _spawn() raises
except Exception as exc:
    print("check() raised     :", type(exc).__name__, exc)
print("after failed respawn: wanted=%s proc=%s _retry_at=%s restarts=%s error=%r" %
      (p.wanted, p.proc, p._retry_at, p.restarts, p.error))
print("status()           :", {k: p.status()[k] for k in ("managed", "running", "restart_at", "error", "last_exit")})

for minutes in (1, 5, 60, 60 * 8):
    advance(minutes * 60)
    p.check()
print("spawn attempts after 8+ more hours of check() calls:", len(calls), "(1 = the single failed one)")
print("state still         : wanted=%s running=%s _retry_at=%s error=%r" % (p.wanted, p.running, p._retry_at, p.error))


# ---------------------------------------------------------------- scenario 2
section("2. the same thing under the real watch() loop (what the desk actually runs)")
calls2 = []
def popen_fail2(cmd, **kw):
    calls2.append(cmd)
    raise OSError(24, "Too many open files")

q = NewsbotProcess(log_path=SCRATCH / "E_e2_b.log", command=["x"], popen=popen_fail2, now=now)
q.wanted = True
q.proc = FakeProc(code=1)
q.started_at = now()

async def main():
    task = asyncio.create_task(q.watch(interval=0.01))
    for _ in range(40):          # ~0.4 s of real supervision, 40 check() ticks
        advance(30)              # ...covering 20 simulated minutes
        await asyncio.sleep(0.01)
    task.cancel()
    try: await task
    except asyncio.CancelledError: pass

asyncio.run(main())
print("watch() ran 40 check() ticks over 20 simulated minutes")
print("total spawn attempts:", len(calls2))
print("final status():", q.status())


# ---------------------------------------------------------------- scenario 3
section("3. same failure from an unwritable log directory (no popen involved)")
baddir = SCRATCH / "E_e2_readonly"
baddir.mkdir(exist_ok=True)
os.chmod(baddir, 0o500)
spawned3 = []
r = NewsbotProcess(log_path=baddir / "newsbot.log", command=["x"],
                   popen=lambda *a, **k: spawned3.append(a) or FakeProc(None), now=now)
r.wanted = True
r.proc = FakeProc(code=1)
r.started_at = now()
r.check()
print("retry scheduled at:", r._retry_at)
advance(10)
try:
    r.check()
except Exception as exc:
    print("check() raised    :", type(exc).__name__, exc)
print("after failure     : wanted=%s proc=%s _retry_at=%s error=%r" % (r.wanted, r.proc, r._retry_at, r.error))
advance(3600)
r.check()
print("one hour later, further spawn attempts:", len(spawned3))
os.chmod(baddir, 0o700)


# ---------------------------------------------------------------- scenario 4
section("4. start() itself failing (manual START with a broken spawn)")
s = NewsbotProcess(log_path=SCRATCH / "E_e2_c.log", command=["x"], popen=popen_fail, now=now)
try:
    s.start()
except Exception as exc:
    print("start() raised    :", type(exc).__name__, exc)
print("state             : wanted=%s running=%s _retry_at=%s error=%r" % (s.wanted, s.running, s._retry_at, s.error))
advance(3600)
s.check()
print("status() an hour later:", {k: s.status()[k] for k in ("managed", "running", "restart_at", "error")})
