from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trader.supervisor import MIN_BACKOFF, NewsbotProcess

T0 = datetime(2026, 9, 28, 13, 0, tzinfo=timezone.utc)


class FakeProc:
    count = 0

    def __init__(self):
        FakeProc.count += 1
        self.pid = 1000 + FakeProc.count
        self.code = None
        self.terminated = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = -15

    def wait(self, timeout=None):
        return self.code

    def kill(self):
        self.code = -9


class Clock:
    def __init__(self):
        self.value = T0

    def __call__(self):
        return self.value


def make(tmp_path):
    procs, changes, clock = [], [], Clock()

    def popen(command, **kwargs):
        assert command[-3:] == ["-m", "trader.newsbot", "run"]
        procs.append(FakeProc())
        return procs[-1]

    sup = NewsbotProcess(log_path=tmp_path / "newsbot.log", popen=popen, on_change=changes.append, now=clock)
    return sup, procs, changes, clock


def test_start_status_and_stop(tmp_path):
    sup, procs, changes, clock = make(tmp_path)
    status = sup.start()
    assert status["running"] and status["managed"] and status["pid"] == procs[0].pid
    sup.start()  # already running: no second process
    assert len(procs) == 1
    status = sup.stop()
    assert procs[0].terminated and not status["running"] and not sup.wanted
    sup.check()  # stopped on purpose: never restarted
    assert len(procs) == 1 and changes


def test_crash_is_restarted_after_a_backoff(tmp_path):
    sup, procs, changes, clock = make(tmp_path)
    sup.start()
    clock.value += timedelta(seconds=30)
    procs[0].code = 1
    sup.check()
    status = sup.status()
    assert not status["running"] and status["last_exit"] == 1 and status["restart_at"]
    sup.check()  # too early
    assert len(procs) == 1
    clock.value += timedelta(seconds=MIN_BACKOFF * 2 + 1)
    sup.check()
    assert len(procs) == 2 and sup.status()["restarts"] == 1 and sup.running


def test_repeated_quick_crashes_back_off_longer(tmp_path):
    sup, procs, changes, clock = make(tmp_path)
    sup.start()
    delays = []
    for _ in range(3):
        procs[-1].code = 1
        sup.check()
        delays.append(sup.backoff)
        clock.value = sup._retry_at
        sup.check()
    assert delays == sorted(delays) and delays[-1] > delays[0]


def test_setup_error_is_reported_not_retried(tmp_path):
    sup, procs, changes, clock = make(tmp_path)
    sup.start()
    sup.log_path.write_text("NEWSBOT_EXECUTION=live is set to trade LIVE but the Alpaca account is PAPER\n")
    procs[0].code = 2
    sup.check()
    clock.value += timedelta(hours=1)
    sup.check()
    status = sup.status()
    assert len(procs) == 1 and not status["running"] and "set to trade LIVE" in status["error"]
    assert changes[-1]["error"] == status["error"]
    sup.start()  # starting again by hand clears the error and retries
    assert len(procs) == 2 and sup.status()["error"] is None
