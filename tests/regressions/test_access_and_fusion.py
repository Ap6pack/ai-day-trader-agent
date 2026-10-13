"""Regressions for desk browser access (F), process lifecycle (E) and fusion (I).

Each test asserts the INTENDED SAFE BEHAVIOUR; unfixed findings carry
`xfail(strict=True)` naming the finding id.

The desk tests run the real FastAPI app with `background=False, start_agents=False`,
so no agent, no poll loop and no broker call is started. The autouse `no_network`
fixture makes any outbound HTTP a hard failure.
"""

from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient

from trader import analysis, desk
from trader.supervisor import NewsbotProcess

HOSTILE = "https://evil.example"


@pytest.fixture
def client(journal):
    with TestClient(desk.create_app(journal=journal, background=False,
                                    start_agents=False)) as c:
        yield c


# --- F: a browser on another origin must not drive the desk ----------------


@pytest.mark.xfail(
    strict=True,
    reason="FINDING-F1a: the /ws/desk handler checks only the token query param and "
           "never the Origin header. WebSockets are exempt from the same-origin "
           "policy, so with DESK_TOKEN unset (the default) a page on any origin "
           "completes the handshake and reads the account stream - live mode, equity, "
           "positions and open order ids. Remove when Origin is validated.",
)
def test_f1a_websocket_rejects_a_hostile_origin(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/desk", headers={"Origin": HOSTILE}):
            pass


@pytest.mark.xfail(
    strict=True,
    reason="FINDING-F1b: no TrustedHostMiddleware and no Host check anywhere, so a "
           "DNS-rebinding page that re-resolves its own domain to 127.0.0.1 becomes "
           "same-origin and reaches every endpoint, including order submission on a "
           "live account. Remove when the Host header is validated.",
)
def test_f1b_requests_for_a_foreign_host_are_refused(client):
    response = client.get("/api/desk/config", headers={"Host": "rebind.attacker.tld"})
    assert response.status_code in (400, 421), (
        f"served {response.status_code} to Host: rebind.attacker.tld"
    )


@pytest.mark.xfail(
    strict=True,
    reason="FINDING-F1c: POST /api/desk/orders/{id}/cancel takes no body, so it is a "
           "CORS *simple request* - no preflight, and the side effect lands even "
           "though the browser hides the response. Chained with F1a (which leaks open "
           "order ids) this cancels a protective stop-loss from a hostile page with no "
           "rebinding at all. Remove when state-changing routes require a header or "
           "token a cross-origin simple request cannot set.",
)
def test_f1c_bodyless_cross_origin_cancel_is_refused(client, monkeypatch):
    reached = []
    monkeypatch.setattr(desk.alpaca, "cancel_order", lambda oid: reached.append(oid))

    client.post("/api/desk/orders/9b1f2c44-dead-beef/cancel",
                headers={"Origin": HOSTILE})

    assert reached == [], (
        f"a cross-origin bodyless POST cancelled order {reached[0]!r}"
    )


def test_f1d_confirm_live_is_a_human_confirmation_not_an_access_control(client):
    """Not a defect to fix here - a property to state plainly so nobody mistakes
    the LIVE prompt for a security boundary. confirm_live is a request field any
    programmatic caller sets; only alpaca.check_live_buy is a real guard, and it
    caps buys only, never sells."""
    fields = desk.OrderRequest.model_fields
    assert "confirm_live" in fields and fields["confirm_live"].default is False


# --- E: one news bot, and a restart loop that does not give up -------------


@pytest.mark.xfail(
    strict=True,
    reason="FINDING-E1: NewsbotProcess.start() is an unsynchronised check-then-spawn "
           "and _spawn assigns self.proc only after Popen returns, so two concurrent "
           "starts launch two processes and the handle to the first is overwritten. "
           "stop() then kills one and reports stopped while the orphan keeps running - "
           "and Alpaca allows one news stream per account. Remove when the lifecycle "
           "is serialised under a lock.",
)
def test_e1_concurrent_starts_launch_one_process(tmp_path):
    import threading

    spawned = []

    def recording_popen(command, **kwargs):
        proc = __import__("subprocess").Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        spawned.append(proc)
        return proc

    supervisor = NewsbotProcess(log_path=tmp_path / "newsbot.log",
                                command=[sys.executable, "-c", "pass"],
                                popen=recording_popen)
    threads = [threading.Thread(target=supervisor.start) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    supervisor.stop()

    alive = [p for p in spawned if p.poll() is None]
    for p in alive:                                   # never leak a child process
        p.kill()
        p.wait(timeout=5)

    assert len(spawned) == 1, f"{len(spawned)} news bot processes were launched"
    assert alive == [], f"{len(alive)} process(es) survived stop()"


@pytest.mark.xfail(
    strict=True,
    reason="FINDING-E2: check() clears self._retry_at BEFORE calling _spawn(), so a "
           "spawn that raises (Popen OSError, unwritable log) leaves wanted=True, "
           "proc=None, _retry_at=None - a state neither branch of check() can ever "
           "act on again. The supervisor is permanently stopped with error=None and "
           "the UI shows no fault. Remove when a failed respawn reschedules.",
)
def test_e2_a_failed_respawn_schedules_another_attempt(tmp_path):
    from datetime import datetime, timedelta, timezone

    clock = {"t": datetime(2026, 9, 28, 14, 0, tzinfo=timezone.utc)}
    attempts = {"n": 0}

    def failing_popen(command, **kwargs):
        attempts["n"] += 1
        raise OSError(24, "Too many open files")

    supervisor = NewsbotProcess(log_path=tmp_path / "newsbot.log",
                                command=[sys.executable, "-c", "pass"],
                                popen=failing_popen, now=lambda: clock["t"])
    supervisor.wanted = True
    supervisor._retry_at = clock["t"]

    for _ in range(5):                                # five supervision ticks
        try:
            supervisor.check()
        except OSError:
            pass                                      # watch() swallows it too
        clock["t"] += timedelta(minutes=10)

    assert attempts["n"] > 1, (
        f"only {attempts['n']} spawn attempt after a failure; the supervisor is "
        f"stuck with wanted={supervisor.wanted}, retry_at={supervisor._retry_at}, "
        f"error={supervisor.error}"
    )


# --- I1: agreement must not erase a conflicting signal ---------------------


@pytest.mark.xfail(
    strict=True,
    reason="FINDING-I1: fuse() uses `elif confirming:`, so the conflicting-signal "
           "penalty branch is unreachable whenever any strategy agrees. A maximally "
           "bearish news signal leaves confidence unchanged - it still goes UP by the "
           "confirmation bonus. Asserts only the SIGN (a conflict must not raise "
           "confidence), not a particular curve, because the position_size double "
           "application of confidence is a separate open decision. Remove when "
           "conflicts are applied additively alongside confirmations.",
)
def test_i1_a_conflicting_signal_never_raises_confidence():
    def sig(direction, strength):
        return {"signal": direction, "strength": strength, "reason": direction}

    with_conflict = analysis.fuse({
        "technical": sig("BUY", 0.8),
        "dividend": sig("BUY", 0.5),
        "sentiment": sig("SELL", 1.0),        # maximally bearish news
    })
    without_conflict = analysis.fuse({
        "technical": sig("BUY", 0.8),
        "dividend": sig("BUY", 0.5),
        "sentiment": sig("HOLD", 0.0),        # same, but news is neutral
    })

    assert with_conflict["conflicting"] == 1          # it is counted...
    assert with_conflict["confidence"] < without_conflict["confidence"], (
        f"confidence with a maximally bearish news signal ({with_conflict['confidence']}) "
        f"is not below confidence with neutral news ({without_conflict['confidence']}); "
        "the conflict is counted and reported but never applied"
    )


def test_conflict_penalty_does_apply_when_nothing_confirms():
    """The penalty branch is reachable - only when confirming == 0. Pinned so a
    fix for I1 is visibly a change in the confirming case alone."""
    def sig(direction, strength):
        return {"signal": direction, "strength": strength, "reason": direction}

    fused = analysis.fuse({
        "technical": sig("HOLD", 0.0),
        "dividend": sig("BUY", 0.5),
        "sentiment": sig("SELL", 1.0),
    })
    assert fused["confirming"] == 0 and fused["conflicting"] == 1
    assert fused["confidence"] == pytest.approx(0.35)   # 0.5 - 0.15
