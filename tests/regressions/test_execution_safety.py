"""Regressions for the execution-path findings: A (stop/start), B (flatten), C (caps).

Each test asserts the INTENDED SAFE BEHAVIOUR. Unfixed findings carry
`xfail(strict=True)`, so the moment the behaviour is corrected the test reports
XPASS and the build goes red until the marker is removed. Remove the marker in
the same commit as the fix.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from trader import alpaca
from trader.autotrader import AutoTrader, Limits, RunConfig
from .conftest import NOW, FakeBroker, decision, wait_for

LIMITS = Limits(min_confidence=0.5, max_trades_per_day=5, cooldown_minutes=0,
                min_price=5, entry_cutoff_minutes=15, flatten_minutes=10)


def _pilot(journal, portfolios, broker, analyze):
    return AutoTrader(journal, portfolios, analyze=analyze, broker=broker,
                      limits=LIMITS, now=lambda: NOW,
                      quotes=lambda syms: {s: {"last": 100.0} for s in syms},
                      universe=lambda: ["NVDA"])


# --- A1: the autopilot must not trade after it is switched OFF --------------


@pytest.mark.xfail(
    strict=True,
    reason="FINDING-A1: stop() cancels the asyncio task, but run_cycle is running in "
           "an asyncio.to_thread worker that cancellation cannot reach. The worker "
           "finishes the cycle and keeps submitting orders after status() reports "
           "running=False. Remove this marker when act() checks for cancellation "
           "immediately before submitting.",
)
def test_a1_no_order_is_placed_after_stop(journal, portfolios):
    broker = FakeBroker(mode="paper")
    entered, release = threading.Event(), threading.Event()

    def slow_analyze(symbol, capital, held, capital_source=None):
        entered.set()
        release.wait(timeout=30)      # the worker is mid-cycle while OFF is pressed
        return decision(symbol=symbol)

    pilot = _pilot(journal, portfolios, broker, slow_analyze)

    async def drive():
        await pilot.start(RunConfig(["AAA"], 60, "paper", "default"))
        # Yield to the loop so the _loop task can reach run_cycle. A blocking
        # Event.wait() here would stall the event loop and deadlock the test.
        for _ in range(200):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set(), "the cycle never started"

        await pilot.stop()
        # Observed inside the loop: asyncio.run's teardown waits for the
        # executor, so anything measured after it would already include the
        # orphaned worker's output.
        at_stop = (pilot.status()["running"], len(broker.orders))
        release.set()                           # the orphaned worker resumes
        placed = await asyncio.to_thread(broker.submitted.wait, 1.0)
        return at_stop, placed

    (running, orders_at_stop), placed = asyncio.run(drive())

    assert running is False and orders_at_stop == 0
    assert not placed, (
        f"autopilot submitted {[o['symbol'] for o in broker.orders]} after stop() "
        "returned and status() reported running=False"
    )


@pytest.mark.xfail(
    strict=True,
    reason="FINDING-A1b: start() awaits stop() before assigning self._task, so two "
           "concurrent starts leave one _loop task orphaned - still trading, no "
           "longer referenced, so no later stop() can cancel it. Remove when start() "
           "serialises its lifecycle.",
)
def test_a1b_concurrent_starts_leave_no_orphan_task(journal, portfolios):
    """After stop(), no autopilot loop may still be alive.

    The orphan needs a task to already exist, so start()'s `await self.stop()`
    genuinely suspends: the second start then sees self._task is None, installs
    its own task, and the first start overwrites the handle on resume. The
    stranded loop is asleep in its interval rather than visibly trading, so the
    assertion is on the surviving task, not on a cycle count.
    """
    broker = FakeBroker(mode="paper")
    pilot = _pilot(journal, portfolios, broker,
                   lambda *a, **k: decision(rec="HOLD", qty=0))

    def live_loops():
        return [t for t in asyncio.all_tasks()
                if t is not asyncio.current_task() and not t.done()
                and t.get_coro().__qualname__ == "AutoTrader._loop"]

    async def drive():
        await pilot.start(RunConfig(["AAA"], 60, "signals", "default"))
        await asyncio.sleep(0.05)                   # a task now exists to await
        await asyncio.gather(
            pilot.start(RunConfig(["BBB"], 60, "signals", "default")),
            pilot.start(RunConfig(["CCC"], 60, "signals", "default")),
        )
        await asyncio.sleep(0.05)
        await pilot.stop()
        await asyncio.sleep(0.05)
        survivors = live_loops()
        for task in survivors:                      # never leak into other tests
            task.cancel()
        return survivors

    survivors = asyncio.run(drive())
    assert pilot.status()["running"] is False

    assert survivors == [], (
        f"{len(survivors)} AutoTrader._loop task(s) still running after stop(), "
        "with status() reporting running=False; nothing holds a reference, so no "
        "later stop() can reach them"
    )


# --- B: flattening must not double-sell or strand a working entry -----------


@pytest.mark.xfail(
    strict=True,
    reason="FINDING-B1: flatten_owned reads owned_positions once, swallows cancel "
           "failures with `except Exception: pass`, then sells the ORIGINAL quantity. "
           "If a bracket exit fills between the read and the cancel, the position is "
           "already flat and the market sell duplicates the exit - selling the "
           "operator's own shares, or opening a short. Remove when flatten "
           "recomputes the remaining quantity after reconciling cancellations.",
)
def test_b1_flatten_does_not_sell_an_already_exited_position(monkeypatch):
    """The stop-loss leg fills in the window between the read and the cancel."""
    state = {"leg_filled": False}

    def fake_request(method, url, **kwargs):
        if method == "GET" and "/v2/orders" in url:
            return [{
                "id": "parent", "symbol": "NVDA", "side": "buy", "status": "filled",
                "filled_qty": "5", "client_order_id": "newsbot-abc",
                "legs": [{"id": "sl", "symbol": "NVDA", "side": "sell",
                          "status": "filled" if state["leg_filled"] else "held",
                          "filled_qty": "5" if state["leg_filled"] else "0"}],
            }]
        if method == "DELETE" and "/v2/orders/" in url:
            state["leg_filled"] = True        # the leg had already filled
            raise alpaca.requests.HTTPError("422 order is not cancelable")
        if method == "POST" and url.endswith("/v2/orders"):
            sent.append(kwargs.get("json") or {})
            return {"id": "sell-1", "status": "accepted"}
        return {}

    sent: list[dict] = []
    monkeypatch.setattr(alpaca, "_request", fake_request)
    monkeypatch.delenv("ALPACA_TRADING_BASE_URL", raising=False)

    alpaca.flatten_owned("newsbot-", wait_seconds=0)

    assert sent == [], (
        f"flatten sold {sent[0]['qty'] if sent else 0} shares after the bracket exit "
        "had already closed the position"
    )


@pytest.mark.xfail(
    strict=True,
    reason="FINDING-B2: owned_positions only records an order id when side == 'sell', "
           "so an unfilled BUY parent is never cancelled; a symbol with no fill and no "
           "sell legs is dropped entirely by the qty/open-ids filter. flatten reports "
           "success while a working entry can still fill. Remove when flatten cancels "
           "pending entries first.",
)
def test_b2_flatten_cancels_a_working_buy_entry(monkeypatch):
    cancelled: list[str] = []

    def fake_request(method, url, **kwargs):
        if method == "GET" and "/v2/orders" in url:
            return [{"id": "entry-1", "symbol": "NVDA", "side": "buy", "status": "new",
                     "filled_qty": "0", "client_order_id": "newsbot-xyz", "legs": []}]
        if method == "DELETE" and "/v2/orders/" in url:
            cancelled.append(url.rsplit("/", 1)[-1])
            return {}
        return {}

    monkeypatch.setattr(alpaca, "_request", fake_request)
    monkeypatch.delenv("ALPACA_TRADING_BASE_URL", raising=False)

    alpaca.flatten_owned("newsbot-", wait_seconds=0)

    assert "entry-1" in cancelled, (
        "the unfilled buy entry was left working; it can still fill after the "
        "agent considers the session flat"
    )


@pytest.mark.xfail(
    strict=True,
    reason="FINDING-B3: flatten_owned's retry loop builds a new client_order_id on "
           "every attempt, so a 5xx returned AFTER the broker accepted the sell makes "
           "the retry a second, distinct order. alpaca.client_order_id(key) already "
           "exists for exactly this. Remove when the retry reuses one idempotency key.",
)
def test_b3_flatten_retry_reuses_one_idempotency_key(monkeypatch):
    sent: list[dict] = []
    calls = {"n": 0}

    def fake_request(method, url, **kwargs):
        if method == "GET" and "/v2/orders" in url:
            return [{"id": "p", "symbol": "NVDA", "side": "buy", "status": "filled",
                     "filled_qty": "5", "client_order_id": "newsbot-abc", "legs": []}]
        if method == "POST" and url.endswith("/v2/orders"):
            sent.append(kwargs.get("json") or {})
            calls["n"] += 1
            if calls["n"] == 1:
                raise alpaca.requests.HTTPError("502 Bad Gateway")  # accepted, then lost
            return {"id": "sell-2", "status": "accepted"}
        return {}

    monkeypatch.setattr(alpaca, "_request", fake_request)
    monkeypatch.delenv("ALPACA_TRADING_BASE_URL", raising=False)

    alpaca.flatten_owned("newsbot-", wait_seconds=1.0)

    ids = [o.get("client_order_id") for o in sent]
    assert len(set(ids)) == 1, (
        f"retry used {len(set(ids))} different client_order_ids ({ids}); the broker "
        "cannot dedupe them, so an accepted-then-retried sell executes twice"
    )


# --- C: the account-wide live cap must actually bound the account -----------


@pytest.mark.xfail(
    strict=True,
    reason="FINDING-C1a: check_live_buy counts the day's orders and submit_order sends "
           "them as two separate operations with no shared reservation. Concurrent "
           "callers (news bot process, autopilot thread, desk ticket) each see the same "
           "remaining allowance. Instance-local locks cannot help across processes. "
           "Remove when submission holds a durable cross-process reservation.",
)
def test_c1a_concurrent_live_buys_cannot_exceed_the_daily_cap(monkeypatch):
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets")
    monkeypatch.setenv("ALPACA_LIVE_TRADING", "true")
    monkeypatch.setenv("ALPACA_LIVE_MAX_ORDERS_PER_DAY", "1")
    monkeypatch.setenv("ALPACA_LIVE_MAX_ORDER_USD", "100000")

    accepted: list[dict] = []
    gate = threading.Barrier(2)

    def fake_request(method, url, **kwargs):
        if "/v2/orders" in url and method == "GET":
            # The race window is between reading the day's orders and POSTing.
            # Hold both callers here so each sees the same remaining allowance,
            # which is exactly what two agents in separate processes do.
            snapshot = list(accepted)
            gate.wait(timeout=5)
            return snapshot
        if "/trades/latest" in url:
            return {"trade": {"p": 10.0}}
        if url.endswith("/v2/account"):
            return {"equity": "100000", "last_equity": "100000"}
        if method == "POST" and url.endswith("/v2/orders"):
            accepted.append({"side": "buy", "status": "accepted"})
            return {"id": f"o{len(accepted)}", "status": "accepted"}
        return {}

    monkeypatch.setattr(alpaca, "_request", fake_request)

    def submit():
        try:
            alpaca.submit_order({"symbol": "NVDA", "qty": "1", "side": "buy",
                                 "type": "market", "time_in_force": "day"})
        except alpaca.LiveLimitError:
            pass

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(accepted) <= 1, (
        f"{len(accepted)} live buys were accepted against "
        "ALPACA_LIVE_MAX_ORDERS_PER_DAY=1"
    )


@pytest.mark.xfail(
    strict=True,
    reason="FINDING-C1b: check_live_buy excludes status 'canceled' from the daily buy "
           "count, but a buy cancelled AFTER a partial fill bought real shares with "
           "real money. It is counted the same as a rejected order that never reached "
           "the market. Remove when the count keys off filled_qty, not status alone.",
)
def test_c1b_a_partially_filled_then_cancelled_buy_still_counts(monkeypatch):
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets")
    monkeypatch.setenv("ALPACA_LIVE_TRADING", "true")
    monkeypatch.setenv("ALPACA_LIVE_MAX_ORDERS_PER_DAY", "1")
    monkeypatch.setenv("ALPACA_LIVE_MAX_ORDER_USD", "100000")

    def fake_request(method, url, **kwargs):
        if "/v2/orders" in url and method == "GET":
            # 25 shares really were bought before the operator cancelled the rest.
            return [{"side": "buy", "status": "canceled", "filled_qty": "25"}]
        if "/trades/latest" in url:
            return {"trade": {"p": 10.0}}
        if url.endswith("/v2/account"):
            return {"equity": "100000", "last_equity": "100000"}
        return {}

    monkeypatch.setattr(alpaca, "_request", fake_request)

    with pytest.raises(alpaca.LiveLimitError, match="daily buy limit"):
        alpaca.check_live_buy({"symbol": "NVDA", "qty": "1", "side": "buy"})


# --- C: fixed behaviour that must stay fixed --------------------------------


def test_live_sells_are_never_blocked_by_the_buy_limits(monkeypatch):
    """Not a defect - a deliberate property worth pinning: an exit must always
    be possible, so check_live_buy returns early for sells."""
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets")
    monkeypatch.setenv("ALPACA_LIVE_TRADING", "true")
    monkeypatch.setenv("ALPACA_LIVE_MAX_ORDERS_PER_DAY", "0")
    alpaca.check_live_buy({"symbol": "NVDA", "qty": "999", "side": "sell"})
