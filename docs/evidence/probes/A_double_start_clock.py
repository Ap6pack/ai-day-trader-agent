"""Incidental: two near-simultaneous start() calls (two POST /api/desk/autopilot) can leak an
UNTRACKED _loop task that no stop() can ever cancel."""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from A_common import NOW, FakeBroker, decision, dump, log  # noqa: E402

from datetime import timedelta
from trader.autotrader import AutoTrader, Limits, RunConfig  # noqa: E402
from trader.journal import Journal  # noqa: E402
from trader.portfolios import Portfolios  # noqa: E402

DB = Path(__file__).parent / "A_double_start_clock.db"
if DB.exists():
    DB.unlink()
LIMITS = Limits(min_confidence=0.5, max_trades_per_day=50, cooldown_minutes=0, min_price=5,
                entry_cutoff_minutes=15, flatten_minutes=10)


def loops():
    return [t for t in asyncio.all_tasks() if getattr(t.get_coro(), "__qualname__", "").endswith("_loop")]


T_START = time.monotonic()


async def main():
    journal, portfolios = Journal(DB), Portfolios(DB)
    portfolios.create("default", 10000)
    broker = FakeBroker(mode="paper")
    cycles = []

    def analyze(symbol, capital, held, capital_source):
        cycles.append((time.monotonic(), symbol))
        log(f'analyze({symbol}) in a worker thread')
        time.sleep(0.3)
        return decision(symbol, rec="BUY")

    pilot = AutoTrader(journal, portfolios, publish=lambda e: None, analyze=analyze, broker=broker,
                       quotes=lambda s: {}, limits=LIMITS,
                       now=lambda: NOW + timedelta(minutes=time.monotonic() - T_START))

    await pilot.start(RunConfig(["AAA"], 60, "paper"))
    await asyncio.sleep(0.05)                       # a cycle is in flight, so stop() must await
    print(f"after 1 start: live _loop tasks = {len(loops())}, pilot._task alive = {pilot.running}")

    print("two concurrent start() calls (double-clicked panel / two POSTs):")
    await asyncio.gather(pilot.start(RunConfig(["AAA"], 60, "paper")),
                         pilot.start(RunConfig(["BBB"], 60, "paper")))
    live = loops()
    print(f"  live _loop tasks now = {len(live)}; pilot._task is one of them = {pilot._task in live}")
    print(f"  untracked loop tasks = {[t.get_name() for t in live if t is not pilot._task]}")

    log('await pilot.stop()')
    st = await pilot.stop()
    log('stop() returned, running=False')
    await asyncio.sleep(0.1)
    still = loops()
    print(f"after stop(): status running={st['running']!r}; live _loop tasks STILL running = {len(still)}")
    before, orders_before = len(cycles), len(broker.orders)
    t0 = time.monotonic()
    print(f"orders at stop(): {orders_before}. Waiting out the orphan's 60s nap ...")
    await asyncio.sleep(70.0)
    print(f"+70s after stop(): analyze() calls = {len(cycles) - before}, "
          f"NEW broker orders = {len(broker.orders) - orders_before} "
          f"{[(o['side'], o['qty'], o['symbol']) for o in broker.orders[orders_before:]]}")
    print(f"  status()['running'] = {pilot.status()['running']!r}, "
          f"status()['config'] = {pilot.status()['config']}")
    dump('leaked loop task keeps trading after stop()')
    for t in still:
        t.cancel()


asyncio.run(main())
