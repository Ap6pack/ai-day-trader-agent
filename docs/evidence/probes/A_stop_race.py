"""A1: does a broker order get submitted AFTER stop() returns and status()['running'] is False?"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from A_common import NOW, FakeBroker, decision, dump, log  # noqa: E402

from trader.autotrader import AutoTrader, Limits, RunConfig  # noqa: E402
from trader.journal import Journal  # noqa: E402
from trader.portfolios import Portfolios  # noqa: E402

SCRATCH = Path(__file__).parent
DB = SCRATCH / "A_stop_race.db"
if DB.exists():
    DB.unlink()

LIMITS = Limits(min_confidence=0.5, max_trades_per_day=50, cooldown_minutes=0, min_price=5,
                entry_cutoff_minutes=15, flatten_minutes=10)


def slow_analyze(symbol, capital, held, capital_source):
    log(f"analyze({symbol}) started  (simulated 1.0s of network I/O)")
    time.sleep(1.0)
    log(f"analyze({symbol}) returned BUY 5 @ 100")
    return decision(symbol)


async def main():
    journal = Journal(DB)
    portfolios = Portfolios(DB)
    portfolios.create("default", 10000)
    broker = FakeBroker(mode="paper")
    events = []

    def publish(e):
        events.append((time.monotonic(), e))
        log(f"publish event type={e['type']!r} msg={e['message'][:70]!r}")

    pilot = AutoTrader(journal, portfolios, publish=publish, analyze=slow_analyze, broker=broker,
                       quotes=lambda s: {}, limits=LIMITS, now=lambda: NOW)

    log("await pilot.start(paper, [AAA, BBB, CCC], 300s)")
    await pilot.start(RunConfig(["AAA", "BBB", "CCC"], 300, "paper"))
    log(f"start() returned; status running={pilot.status()['running']}")

    await asyncio.sleep(0.4)  # worker thread is mid-analysis of AAA; no order placed yet
    log(f"--- user flips autopilot OFF; orders so far = {len(broker.orders)}")
    log("await pilot.stop()  [the DELETE /api/desk/autopilot handler]")
    st = await pilot.stop()
    log(f"stop() RETURNED. status()['running']={st['running']!r} "
        f"current_symbol={st['current_symbol']!r} task={pilot._task!r}")
    log(f"orders at the moment stop() returned = {len(broker.orders)}")

    for _ in range(40):  # watch what the orphaned worker thread keeps doing
        await asyncio.sleep(0.25)
    log(f"FINAL: broker.orders = {len(broker.orders)}  running={pilot.status()['running']!r} "
        f"cycles={pilot.cycles}")

    dump("A1  stop() vs. the asyncio.to_thread worker")
    print()
    print(f"orders submitted BEFORE stop() returned: 0")
    print(f"orders submitted AFTER  stop() returned: {len(broker.orders)}")
    print(f"symbols ordered after OFF: {[o['symbol'] for o in broker.orders]}")
    print(f"journal autopilot_order rows: {journal.count_events_today('autopilot_order', now=NOW)}")
    print(f"status()['running'] throughout the post-stop window: {pilot.status()['running']!r}")


asyncio.run(main())
