"""A1 part 2: start() while a cycle is in flight -> the OLD worker keeps trading the OLD config.
Scenario: autopilot is running in LIVE mode; the user switches it to SIGNALS (analyze-only).
Does the old thread keep sending LIVE (real money) orders?"""
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

DB = Path(__file__).parent / "A_start_switch.db"
if DB.exists():
    DB.unlink()

LIMITS = Limits(min_confidence=0.5, max_trades_per_day=50, cooldown_minutes=0, min_price=5,
                entry_cutoff_minutes=15, flatten_minutes=10)


def slow_analyze(symbol, capital, held, capital_source):
    log(f"analyze({symbol}) started (1.0s of network I/O)")
    time.sleep(1.0)
    return decision(symbol)


async def main():
    journal, portfolios = Journal(DB), Portfolios(DB)
    portfolios.create("default", 10000)
    broker = FakeBroker(mode="live")          # the desk is pointed at the LIVE account
    pilot = AutoTrader(journal, portfolios, publish=lambda e: None, analyze=slow_analyze,
                       broker=broker, quotes=lambda s: {}, limits=LIMITS, now=lambda: NOW)

    log("start(mode=live, symbols=[AAA,BBB])")
    await pilot.start(RunConfig(["AAA", "BBB"], 300, "live"))
    await asyncio.sleep(0.4)                  # mid-analysis of AAA, nothing ordered yet
    log(f"user switches the panel to SIGNALS; orders so far={len(broker.orders)}")
    st = await pilot.start(RunConfig(["AAA", "BBB"], 300, "signals"))
    log(f"start(signals) returned: running={st['running']} mode={st['config']['mode']!r}")
    for _ in range(24):
        await asyncio.sleep(0.25)
    log(f"FINAL orders={len(broker.orders)} status.mode={pilot.status()['config']['mode']!r} "
        f"cycles={pilot.cycles}")
    dump("A1b  live -> signals switch while a cycle is in flight")
    print()
    print(f"broker orders placed while the panel says mode=signals: {len(broker.orders)}")
    print(f"  {[ (o['side'], o['qty'], o['symbol']) for o in broker.orders ]}")
    print("journal rows written by the orphaned LIVE worker:")
    for r in journal.events(limit=10):
        if r["kind"] == "autopilot_order":
            print(f"   {r['kind']} {r['detail']} payload_mode="
                  f"{__import__('json').loads(r['payload'])['mode']!r}")


asyncio.run(main())
