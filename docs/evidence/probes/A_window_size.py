"""How big is the post-OFF exposure in realistic conditions?
AUTO universe (max_symbols=10), PRODUCTION limits (start() overwrites any injected Limits with
Limits.from_env()), ~0.8s of network per symbol, OFF pressed during the first symbol."""
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

DB = Path(__file__).parent / "A_window.db"
if DB.exists():
    DB.unlink()
UNIVERSE = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ"]


def slow_analyze(symbol, capital, held, capital_source):
    time.sleep(0.8)   # one symbol = bars + quote + news + Jev + dividends
    return decision(symbol)


async def main():
    journal, portfolios = Journal(DB), Portfolios(DB)
    portfolios.create("default", 10000)
    broker = FakeBroker(mode="live")                       # REAL MONEY account
    pilot = AutoTrader(journal, portfolios, publish=lambda e: None, analyze=slow_analyze,
                       broker=broker, quotes=lambda s: {}, now=lambda: NOW,
                       universe=lambda: list(UNIVERSE))
    await pilot.start(RunConfig(["AUTO"], 300, "live"))
    print(f"limits actually in force (Limits.from_env, set by start()): {pilot.limits}")
    await asyncio.sleep(0.3)
    log("OFF pressed during the first symbol")
    st = await pilot.stop()
    log(f"stop() returned running={st['running']!r}; orders so far={len(broker.orders)}")
    t_off = time.monotonic()
    while True:
        await asyncio.sleep(0.2)
        if pilot.current_symbol is None and time.monotonic() - t_off > 1:
            break
        if time.monotonic() - t_off > 30:
            break
    log(f"worker thread finished the cycle; LIVE orders placed after OFF = {len(broker.orders)}")
    dump("A1d  how much can leak after OFF with production limits")
    print()
    print(f"LIVE orders after OFF: {[o['symbol'] for o in broker.orders]}")
    print(f"exposure window: {time.monotonic() - t_off:.1f}s of trading after the desk said OFF")


asyncio.run(main())
