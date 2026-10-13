"""A1 part 4: the desk's own shutdown (lifespan: `await pilot.stop(announce=False)`) does not
stop the worker either - the order lands after the event loop is closed, during interpreter exit."""
from __future__ import annotations

import asyncio
import atexit
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from A_common import NOW, FakeBroker, decision, log  # noqa: E402

from trader.autotrader import AutoTrader, Limits, RunConfig  # noqa: E402
from trader.journal import Journal  # noqa: E402
from trader.portfolios import Portfolios  # noqa: E402

DB = Path(__file__).parent / "A_shutdown.db"
if DB.exists():
    DB.unlink()
LIMITS = Limits(min_confidence=0.5, max_trades_per_day=50, cooldown_minutes=0, min_price=5,
                entry_cutoff_minutes=15, flatten_minutes=10)
broker = FakeBroker(mode="paper")


def slow_analyze(symbol, capital, held, capital_source):
    print(f"  [worker] analyze({symbol}) running 2.0s")
    time.sleep(2.0)
    return decision(symbol)


async def main():
    journal, portfolios = Journal(DB), Portfolios(DB)
    portfolios.create("default", 10000)
    pilot = AutoTrader(journal, portfolios, publish=lambda e: None, analyze=slow_analyze,
                       broker=broker, quotes=lambda s: {}, limits=LIMITS, now=lambda: NOW)
    await pilot.start(RunConfig(["AAA"], 300, "paper"))
    await asyncio.sleep(0.4)
    print("[main] desk lifespan shutdown: await pilot.stop(announce=False)")
    await pilot.stop(announce=False)
    print(f"[main] stop() returned, running={pilot.status()['running']}, orders={len(broker.orders)}")


atexit.register(lambda: print(f"[atexit] orders submitted by the end of the process: "
                              f"{len(broker.orders)} -> "
                              f"{[(o['side'], o['qty'], o['symbol']) for o in broker.orders]}"))
t = time.monotonic()
asyncio.run(main())
print(f"[main] asyncio.run() returned at +{time.monotonic()-t:.3f}s, event loop closed; "
      f"orders={len(broker.orders)}; falling off the end of the program now")
