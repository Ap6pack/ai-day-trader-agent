"""A1 part 3: same hazard on the pre-close flatten loop?
_loop() also does `await asyncio.to_thread(self.maybe_flatten, config)`.
Real constants used: FLATTEN_CHECK_SECONDS=30, interval 60s (the project minimum)."""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from A_common import NOW, FakeBroker, decision, dump, log  # noqa: E402

from trader import autotrader  # noqa: E402
from trader.autotrader import AutoTrader, Limits, RunConfig  # noqa: E402
from trader.journal import Journal  # noqa: E402
from trader.portfolios import Portfolios  # noqa: E402

DB = Path(__file__).parent / "A_flatten_race.db"
if DB.exists():
    DB.unlink()

# NOW is 2026-09-28T15:00Z; a close 5 minutes out is inside flatten_minutes=10.
CLOSING = {"is_open": True, "next_close": "2026-09-28T15:05:00+00:00"}
LIMITS = Limits(min_confidence=0.5, max_trades_per_day=50, cooldown_minutes=0, min_price=5,
                entry_cutoff_minutes=15, flatten_minutes=10)
entered = threading.Event()


class FlattenBroker(FakeBroker):
    def flatten_owned(self, prefix):
        entered.set()
        return super().flatten_owned(prefix)


def hold_analyze(symbol, capital, held, capital_source):
    return decision(symbol, rec="HOLD")   # keep the cycle quiet; only flatten should trade


async def main():
    journal, portfolios = Journal(DB), Portfolios(DB)
    portfolios.create("default", 10000)
    broker = FlattenBroker(mode="paper", clock=CLOSING, slow_flatten=3.0)
    broker.owned = {"AAA": 7.0}
    pilot = AutoTrader(journal, portfolios, publish=lambda e: None, analyze=hold_analyze,
                       broker=broker, quotes=lambda s: {}, limits=LIMITS, now=lambda: NOW)

    log(f"start(paper, interval=60). FLATTEN_CHECK_SECONDS={autotrader.FLATTEN_CHECK_SECONDS}")
    await pilot.start(RunConfig(["AAA"], 60, "paper"))
    while not entered.is_set():               # wait for the flatten worker thread to start
        await asyncio.sleep(0.05)
    log("maybe_flatten worker is inside broker.flatten_owned()")
    await asyncio.sleep(0.3)
    log("user flips autopilot OFF -> await pilot.stop()")
    st = await pilot.stop()
    log(f"stop() RETURNED running={st['running']!r} orders so far={len(broker.orders)}")
    for _ in range(24):
        await asyncio.sleep(0.25)
    log(f"FINAL orders={len(broker.orders)} flatten_calls={broker.flatten_calls} "
        f"_flattened_for={pilot._flattened_for!r}")
    dump("A1c  pre-close flatten vs. stop()")
    print()
    print(f"sell orders submitted after stop() returned: "
          f"{[(o['side'], o['qty'], o['symbol']) for o in broker.orders]}")


asyncio.run(main())
