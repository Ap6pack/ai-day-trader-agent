"""Incidental: AutoTrader sizes a live buy from the ANALYSIS price via max_live_qty(price),
but check_live_buy re-prices the same order from a FRESH latest_price(). Any upward drift
between the two turns the order into a hard LiveLimitError, not a resize.
"""
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ["ALPACA_TRADING_BASE_URL"] = "https://api.alpaca.markets"
os.environ["ALPACA_LIVE_TRADING"] = "true"
os.environ["ALPACA_LIVE_MAX_ORDER_USD"] = "500"
os.environ["ALPACA_LIVE_MAX_ORDERS_PER_DAY"] = "10"
os.environ["APCA_API_KEY_ID"] = "stub"
os.environ["APCA_API_SECRET_KEY"] = "stub"
SCRATCH = "docs/evidence/probes"
sys.path.insert(0, SCRATCH)
import C_fake_alpaca  # noqa: E402
C_fake_alpaca.GET_LATENCY = C_fake_alpaca.POST_LATENCY = 0.0
from C_fake_alpaca import FakeAlpaca  # noqa: E402

from trader import alpaca  # noqa: E402
from trader.autotrader import AutoTrader, Limits, RunConfig  # noqa: E402
from trader.journal import Journal  # noqa: E402
from trader.portfolios import Portfolios  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="C_drift_"))
NOW = datetime(2026, 10, 13, 15, 0, tzinfo=timezone.utc)

ANALYSIS_PRICE = 10.00   # price in the analysis result the autopilot acts on
LIVE_PRICE = 10.20       # price when check_live_buy re-fetches, 2% later

fake = FakeAlpaca(price=LIVE_PRICE)
alpaca._request = fake.request

pilot = AutoTrader(journal=Journal(tmp / "d.db"), portfolios=Portfolios(tmp / "pf.json"),
                   broker=alpaca,
                   limits=Limits(min_confidence=0.5, max_trades_per_day=5, cooldown_minutes=0,
                                 min_price=1.0, entry_cutoff_minutes=15),
                   now=lambda: NOW)
cfg = RunConfig(symbols=["NVDA"], mode="live").validate()

print(f"ALPACA_LIVE_MAX_ORDER_USD = $500")
print(f"analysis price ${ANALYSIS_PRICE:.2f} -> max_live_qty = {alpaca.max_live_qty(ANALYSIS_PRICE)} shares")
print(f"price when check_live_buy re-fetches: ${LIVE_PRICE:.2f}")
print(f"  {alpaca.max_live_qty(ANALYSIS_PRICE)} x ${LIVE_PRICE:.2f} = "
      f"${alpaca.max_live_qty(ANALYSIS_PRICE) * LIVE_PRICE:,.2f}\n")

out = pilot.act({"symbol": "NVDA", "recommendation": "BUY", "quantity": 999, "confidence": 0.8,
                 "current_price": ANALYSIS_PRICE, "primary_strategy": "technical",
                 "reason": "net indicators", "risk_parameters": {"stop_loss": 9.6, "take_profit": 10.6}},
                cfg)
print("AutoTrader.act ->", out)
print("orders actually sent:", len(fake.orders))
print("\nThe autopilot does not resize and does not retry: a 2% move between analysis and")
print("submission turns the whole live BUY into an order_failed event.")
