"""C1a, realistic actors: the REAL NewsBot.handle() and the REAL AutoTrader.act(),
each holding its own lock, racing in one process against the same live cap.

Shows the two locks are different objects guarding different critical sections,
so neither excludes the other.
"""
import os
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

os.environ["ALPACA_TRADING_BASE_URL"] = "https://api.alpaca.markets"
os.environ["ALPACA_LIVE_TRADING"] = "true"
os.environ["ALPACA_LIVE_MAX_ORDERS_PER_DAY"] = "1"
os.environ["ALPACA_LIVE_MAX_ORDER_USD"] = "500"
os.environ["ALPACA_LIVE_MAX_DAILY_LOSS_USD"] = "200"
os.environ["APCA_API_KEY_ID"] = "stub"
os.environ["APCA_API_SECRET_KEY"] = "stub"

SCRATCH = "docs/evidence/probes"
sys.path.insert(0, SCRATCH)
from C_fake_alpaca import FakeAlpaca  # noqa: E402

from trader import alpaca, newsbot  # noqa: E402
from trader.autotrader import AutoTrader, Limits, RunConfig  # noqa: E402
from trader.journal import Journal  # noqa: E402
from trader.portfolios import Portfolios  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="C_agents_"))
NOW = datetime(2026, 10, 13, 15, 0, tzinfo=timezone.utc)  # 11:00 ET, market open

fake = FakeAlpaca(price=10.0)
alpaca._request = fake.request  # the single transport both agents share

# --- the news bot: its own process in production, its own Journal, its own _order_lock
bot = newsbot.NewsBot(
    settings=newsbot.Settings(execution="live", symbols=frozenset({"ACME"}), order_usd=100.0,
                              min_price=1.0),
    journal=Journal(tmp / "newsbot.db"),
    judge=lambda sym, art: {"relevance": 0.95, "materiality": 0.8, "p_bullish": 0.9,
                            "p_bearish": 0.02, "event_type": "product", "direction": 0.9},
    broker=alpaca,
    now=lambda: NOW,
)

# --- the desk autopilot: a task inside the desk process, its own Journal, its own _trade_lock
pilot = AutoTrader(
    journal=Journal(tmp / "desk.db"),
    portfolios=Portfolios(tmp / "pf.json"),
    broker=alpaca,
    limits=Limits(min_confidence=0.5, max_trades_per_day=5, cooldown_minutes=0,
                  min_price=1.0, entry_cutoff_minutes=15),
    now=lambda: NOW,
)
pilot_cfg = RunConfig(symbols=["NVDA"], mode="live").validate()
decision = {"symbol": "NVDA", "recommendation": "BUY", "quantity": 10, "confidence": 0.8,
            "current_price": 10.0, "primary_strategy": "technical", "reason": "net indicators",
            "risk_parameters": {"stop_loss": 9.6, "take_profit": 10.6}}

print("NewsBot._order_lock   id =", hex(id(bot._order_lock)))
print("AutoTrader._trade_lock id =", hex(id(pilot._trade_lock)))
print("same object?", bot._order_lock is pilot._trade_lock)
print("cap =", alpaca.LiveLimits.from_env().max_orders_per_day, "live buy(s) per day\n")

out = {}
barrier = threading.Barrier(2)


def run_newsbot():
    barrier.wait()
    out["newsbot"] = bot.handle("ACME", {"title": "Acme wins $2B contract", "description": "",
                                         "publishedAt": NOW.isoformat().replace("+00:00", "Z"),
                                         "symbols": ["ACME"]},
                                received_at=NOW, news_id="news-1")


def run_autopilot():
    barrier.wait()
    out["autopilot"] = pilot.act(decision, pilot_cfg)


threads = [threading.Thread(target=run_newsbot, name="newsbot-proc"),
           threading.Thread(target=run_autopilot, name="desk-autopilot")]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("wire trace:")
for line in fake.log:
    print("  ", line)
print()
nb = out["newsbot"]
print("newsbot  result:", nb.get("kind"), "order_id=", nb.get("order_id"), "blocked=", nb.get("blocked"))
print("autopilot result:", out["autopilot"])
print(f"\nbroker received {len(fake.orders)} LIVE buy order(s) against a cap of 1:")
for o in fake.orders:
    print(f"   {o['id']} {o['side']} {o['qty']} {o['symbol']}  client_order_id={o['client_order_id']}")
print("\nVERDICT:", "CAP BREACHED" if len(fake.orders) > 1 else "cap held")
