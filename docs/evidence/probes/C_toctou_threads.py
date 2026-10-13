"""C1a: TOCTOU on ALPACA_LIVE_MAX_ORDERS_PER_DAY, in one process, two threads.

Models the real two-actor case inside the desk process:
  thread "autopilot" -> AutoTrader.act()  (holds AutoTrader._trade_lock)
  thread "desk-tkt"  -> POST /api/desk/orders (holds NO lock, runs on Starlette's threadpool)
Both end up in trader.alpaca.submit_order -> check_live_buy -> POST.
"""
import os
import sys
import threading

os.environ["ALPACA_TRADING_BASE_URL"] = "https://api.alpaca.markets"
os.environ["ALPACA_LIVE_TRADING"] = "true"
os.environ["ALPACA_LIVE_MAX_ORDERS_PER_DAY"] = "1"
os.environ["ALPACA_LIVE_MAX_ORDER_USD"] = "500"
os.environ["ALPACA_LIVE_MAX_DAILY_LOSS_USD"] = "200"
os.environ["APCA_API_KEY_ID"] = "stub"
os.environ["APCA_API_SECRET_KEY"] = "stub"

sys.path.insert(0, "docs/evidence/probes")
from C_fake_alpaca import FakeAlpaca  # noqa: E402

from trader import alpaca  # noqa: E402

print("account_mode() =", alpaca.account_mode())
print("limits          =", alpaca.LiveLimits.from_env())
print()

fake = FakeAlpaca(price=10.0)
alpaca._request = fake.request

results = {}
barrier = threading.Barrier(2)


def actor(name, symbol):
    barrier.wait()
    payload = alpaca.market_order_payload(symbol, "buy", 10)  # 10 x $10 = $100, under the $500 cap
    try:
        order = alpaca.submit_order(payload)
        results[name] = f"SUBMITTED {order['id']} {order['side']} {order['qty']} {order['symbol']}"
    except Exception as exc:
        results[name] = f"BLOCKED {type(exc).__name__}: {exc}"


print("=== CONTROL: the two calls run SEQUENTIALLY ===")
t = threading.Thread(target=actor, name="seq-1", args=("seq-1", "AAPL"))
t.start(); barrier.wait(); t.join()
t = threading.Thread(target=actor, name="seq-2", args=("seq-2", "MSFT"))
t.start(); barrier.wait(); t.join()
for k in ("seq-1", "seq-2"):
    print(f"  {k}: {results[k]}")
print(f"  broker received {len(fake.orders)} order(s) -> cap of 1 HELD\n")

# reset
fake.orders.clear()
fake.log.clear()
results.clear()

print("=== RACE: autopilot thread and desk manual ticket thread, concurrently ===")
threads = [
    threading.Thread(target=actor, name="autopilot", args=("autopilot", "AAPL")),
    threading.Thread(target=actor, name="desk-tkt", args=("desk-tkt", "MSFT")),
]
for t in threads:
    t.start()
for t in threads:
    t.join()

print("  wire trace:")
for line in fake.log:
    print("   ", line)
print()
for k, v in results.items():
    print(f"  {k}: {v}")
print(f"\n  ALPACA_LIVE_MAX_ORDERS_PER_DAY=1, broker received {len(fake.orders)} live buy order(s):")
for o in fake.orders:
    print(f"    {o['id']} {o['side']} {o['qty']} {o['symbol']}")
print()
print("VERDICT:", "CAP BREACHED" if len(fake.orders) > 1 else "cap held")
