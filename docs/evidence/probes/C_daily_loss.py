"""C1b part 4: the daily-loss stop and the per-order notional check have the same
read-then-act shape as the order-count cap."""
import os
import sys
import threading

os.environ["ALPACA_TRADING_BASE_URL"] = "https://api.alpaca.markets"
os.environ["ALPACA_LIVE_TRADING"] = "true"
os.environ["ALPACA_LIVE_MAX_ORDER_USD"] = "500"
os.environ["ALPACA_LIVE_MAX_DAILY_LOSS_USD"] = "200"
os.environ["ALPACA_LIVE_MAX_ORDERS_PER_DAY"] = "10"
os.environ["APCA_API_KEY_ID"] = "stub"
os.environ["APCA_API_SECRET_KEY"] = "stub"

SCRATCH = "docs/evidence/probes"
sys.path.insert(0, SCRATCH)
from C_fake_alpaca import FakeAlpaca  # noqa: E402

from trader import alpaca  # noqa: E402

fake = FakeAlpaca(price=10.0, equity=99_801.0, last_equity=100_000.0)  # day P&L = -$199
alpaca._request = fake.request
print(f"equity - last_equity = ${fake.equity - fake.last_equity:,.2f}; "
      f"ALPACA_LIVE_MAX_DAILY_LOSS_USD = $200; order cap raised to 10 so only the loss stop is in play")

res = {}
bar = threading.Barrier(3)


def one(name, sym):
    bar.wait()
    try:
        alpaca.submit_order(alpaca.market_order_payload(sym, "buy", 50))  # 50 x $10 = $500 each
        res[name] = "SUBMITTED"
    except Exception as exc:
        res[name] = f"BLOCKED {exc}"


ts = [threading.Thread(target=one, name=n, args=(n, s))
      for n, s in (("newsbot", "ACME"), ("autopilot", "NVDA"), ("desk-ticket", "MSFT"))]
for t in ts:
    t.start()
for t in ts:
    t.join()
for k, v in sorted(res.items()):
    print(f"   {k:<12} {v}")
print(f"\n{len(fake.orders)} live buys accepted at once ($500 of fresh risk each), all cleared against")
print("the same equity snapshot read before any of them existed.")
print("\nSequential control (same state, one at a time):")
fake.orders.clear()
for n, s in (("newsbot", "ACME"), ("autopilot", "NVDA"), ("desk-ticket", "MSFT")):
    try:
        alpaca.submit_order(alpaca.market_order_payload(s, "buy", 50))
        print(f"   {n:<12} SUBMITTED  (equity is still stale: account() is a fresh GET each time,")
        print("                but a just-submitted order has not settled into equity yet)")
    except Exception as exc:
        print(f"   {n:<12} BLOCKED {exc}")
