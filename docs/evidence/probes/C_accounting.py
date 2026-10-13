"""C1b: what check_live_buy() counts, and what it misses.

  1. a buy that PARTIALLY FILLED and was then canceled -> excluded from the cap
  2. a buy that EXPIRED at the close with a partial fill -> still counted (peer did not test this)
  3. >500 orders in a day -> orders(limit=500, direction=desc) silently truncates
  4. the daily-loss stop has the same read-then-act shape
"""
import os
import sys
import threading
from datetime import datetime, timezone

os.environ["ALPACA_TRADING_BASE_URL"] = "https://api.alpaca.markets"
os.environ["ALPACA_LIVE_TRADING"] = "true"
os.environ["ALPACA_LIVE_MAX_ORDER_USD"] = "500"
os.environ["ALPACA_LIVE_MAX_DAILY_LOSS_USD"] = "200"
os.environ["APCA_API_KEY_ID"] = "stub"
os.environ["APCA_API_SECRET_KEY"] = "stub"

SCRATCH = "docs/evidence/probes"
sys.path.insert(0, SCRATCH)
from C_fake_alpaca import FakeAlpaca  # noqa: E402

import C_fake_alpaca  # noqa: E402
C_fake_alpaca.GET_LATENCY = 0.0
C_fake_alpaca.POST_LATENCY = 0.0

from trader import alpaca  # noqa: E402


def buy(oid, status, qty="50", filled="50"):
    return {"id": oid, "symbol": "ACME", "qty": qty, "filled_qty": filled, "side": "buy",
            "status": status, "filled_avg_price": "10.00"}


def check(seed, cap, label):
    os.environ["ALPACA_LIVE_MAX_ORDERS_PER_DAY"] = str(cap)
    fake = FakeAlpaca(price=10.0, seed_orders=seed)
    alpaca._request = fake.request
    payload = alpaca.market_order_payload("MSFT", "buy", 10)
    try:
        alpaca.submit_order(payload)
        verdict = "PASSED -> order sent"
    except alpaca.LiveLimitError as exc:
        verdict = f"BLOCKED -> {exc}"
    counted = [o for o in seed if o.get("side") == "buy" and o.get("status") not in ("rejected", "canceled")]
    print(f"  {label}\n     counted by check_live_buy: {len(counted)}/{cap}   {verdict}")
    return verdict.startswith("PASSED")


print("=== 1. ALPACA_LIVE_MAX_ORDERS_PER_DAY=1, one prior buy in each state ===")
for status, filled in [("filled", "50"), ("partially_filled", "25"), ("accepted", "0"),
                       ("expired", "25"), ("rejected", "0"), ("canceled", "0"), ("canceled", "25")]:
    tag = f'status={status!r:<20} filled_qty={filled:>3}'
    check([buy("o1", status, filled=filled)], 1, tag)

print("\n   -> a buy that filled 25 shares and was then CANCELED is treated exactly like a")
print("      rejected order that never touched the market.\n")

print("=== 2. how far the hole goes: three partial-then-canceled buys, cap of 3 ===")
seed = [buy(f"o{i}", "canceled", qty="50", filled="25") for i in range(1, 4)]
shares = sum(float(o["filled_qty"]) for o in seed)
notional = shares * 10.0
check(seed, 3, f"3 canceled-after-partial buys: {shares:g} shares already bought, ${notional:,.2f} spent")
print(f"     real exposure already taken: ${notional:,.2f}; cap thinks 0/3 used, so 3 more")
print(f"     buys of up to ALPACA_LIVE_MAX_ORDER_USD=$500 can follow -> up to ${notional + 1500:,.2f}\n")

print("=== 3. pagination: more than 500 orders today ===")
old_buys = [buy(f"old-{i}", "filled") for i in range(5)]
newer_sells = [{"id": f"s{i}", "symbol": "ACME", "qty": "1", "filled_qty": "1", "side": "sell",
                "status": "filled"} for i in range(600)]
# Alpaca returns direction=desc (newest first) and caps limit at 500; model that truncation.
book = old_buys + newer_sells                      # oldest -> newest
returned_desc = list(reversed(book))[:500]         # what the API actually hands back
print(f"  orders today: {len(book)}   API returns: {len(returned_desc)} (limit=500, direction=desc)")
print(f"  real buys today: {len(old_buys)}   buys visible in the page: "
      f"{len([o for o in returned_desc if o['side'] == 'buy'])}")
os.environ["ALPACA_LIVE_MAX_ORDERS_PER_DAY"] = "3"
fake = FakeAlpaca(price=10.0, seed_orders=returned_desc)
alpaca._request = fake.request
try:
    alpaca.submit_order(alpaca.market_order_payload("MSFT", "buy", 10))
    print("  cap of 3 with 5 real buys already placed: PASSED -> order sent\n")
except alpaca.LiveLimitError as exc:
    print(f"  BLOCKED -> {exc}\n")
print("  (code: trader/alpaca.py:312  params limit = max(1, min(int(limit), 500)), no paging loop)\n")

print("=== 4. the daily-loss stop has the same read-then-act shape ===")
os.environ["ALPACA_LIVE_MAX_ORDERS_PER_DAY"] = "10"
C_fake_alpaca.GET_LATENCY = 0.06
C_fake_alpaca.POST_LATENCY = 0.04
fake = FakeAlpaca(price=10.0, equity=99_801.0, last_equity=100_000.0)  # day P&L = -$199, stop at -$200
alpaca._request = fake.request
print(f"  equity - last_equity = ${fake.equity - fake.last_equity:,.2f}, "
      f"ALPACA_LIVE_MAX_DAILY_LOSS_USD = $200")
res = {}
bar = threading.Barrier(3)


def one(name, sym):
    bar.wait()
    try:
        alpaca.submit_order(alpaca.market_order_payload(sym, "buy", 50))  # $500 each
        res[name] = "SUBMITTED"
    except Exception as exc:
        res[name] = f"BLOCKED {exc}"


ts = [threading.Thread(target=one, name=n, args=(n, s))
      for n, s in (("newsbot", "ACME"), ("autopilot", "NVDA"), ("desk-ticket", "MSFT"))]
for t in ts:
    t.start()
bar.wait()
for t in ts:
    t.join()
for k, v in res.items():
    print(f"    {k:<12} {v}")
print(f"  {len(fake.orders)} orders accepted at once, $500 of fresh risk each, all on the same")
print("  equity snapshot read before any of them existed.")
