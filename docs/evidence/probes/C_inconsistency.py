"""C1b corollary: the SAME canceled-after-partial-fill buy is counted as real shares by
owned_positions() (which reads filled_qty) and as nothing by check_live_buy() (which reads status).
"""
import os
import sys

os.environ["ALPACA_TRADING_BASE_URL"] = "https://api.alpaca.markets"
os.environ["ALPACA_LIVE_TRADING"] = "true"
os.environ["ALPACA_LIVE_MAX_ORDERS_PER_DAY"] = "1"
os.environ["APCA_API_KEY_ID"] = "stub"
os.environ["APCA_API_SECRET_KEY"] = "stub"
SCRATCH = "docs/evidence/probes"
sys.path.insert(0, SCRATCH)
import C_fake_alpaca  # noqa: E402
C_fake_alpaca.GET_LATENCY = C_fake_alpaca.POST_LATENCY = 0.0
from C_fake_alpaca import FakeAlpaca  # noqa: E402
from trader import alpaca  # noqa: E402

order = {"id": "o1", "symbol": "ACME", "qty": "50", "filled_qty": "25", "side": "buy",
         "status": "canceled", "filled_avg_price": "10.00",
         "client_order_id": "newsbot-abc", "legs": []}
fake = FakeAlpaca(price=10.0, seed_orders=[order])
alpaca._request = fake.request

print("the order:", {k: order[k] for k in ("side", "qty", "filled_qty", "status")})
print()
print("owned_positions('newsbot-') ->", alpaca.owned_positions("newsbot-"))
print("  (alpaca.py:255 adds filled_qty with no status filter: 25 real shares are held)")
print()
buys = [o for o in [order] if o.get("side") == "buy" and o.get("status") not in ("rejected", "canceled")]
print(f"check_live_buy's filter (alpaca.py:167) counts it as {len(buys)} buy(s) toward the 1/day cap")
try:
    alpaca.submit_order(alpaca.market_order_payload("MSFT", "buy", 10))
    print("  -> next live buy PASSED, while 25 ACME shares bought today are still on the books")
except alpaca.LiveLimitError as exc:
    print("  -> BLOCKED:", exc)
