"""ALPACA_LIVE_MAX_ORDER_USD=nan: does a huge REAL-MONEY buy pass alpaca.submit_order()?
No network at all: _request, latest_price, account, orders and account_mode are stubbed."""
import os
from trader import alpaca

os.environ["ALPACA_LIVE_MAX_ORDER_USD"] = "nan"
os.environ["ALPACA_LIVE_MAX_DAILY_LOSS_USD"] = "nan"

sent = []
alpaca.account_mode = lambda: "live"
alpaca.latest_price = lambda s: 250.0
alpaca.account = lambda: {"equity": "5000", "last_equity": "30000"}  # -$25,000 day P&L
alpaca.orders = lambda **kw: []
def fake_request(method, url, **kw):
    sent.append((method, url, kw.get("json")))
    return {"id": "STUB-ORDER", "status": "accepted"}
alpaca._request = fake_request

print("LiveLimits.from_env() =", alpaca.LiveLimits.from_env())
payload = alpaca.market_order_payload("AAPL", "buy", 4000)   # 4000 x $250 = $1,000,000
print("payload =", payload)
res = alpaca.submit_order(payload)
print("submit_order returned:", res)
print("HTTP calls the stub captured:", sent)
print()
print("control: same thing with the default $500 cap")
os.environ["ALPACA_LIVE_MAX_ORDER_USD"] = "500"
os.environ["ALPACA_LIVE_MAX_DAILY_LOSS_USD"] = "200"
try:
    alpaca.submit_order(payload)
    print("  NOT BLOCKED")
except alpaca.LiveLimitError as e:
    print("  LiveLimitError:", e)
