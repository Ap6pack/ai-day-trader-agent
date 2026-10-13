import os
os.environ["DESK_DEMO_MODE"] = "true"
os.environ["ANALYSIS_TIMEFRAME"] = "15Min"
from trader import market_feed, technicals

for sym in list(market_feed.DemoMarket.BASE_PRICES):
    for tf in ("15Min", "5Min", "1Hour"):
        payload = market_feed.get_bars(sym, tf, 300)
        sig = technicals.technical_signal(payload["bars"])
        if sig["signal"] == "BUY" and sig["strength"] >= 0.5:
            print(f"{sym:6s} {tf:6s} source={payload['source']!r} "
                  f"{sig['signal']} strength={sig['strength']} price={sig['current_price']} :: {sig['reason']}")
