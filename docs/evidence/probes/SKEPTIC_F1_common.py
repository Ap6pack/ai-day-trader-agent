"""Shared harness: build the desk app with NO repo modification, no network."""
import os, sys, json
sys.path.insert(0, "/home/user/ai-day-trader-agent")

# default-as-shipped: no DESK_TOKEN
for k in ("DESK_TOKEN",):
    os.environ.pop(k, None)

CALLS = []

def install_recorder(live=False, keys=True):
    import requests
    from trader import alpaca, market_feed
    def boom(*a, **k):
        raise AssertionError("REAL NETWORK ATTEMPTED: %r %r" % (a, k))
    requests.request = boom
    requests.get = boom
    requests.post = boom
    def rec(method, path, **kw):
        CALLS.append({"method": method, "path": path, "kw": kw})
        if path == "/v2/account":
            return {"equity": "100000", "buying_power": "200000", "cash": "100000",
                    "status": "ACTIVE", "daytrade_count": 0, "last_equity": "99000",
                    "pattern_day_trader": False}
        if path == "/v2/positions":
            return []
        if path.startswith("/v2/orders") and method == "POST":
            return {"id": "ord-abc-123", "status": "accepted", "symbol": "AAPL"}
        if path.startswith("/v2/orders"):
            return []
        if path == "/v2/clock":
            return {"is_open": True, "next_close": "2026-10-13T20:00:00Z"}
        return {}
    alpaca._request = rec
    if keys:
        os.environ["ALPACA_API_KEY"] = "PLACEHOLDER_KEY_ID"
        os.environ["ALPACA_SECRET_KEY"] = "testsecret"
        os.environ["ALPACA_API_SECRET"] = "testsecret"
    if live:
        os.environ["ALPACA_TRADING_BASE_URL"] = "https://api.alpaca.markets"
        os.environ["ALPACA_LIVE_TRADING"] = "true"
    else:
        os.environ["ALPACA_TRADING_BASE_URL"] = "https://paper-api.alpaca.markets"
    os.environ["DESK_DEMO_MODE"] = "true"   # keep market_feed off the wire
    return CALLS
