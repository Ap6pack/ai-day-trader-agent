"""Shared harness for claim group F probes. Read-only against the repo."""
import os, tempfile, pathlib

SCRATCH = pathlib.Path("docs/evidence/probes")
_D = SCRATCH / "F_data"
_D.mkdir(parents=True, exist_ok=True)

# Nothing may touch the repo or the network.
os.environ["TRADER_JOURNAL_PATH"] = str(_D / "F_journal.db")
os.environ.pop("DESK_TOKEN", None)                  # <-- the DEFAULT tokenless dashboard
os.environ["DESK_START_NEWSBOT"] = "false"
os.environ["DESK_AUTOPILOT_MODE"] = "off"
os.environ["DESK_DEMO_MODE"] = "1"                  # no market network
os.environ.setdefault("ALPACA_API_KEY", "PLACEHOLDER")
os.environ.setdefault("ALPACA_SECRET_KEY", "PLACEHOLDER")

CALLS = []

def arm(live=False):
    """Import the app with alpaca's HTTP layer replaced by a recorder."""
    if live:
        os.environ["ALPACA_TRADING_BASE_URL"] = "https://api.alpaca.markets"
        os.environ["ALPACA_LIVE_TRADING"] = "true"
    else:
        os.environ["ALPACA_TRADING_BASE_URL"] = "https://paper-api.alpaca.markets"
        os.environ.pop("ALPACA_LIVE_TRADING", None)

    from trader import alpaca

    def fake_request(method, url, **kwargs):
        CALLS.append({"method": method, "url": url, "json": kwargs.get("json")})
        if url.endswith("/v2/orders") and method == "POST":
            return {"id": "ORD-FAKE-0001", "status": "accepted", **(kwargs.get("json") or {})}
        if "/v2/orders/" in url and method == "DELETE":
            return None
        if url.endswith("/v2/account"):
            return {"equity": "250000", "cash": "100000", "buying_power": "500000",
                    "account_number": "REAL-ACCT-9999"}
        if url.endswith("/v2/positions"):
            return [{"symbol": "NVDA", "qty": "500", "market_value": "90000",
                     "unrealized_pl": "12345", "avg_entry_price": "155.00"}]
        if url.endswith("/v2/clock"):
            return {"is_open": True, "next_open": "", "next_close": "", "timestamp": ""}
        return {}

    alpaca._request = fake_request
    import requests
    def boom(*a, **k):
        raise AssertionError("REAL NETWORK CALL ATTEMPTED: %r %r" % (a, k))
    requests.request = boom
    requests.post = boom
    requests.get = boom
    return alpaca

def app_and_client():
    from fastapi.testclient import TestClient
    from trader.desk import create_app
    app = create_app(background=False, start_agents=False)
    return app, TestClient(app)
