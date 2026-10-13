"""H2: paper flatten counts per-position failures as success, then never retries."""
import os, json, types
from datetime import datetime, timedelta, timezone

SCRATCH = "docs/evidence/probes"
DB = os.path.join(SCRATCH, "H_h2_journal.db")
if os.path.exists(DB):
    os.remove(DB)
os.environ["TRADER_JOURNAL_PATH"] = DB
os.environ["TRADER_MODE"] = "review"
os.environ["ALPACA_TRADING_BASE_URL"] = "https://paper-api.alpaca.markets"   # paper; requests is stubbed below
os.environ["ALPACA_API_KEY"] = "stub"
os.environ["ALPACA_SECRET_KEY"] = "stub"

from trader import alpaca, newsbot
from trader.journal import Journal

# ---------------------------------------------------------------- part 1: the REST layer
# Real Alpaca DELETE /v2/positions?cancel_orders=true answers HTTP 207 Multi-Status with one
# element per symbol, each carrying its OWN status code.
MULTI_STATUS_BODY = [
    {"symbol": "AAPL", "status": 200,
     "body": {"id": "ord-1", "symbol": "AAPL", "qty": "10", "side": "sell", "status": "accepted"}},
    {"symbol": "TSLA", "status": 500,
     "body": {"code": 50010000, "message": "internal server error: could not liquidate TSLA"}},
    {"symbol": "NVDA", "status": 403,
     "body": {"code": 40310000, "message": "insufficient qty available for order (requested: 5, available: 0)"}},
]


class FakeResponse:
    status_code = 207
    content = b"x"

    def raise_for_status(self):
        print(f"  [requests stub] raise_for_status() on HTTP {self.status_code} -> no exception "
              f"(requests only raises for 4xx/5xx)")

    def json(self):
        return MULTI_STATUS_BODY


CALLS = []


def fake_request(method, url, **kw):
    assert "alpaca" not in url or True
    CALLS.append((method, url, kw.get("params")))
    print(f"  [requests stub] intercepted {method} {url} params={kw.get('params')}  (no network)")
    return FakeResponse()


alpaca.requests = types.SimpleNamespace(request=fake_request,
                                        HTTPError=__import__("requests").HTTPError)

print("=== part 1: alpaca.close_all_positions() against a 207 multi-status body ===")
print("  account_mode() =", alpaca.account_mode())
closed = alpaca.close_all_positions()
print("  returned list length  len(closed) =", len(closed))
print("  per-element statuses  =", [e["status"] for e in closed])
print("  actually succeeded    =", [e["symbol"] for e in closed if e["status"] == 200])
print("  actually FAILED       =", [e["symbol"] for e in closed if e["status"] != 200])
print("  -> close_all_positions() returns every element; no status is inspected anywhere.")

# ---------------------------------------------------------------- part 2: NewsBot.maybe_flatten
print("\n=== part 2: NewsBot.maybe_flatten() with that response ===")

CLOSE = datetime(2026, 10, 13, 16, 0, tzinfo=timezone.utc)   # pretend 'next_close'
NOW = {"t": CLOSE - timedelta(minutes=9)}                    # inside the 10-minute flatten window


class StubBroker:
    """Positions ledger: only AAPL is really closed by the multi-status response."""
    def __init__(self):
        self.live_positions = {"AAPL": 10, "TSLA": 7, "NVDA": 5}
        self.close_calls = 0

    def market_clock(self):
        return {"is_open": True, "next_close": CLOSE.isoformat(), "timestamp": NOW["t"].isoformat()}

    def close_all_positions(self):
        self.close_calls += 1
        for e in MULTI_STATUS_BODY:
            if e["status"] == 200:
                self.live_positions.pop(e["symbol"], None)
        return list(MULTI_STATUS_BODY)

    def flatten_owned(self, prefix):
        raise AssertionError("paper mode must not call flatten_owned")


broker = StubBroker()
settings = newsbot.Settings(execution="paper", flatten_minutes=10, entry_cutoff_minutes=15)
jr = Journal()
bot = newsbot.NewsBot(settings, journal=jr, judge=lambda s, a: None, broker=broker,
                      now=lambda: NOW["t"], monotonic=lambda: 0.0)

res = bot.maybe_flatten()
print("  maybe_flatten() ->", res)
print("  journal event detail:", [dict(r)["detail"] for r in jr.events(5) if dict(r)["kind"] == "newsbot_flatten"])
print("  broker positions STILL OPEN after 'successful' flatten:", broker.live_positions)
print("  bot._flattened_for =", repr(bot._flattened_for))

# ---------------------------------------------------------------- part 3: no retry
print("\n=== part 3: flatten_loop keeps checking every 30s until the close. Does it retry? ===")
for i in range(1, 19):                       # 18 further checks x 30s = the last 9 minutes
    NOW["t"] = NOW["t"] + timedelta(seconds=30)
    out = bot.maybe_flatten()
    if i in (1, 2, 17, 18):
        print(f"  t-{(CLOSE - NOW['t']).total_seconds()/60:4.1f}min  maybe_flatten() -> {out}")
print("  broker.close_all_positions() call count over the whole session:", broker.close_calls)
print("  positions left open at the close:", broker.live_positions)

# ---------------------------------------------------------------- part 4: total failure IS retried
print("\n=== part 4: control - a TOTAL failure (exception) is retried, so the gap is partial failure only ===")


class RaisingBroker(StubBroker):
    def close_all_positions(self):
        self.close_calls += 1
        raise RuntimeError("503 Service Unavailable")


b2 = RaisingBroker()
NOW["t"] = CLOSE - timedelta(minutes=9)
bot2 = newsbot.NewsBot(settings, journal=jr, judge=lambda s, a: None, broker=b2,
                       now=lambda: NOW["t"], monotonic=lambda: 0.0)
for i in range(3):
    try:
        bot2.maybe_flatten()
    except Exception as exc:
        print(f"  check {i+1}: raised {exc!r}; _flattened_for={bot2._flattened_for!r} -> loop logs and retries")
    NOW["t"] = NOW["t"] + timedelta(seconds=30)
print("  close_all_positions attempts after 3 checks:", b2.close_calls)
