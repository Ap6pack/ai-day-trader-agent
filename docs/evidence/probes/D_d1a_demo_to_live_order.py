"""D1a: DESK_DEMO_MODE=true + live-configured Alpaca -> does a fabricated demo
price reach alpaca.submit_order()?  All HTTP is intercepted; nothing leaves."""
import json, os, sys, pathlib

SCRATCH = pathlib.Path("docs/evidence/probes")

os.environ.update({
    "DESK_DEMO_MODE": "true",                                  # <-- simulator ON
    "ALPACA_TRADING_BASE_URL": "https://api.alpaca.markets",    # <-- REAL MONEY host
    "ALPACA_LIVE_TRADING": "true",                              # <-- live opt-in
    "ALPACA_API_KEY": "PLACEHOLDER", "ALPACA_SECRET_KEY": "PLACEHOLDER",
    "ANALYSIS_TIMEFRAME": "15Min",
    "TRADER_JOURNAL_PATH": str(SCRATCH / "D_d1a_journal.db"),
})
for f in (SCRATCH / "D_d1a_journal.db",):
    f.unlink(missing_ok=True)

import requests
from datetime import datetime, timedelta, timezone

REAL_MARKET_PRICE = 58.00   # what the live tape says
CAPTURED = []
SEEN = []

class Resp:
    def __init__(self, payload): self._p = payload; self.content = b"x"; self.status_code = 200
    def raise_for_status(self): pass
    def json(self): return self._p

def _route(method, url, **kw):
    SEEN.append(f"{method} {url}")
    if "api.alpaca.markets" not in url and "data.alpaca.markets" not in url:
        raise AssertionError(f"unexpected host: {url}")
    now = datetime.now(timezone.utc)
    if url.endswith("/v2/clock"):
        return Resp({"is_open": True, "timestamp": now.isoformat(),
                     "next_open": (now + timedelta(hours=20)).isoformat(),
                     "next_close": (now + timedelta(hours=3)).isoformat()})
    if url.endswith("/v2/account"):
        return Resp({"equity": "100000", "last_equity": "100000"})
    if url.endswith("/trades/latest"):
        return Resp({"trade": {"p": REAL_MARKET_PRICE}})
    if "/v1/corporate-actions" in url:
        return Resp({"corporate_actions": {"cash_dividends": []}})
    if url.endswith("/v2/orders") and method == "GET":
        return Resp([])
    if url.endswith("/v2/orders") and method == "POST":
        CAPTURED.append(kw.get("json"))
        return Resp({"id": "ORDER-ID-0001", "status": "accepted"})
    if url.endswith("/v2/positions"):
        return Resp([])
    raise AssertionError(f"unrouted: {method} {url}")

requests.request = lambda method, url, **kw: _route(method.upper(), url, **kw)
requests.get = lambda url, **kw: _route("GET", url, **kw)
requests.post = lambda url, **kw: _route("POST", url, **kw)

from trader import alpaca, analysis, autotrader, market_feed
from trader.journal import Journal
from trader.portfolios import Portfolios

SYM = "O"
print("account_mode()        =", alpaca.account_mode())
print("demo_mode_enabled()   =", market_feed.demo_mode_enabled())
feed = market_feed.get_bars(SYM, "15Min", 300)
print("get_bars source       =", repr(feed["source"]), " last bar close =", feed["bars"][-1]["close"])

res = analysis.analyze(SYM, capital=100000.0, held=0, capital_source="alpaca LIVE equity")
print("analyze() keys        =", sorted(res))
print("  'source' in result  =", "source" in res)
print("  current_price       =", res["current_price"])
print("  recommendation      =", res["recommendation"], " qty", res["quantity"], " conf", res["confidence"])
print("  risk_parameters     =", json.dumps(res["risk_parameters"]))

journal = Journal(SCRATCH / "D_d1a_journal.db")
pilot = autotrader.AutoTrader(journal, Portfolios(SCRATCH / "D_d1a_journal.db"),
                              universe=lambda: [SYM])
cfg = autotrader.RunConfig([SYM], 300, "live").validate()
pilot.broker.require_mode("live", who="probe")        # same gate desk/start uses
out = pilot.run_cycle(cfg)
print("run_cycle result      =", out)
print()
print("POST /v2/orders bodies captured:", len(CAPTURED))
for p in CAPTURED:
    print(json.dumps(p, indent=2))
print()
if CAPTURED:
    p = CAPTURED[0]
    tp = float(p["take_profit"]["limit_price"]); sl = float(p["stop_loss"]["stop_price"])
    print(f"demo price used to build order : {res['current_price']}")
    print(f"REAL tape price (latest_price) : {REAL_MARKET_PRICE}")
    print(f"bracket take_profit limit      : {tp}   ({'ABOVE' if tp > REAL_MARKET_PRICE else 'below'} the real price)")
    print(f"bracket stop_loss stop         : {sl}   ({'ABOVE' if sl > REAL_MARKET_PRICE else 'below'} the real price)")
    print(f"stop above market => stop-loss leg triggers immediately: {sl > REAL_MARKET_PRICE}")
