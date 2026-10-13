"""D1b: real (non-demo) Alpaca bars whose newest bar is 5 DAYS old.
Does analyze() report that ancient close as current_price, with no freshness check,
and does the autopilot build a LIVE bracket order from it?  All HTTP intercepted."""
import json, os, pathlib, time
from datetime import datetime, timedelta, timezone
SCRATCH = pathlib.Path("docs/evidence/probes")
DB = SCRATCH / "D_d1b_journal.db"; DB.unlink(missing_ok=True)

os.environ.update({
    "DESK_DEMO_MODE": "false",                                 # <-- NOT demo. Real feed.
    "ALPACA_TRADING_BASE_URL": "https://api.alpaca.markets",
    "ALPACA_LIVE_TRADING": "true",
    "ALPACA_API_KEY": "PLACEHOLDER", "ALPACA_SECRET_KEY": "PLACEHOLDER",
    "ANALYSIS_TIMEFRAME": "15Min",
    "TRADER_JOURNAL_PATH": str(DB),
})
import requests

NOW = datetime.now(timezone.utc)
STALE_AGE_DAYS = 5
LAST_BAR_TS = NOW - timedelta(days=STALE_AGE_DAYS)
REAL_TAPE_PRICE = 71.40          # where the stock trades RIGHT NOW
# 120 15-minute bars ending 5 days ago, drifting down to 100.00 (RSI low -> BUY)
N = 120
bars_desc = []
for i in range(N):                                  # newest first, as Alpaca sort=desc
    t = LAST_BAR_TS - timedelta(minutes=15 * i)
    c = 100.00 + 0.25 * i                           # older bars higher -> falling market -> RSI low
    bars_desc.append({"t": t.isoformat().replace("+00:00", "Z"),
                      "o": round(c + 0.10, 2), "h": round(c + 0.30, 2),
                      "l": round(c - 0.30, 2), "c": round(c, 2), "v": 50000})

CAPTURED = []; CALLS = []
class Resp:
    def __init__(self, p): self._p=p; self.content=b"x"; self.status_code=200
    def raise_for_status(self): pass
    def json(self): return self._p
def _route(method, url, **kw):
    CALLS.append(f"{method} {url.split('?')[0]}")
    if "alpaca.markets" not in url: raise AssertionError("unexpected host " + url)
    if "/bars" in url:  return Resp({"bars": bars_desc})
    if url.endswith("/v2/clock"):
        return Resp({"is_open": True, "timestamp": NOW.isoformat(),
                     "next_open": (NOW+timedelta(hours=20)).isoformat(),
                     "next_close": (NOW+timedelta(hours=3)).isoformat()})
    if url.endswith("/v2/account"): return Resp({"equity":"100000","last_equity":"100000"})
    if url.endswith("/trades/latest"): return Resp({"trade":{"p":REAL_TAPE_PRICE}})
    if "/v1/corporate-actions" in url: return Resp({"corporate_actions":{"cash_dividends":[]}})
    if url.endswith("/v2/orders") and method=="GET": return Resp([])
    if url.endswith("/v2/orders") and method=="POST":
        CAPTURED.append(kw.get("json")); return Resp({"id":"STALE-ORDER-1","status":"accepted"})
    if url.endswith("/v2/positions"): return Resp([])
    if "/snapshots" in url: return Resp({})
    raise AssertionError("unrouted " + method + " " + url)
requests.request = lambda m,u,**k: _route(m.upper(),u,**k)
requests.get = lambda u,**k: _route("GET",u,**k)
requests.post = lambda u,**k: _route("POST",u,**k)

from trader import alpaca, analysis, autotrader, market_feed
from trader.journal import Journal
from trader.portfolios import Portfolios

SYM = "THIN"
feed = market_feed.get_bars(SYM, "15Min", 300)
newest = feed["bars"][-1]
age_h = (NOW - datetime.fromtimestamp(newest["time"], timezone.utc)).total_seconds()/3600
print(f"get_bars source           = {feed['source']!r}   (demo_mode={market_feed.demo_mode_enabled()})")
print(f"newest bar time           = {datetime.fromtimestamp(newest['time'], timezone.utc).isoformat()}  "
      f"-> {age_h:.1f} HOURS old")
print(f"newest bar close          = {newest['close']}")
print(f"REAL tape price now       = {REAL_TAPE_PRICE}")

res = analysis.analyze(SYM, capital=100000.0, held=0, capital_source="alpaca LIVE equity")
print()
print(f"analyze() current_price   = {res['current_price']}   <-- the {age_h:.0f}h-old bar close")
print(f"analyze() timestamp       = {res['timestamp']}  (wall clock NOW, not the data's time)")
print(f"any bar timestamp in out? = "
      f"{[k for k in list(res)+list(res['technical_indicators']) if 'time' in k.lower() or 'age' in k.lower()]}")
print(f"'source' key in result    = {'source' in res}")
print(f"recommendation            = {res['recommendation']} qty={res['quantity']} conf={res['confidence']}")
print(f"risk_parameters           = {json.dumps(res['risk_parameters'])}")

journal = Journal(DB)
pilot = autotrader.AutoTrader(journal, Portfolios(DB), universe=lambda: [SYM])
cfg = autotrader.RunConfig([SYM], 300, "live").validate()
print()
print("market_clock().is_open    =", alpaca.market_clock()["is_open"], "  <- the only data-related gate in _blocked()")
print("run_cycle ->", pilot.run_cycle(cfg))
print()
print("POST /v2/orders captured:", len(CAPTURED))
for p in CAPTURED: print(json.dumps(p, indent=2))
if CAPTURED:
    p = CAPTURED[0]
    qty = int(p["qty"]); tp=float(p["take_profit"]["limit_price"]); sl=float(p["stop_loss"]["stop_price"])
    print()
    print(f"max_live_qty(stale {res['current_price']})      = {alpaca.max_live_qty(res['current_price'])}  "
          f"(sized off the STALE price)")
    print(f"check_live_buy notional at REAL {REAL_TAPE_PRICE} = ${qty*REAL_TAPE_PRICE:,.2f} "
          f"vs cap ${alpaca.LiveLimits.from_env().max_order_usd:,.2f}  -> passed")
    print(f"bracket take_profit {tp}  vs real price {REAL_TAPE_PRICE}: "
          f"{'ABOVE' if tp>REAL_TAPE_PRICE else 'BELOW -> limit sell fills instantly at the open'}")
    print(f"bracket stop_loss   {sl}  vs real price {REAL_TAPE_PRICE}: "
          f"{'ABOVE -> stop triggers instantly' if sl>REAL_TAPE_PRICE else 'below'}")
    print(f"neither leg was re-priced by check_live_buy: {tp!=REAL_TAPE_PRICE and sl!=REAL_TAPE_PRICE}")
print()
print("HTTP calls made:", sorted(set(CALLS)))
