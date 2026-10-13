"""E (skeptic): with the SHIPPED default watchlist, does demo mode actually
produce tradeable BUY/SELL signals for the autopilot, and would the live
notional cap then refuse them?  No network: requests is replaced.
"""
import os, sys, json, pathlib, math

SCRATCH = pathlib.Path("docs/evidence/probes")
os.environ.update({
    "DESK_DEMO_MODE": "true",
    "ALPACA_TRADING_BASE_URL": "https://api.alpaca.markets",
    "ALPACA_LIVE_TRADING": "true",
    "ALPACA_API_KEY": "K", "ALPACA_SECRET_KEY": "S",
    "ANALYSIS_TIMEFRAME": "15Min",
})

import requests
class Resp:
    def __init__(self, p): self._p = p; self.content = b"x"; self.status_code = 200
    def raise_for_status(self): pass
    def json(self): return self._p
def _route(method, url, **kw):
    if "alpaca.markets" not in url:
        raise AssertionError("unexpected host " + url)
    if "/v1/corporate-actions" in url:
        return Resp({"corporate_actions": {"cash_dividends": []}})
    raise AssertionError("unrouted " + method + " " + url)
requests.request = lambda m, u, **kw: _route(m.upper(), u, **kw)
requests.get = lambda u, **kw: _route("GET", u, **kw)
requests.post = lambda u, **kw: _route("POST", u, **kw)

from trader import analysis, market_feed, technicals, alpaca

WATCH = "SPY,QQQ,AAPL,MSFT,NVDA,TSLA,AMZN,META,GOOGL,AMD".split(",")
MINCONF = 0.5

print("DEFAULT_WATCHLIST demo signals at 15Min (what the desk's demo AUTO universe is)")
print(f"{'sym':6} {'demo px':>9} {'signal':7} {'str':>5} {'conf':>5} {'qty':>5} {'maxliveqty':>10} {'stop':>8} {'target':>8}  tradeable?")
tradeable = []
for sym in WATCH:
    res = analysis.analyze(sym, capital=100000.0, held=0)
    px = res["current_price"] or 0
    mq = alpaca.max_live_qty(px) if px else 0
    qty = min(res["quantity"], mq)
    rp = res.get("risk_parameters") or {}
    ok = (res["recommendation"] in ("BUY", "SELL") and res["confidence"] >= MINCONF
          and res["quantity"] >= 1 and px >= 5.0 and qty >= 1)
    print(f"{sym:6} {px:9.2f} {res['recommendation']:7} "
          f"{(res['all_signals']['technical'].get('strength') or 0):5.2f} {res['confidence']:5.2f} "
          f"{res['quantity']:5d} {mq:10d} {rp.get('stop_loss', 0) or 0:8.2f} {rp.get('take_profit', 0) or 0:8.2f}  {ok}")
    if ok:
        tradeable.append((sym, px, qty, rp.get("stop_loss"), rp.get("take_profit")))

print()
print(f"tradeable in demo+live: {len(tradeable)}/{len(WATCH)} -> {[t[0] for t in tradeable]}")
print()
print("determinism check (same symbol 3x, 15Min):")
for sym in WATCH[:4]:
    sigs = []
    for _ in range(3):
        bars = market_feed.get_bars(sym, "15Min", 300)["bars"]
        t = technicals.technical_signal(bars)
        sigs.append((t["signal"], round(t["strength"], 3), round(t["current_price"], 2)))
    print(" ", sym, sigs)

print()
print("would the live notional cap refuse these, as a function of the real price?")
print("  (qty is sized off the DEMO price; check_live_buy re-prices at the real tape)")
for sym, px, qty, sl, tp in tradeable:
    breakeven = 500.0 / qty if qty else float("inf")
    print(f"  {sym:6} demo {px:8.2f} qty {qty:3d} -> order PASSES the $500 cap only while the real price <= "
          f"{breakeven:8.2f} (i.e. real/demo <= {breakeven/px:5.2f}x)")
