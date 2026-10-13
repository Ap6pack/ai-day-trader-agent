"""E (skeptic) 2: how often does the seeded demo walk produce a TRADEABLE BUY
(conf >= AUTOPILOT_MIN_CONFIDENCE 0.5) across (a) the shipped watchlist at every
timeframe and (b) a broad population of tickers?  No network.
"""
import os, pathlib, string, itertools

os.environ.update({"DESK_DEMO_MODE": "true"})

import requests
def _boom(*a, **k):
    raise AssertionError("no network in this probe")
requests.request = requests.get = requests.post = _boom

from trader import market_feed, technicals

WATCH = "SPY,QQQ,AAPL,MSFT,NVDA,TSLA,AMZN,META,GOOGL,AMD".split(",")
TFS = ["1Min", "5Min", "15Min", "1Hour", "1Day"]

def sig(sym, tf):
    bars = market_feed.demo_market.bars(sym, tf, 300)
    t = technicals.technical_signal(bars)
    return t["signal"], round(t["strength"], 3)

print("shipped DEFAULT_WATCHLIST, every timeframe  (BUY with strength>=0.5 == a live buy)")
print(f"{'sym':7}" + "".join(f"{tf:>16}" for tf in TFS))
buyable = 0
for sym in WATCH:
    cells = []
    for tf in TFS:
        s, st = sig(sym, tf)
        flag = "*" if (s == "BUY" and st >= 0.5) else " "
        if flag == "*":
            buyable += 1
        cells.append(f"{s}/{st:.2f}{flag}".rjust(16))
    print(f"{sym:7}" + "".join(cells))
print(f"\ntradeable demo BUYs in the shipped watchlist across all 5 timeframes: {buyable}/{len(WATCH)*len(TFS)}")

# Broad population: all 3-letter tickers (a stand-in for "whatever the operator lists").
pop = ["".join(c) for c in itertools.product(string.ascii_uppercase, repeat=3)]
for tf in ("15Min", "5Min"):
    n_buy = n_strong = n_sell = 0
    strong = []
    for sym in pop:
        s, st = sig(sym, tf)
        if s == "BUY":
            n_buy += 1
            if st >= 0.5:
                n_strong += 1
                if len(strong) < 12:
                    strong.append(f"{sym}:{st:.2f}")
        elif s == "SELL":
            n_sell += 1
    print(f"\n{tf}: of {len(pop)} seeded symbols -> BUY {n_buy} ({n_buy/len(pop):.1%}), "
          f"of which strength>=0.5 (tradeable): {n_strong} ({n_strong/len(pop):.1%}); SELL {n_sell}")
    print("   e.g.", ", ".join(strong))

# Stability across processes: the walk is seeded by (symbol, timeframe) only.
print("\nstability: random.Random(f'{symbol}:{timeframe}') seeding means the signal is the SAME")
print("for every operator and every cycle; only the price level drifts (scale-invariant indicators).")
for sym in ("O", "KO", "VZ", "T", "JPM"):
    print("  ", sym, [sig(sym, "15Min") for _ in range(3)])
