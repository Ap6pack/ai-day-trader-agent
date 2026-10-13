"""I1 end-to-end: real analyze() -> real AutoTrader.act() with a stub broker (NO network).
Shows that maximally bearish news does not stop (or shrink) the order."""
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone

os.environ["MIN_POSITION_PERCENTAGE"] = "0.02"
os.environ["MAX_POSITION_PERCENTAGE"] = "0.10"
os.environ["SENTIMENT_THRESHOLD"] = "0.3"
os.environ["DIVIDEND_STRATEGY_ENABLED"] = "true"
os.environ["DIVIDEND_CAPTURE_WINDOW_DAYS"] = "7"
os.environ["MIN_DIVIDEND_YIELD"] = "6.0"
os.environ["ALPACA_TRADING_BASE_URL"] = "http://127.0.0.1:1/never-used"  # belt and braces: no real host

from trader import alpaca, analysis
from trader.autotrader import AutoTrader, Limits, RunConfig
from trader.journal import Journal
from trader.portfolios import Portfolios

TODAY = date.today()
NOW = datetime.now(timezone.utc)


def bars_from(closes, spread=1.0):
    return [{"time": i, "open": c, "high": c + spread, "low": c - spread, "close": c, "volume": 1000}
            for i, c in enumerate(closes)]


BULLISH_BARS = bars_from([100 - i * 0.8 for i in range(60)] + [52.5, 53.5, 54.8])  # oversold bounce = technical BUY
DIV_ROWS = [  # trailing 12m: 4 x $0.90 = $3.60 on a ~$54.80 price = 6.6% yield
    {"ex_date": (TODAY - timedelta(days=d)).isoformat(), "rate": 0.90} for d in (300, 210, 120, 30)
] + [{"ex_date": (TODAY + timedelta(days=1)).isoformat(), "rate": 0.90}]


def bearish_judge(symbol, articles, max_articles):
    return {"score": -1.0, "evidence_weight": 1.0, "articles": [{}] * 8,
            "rationale": "SEC fraud probe, CFO resigns, guidance pulled"}


def neutral_judge(symbol, articles, max_articles):
    return {"score": 0.0, "evidence_weight": 0.1, "articles": [{}] * 8, "rationale": "nothing material"}


class StubBroker:
    """Never touches the network. Mirrors the surface AutoTrader uses."""
    def __init__(self):
        self.orders = []
    def require_mode(self, expected, who="x"): pass
    def owned_positions(self, prefix): return {}
    def owned_qty(self, prefix, symbol): return 0.0
    def max_live_qty(self, price): return 10_000
    def flatten_owned(self, prefix): return []
    def market_clock(self):
        return {"is_open": True, "next_close": (NOW + timedelta(hours=3)).isoformat()}
    def account(self): return {"equity": "10000"}
    def positions(self): return []
    bracket_order_payload = staticmethod(alpaca.bracket_order_payload)
    market_order_payload = staticmethod(alpaca.market_order_payload)
    def submit_order(self, payload):
        self.orders.append(payload)
        return {"id": f"stub-{len(self.orders)}", "status": "accepted"}


def run(label, judge):
    res = analysis.analyze("ACME", capital=10000, bars=BULLISH_BARS, articles=[{"title": "x"}],
                           dividend_rows=DIV_ROWS, judge=judge)
    print(f"\n--- {label} ---")
    for n, s in res["all_signals"].items():
        print(f"   {n:10} {s['signal']:5} strength {s.get('strength'):<5} | {s['reason'][:80]}")
    print(f"   FUSED: {res['recommendation']} qty={res['quantity']} confidence={res['confidence']} "
          f"primary={res['primary_strategy']} confirming={res['confirming_strategies']} "
          f"conflicting={res['conflicting_strategies']}")
    print(f"   price ${res['current_price']:.2f}  position_value "
          f"${res['risk_parameters'].get('position_value')}")
    return res


bear = run("maximally BEARISH news (Jev score -1.00) + bullish technical + dividend capture", bearish_judge)
neut = run("NEUTRAL news (Jev score 0.00), everything else identical", neutral_judge)

print()
print("=" * 78)
print(f"confidence  bearish-news {bear['confidence']}  vs  neutral-news {neut['confidence']}  "
      f"-> delta {round(bear['confidence'] - neut['confidence'], 6)}")
print(f"quantity    bearish-news {bear['quantity']}  vs  neutral-news {neut['quantity']}")
print("=" * 78)

# Now push the bearish-news result through the REAL AutoTrader.act() in paper mode.
import tempfile
tmp = tempfile.mkdtemp()
journal = Journal(os.path.join(tmp, "j.db"))
pf = Portfolios(os.path.join(tmp, "j.db"))
pf.create("default", 10000)
events = []
broker = StubBroker()
at = AutoTrader(journal, pf, publish=events.append, broker=broker,
                limits=Limits(min_confidence=0.5, max_trades_per_day=5, cooldown_minutes=30,
                              min_price=5, entry_cutoff_minutes=15),
                now=lambda: NOW)
cfg = RunConfig(symbols=["ACME"], mode="paper").validate()
print("\nAutoTrader.act(bearish-news result, mode=paper):")
print("   ->", at.act(bear, cfg))
print("   broker orders submitted:", json.dumps(broker.orders, indent=2)[:600])

# And what a sane additive confidence would have done.
sane_conf = round(max(0.0, min(0.7 + 0.1 * 1 - 0.15 * 1, 0.95)), 3)
sane_qty = analysis.position_size("BUY", sane_conf, bear["current_price"], 10000,
                                  bear["technical_indicators"].get("volatility"), 900)
sane_res = {**bear, "confidence": sane_conf, "quantity": sane_qty}
broker2 = StubBroker()
journal2 = Journal(os.path.join(tmp, "j2.db"))
pf2 = Portfolios(os.path.join(tmp, "j2.db"))
pf2.create("default", 10000)
at2 = AutoTrader(journal2, pf2, publish=lambda e: None, broker=broker2,
                 limits=Limits(min_confidence=0.5, max_trades_per_day=5, cooldown_minutes=30,
                               min_price=5, entry_cutoff_minutes=15), now=lambda: NOW)
print(f"\nWith a sane additive confidence ({sane_conf}) the same inputs give qty {sane_qty}:")
print("   ->", at2.act(sane_res, cfg))
print(f"   notional actual ${bear['quantity'] * bear['current_price']:,.2f} vs "
      f"sane ${sane_qty * bear['current_price']:,.2f}")
