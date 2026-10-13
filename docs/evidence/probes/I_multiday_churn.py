"""I2 amplification: the capture BUY is re-issued every day inside the window and
flattened every day, so the strategy churns 1 round trip per day and never holds
through the ex-date. Also probes the 'cycle overran the flatten window' path."""
import os
import tempfile
from datetime import date, datetime, timedelta, timezone

os.environ.update({"MIN_POSITION_PERCENTAGE": "0.02", "MAX_POSITION_PERCENTAGE": "0.10",
                   "DIVIDEND_STRATEGY_ENABLED": "true", "DIVIDEND_CAPTURE_WINDOW_DAYS": "7",
                   "MIN_DIVIDEND_YIELD": "6.0", "SENTIMENT_THRESHOLD": "0.3",
                   "ALPACA_TRADING_BASE_URL": "http://127.0.0.1:1/never-used"})

from trader import alpaca, analysis, dividends
from trader.autotrader import ORDER_PREFIX, AutoTrader, Limits, RunConfig
from trader.journal import Journal
from trader.portfolios import Portfolios

START = date(2026, 10, 13)
EX = date(2026, 10, 18)          # 5 calendar days out on day 1
PRICE, DIV, SPREAD = 54.80, 0.90, 0.02


def bars_from(cl, sp=1.0):
    return [{"time": i, "open": c, "high": c + sp, "low": c - sp, "close": c, "volume": 1000}
            for i, c in enumerate(cl)]


BARS = bars_from([100 - i * 0.8 for i in range(60)] + [52.5, 53.5, PRICE])  # technical BUY, confirming


def rows_for(today):
    r = [{"ex_date": (today - timedelta(days=d)).isoformat(), "rate": DIV} for d in (300, 210, 120, 30)]
    r.append({"ex_date": EX.isoformat(), "rate": DIV})
    return r


class StubBroker:
    def __init__(self):
        self.orders, self.held = [], {}
        self.clock = {"is_open": True, "next_close": None}
    def require_mode(self, *a, **k): pass
    def owned_positions(self, prefix):
        return {s: {"qty": q, "open_order_ids": []} for s, q in self.held.items() if q}
    def owned_qty(self, prefix, symbol): return self.held.get(symbol, 0.0)
    def max_live_qty(self, price): return 10_000
    def market_clock(self): return self.clock
    def account(self): return {"equity": "10000"}
    def positions(self): return []
    bracket_order_payload = staticmethod(alpaca.bracket_order_payload)
    market_order_payload = staticmethod(alpaca.market_order_payload)
    def submit_order(self, p):
        self.orders.append(p)
        q = float(p["qty"])
        self.held[p["symbol"]] = self.held.get(p["symbol"], 0.0) + (q if p["side"] == "buy" else -q)
        return {"id": f"o{len(self.orders)}", "status": "accepted"}
    def flatten_owned(self, prefix):
        out = []
        for s, e in list(self.owned_positions(prefix).items()):
            p = self.market_order_payload(s, "sell", int(e["qty"]))
            p["client_order_id"] = f"{prefix}flat"
            out.append(self.submit_order(p))
        return out


tmp = tempfile.mkdtemp()
journal = Journal(os.path.join(tmp, "j.db"))
pf = Portfolios(os.path.join(tmp, "j.db"))
pf.create("default", 10000)
broker = StubBroker()
clock = {"t": None}
day_state = {"today": START}
at = AutoTrader(journal, pf, publish=lambda e: None, broker=broker,
                analyze=lambda sym, cap, held, capital_source="": analysis.analyze(
                    sym, cap, held, bars=BARS, articles=[{"title": "x"}],
                    dividend_rows=rows_for(day_state["today"]),
                    judge=lambda *a, **k: {"score": 0.0, "articles": []}, capital_source=capital_source),
                limits=Limits(min_confidence=0.5, max_trades_per_day=5, cooldown_minutes=30,
                              min_price=5, entry_cutoff_minutes=15, flatten_minutes=10),
                now=lambda: clock["t"])
cfg = RunConfig(symbols=["HIYLD"], interval_seconds=300, mode="paper").validate()

print("=" * 86)
print(f"Five consecutive sessions, ex-date {EX}, DIVIDEND_CAPTURE_WINDOW_DAYS=7, mode=paper")
print("=" * 86)
print(f"{'session':12} {'days_to_ex':>10} {'conf':>6} {'action':>22} {'flattened':>10} {'net pos':>8}")
round_trips, shares = 0, 0
for i in range(5):
    today = START + timedelta(days=i)
    day_state["today"] = today
    broker.clock["next_close"] = f"{today.isoformat()}T16:00:00-04:00"
    at._flattened_for = None
    clock["t"] = datetime(today.year, today.month, today.day, 15, 0, tzinfo=timezone.utc)   # 11:00 ET
    sig = dividends.dividend_signal("HIYLD", PRICE, today=today, rows=rows_for(today))
    res = at.run_cycle(cfg)[0]
    conf = analysis.analyze("HIYLD", 10000, 0, bars=BARS, articles=[{"title": "x"}],
                            dividend_rows=rows_for(today),
                            judge=lambda *a, **k: {"score": 0.0, "articles": []})["confidence"]
    clock["t"] = datetime(today.year, today.month, today.day, 19, 52, tzinfo=timezone.utc)  # 15:52 ET
    n = at.maybe_flatten(cfg)
    if res.get("action") == "buy":
        round_trips += 1
        shares += res["qty"]
    print(f"{today.isoformat():12} {sig.get('days_to_ex_dividend'):>10} {conf:>6.3f} "
          f"{str(res.get('action')) + ' ' + str(res.get('qty', res.get('reason', ''))):>22} "
          f"{str(n):>10} {broker.held.get('HIYLD', 0.0):>8}")

print()
print(f"  round trips over the capture window : {round_trips}")
print(f"  shares bought and resold same-day   : {shares}")
print(f"  spread paid @ ${SPREAD:.2f}/sh, 2 crossings  : ${shares * SPREAD * 2:.2f}")
print(f"  dividends received                  : $0.00   (flat at every close; ex-date {EX})")
print(f"  dividend the thesis chased          : ${shares / max(round_trips,1) * DIV:.2f} per round trip")
print(f"  position held on the night before the ex-date: {broker.held.get('HIYLD', 0.0)} shares")

print()
print("=" * 86)
print("Bonus: if a run_cycle overruns the whole flatten window, is the position still closed?")
print("=" * 86)
broker2 = StubBroker()
broker2.clock = {"is_open": True, "next_close": "2026-10-20T16:00:00-04:00"}
j2 = Journal(os.path.join(tmp, "k.db")); p2 = Portfolios(os.path.join(tmp, "k.db")); p2.create("default", 10000)
clk = {"t": datetime(2026, 10, 20, 15, 0, tzinfo=timezone.utc)}
at2 = AutoTrader(j2, p2, publish=lambda e: None, broker=broker2,
                 analyze=lambda sym, cap, held, capital_source="": analysis.analyze(
                     sym, cap, held, bars=BARS, articles=[{"title": "x"}], dividend_rows=rows_for(date(2026, 10, 20)),
                     judge=lambda *a, **k: {"score": 0.0, "articles": []}, capital_source=capital_source),
                 limits=Limits(min_confidence=0.5, flatten_minutes=10, entry_cutoff_minutes=15, min_price=5),
                 now=lambda: clk["t"])
cfg2 = RunConfig(symbols=["HIYLD"], interval_seconds=300, mode="paper").validate()
at2.run_cycle(cfg2)
print("  position after the 11:00 ET cycle:", broker2.owned_positions(ORDER_PREFIX))
clk["t"] = datetime(2026, 10, 20, 20, 1, tzinfo=timezone.utc)   # 16:01 ET, one minute after the close
broker2.clock = {"is_open": False, "next_close": "2026-10-21T16:00:00-04:00"}
print("  maybe_flatten at 16:01 ET (market now closed) ->", at2.maybe_flatten(cfg2))
print("  position carried overnight        :", broker2.owned_positions(ORDER_PREFIX))
print("  (bracket take-profit/stop legs are time_in_force=day, so they expired at the close)")
