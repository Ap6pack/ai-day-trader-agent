"""I2 probe: dividend-capture BUY vs the autopilot's same-day pre-close flatten.
Stub broker only -- nothing leaves this process."""
import os
import tempfile
from datetime import date, datetime, timedelta, timezone

os.environ["MIN_POSITION_PERCENTAGE"] = "0.02"
os.environ["MAX_POSITION_PERCENTAGE"] = "0.10"
os.environ["DIVIDEND_STRATEGY_ENABLED"] = "true"
os.environ["DIVIDEND_CAPTURE_WINDOW_DAYS"] = "7"
os.environ["MIN_DIVIDEND_YIELD"] = "6.0"
os.environ["SENTIMENT_THRESHOLD"] = "0.3"
os.environ["ALPACA_TRADING_BASE_URL"] = "http://127.0.0.1:1/never-used"

from trader import alpaca, analysis, dividends
from trader.autotrader import ORDER_PREFIX, AutoTrader, Limits, RunConfig
from trader.journal import Journal
from trader.portfolios import Portfolios

TODAY = date(2026, 10, 13)
OPEN_ET = datetime(2026, 10, 13, 15, 0, tzinfo=timezone.utc)   # 11:00 ET
PRE_CLOSE = datetime(2026, 10, 13, 19, 52, tzinfo=timezone.utc)  # 15:52 ET, 8 min before close
NEXT_CLOSE = "2026-10-13T16:00:00-04:00"
PRICE = 54.80
DIV = 0.90


def div_rows(days_out):
    rows = [{"ex_date": (TODAY - timedelta(days=d)).isoformat(), "rate": DIV} for d in (300, 210, 120, 30)]
    rows.append({"ex_date": (TODAY + timedelta(days=days_out)).isoformat(), "rate": DIV})
    return rows


def bars_from(closes, spread=1.0):
    return [{"time": i, "open": c, "high": c + spread, "low": c - spread, "close": c, "volume": 1000}
            for i, c in enumerate(closes)]


FLAT_BARS = bars_from([PRICE] * 60)                                   # technical HOLD
BULL_BARS = bars_from([100 - i * 0.8 for i in range(60)] + [52.5, 53.5, PRICE])


def judge_factory(score):
    def judge(symbol, articles, max_articles):
        return {"score": score, "articles": [{}] * 3, "rationale": f"score {score}"}
    return judge


class StubBroker:
    def __init__(self):
        self.orders, self.held, self.log = [], {}, []
        self.clock = {"is_open": True, "next_close": NEXT_CLOSE}
    def require_mode(self, expected, who="x"): pass
    def owned_positions(self, prefix):
        assert prefix == ORDER_PREFIX
        return {s: {"qty": q, "open_order_ids": []} for s, q in self.held.items() if q}
    def owned_qty(self, prefix, symbol): return self.held.get(symbol, 0.0)
    def max_live_qty(self, price): return 10_000
    def market_clock(self): return self.clock
    def account(self): return {"equity": "10000"}
    def positions(self):
        return [{"symbol": s, "qty": str(q), "side": "long"} for s, q in self.held.items() if q]
    bracket_order_payload = staticmethod(alpaca.bracket_order_payload)
    market_order_payload = staticmethod(alpaca.market_order_payload)
    def submit_order(self, payload):
        self.orders.append(payload)
        sym, qty = payload["symbol"], float(payload["qty"])
        self.held[sym] = self.held.get(sym, 0.0) + (qty if payload["side"] == "buy" else -qty)
        self.log.append(f"{payload['side'].upper():4} {qty:g} {sym}  cid={payload['client_order_id'][:18]}")
        return {"id": f"stub-{len(self.orders)}", "status": "accepted"}
    def flatten_owned(self, prefix):
        out = []
        for sym, entry in list(self.owned_positions(prefix).items()):
            p = self.market_order_payload(sym, "sell", int(entry["qty"]))
            p["client_order_id"] = f"{prefix}flat-x"
            out.append(self.submit_order(p))
        return out


def scenario(label, days_out, bars, score, mode="paper"):
    print()
    print("=" * 78)
    print(f"SCENARIO: {label}")
    print("=" * 78)
    rows = div_rows(days_out)
    ex = TODAY + timedelta(days=days_out)
    sig = dividends.dividend_signal("HIYLD", PRICE, today=TODAY, rows=rows)
    print(f"  dividend_signal: {sig['signal']} strength={sig['strength']} "
          f"yield={sig['yield_pct']}% next_ex_date={sig['next_ex_date']} "
          f"days_to_ex_dividend={sig['days_to_ex_dividend']}")
    tmp = tempfile.mkdtemp()
    journal = Journal(os.path.join(tmp, "j.db"))
    pf = Portfolios(os.path.join(tmp, "j.db"))
    pf.create("default", 10000)
    broker = StubBroker()
    clock = {"t": OPEN_ET}
    at = AutoTrader(journal, pf, publish=lambda e: None, broker=broker,
                    analyze=lambda sym, cap, held, capital_source="": analysis.analyze(
                        sym, cap, held, bars=bars, articles=[{"title": "x"}], dividend_rows=rows,
                        judge=judge_factory(score), capital_source=capital_source),
                    limits=Limits(min_confidence=0.5, max_trades_per_day=5, cooldown_minutes=30,
                                  min_price=5, entry_cutoff_minutes=15, flatten_minutes=10),
                    now=lambda: clock["t"])
    cfg = RunConfig(symbols=["HIYLD"], interval_seconds=300, mode=mode).validate()
    full = analysis.analyze("HIYLD", 10000, 0, bars=bars, articles=[{"title": "x"}],
                            dividend_rows=rows, judge=judge_factory(score))
    print(f"  fused analysis : {full['recommendation']} qty={full['quantity']} "
          f"conf={full['confidence']} primary={full['primary_strategy']} "
          f"confirming={full['confirming_strategies']} conflicting={full['conflicting_strategies']}")
    print(f"  per-strategy   : " + " | ".join(
        f"{n}={s['signal']}/{s.get('strength')}" for n, s in full["all_signals"].items()))
    print(f"  11:00 ET run_cycle -> {at.run_cycle(cfg)}")
    print(f"             broker : {broker.log or '(no orders)'}")
    print(f"             held   : {broker.owned_positions(ORDER_PREFIX)}")
    clock["t"] = PRE_CLOSE
    closed = at.maybe_flatten(cfg)
    print(f"  15:52 ET maybe_flatten -> {closed} position(s) closed")
    print(f"             broker : {broker.log or '(no orders)'}")
    print(f"             held   : {broker.owned_positions(ORDER_PREFIX)}")
    if broker.orders and broker.orders[0]["side"] == "buy":
        q = float(broker.orders[0]["qty"])
        print(f"  OUTCOME: bought {q:g} sh @ ${PRICE:.2f} (${q * PRICE:,.2f}) and sold the same day "
              f"{at.limits.flatten_minutes} min before the close.")
        print(f"           ex-date {ex} is {days_out} day(s) AFTER the position was closed.")
        print(f"           dividend chased:  {q:g} x ${DIV:.2f} = ${q * DIV:.2f}")
        print(f"           dividend received: $0.00")
        print(f"           round-trip spread @ $0.02/sh: -${q * 0.02 * 2:.2f}  "
              f"(two crossings), plus ~5h of unhedged directional risk")
    return broker


scenario("ex-date 3 days out, technical HOLD, news neutral (dividend thesis alone)",
         days_out=3, bars=FLAT_BARS, score=0.0)
scenario("ex-date 5 days out, technical HOLD, news neutral",
         days_out=5, bars=FLAT_BARS, score=0.0)
scenario("ex-date 5 days out, technical BUY confirming, news neutral",
         days_out=5, bars=BULL_BARS, score=0.0)
b = scenario("ex-date 3 days out, autopilot in RECORD mode (local portfolio, no flatten)",
             days_out=3, bars=FLAT_BARS, score=0.0, mode="record")

print()
print("=" * 78)
print("Economics even if the position HAD been held through the ex-date (9 shares)")
print("=" * 78)
q = 9
print(f"  dividend received            +${q * DIV:.2f}")
print(f"  expected ex-date price drop  -${q * DIV:.2f}   (price opens ~ex-dividend)")
print(f"  pre-tax expected value        ${0.0:.2f}")
print(f"  qualified-dividend tax @15%  -${q * DIV * 0.15:.2f}   (the offsetting capital loss is only deferred)")
print(f"  after-tax expected value     -${q * DIV * 0.15:.2f}")
