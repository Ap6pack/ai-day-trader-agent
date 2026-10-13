"""I2 part 2: does PRIORITY=3 let the dividend signal override bearish technical/sentiment?
I2 part 3: dividends.py:91 excludes specials from `past`, :92 does not exclude them from `upcoming`."""
import os
from datetime import date, timedelta

os.environ["MIN_POSITION_PERCENTAGE"] = "0.02"
os.environ["MAX_POSITION_PERCENTAGE"] = "0.10"
os.environ["DIVIDEND_STRATEGY_ENABLED"] = "true"
os.environ["DIVIDEND_CAPTURE_WINDOW_DAYS"] = "7"
os.environ["MIN_DIVIDEND_YIELD"] = "6.0"

from trader import analysis, dividends

TODAY = date(2026, 10, 13)
MIN_CONF = 0.5  # AUTOPILOT_MIN_CONFIDENCE default


def s(sig, strength, reason="r"):
    return {"signal": sig, "strength": strength, "reason": reason}


print("=" * 78)
print("A. PRIORITY override: dividend=3 > technical=2 > sentiment=1")
print("=" * 78)
cases = [
    ("dividend BUY 0.70 (ex tomorrow) | technical SELL 0.90 | sentiment SELL 1.00",
     {"dividend": s("BUY", 0.70), "technical": s("SELL", 0.90), "sentiment": s("SELL", 1.00)}),
    ("dividend BUY 0.70 (ex tomorrow) | technical SELL 0.90 | sentiment HOLD 0.00",
     {"dividend": s("BUY", 0.70), "technical": s("SELL", 0.90), "sentiment": s("HOLD", 0.00)}),
    ("dividend BUY 0.70 (ex tomorrow) | technical SELL 1.00 | sentiment BUY 0.40",
     {"dividend": s("BUY", 0.70), "technical": s("SELL", 1.00), "sentiment": s("BUY", 0.40)}),
    ("dividend BUY 0.30 (ex in 7d)    | technical SELL 1.00 | sentiment SELL 1.00",
     {"dividend": s("BUY", 0.30), "technical": s("SELL", 1.00), "sentiment": s("SELL", 1.00)}),
]
for label, sigs in cases:
    f = analysis.fuse(sigs)
    q = analysis.position_size(f["signal"], f["confidence"], 54.80, 10000, None, 900)
    gate = "ORDER PLACED" if f["confidence"] >= MIN_CONF and q >= 1 else \
        f"blocked (confidence {f['confidence']:.0%} < {MIN_CONF:.0%})"
    print(f"\n  {label}")
    print(f"    -> recommendation {f['signal']} via {f['primary_strategy']}, confidence {f['confidence']}, "
          f"confirming {f['confirming']}, conflicting {f['conflicting']}")
    print(f"    -> qty {q} (${q * 54.80:,.2f}) at AUTOPILOT_MIN_CONFIDENCE=0.5: {gate}")
    print(f"    -> journal.add_decision() records a BUY thesis regardless of the gate "
          f"(autotrader.py:330-334)")

print()
print("=" * 78)
print("B. Special dividends: `past` filters them out, `upcoming` does not")
print("=" * 78)
print("  dividends.py:91  past     = [... and not r.get('special')]")
print("  dividends.py:92  upcoming = sorted((r for r in rows if ex_date(r) > today), key=ex_date)")
print()


def show(label, rows, price):
    sig = dividends.dividend_signal("X", price, today=TODAY, rows=rows)
    print(f"  {label}")
    print(f"    {sig['signal']:4} strength={sig.get('strength')} yield_pct={sig['yield_pct']} "
          f"trailing_annual_dividend={sig['trailing_annual_dividend']}")
    print(f"    next_ex_date={sig['next_ex_date']} next_amount={sig['next_amount']} "
          f"days={sig.get('days_to_ex_dividend')}")
    print(f"    reason: {sig['reason']}")
    print()
    return sig


regular_past = [{"ex_date": (TODAY - timedelta(days=d)).isoformat(), "rate": 0.90, "special": False}
                for d in (300, 210, 120, 30)]

# 1. upcoming SPECIAL inside the window, trailing yield from REGULARS only
rows1 = regular_past + [{"ex_date": (TODAY + timedelta(days=2)).isoformat(), "rate": 7.50, "special": True}]
a = show("1) upcoming row is a SPECIAL $7.50 in 2 days; trailing yield is the regular 6.6%",
         rows1, 54.80)

# 2. the special PREEMPTS the regular ex-date that is also upcoming
rows2 = regular_past + [{"ex_date": (TODAY + timedelta(days=2)).isoformat(), "rate": 7.50, "special": True},
                        {"ex_date": (TODAY + timedelta(days=6)).isoformat(), "rate": 0.90, "special": False}]
b = show("2) a SPECIAL in 2d and the real quarterly in 6d -> which one does the signal time against?",
         rows2, 54.80)

# 3. a company whose ONLY dividends are specials: yield 0 -> no BUY (the exclusion does bite here)
rows3 = [{"ex_date": (TODAY - timedelta(days=200)).isoformat(), "rate": 10.0, "special": True},
         {"ex_date": (TODAY + timedelta(days=2)).isoformat(), "rate": 10.0, "special": True}]
c = show("3) ONLY special dividends (past $10 special excluded from the yield)", rows3, 54.80)

# 4. same rows but with the past special counted -> what a consistent implementation would do
annual_with_special = 10.0
print(f"    had the past special been counted, yield would be "
      f"{annual_with_special / 54.80 * 100:.1f}% >= 6.0% -> a capture BUY. "
      f"As written the yield is {c['yield_pct']}% -> {c['signal']}.")

print()
print("  VERDICT on the asymmetry:")
print(f"    case 1: the special DID trigger the capture window check and became next_ex_date/next_amount")
print(f"            (signal={a['signal']}, next_amount={a['next_amount']}) while the printed "
         f"'{a['yield_pct']}% yield' came from regulars only.")
print(f"    case 2: the special at 2d displaced the real quarterly at 6d as next_ex_date "
      f"({b['next_ex_date']}, ${b['next_amount']}).")
print(f"    case 3: a special-only payer gets yield 0% and is NOT bought, "
      f"so the exclusion is directionally conservative there.")
