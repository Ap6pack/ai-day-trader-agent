"""I1 probe: does a confirming signal suppress the conflicting-signal penalty in analysis.fuse()?"""
import itertools
import os

os.environ.setdefault("MIN_POSITION_PERCENTAGE", "0.02")
os.environ.setdefault("MAX_POSITION_PERCENTAGE", "0.10")

from trader import analysis


def s(sig, strength):
    return {"signal": sig, "strength": strength, "reason": f"{sig}@{strength}"}


def sane(base, confirming, conflicting):
    """What an additive formula would produce."""
    return round(max(0.0, min(base + 0.1 * confirming - 0.15 * conflicting, 0.95)), 3)


print("=" * 78)
print("A. The exact scenario from the claim: technical BUY .8, dividend BUY .5, sentiment SELL 1.0")
print("=" * 78)
sig_a = {"technical": s("BUY", 0.8), "dividend": s("BUY", 0.5), "sentiment": s("SELL", 1.0)}
f_a = analysis.fuse(sig_a)
print("fuse ->", f_a)
print("sane additive formula would give:", sane(f_a_base := 0.5, f_a["confirming"], f_a["conflicting"]))

print()
print("B. Same, but the bearish news signal replaced by a NEUTRAL one (sentiment HOLD 0.0)")
sig_b = {"technical": s("BUY", 0.8), "dividend": s("BUY", 0.5), "sentiment": s("HOLD", 0.0)}
f_b = analysis.fuse(sig_b)
print("fuse ->", f_b)
print(f"  confidence with maximally BEARISH news = {f_a['confidence']}")
print(f"  confidence with NEUTRAL news           = {f_b['confidence']}")
print(f"  delta from a maximally bearish news signal = {round(f_a['confidence'] - f_b['confidence'], 6)}")

print()
print("C. Control: remove the confirming technical (technical HOLD), keep the SELL")
sig_c = {"technical": s("HOLD", 0.0), "dividend": s("BUY", 0.5), "sentiment": s("SELL", 1.0)}
f_c = analysis.fuse(sig_c)
print("fuse ->", f_c)
print(f"  penalty IS applied when confirming == 0: {f_c['confidence']} = max(0.5 - 0.15*1, 0.2)")
print(f"  so one bullish confirmation swings confidence by "
      f"{round(f_a['confidence'] - f_c['confidence'], 3)} (+0.10 bonus AND -0.15 penalty erased)")

print()
print("=" * 78)
print("D. Exhaustive sweep: every (primary-non-HOLD) combination with conflicting>=1")
print("   Does confidence EVER drop below base when confirming>=1?")
print("=" * 78)
opts = [("BUY", 0.8), ("SELL", 0.8), ("HOLD", 0.0), ("BUY", 0.5), ("SELL", 1.0)]
rows = []
hold_counted_as_confirming = []
penalty_applied_with_confirmation = []
for t, d, n in itertools.product(opts, repeat=3):
    sigs = {"technical": s(*t), "dividend": s(*d), "sentiment": s(*n)}
    f = analysis.fuse(sigs)
    prim = f["primary_strategy"]
    base = sigs[prim]["strength"]
    others = [v["signal"] for k, v in sigs.items() if k != prim]
    if f["confirming"] > sum(1 for o in others if o == f["signal"] and f["signal"] != "HOLD"):
        hold_counted_as_confirming.append((sigs, f))
    if f["confirming"] >= 1 and f["conflicting"] >= 1:
        rows.append((f["signal"], prim, base, f["confirming"], f["conflicting"], f["confidence"],
                     sane(base, f["confirming"], f["conflicting"])))
        if f["confidence"] < base:
            penalty_applied_with_confirmation.append((sigs, f))
print(f"combinations with confirming>=1 AND conflicting>=1: {len(rows)}")
seen = set()
print(f"{'signal':6} {'primary':10} {'base':>5} {'conf#':>5} {'confl#':>6} {'ACTUAL':>7} {'SANE':>6}  penalty?")
for r in rows:
    if r in seen:
        continue
    seen.add(r)
    print(f"{r[0]:6} {r[1]:10} {r[2]:5.2f} {r[3]:5d} {r[4]:6d} {r[5]:7.3f} {r[6]:6.3f}  "
          f"{'APPLIED' if r[5] < r[2] else 'SUPPRESSED'}")
print()
print("cases where a conflict lowered confidence while confirming>=1:",
      len(penalty_applied_with_confirmation))
print("cases where a HOLD was miscounted as confirming:", len(hold_counted_as_confirming))

print()
print("=" * 78)
print("E. Does the confirming counter count a HOLD? (generator-expression check)")
print("=" * 78)
f_hold = analysis.fuse({"technical": s("HOLD", 0.0), "dividend": s("HOLD", 0.0), "sentiment": s("HOLD", 0.0)})
print("all three HOLD ->", f_hold)
f_bh = analysis.fuse({"technical": s("BUY", 0.6), "dividend": s("HOLD", 0.0), "sentiment": s("HOLD", 0.0)})
print("BUY + two HOLD ->", f_bh)
print("  -> confirming counts only signals EQUAL to a non-HOLD primary, so a HOLD can never be counted.")

print()
print("=" * 78)
print("F. Dollar effect through position_size() (capital $10,000, price $100.00, no vol input)")
print("=" * 78)
CAP, PRICE = 10000.0, 100.0
for label, strengths in [("dividend BUY 0.70 (ex-date tomorrow) + technical BUY 0.80 + sentiment SELL 1.00",
                          {"dividend": ("BUY", 0.7), "technical": ("BUY", 0.8), "sentiment": ("SELL", 1.0)}),
                         ("dividend BUY 0.50 + technical BUY 0.80 + sentiment SELL 1.00",
                          {"dividend": ("BUY", 0.5), "technical": ("BUY", 0.8), "sentiment": ("SELL", 1.0)})]:
    sigs = {k: s(*v) for k, v in strengths.items()}
    f = analysis.fuse(sigs)
    base_strength = sigs[f["primary_strategy"]]["strength"]
    sane_conf = sane(base_strength, f["confirming"], f["conflicting"])
    q_actual = analysis.position_size("BUY", f["confidence"], PRICE, CAP, None, 900)
    q_sane = analysis.position_size("BUY", sane_conf, PRICE, CAP, None, 900)
    tier = lambda c: "MAX 10%" if c >= 0.7 else "MID 6%" if c >= 0.5 else "MIN 2%"
    print(f"\n{label}")
    print(f"  actual confidence {f['confidence']:.3f} [{tier(f['confidence'])}] -> qty {q_actual} "
          f"= ${q_actual * PRICE:,.2f}")
    print(f"  sane   confidence {sane_conf:.3f} [{tier(sane_conf)}] -> qty {q_sane} "
          f"= ${q_sane * PRICE:,.2f}")
    print(f"  AUTOPILOT_MIN_CONFIDENCE=0.5 gate: actual {'PASSES' if f['confidence'] >= 0.5 else 'BLOCKS'}, "
          f"sane {'PASSES' if sane_conf >= 0.5 else 'BLOCKS'}")
    print(f"  extra exposure caused by the suppressed penalty: ${(q_actual - q_sane) * PRICE:,.2f}")
