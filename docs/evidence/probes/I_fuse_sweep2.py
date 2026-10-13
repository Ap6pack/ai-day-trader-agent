"""I1 addendum: (a) prove the conflicting count has ZERO effect whenever confirming>=1,
(b) establish which signal shapes can even reach that branch with the 3 strategies that exist."""
import itertools

from trader import analysis


def s(sig, strength):
    return {"signal": sig, "strength": strength, "reason": f"{sig}@{strength}"}


STRENGTHS = [0.0, 0.3, 0.5, 0.7, 0.8, 1.0]
SIGS = ["BUY", "SELL", "HOLD"]
names = ["technical", "sentiment", "dividend"]

hits, mismatches, shapes = 0, [], {}
for combo in itertools.product([(a, b) for a in SIGS for b in STRENGTHS], repeat=3):
    sigs = {n: s(*c) for n, c in zip(names, combo)}
    f = analysis.fuse(sigs)
    if f["confirming"] >= 1 and f["conflicting"] >= 1:
        hits += 1
        shapes[f["primary_strategy"]] = shapes.get(f["primary_strategy"], 0) + 1
        # neutralize every conflicting signal -> HOLD, refuse to change anything else
        neutral = {n: (s("HOLD", 0.0) if v["signal"] not in ("HOLD", f["signal"]) else v)
                   for n, v in sigs.items()}
        g = analysis.fuse(neutral)
        if g["confidence"] != f["confidence"]:
            mismatches.append((sigs, f, g))

print(f"combinations reaching confirming>=1 AND conflicting>=1 : {hits}")
print(f"  of those, confidence CHANGED when the conflicting signal was neutralized: {len(mismatches)}")
print(f"  primary strategy in those combinations: {shapes}")
print()
print("Can the dividend strategy ever emit SELL?  grep of trader/dividends.py returns:")
import re
src = open("trader/dividends.py").read()
print("   ", sorted(set(re.findall(r'"signal": "(\w+)"', src))))
print()
print("So with primary=dividend the dividend signal must be BUY, and the split must be")
print("1 confirming BUY + 1 conflicting SELL among {technical, sentiment}.")
print()
print("Control: with the dividend strategy OFF (HOLD) a conflict is always penalized:")
for t, n in [(("BUY", 0.8), ("SELL", 1.0)), (("SELL", 0.9), ("BUY", 1.0))]:
    f = analysis.fuse({"technical": s(*t), "sentiment": s(*n), "dividend": s("HOLD", 0.0)})
    print(f"   technical {t} + sentiment {n} + dividend HOLD -> {f}")
