"""G2: NaN / inf / negative in numeric config."""
import math, tempfile, json
from datetime import datetime, timezone
from pathlib import Path
from trader import config, guard
from trader.journal import Journal

NOW = datetime(2026, 10, 13, 15, 0, 0, tzinfo=timezone.utc)
BIG_BUY = {"symbol": "AAPL", "side": "buy", "dollar_amount": 10000.0}

def try_limits(envextra):
    env = {"TRADER_MODE": "live"}
    env.update(envextra)
    try:
        return config.load_limits(env), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

print("=" * 72)
print("PART 1  TRADER_MAX_ORDER_USD")
print("=" * 72)
for val in ["nan", "NaN", "inf", "infinity", "-inf", "-1", "500"]:
    lim, err = try_limits({"TRADER_MAX_ORDER_USD": val})
    if err:
        print(f"TRADER_MAX_ORDER_USD={val!r:12} -> load_limits raised {err}")
        continue
    with tempfile.TemporaryDirectory() as td:
        j = Journal(Path(td) / "j.db")
        j.log_event("review", symbol="AAPL", side="buy", now=NOW)   # satisfy review gate
        v = guard.evaluate_order("place_equity_order", BIG_BUY, lim, j, now=NOW)
        print(f"TRADER_MAX_ORDER_USD={val!r:12} -> limits.max_order_usd={lim.max_order_usd!r:8} "
              f" $10,000 buy: allowed={v.allowed}  reason={v.reason}")

print()
print("  end-to-end through guard.handle() with TRADER_MAX_ORDER_USD=nan:")
lim, _ = try_limits({"TRADER_MAX_ORDER_USD": "nan"})
with tempfile.TemporaryDirectory() as td:
    j = Journal(Path(td) / "j.db")
    j.log_event("review", symbol="AAPL", side="buy", now=NOW)
    out, code = guard.handle({"hook_event_name": "PreToolUse",
                              "tool_name": "mcp__Robinhood__place_equity_order",
                              "tool_input": BIG_BUY}, journal=j, limits=lim, now=NOW)
    print("   guard.handle ->", out, code, "  (None == no deny, order proceeds to the permission prompt)")
    print("   journal:", [ (dict(r)['kind'], dict(r)['detail']) for r in j.events(1) ])

print()
print("=" * 72)
print("PART 2  every other trader.config limit")
print("=" * 72)
for name, val in [("TRADER_MAX_ORDERS_PER_DAY", "nan"), ("TRADER_MAX_ORDERS_PER_DAY", "inf"),
                  ("TRADER_MAX_ORDERS_PER_DAY", "1e9"), ("TRADER_MAX_ORDERS_PER_DAY", "-1"),
                  ("TRADER_REVIEW_WINDOW_MINUTES", "nan"), ("TRADER_REVIEW_WINDOW_MINUTES", "inf"),
                  ("TRADER_REVIEW_WINDOW_MINUTES", "-1"),
                  ("TRADER_REVIEW_WINDOW_MINUTES", "999999999999")]:
    lim, err = try_limits({name: val})
    if err:
        print(f"{name}={val!r:16} -> load_limits raised {err}")
        continue
    with tempfile.TemporaryDirectory() as td:
        j = Journal(Path(td) / "j.db")
        j.log_event("review", symbol="AAPL", side="buy", now=NOW)
        try:
            v = guard.evaluate_order("place_equity_order",
                                     {"symbol": "AAPL", "side": "buy", "dollar_amount": 100.0},
                                     lim, j, now=NOW)
            print(f"{name}={val!r:16} -> parsed; $100 buy allowed={v.allowed} reason={v.reason}")
        except Exception as e:
            print(f"{name}={val!r:16} -> parsed; evaluate_order raised {type(e).__name__}: {e}")

print()
print("  NaN via config._int / _float directly:")
for fn, nm in ((config._int, "_int"), (config._float, "_float")):
    for v in ["nan", "inf", "-inf", "1e3"]:
        try:
            print(f"    config.{nm}({{X:{v!r}}}) = {fn({'X': v}, 'X', 1)!r}")
        except Exception as e:
            print(f"    config.{nm}({{X:{v!r}}}) raised {type(e).__name__}: {e}")
