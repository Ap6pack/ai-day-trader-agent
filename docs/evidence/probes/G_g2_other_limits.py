"""G2 part 3: NaN/inf in alpaca LiveLimits, newsbot and autotrader. No network."""
import math, os, sys
from trader import alpaca, newsbot, autotrader, config

def reset(**kv):
    for k in list(os.environ):
        if k.startswith(("ALPACA_LIVE_", "AUTOPILOT_")):
            del os.environ[k]
    os.environ.update(kv)

print("=" * 72); print("alpaca._env_float / LiveLimits.from_env"); print("=" * 72)
for name in ("ALPACA_LIVE_MAX_ORDER_USD", "ALPACA_LIVE_MAX_ORDERS_PER_DAY", "ALPACA_LIVE_MAX_DAILY_LOSS_USD"):
    for val in ("nan", "inf", "-1"):
        reset(**{name: val})
        try:
            print(f"  {name}={val!r:6} -> {alpaca.LiveLimits.from_env()}")
        except Exception as e:
            print(f"  {name}={val!r:6} -> from_env raised {type(e).__name__}: {e}")

print()
print("check_live_buy with a NaN cap (price/account stubbed, NO network):")
alpaca.latest_price = lambda s: 100.0
alpaca.account = lambda: {"equity": "10000", "last_equity": "12000"}   # -$2,000 day P&L
alpaca.orders = lambda **kw: [{"side": "buy", "status": "filled"}] * 50
reset(ALPACA_LIVE_MAX_ORDER_USD="nan", ALPACA_LIVE_MAX_DAILY_LOSS_USD="nan")
lim = alpaca.LiveLimits(max_order_usd=float("nan"), max_orders_per_day=3,
                        max_daily_loss_usd=float("nan"))
try:
    alpaca.check_live_buy({"symbol": "AAPL", "side": "buy", "qty": "1000"}, limits=lim)
    print("  1000 AAPL @ $100 = $100,000 LIVE buy -> NO LiveLimitError raised for size/loss")
except alpaca.LiveLimitError as e:
    print("  blocked:", e)
alpaca.orders = lambda **kw: []
try:
    alpaca.check_live_buy({"symbol": "AAPL", "side": "buy", "qty": "1000"}, limits=lim)
    print("  with 0 prior orders -> check_live_buy returned cleanly: $100,000 LIVE buy PASSES")
except alpaca.LiveLimitError as e:
    print("  blocked:", e)
print("  max_live_qty($100) with NaN cap =", end=" ")
try:
    print(alpaca.max_live_qty(100.0, lim))
except Exception as e:
    print(f"raised {type(e).__name__}: {e}")

print()
print("=" * 72); print("newsbot NEWSBOT_ORDER_USD"); print("=" * 72)
base = {"NEWSBOT_EXECUTION": "paper", "NEWSBOT_SYMBOLS": "AAPL"}
for val in ("nan", "inf", "-1", "0"):
    env = dict(base, NEWSBOT_ORDER_USD=val)
    try:
        s = newsbot.load_settings(env)
        print(f"  NEWSBOT_ORDER_USD={val!r:5} -> load_settings OK, order_usd={s.order_usd!r}"
              f"   (guard `order_usd <= 0` = {s.order_usd <= 0})")
        try:
            print(f"      math.floor(order_usd/100.0) = {math.floor(s.order_usd/100.0)}")
        except Exception as e:
            print(f"      math.floor(order_usd/100.0) raised {type(e).__name__}: {e}")
    except Exception as e:
        print(f"  NEWSBOT_ORDER_USD={val!r:5} -> load_settings raised {type(e).__name__}: {e}")
for key in ("NEWSBOT_MIN_RELEVANCE", "NEWSBOT_MIN_MATERIALITY", "NEWSBOT_MIN_PROBABILITY",
            "NEWSBOT_TAKE_PROFIT_PCT", "NEWSBOT_STOP_LOSS_PCT", "NEWSBOT_MIN_PRICE"):
    env = dict(base, **{key: "nan"})
    try:
        s = newsbot.load_settings(env)
        print(f"  {key}=nan -> accepted ({getattr(s, {'NEWSBOT_MIN_RELEVANCE':'min_relevance','NEWSBOT_MIN_MATERIALITY':'min_materiality','NEWSBOT_MIN_PROBABILITY':'min_probability','NEWSBOT_TAKE_PROFIT_PCT':'take_profit_pct','NEWSBOT_STOP_LOSS_PCT':'stop_loss_pct','NEWSBOT_MIN_PRICE':'min_price'}[key])!r})")
    except Exception as e:
        print(f"  {key}=nan -> rejected: {type(e).__name__}: {e}")

print()
print("=" * 72); print("autotrader AUTOPILOT_*"); print("=" * 72)
for name in ("AUTOPILOT_MIN_CONFIDENCE", "AUTOPILOT_MAX_TRADES_PER_DAY", "AUTOPILOT_MIN_PRICE",
             "AUTOPILOT_MAX_SYMBOLS"):
    for val in ("nan", "inf"):
        reset(**{name: val})
        try:
            print(f"  {name}={val!r:5} -> {autotrader.Limits.from_env()}")
        except Exception as e:
            print(f"  {name}={val!r:5} -> from_env raised {type(e).__name__}: {e}")
reset(AUTOPILOT_MIN_CONFIDENCE="nan")
lim = autotrader.Limits.from_env()
print(f"  gate `confidence < min_confidence` for confidence=0.01: {0.01 < lim.min_confidence}"
      f"  -> a 1%-confidence signal is NOT skipped")
