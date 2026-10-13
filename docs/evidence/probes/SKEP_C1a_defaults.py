"""Do the SHIPPED DEFAULTS actually put two live submitters on the wire?"""
import os
for k in list(os.environ):
    if k.startswith(("NEWSBOT_", "DESK_", "AUTOPILOT_", "ALPACA_")):
        del os.environ[k]
os.environ.update({"ALPACA_TRADING_BASE_URL": "https://api.alpaca.markets", "ALPACA_LIVE_TRADING": "true",
                   "ALPACA_API_KEY": "PLACEHOLDER", "ALPACA_SECRET_KEY": "PLACEHOLDER"})

from trader import newsbot, desk, autotrader, alpaca

s = newsbot.load_settings()
print("NEWSBOT_EXECUTION default        =", repr(s.execution))
a = desk.agent_settings()
print("DESK_AUTOPILOT_MODE default      =", repr(a.autopilot_mode))
print("DESK_START_NEWSBOT default       =", a.start_newsbot)
print("LiveLimits defaults              =", alpaca.LiveLimits.from_env())
print()

calls = []
alpaca._request = lambda m, u, **k: calls.append((m, u))

# news bot, default settings, bullish signal, live account
bot = newsbot.NewsBot(s, journal=__import__("trader.journal", fromlist=["Journal"]).Journal(
    __import__("pathlib").Path("docs/evidence/probes/SKEP_def.db")))
print("newsbot._check_limits at defaults ->", bot._check_limits("AAPL", 10.0))

# autopilot, default mode
pilot = autotrader.AutoTrader.__new__(autotrader.AutoTrader)
pilot.limits = autotrader.Limits.from_env()
cfg = autotrader.RunConfig(["AAPL"], 300, a.autopilot_mode, "default")
res = {"symbol": "AAPL", "recommendation": "BUY", "confidence": 0.9, "current_price": 10.0, "quantity": 10}
print("autopilot._blocked at defaults    ->", repr(pilot._blocked(res, cfg)))
print("\nbroker calls made by either agent at defaults:", calls)
