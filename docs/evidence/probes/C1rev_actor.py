"""C1 adversarial re-check: ONE real agent, in its OWN OS process, talking real
HTTP (requests -> 127.0.0.1) to a local fake broker. No alpaca.markets socket:
the only rewrite is the URL host, asserted to stay on 127.0.0.1.

argv: <role: newsbot|autopilot|desk> <port> <start_at_epoch> <journal_db> <cap>
prints one JSON line with the outcome.
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

ROLE, PORT, START_AT, DB, CAP = (sys.argv[1], sys.argv[2], float(sys.argv[3]),
                                 sys.argv[4], sys.argv[5])
LOCAL = f"http://127.0.0.1:{PORT}"

# ---- the operator's real-money opt-in, exactly as documented ----
os.environ["ALPACA_TRADING_BASE_URL"] = "https://api.alpaca.markets"
os.environ["ALPACA_LIVE_TRADING"] = "true"
os.environ["ALPACA_LIVE_MAX_ORDERS_PER_DAY"] = CAP
os.environ["ALPACA_LIVE_MAX_ORDER_USD"] = "500"
os.environ["ALPACA_LIVE_MAX_DAILY_LOSS_USD"] = "200"
os.environ["ALPACA_API_KEY"] = "stub"
os.environ["ALPACA_SECRET_KEY"] = "stub"
os.environ["ALPACA_DATA_BASE_URL"] = LOCAL          # quotes go to the local stub too
os.environ["TRADER_JOURNAL_PATH"] = DB              # never touch the repo's data dir

import requests  # noqa: E402

_real_request = requests.request


def _rewrite(method, url, **kw):
    if url.startswith("https://api.alpaca.markets"):
        url = LOCAL + url[len("https://api.alpaca.markets"):]
    assert url.startswith(LOCAL), f"REFUSING non-local request: {url}"
    return _real_request(method, url, **kw)


requests.request = _rewrite

from trader import alpaca, newsbot  # noqa: E402
from trader.autotrader import AutoTrader, Limits, RunConfig  # noqa: E402
from trader.journal import Journal  # noqa: E402
from trader.portfolios import Portfolios  # noqa: E402

assert alpaca.account_mode() == "live", alpaca.account_mode()
NOW = datetime.now(timezone.utc)


def wait_for_start():
    while time.time() < START_AT:
        time.sleep(0.0005)


def run_newsbot():
    bot = newsbot.NewsBot(
        settings=newsbot.Settings(execution="live", symbols=frozenset({"ACME"}),
                                  order_usd=100.0, min_price=1.0),
        journal=Journal(DB),
        judge=lambda sym, art: {"relevance": 0.95, "materiality": 0.8, "p_bullish": 0.9,
                                "p_bearish": 0.02, "event_type": "product", "direction": 0.9},
        broker=alpaca,
    )
    article = {"title": "Acme wins a $2B contract", "description": "",
               "publishedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
               "symbols": ["ACME"]}
    wait_for_start()
    rec = bot.handle("ACME", article, received_at=datetime.now(timezone.utc), news_id="news-1")
    return {"role": "newsbot", "kind": rec.get("kind"), "order_id": rec.get("order_id"),
            "blocked": rec.get("blocked")}


def run_autopilot():
    pilot = AutoTrader(
        journal=Journal(DB),
        portfolios=Portfolios(DB + ".pf.json"),
        broker=alpaca,
        limits=Limits(min_confidence=0.5, max_trades_per_day=5, cooldown_minutes=0,
                      min_price=1.0, entry_cutoff_minutes=15),
    )
    cfg = RunConfig(symbols=["NVDA"], mode="live").validate()
    result = {"symbol": "NVDA", "recommendation": "BUY", "quantity": 10, "confidence": 0.8,
              "current_price": 10.0, "primary_strategy": "technical", "reason": "net indicators",
              "risk_parameters": {"stop_loss": 9.6, "take_profit": 10.6}}
    wait_for_start()
    out = pilot.act(result, cfg)
    return {"role": "autopilot", **out}


def run_desk():
    """The desk's manual ticket: alpaca.market_order_payload + alpaca.submit_order,
    the same two calls desk.py's POST /api/desk/orders makes."""
    payload = alpaca.market_order_payload("TSLA", "buy", 10)
    wait_for_start()
    try:
        order = alpaca.submit_order(payload)
        return {"role": "desk", "submitted": True, "order_id": order.get("id")}
    except Exception as exc:
        return {"role": "desk", "submitted": False, "skipped_reason": f"{type(exc).__name__}: {exc}"}


try:
    out = {"newsbot": run_newsbot, "autopilot": run_autopilot, "desk": run_desk}[ROLE]()
except Exception as exc:  # noqa: BLE001
    out = {"role": ROLE, "error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(out))
