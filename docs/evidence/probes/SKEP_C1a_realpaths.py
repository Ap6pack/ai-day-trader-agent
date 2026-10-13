"""SKEPTIC probe for C1a: drive the REAL callers, not alpaca.submit_order directly.

Actor 1: the actual FastAPI route POST /api/desk/orders (confirm_live=true), via TestClient.
Actor 2: the actual AutoTrader.act() in live mode (which DOES hold _trade_lock).

Both must reach the broker through trader.alpaca.submit_order.
Latency of the fake broker is a parameter so we can see how wide the window must be.
No socket is ever opened to Alpaca.
"""
import os, sys, threading, time, json

LAT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.06   # one-way-ish RTT per call

os.environ["ALPACA_TRADING_BASE_URL"] = "https://api.alpaca.markets"
os.environ["ALPACA_LIVE_TRADING"] = "true"
os.environ["ALPACA_LIVE_MAX_ORDERS_PER_DAY"] = "1"
os.environ["ALPACA_LIVE_MAX_ORDER_USD"] = "500"
os.environ["ALPACA_LIVE_MAX_DAILY_LOSS_USD"] = "200"
os.environ["ALPACA_API_KEY"] = "stub"
os.environ["ALPACA_SECRET_KEY"] = "stub"
os.environ.pop("DESK_TOKEN", None)
os.environ.pop("DESK_DEMO_MODE", None)
os.environ["DESK_AUTOPILOT_MODE"] = "off"
os.environ["DESK_START_NEWSBOT"] = "false"

from trader import alpaca, autotrader, desk          # noqa: E402
from trader.journal import Journal                   # noqa: E402
from trader.portfolios import Portfolios             # noqa: E402
from fastapi.testclient import TestClient            # noqa: E402

T0 = time.monotonic()
LOG, ORDERS = [], []
BLOCK = threading.Lock()
N = [0]

def note(m):
    with BLOCK:
        LOG.append(f"[{time.monotonic()-T0:6.3f}s] {threading.current_thread().name:<14} {m}")

def fake_request(method, url, **kw):
    if "/trades/latest" in url:
        return {"trade": {"p": 10.0}}
    if url.endswith("/v2/account"):
        return {"equity": "100000", "last_equity": "100000", "buying_power": "100000", "status": "ACTIVE"}
    if url.endswith("/v2/clock"):
        return {"is_open": True, "next_close": "2099-01-01T16:00:00-05:00"}
    if method == "GET" and url.endswith("/v2/orders"):
        note("GET /v2/orders  ->")
        time.sleep(LAT)
        with BLOCK:
            snap = [dict(o) for o in ORDERS]
        note(f"GET /v2/orders  <- {len(snap)} order(s)")
        return snap
    if method == "POST" and url.endswith("/v2/orders"):
        b = kw.get("json") or {}
        note(f"POST /v2/orders {b.get('side')} {b.get('qty')} {b.get('symbol')} ->")
        time.sleep(LAT)
        with BLOCK:
            N[0] += 1
            o = {"id": f"ord-{N[0]}", "symbol": b.get("symbol"), "qty": b.get("qty"), "side": b.get("side"),
                 "status": "accepted", "filled_qty": "0", "client_order_id": b.get("client_order_id")}
            ORDERS.append(o)
        note(f"POST /v2/orders <- accepted {o['id']} (book={len(ORDERS)})")
        return o
    if method == "GET" and url.endswith("/v2/positions"):
        return []
    raise AssertionError(f"unexpected {method} {url}")

alpaca._request = fake_request
print("account_mode() =", alpaca.account_mode())
print("limits         =", alpaca.LiveLimits.from_env())
print(f"fake broker latency per call = {LAT*1000:.0f} ms\n")

# --- build the REAL desk app and the REAL AutoTrader in live mode -------------------
import tempfile, pathlib
tmp = pathlib.Path(tempfile.mkdtemp(prefix="skep_c1a_"))
journal = Journal(tmp / "SKEP_desk.db")
app = desk.create_app(journal=journal, background=False, start_agents=False)
pilot = app.state.autopilot
cfg = autotrader.RunConfig(["AAPL"], 60, "live", "default").validate()
pilot.config = cfg
pilot.limits = autotrader.Limits.from_env()

analysis_result = {
    "symbol": "AAPL", "recommendation": "BUY", "confidence": 0.9, "current_price": 10.0,
    "quantity": 10, "reason": "probe", "risk_parameters": {"stop_loss": 9.9, "take_profit": 10.2},
}

results = {}
bar = threading.Barrier(2)

def run_autopilot():
    bar.wait()
    try:
        results["autopilot"] = pilot.act(dict(analysis_result), cfg)
    except Exception as exc:
        results["autopilot"] = f"EXC {type(exc).__name__}: {exc}"

def run_desk_ticket(client):
    bar.wait()
    r = client.post("/api/desk/orders",
                    json={"symbol": "MSFT", "side": "BUY", "qty": 10,
                          "destination": "alpaca", "confirm_live": True})
    results["desk-ticket"] = (r.status_code, r.json())

with TestClient(app) as client:
    # warm the route once in PAPER-free way? no - go straight to the race.
    ta = threading.Thread(target=run_autopilot, name="autopilot")
    td = threading.Thread(target=run_desk_ticket, args=(client,), name="desk-ticket")
    ta.start(); td.start(); ta.join(); td.join()

print("wire trace (real code paths):")
for line in LOG:
    print("  ", line)
print()
print("autopilot   ->", json.dumps(results.get("autopilot"), default=str)[:300])
print("desk ticket ->", json.dumps(results.get("desk-ticket"), default=str)[:300])
print(f"\nALPACA_LIVE_MAX_ORDERS_PER_DAY=1; broker book holds {len(ORDERS)} LIVE buy order(s):")
for o in ORDERS:
    print("   ", o["id"], o["side"], o["qty"], o["symbol"], o["client_order_id"])
print("\nVERDICT:", "CAP BREACHED via real callers" if len(ORDERS) > 1 else "cap HELD")
