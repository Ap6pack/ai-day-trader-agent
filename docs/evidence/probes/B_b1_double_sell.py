"""B1: stop-loss leg fills between owned_positions() read and cancel_order().
No network: trader.alpaca._request is replaced by a ledger-backed fake."""
import json, os, sys
import requests

os.environ["ALPACA_TRADING_BASE_URL"] = "https://api.alpaca.markets"   # LIVE path (flatten_owned is live-only)
os.environ["ALPACA_LIVE_TRADING"] = "true"

from trader import alpaca

assert alpaca.account_mode() == "live"

# ---- the broker ledger (what the account really holds) -------------------------------
agent_shares = {"NVDA": 5.0}      # 5 shares the newsbot bought today
my_own_shares = {"NVDA": 0.0}     # personal long in the same name (0 here -> short on over-sell)
call_log = []
stop_has_fired = {"yet": False}

# Alpaca order JSON as returned by GET /v2/orders?nested=true BEFORE the stop fires.
# Realistic bracket: parent = market BUY (filled), legs = sell limit (take profit) + sell stop (stop loss), OCO.
SNAPSHOT_BEFORE = [{
    "id": "b1a2c3d4-0000-4000-8000-000000000001",
    "client_order_id": "newsbot-6c0d1e2f-aaaa-4bbb-8ccc-dddddddddddd",
    "symbol": "NVDA", "side": "buy", "type": "market", "order_class": "bracket",
    "qty": "5", "filled_qty": "5", "filled_avg_price": "182.40", "status": "filled",
    "legs": [
        {"id": "b1a2c3d4-0000-4000-8000-00000000tp01", "client_order_id": "tp-leg",
         "symbol": "NVDA", "side": "sell", "type": "limit", "limit_price": "186.05",
         "qty": "5", "filled_qty": "0", "status": "new", "order_class": "bracket"},
        {"id": "b1a2c3d4-0000-4000-8000-00000000sl01", "client_order_id": "sl-leg",
         "symbol": "NVDA", "side": "sell", "type": "stop", "stop_price": "180.58",
         "qty": "5", "filled_qty": "0", "status": "held", "order_class": "bracket"},
    ],
}]

def fire_the_stop():
    """The market ticks through 180.58 right after flatten_owned() read the orders."""
    stop_has_fired["yet"] = True
    agent_shares["NVDA"] -= 5.0          # the stop leg SOLD the 5 shares
    print("   [market] stop leg sl01 FILLED 5 NVDA @ 180.55 -> agent position now "
          f"{agent_shares['NVDA']:g}; OCO sibling tp01 auto-canceled")

def fake_request(method, url, **kw):
    call_log.append((method, url.split("alpaca.markets")[-1], kw.get("json")))
    if method == "GET" and url.endswith("/v2/orders"):
        print(f"   [api ] GET /v2/orders (nested) -> snapshot: NVDA qty 5, legs tp01/sl01 open")
        out = json.loads(json.dumps(SNAPSHOT_BEFORE))
        fire_the_stop()                   # the race: fill lands after the read, before the cancels
        return out
    if method == "DELETE" and "/v2/orders/" in url:
        oid = url.rsplit("/", 1)[-1][-4:]
        if stop_has_fired["yet"]:
            why = "order is already filled" if oid == "sl01" else "order is already canceled (OCO sibling filled)"
            print(f"   [api ] DELETE /v2/orders/...{oid} -> 422 {why}")
            raise requests.HTTPError(f"422 Client Error: Unprocessable Entity ({why})")
        print(f"   [api ] DELETE /v2/orders/...{oid} -> 204")
        return None
    if method == "POST" and url.endswith("/v2/orders"):
        p = kw["json"]
        q = float(p["qty"])
        print(f"   [api ] POST /v2/orders  {p['side'].upper()} {p['qty']} {p['symbol']} "
              f"{p['type']}  client_order_id={p['client_order_id']}")
        agent_shares[p["symbol"]] -= q
        return {"id": "flat-order-1", "status": "accepted", **p}
    if url.endswith("/v2/account"):
        return {"equity": "10000", "last_equity": "10000"}
    raise AssertionError(f"unexpected call {method} {url}")

alpaca._request = fake_request

print("=== B1: stop fills between the owned_positions() read and the cancel ===")
print(f"start: agent-owned NVDA = {agent_shares['NVDA']:g}, personal NVDA = {my_own_shares['NVDA']:g}")
print("--- flatten_owned('newsbot-') ---")
results = alpaca.flatten_owned("newsbot-", wait_seconds=0.0)
print("--- returned ---")
print("flatten_owned returned", len(results), "order(s):",
      [(r["side"], r["qty"], r["symbol"]) for r in results])
print(f"END STATE: agent NVDA position = {agent_shares['NVDA']:g}   "
      f"(0 = flat as intended, negative = UNINTENDED SHORT)")
print()
print("call sequence seen by the broker:")
for m, u, body in call_log:
    print(f"   {m:6} {u}" + (f"   qty={body['qty']} side={body['side']}" if body else ""))
print()
reads_after_first = [c for c in call_log[1:] if c[0] == "GET" and c[1].endswith("/v2/orders")]
print(f"re-reads of /v2/orders between the snapshot and the market sell: {len(reads_after_first)}"
      "  -> entry['qty'] is NEVER recomputed" if not reads_after_first else "RECOMPUTED")
