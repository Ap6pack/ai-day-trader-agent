"""B1 part 2: how wide the stale-snapshot window is, and the no-race variant
(cancel fails for a transport reason, leg survives, sell still fires)."""
import json, os, time
import requests
os.environ["ALPACA_TRADING_BASE_URL"] = "https://api.alpaca.markets"
os.environ["ALPACA_LIVE_TRADING"] = "true"
from trader import alpaca

def parent(sym, oid):
    return {"id": oid, "client_order_id": f"newsbot-{oid}", "symbol": sym, "side": "buy",
            "type": "market", "order_class": "bracket", "qty": "5", "filled_qty": "5",
            "status": "filled",
            "legs": [{"id": oid + "-tp", "side": "sell", "type": "limit", "qty": "5",
                      "filled_qty": "0", "status": "new"},
                     {"id": oid + "-sl", "side": "sell", "type": "stop", "qty": "5",
                      "filled_qty": "0", "status": "held"}]}

print("=== B1b: one snapshot drives every symbol; retries make it stale by seconds ===")
TODAY = [parent("AAA", "o1"), parent("BBB", "o2"), parent("CCC", "o3")]
log, t0 = [], time.monotonic()
attempts = {"n": 0}
def fake(method, url, **kw):
    el = time.monotonic() - t0
    if method == "GET" and url.endswith("/v2/orders"):
        log.append(f"t={el:5.2f}s GET /v2/orders  (the ONLY read; snapshot taken here)")
        return json.loads(json.dumps(TODAY))
    if method == "DELETE":
        log.append(f"t={el:5.2f}s DELETE .../{url.rsplit('/',1)[-1]}")
        return None
    if method == "POST":
        p = kw["json"]
        attempts["n"] += 1
        # AAA's released shares are not settled yet -> Alpaca 403, flatten retries
        if p["symbol"] == "AAA" and attempts["n"] <= 3:
            log.append(f"t={el:5.2f}s POST sell {p['qty']} {p['symbol']} -> 403 insufficient qty (retry)")
            raise requests.HTTPError("403 Client Error: Forbidden (insufficient qty available for order)")
        log.append(f"t={el:5.2f}s POST sell {p['qty']} {p['symbol']} -> accepted")
        return {"id": "x", "status": "accepted", **p}
    return {"equity": "10000", "last_equity": "10000"}
alpaca._request = fake
alpaca.flatten_owned("newsbot-", wait_seconds=5.0)
for line in log:
    print("  ", line)
print("   -> CCC was sold using quantities read", f"{time.monotonic()-t0:.1f}s", "earlier.")
print()

print("=== B1c: no race at all - cancel fails for a transport reason, leg stays LIVE ===")
state = {"tp_live": True, "pos": 5.0}
def fake2(method, url, **kw):
    if method == "GET" and url.endswith("/v2/orders"):
        return [parent("NVDA", "p9")]
    if method == "DELETE":
        raise requests.ConnectionError("Connection aborted: RemoteDisconnected")  # swallowed by except Exception
    if method == "POST":
        p = kw["json"]; state["pos"] -= float(p["qty"])
        print(f"   POST sell {p['qty']} {p['symbol']} -> accepted; position now {state['pos']:g}")
        return {"id": "y", "status": "accepted", **p}
    return {"equity": "10000", "last_equity": "10000"}
alpaca._request = fake2
alpaca.flatten_owned("newsbot-", wait_seconds=0.0)
print(f"   take-profit leg p9-tp still working at the broker: {state['tp_live']}")
state["pos"] -= 5.0
print(f"   later, p9-tp fills 5 -> position {state['pos']:g}  (short, with no race required)")
