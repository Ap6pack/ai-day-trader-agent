"""B2: does flatten_owned() ever cancel a WORKING BUY ENTRY? Three realistic shapes."""
import json, os
import requests
os.environ["ALPACA_TRADING_BASE_URL"] = "https://api.alpaca.markets"
os.environ["ALPACA_LIVE_TRADING"] = "true"
from trader import alpaca

print("OPEN_ORDER_STATUSES =", alpaca.OPEN_ORDER_STATUSES)
print()

def run(title, todays, note=""):
    log = []
    def fake(method, url, **kw):
        tail = url.split("alpaca.markets")[-1]
        if method == "GET" and tail == "/v2/orders":
            log.append("GET /v2/orders")
            return json.loads(json.dumps(todays))
        if method == "DELETE":
            log.append("DELETE /v2/orders/" + url.rsplit("/", 1)[-1])
            raise requests.HTTPError("422 Client Error: order is not cancelable (bracket leg is held)")
        if method == "POST":
            p = kw["json"]
            log.append(f"POST /v2/orders  {p['side']} {p['qty']} {p['symbol']}")
            return {"id": "z", "status": "accepted", **p}
        return {"equity": "10000", "last_equity": "10000"}
    alpaca._request = fake
    print("---", title, "---")
    print("owned_positions('newsbot-') =", alpaca.owned_positions("newsbot-"))
    res = alpaca.flatten_owned("newsbot-", wait_seconds=0.0)
    print("flatten_owned returned:", res)
    print("broker calls made:", log)
    print("DELETEs issued for the BUY PARENT id:",
          [c for c in log if any(p["id"] in c for p in todays)] or "NONE")
    if note:
        print("note:", note)
    print()

# (a) working BUY entry with no legs yet (simple buy, or a bracket still pending_new)
run("(a) unfilled BUY, status 'new', no legs", [{
    "id": "parent-A", "client_order_id": "newsbot-aaaa", "symbol": "NVDA", "side": "buy",
    "type": "market", "order_class": "simple", "qty": "5", "filled_qty": "0",
    "status": "new", "legs": []}],
    "symbol dropped by `if e['qty'] > 1e-9 or e['open_order_ids']` -> flatten is a no-op")

# (b) the realistic newsbot/autotrader shape: bracket parent not yet filled, legs 'held'
run("(b) unfilled bracket BUY parent 'accepted', legs 'held' (sell limit + sell stop, OCO)", [{
    "id": "parent-B", "client_order_id": "newsbot-bbbb", "symbol": "NVDA", "side": "buy",
    "type": "market", "order_class": "bracket", "qty": "5", "filled_qty": "0",
    "status": "accepted",
    "legs": [{"id": "B-tp", "side": "sell", "type": "limit", "qty": "5", "filled_qty": "0", "status": "held"},
             {"id": "B-sl", "side": "sell", "type": "stop", "qty": "5", "filled_qty": "0", "status": "held"}]}],
    "legs are cancel-attempted, the parent entry is not; qty==0 so no sell; returns []")

# (c) partially filled bracket BUY parent - the unfilled remainder keeps working
run("(c) bracket BUY parent 'partially_filled' 2 of 5", [{
    "id": "parent-C", "client_order_id": "newsbot-cccc", "symbol": "NVDA", "side": "buy",
    "type": "market", "order_class": "bracket", "qty": "5", "filled_qty": "2",
    "status": "partially_filled",
    "legs": [{"id": "C-tp", "side": "sell", "type": "limit", "qty": "2", "filled_qty": "0", "status": "new"},
             {"id": "C-sl", "side": "sell", "type": "stop", "qty": "2", "filled_qty": "0", "status": "held"}]}],
    "sells the 2 filled shares; the remaining 3 shares of the BUY stay working")

# control: what the code DOES collect - an open top-level SELL
run("(control) open top-level SELL is collected", [{
    "id": "parent-D", "client_order_id": "newsbot-dddd", "symbol": "NVDA", "side": "sell",
    "type": "market", "qty": "5", "filled_qty": "0", "status": "new", "legs": []}])
