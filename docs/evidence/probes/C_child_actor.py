"""One OS process that submits one live buy, exactly as production does.
argv: <port> <actor-name> <symbol> <start-epoch>
Transport is redirected to 127.0.0.1; trading_base_url() still reads as the live host,
so account_mode() == "live" and check_live_buy() runs for real.
"""
import json
import os
import sys
import time
import urllib.request

os.environ["ALPACA_TRADING_BASE_URL"] = "https://api.alpaca.markets"
os.environ["ALPACA_LIVE_TRADING"] = "true"
os.environ["ALPACA_LIVE_MAX_ORDERS_PER_DAY"] = "1"
os.environ["ALPACA_LIVE_MAX_ORDER_USD"] = "500"
os.environ["APCA_API_KEY_ID"] = "stub"
os.environ["APCA_API_SECRET_KEY"] = "stub"

PORT, ACTOR, SYMBOL, START = int(sys.argv[1]), sys.argv[2], sys.argv[3], float(sys.argv[4])
LOCAL = f"http://127.0.0.1:{PORT}"

from trader import alpaca  # noqa: E402


def local_request(method, url, **kw):
    path = url.split("alpaca.markets", 1)[-1] if "alpaca.markets" in url else url
    if kw.get("params"):
        from urllib.parse import urlencode
        path += ("&" if "?" in path else "?") + urlencode(kw["params"])
    data = json.dumps(kw["json"]).encode() if kw.get("json") is not None else None
    req = urllib.request.Request(LOCAL + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", "X-Actor": ACTOR})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read() or b"null")


alpaca._request = local_request

assert alpaca.account_mode() == "live", alpaca.account_mode()
assert alpaca.trading_base_url() == "https://api.alpaca.markets"

while time.time() < START:          # all children cross the start line together
    time.sleep(0.002)

payload = alpaca.market_order_payload(SYMBOL, "buy", 10)
try:
    order = alpaca.submit_order(payload)
    print(json.dumps({"actor": ACTOR, "pid": os.getpid(), "result": "SUBMITTED", "order": order["id"],
                      "symbol": SYMBOL}))
except Exception as exc:
    print(json.dumps({"actor": ACTOR, "pid": os.getpid(), "result": "BLOCKED",
                      "error": f"{type(exc).__name__}: {exc}"}))
