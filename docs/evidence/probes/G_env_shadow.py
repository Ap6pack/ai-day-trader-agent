import os, tempfile
from pathlib import Path
from trader import config
d = Path(tempfile.mkdtemp())
(d / ".env").write_text("TRADER_MODE=live\n")        # no TRADER_MAX_ORDER_USD line
os.environ["TRADER_MAX_ORDER_USD"] = "nan"           # stale shell export
config.load_env(d / ".env")
print("after load_env, os.environ['TRADER_MAX_ORDER_USD'] =", os.environ["TRADER_MAX_ORDER_USD"])
print("load_limits().max_order_usd =", config.load_limits().max_order_usd)
print('docstring claim: "a stale export must never silently shadow the key or limit in .env"')
