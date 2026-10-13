import json, os, tempfile
os.environ["TRADER_JOURNAL_PATH"] = tempfile.mkdtemp() + "/j.db"
os.environ.update({"NEWSBOT_EXECUTION": "paper", "NEWSBOT_SYMBOLS": "AAPL",
                   "NEWSBOT_MIN_PRICE": "nan", "DESK_DEMO_MODE": "true"})
from fastapi.testclient import TestClient
from trader import desk
app = desk.create_app(background=False)
with TestClient(app, raise_server_exceptions=False) as c:
    r = c.get("/api/desk/config")
    print("GET /api/desk/config with NEWSBOT_MIN_PRICE=nan -> HTTP", r.status_code)
    print("body:", r.text[:120])
