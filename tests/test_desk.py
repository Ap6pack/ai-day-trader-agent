from __future__ import annotations

import asyncio
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from config.api import auth, desk
from config.api.auth import User
from core import market_feed
from core.event_bus import EventBus, event_bus
from core.portfolio_manager_provider import clear_portfolio_manager_cache
from core.trading_workflow import TradingWorkflow


@pytest.fixture
def current_user() -> User:
    return User(
        id=1,
        username="api-user",
        email="api-user@example.com",
        is_active=True,
        is_admin=False,
        created_at="2026-01-01T00:00:00",
    )


@pytest.fixture(autouse=True)
def clean_bus():
    event_bus.clear()
    yield
    event_bus.clear()


# ── Event bus ─────────────────────────────────────────────────────────

def test_event_bus_history_is_scoped_to_user():
    bus = EventBus()
    bus.publish("decision", "global")
    bus.publish("decision", "mine", user_id=1)
    bus.publish("decision", "theirs", user_id=2)

    assert [e["message"] for e in bus.history(user_id=1)] == ["global", "mine"]
    assert [e["message"] for e in bus.history(user_id=2)] == ["global", "theirs"]


def test_event_bus_serializes_pipeline_values():
    np = pytest.importorskip("numpy")
    bus = EventBus()
    event = bus.publish("decision", data={
        "timestamp": datetime(2026, 1, 2, 3, 4, 5),
        "rsi": np.float64(41.5),
        "nan": float("nan"),
        "nested": {"qty": np.int64(3)},
    })

    assert event["data"] == {
        "timestamp": "2026-01-02T03:04:05",
        "rsi": 41.5,
        "nan": None,
        "nested": {"qty": 3},
    }


async def test_event_bus_delivers_events_published_from_threads():
    bus = EventBus()
    queue = bus.subscribe()

    await asyncio.to_thread(bus.publish, "order_submitted", "BUY 1 AAPL", symbol="AAPL")
    event = await asyncio.wait_for(queue.get(), timeout=1)

    assert event["type"] == "order_submitted"
    assert event["symbol"] == "AAPL"
    bus.unsubscribe(queue)


# ── Market feed ───────────────────────────────────────────────────────

def test_demo_quotes_and_bars_are_labelled(monkeypatch):
    monkeypatch.setenv("DESK_DEMO_MODE", "true")

    quotes = market_feed.get_quotes(["aapl", "MSFT", "AAPL"])
    bars = market_feed.get_bars("AAPL", "5Min", 50)

    assert set(quotes) == {"AAPL", "MSFT"}
    assert all(q["source"] == "demo" for q in quotes.values())
    assert quotes["AAPL"]["change_pct"] is not None
    assert bars["source"] == "demo"
    assert len(bars["bars"]) == 50
    times = [b["time"] for b in bars["bars"]]
    assert times == sorted(times)
    assert all(b["low"] <= min(b["open"], b["close"]) for b in bars["bars"])


def test_get_bars_rejects_unknown_timeframe():
    with pytest.raises(ValueError):
        market_feed.get_bars("AAPL", "3Min")


def test_alpaca_snapshot_parsing(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "AAPL": {
                    "latestTrade": {"p": 102.0, "t": "2026-01-02T15:00:00Z"},
                    "latestQuote": {"bp": 101.9, "ap": 102.1, "bs": 3, "as": 4},
                    "dailyBar": {"o": 100.5, "h": 103.0, "l": 99.5, "v": 1000},
                    "prevDailyBar": {"c": 100.0},
                }
            }

    captured = {}

    def fake_get(url, headers, params, timeout):
        captured.update(url=url, params=params)
        return FakeResponse()

    monkeypatch.delenv("DESK_DEMO_MODE", raising=False)
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr(market_feed.requests, "get", fake_get)

    quote = market_feed.get_quotes(["AAPL"])["AAPL"]

    assert captured["url"].endswith("/v2/stocks/snapshots")
    assert captured["params"]["symbols"] == "AAPL"
    assert quote["last"] == 102.0
    assert quote["bid"] == 101.9 and quote["ask"] == 102.1
    assert quote["change"] == pytest.approx(2.0)
    assert quote["change_pct"] == pytest.approx(2.0)
    assert quote["source"] == "alpaca"


# ── REST endpoints ────────────────────────────────────────────────────

async def test_desk_config_reports_demo_mode(monkeypatch, current_user):
    monkeypatch.setenv("DESK_DEMO_MODE", "1")
    monkeypatch.setenv("DESK_WATCHLIST", "aapl, msft")

    config = await desk.get_desk_config(current_user=current_user)

    assert config["demo_mode"] is True
    assert config["watchlist"] == ["AAPL", "MSFT"]
    assert "5Min" in config["timeframes"]


async def test_desk_events_hide_other_users_events(current_user):
    event_bus.publish("decision", "for me", user_id=current_user.id)
    event_bus.publish("decision", "for someone else", user_id=999)

    events = await desk.get_desk_events(limit=50, current_user=current_user)

    assert [e["message"] for e in events] == ["for me"]


async def test_desk_bars_rejects_bad_symbol(current_user):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await desk.get_desk_bars("12$", timeframe="5Min", limit=100, current_user=current_user)
    assert exc.value.status_code == 400


def test_account_snapshot_without_keys(monkeypatch):
    for name in ("ALPACA_API_KEY", "ALPACA_KEY_ID", "ALPACA_SECRET_KEY", "ALPACA_SECRET"):
        monkeypatch.delenv(name, raising=False)

    assert desk.account_snapshot() == {"connected": False, "message": "Alpaca keys not configured"}


def test_account_snapshot_normalizes_alpaca(monkeypatch):
    class FakeExecutor:
        def get_account(self):
            return {"status": "ACTIVE", "equity": "10100", "last_equity": "10000", "cash": "5000",
                    "buying_power": "20000"}

        def get_positions(self):
            return [{"symbol": "AAPL", "qty": "5", "unrealized_pl": "12.5", "unrealized_plpc": "0.025",
                     "avg_entry_price": "100", "current_price": "102.5"}]

        def get_orders(self, limit=25):
            return [{"id": "abc", "symbol": "AAPL", "side": "buy", "qty": "5", "status": "filled"}]

        def get_clock(self):
            return {"is_open": True, "next_close": "2026-01-02T21:00:00Z"}

    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr("core.alpaca_executor_provider.get_alpaca_executor", lambda: FakeExecutor())

    snap = desk.account_snapshot()

    assert snap["connected"] is True
    assert snap["account"]["day_pl"] == pytest.approx(100.0)
    assert snap["account"]["day_pl_pct"] == pytest.approx(1.0)
    assert snap["positions"][0]["unrealized_plpc"] == pytest.approx(2.5)
    assert snap["orders"][0]["status"] == "filled"
    assert snap["clock"]["is_open"] is True


# ── Autopilot ─────────────────────────────────────────────────────────

def test_autopilot_config_validation():
    config = desk.AutopilotConfig(symbols=["aapl", "bad sym", "AAPL", "msft"], interval_seconds=60)
    assert config.symbols == ["AAPL", "MSFT"]

    with pytest.raises(ValueError):
        desk.AutopilotConfig(symbols=["AAPL"], interval_seconds=5)
    with pytest.raises(ValueError):
        desk.AutopilotConfig(symbols=["$$$"])


async def test_autopilot_scans_symbols_and_stops(monkeypatch, portfolio_manager):
    calls = []

    def fake_run(self, symbol, portfolio_name="default", *, record_paper_trade=False,
                 submit_alpaca_paper_order=False, user_id=None):
        calls.append((symbol, record_paper_trade, submit_alpaca_paper_order, user_id))
        from core.trading_workflow import WorkflowResult
        return WorkflowResult(symbol=symbol, portfolio_name=portfolio_name, analysis={"signal": "HOLD"})

    monkeypatch.setattr(TradingWorkflow, "run", fake_run)
    manager = desk.AutopilotManager()
    config = desk.AutopilotConfig(symbols=["AAPL", "MSFT"], interval_seconds=60)

    status = await manager.start(7, config, portfolio_manager)
    assert status["running"] is True
    for _ in range(50):
        if manager.status(7)["cycles"] and manager.status(7)["next_cycle_at"]:
            break
        await asyncio.sleep(0.02)

    assert calls == [("AAPL", False, False, 7), ("MSFT", False, False, 7)]
    stopped = await manager.stop(7)
    assert stopped["running"] is False
    messages = [e["message"] for e in event_bus.history(user_id=7)]
    assert messages[0].startswith("Autopilot ON")
    assert messages[-1] == "Autopilot OFF"


# ── WebSocket ─────────────────────────────────────────────────────────

@pytest.fixture
def desk_client(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "desk.db"))
    monkeypatch.setenv("DESK_DEMO_MODE", "true")
    monkeypatch.setenv("DESK_QUOTE_INTERVAL", "1")
    clear_portfolio_manager_cache()
    from config.api.dependencies import get_portfolio_manager
    from config.api.server import app

    db = get_portfolio_manager()
    db.create_user(username="desk-user", email="desk@example.com",
                   hashed_password=auth.get_password_hash("password123"))
    user = db.get_user_by_username("desk-user")
    token = auth.create_access_token({"sub": user["username"], "user_id": user["id"]})
    with TestClient(app) as client:
        yield client, token, user
    clear_portfolio_manager_cache()


def test_desk_websocket_rejects_missing_token(desk_client):
    from starlette.websockets import WebSocketDisconnect

    client, _, _ = desk_client
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/desk") as ws:
            ws.receive_json()


def test_desk_websocket_streams_hello_quotes_and_events(desk_client):
    client, token, user = desk_client
    event_bus.publish("decision", "earlier call", symbol="AAPL", user_id=user["id"])

    with client.websocket_connect(f"/ws/desk?token={token}") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert hello["data"]["config"]["demo_mode"] is True
        assert [e["message"] for e in hello["data"]["events"]] == ["earlier call"]

        ws.send_json({"type": "watch", "symbols": ["aapl", "not valid!"]})
        # Frames for the default watchlist may already be in flight; after the
        # watch is applied only AAPL is streamed.
        watched = None
        for _ in range(10):
            msg = ws.receive_json()
            if msg["type"] == "quotes" and set(msg["data"]) == {"AAPL"}:
                watched = msg["data"]
                break
        assert watched is not None
        assert watched["AAPL"]["source"] == "demo"

        event_bus.publish("order_submitted", "BUY 1 AAPL", symbol="AAPL")
        event_bus.publish("decision", "someone else's", user_id=user["id"] + 100)
        seen = []
        for _ in range(10):
            msg = ws.receive_json()
            if msg["type"] == "event":
                seen.append(msg["data"]["message"])
                break
        assert seen == ["BUY 1 AAPL"]


def test_desk_page_is_served(desk_client):
    client, _, _ = desk_client
    resp = client.get("/desk")
    assert resp.status_code == 200
    assert "ADT" in resp.text
    assert client.get("/static/js/desk.js").status_code == 200


# ── Pipeline instrumentation ──────────────────────────────────────────

def test_pipeline_publishes_each_analysis_step(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "pipeline.db"))
    monkeypatch.setenv("DESK_DEMO_MODE", "true")
    monkeypatch.setenv("DIVIDEND_STRATEGY_ENABLED", "false")
    clear_portfolio_manager_cache()
    monkeypatch.setattr("core.pipeline.get_news_articles", lambda symbol: [])
    from core.pipeline import EnhancedTradingPipeline

    result = EnhancedTradingPipeline("AAPL", user_id=3).run_analysis({})
    clear_portfolio_manager_cache()

    assert not result.get("error")
    events = event_bus.history(user_id=3)
    assert [e["type"] for e in events] == [
        "analysis_started", "market_data", "strategy_signal", "strategy_signal", "strategy_signal", "decision",
    ]
    assert all(e["symbol"] == "AAPL" and e["user_id"] == 3 for e in events)
    assert events[1]["data"]["source"] == "demo"
    assert events[-1]["data"]["signal"] == result["signal"]
