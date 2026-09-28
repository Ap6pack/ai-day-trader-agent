from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from trader import desk, newsbot
from trader.journal import Journal

NOW = datetime(2026, 9, 28, 15, 0, tzinfo=timezone.utc)


@pytest.fixture
def journal(tmp_path):
    return Journal(tmp_path / "desk.db")


@pytest.fixture
def env(monkeypatch):
    for name in ("DESK_TOKEN", "DESK_DEMO_MODE", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "TYPESAFE_API_KEY",
                 "NEWSBOT_SYMBOLS", "NEWSBOT_EXECUTION", "ALPACA_TRADING_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def client_for(journal):
    return TestClient(desk.create_app(journal=journal, background=False))


def bot_signal(journal, symbol="NVDA", signal="buy", blocked=None, order=False):
    record = {"news_id": 7, "headline": "Nvidia wins $5B contract", "published_at": "2026-09-28T14:59:30Z",
              "signal": signal, "reason": "p_bullish 0.91 >= 0.75", "probability": 0.91,
              "execution": "paper", "jev_ms": 280, "relevance": 0.97, "materiality": 0.8,
              "p_bullish": 0.91, "p_bearish": 0.02, "event_type": "product", "price": 120.0}
    if blocked:
        record["blocked"] = blocked
    if order:
        record.update(order_id="abc12345-order", qty=4)
    journal.log_event(newsbot.ORDER_EVENT if order else newsbot.SIGNAL_EVENT, tool="newsbot", symbol=symbol,
                      side="buy", quantity=4 if order else None, price=120.0,
                      detail=blocked or record["reason"], payload=record, now=NOW)


# --- pages and auth ---------------------------------------------------------------


def test_desk_page_and_assets_are_served(env, journal):
    with client_for(journal) as c:
        page = c.get("/desk")
        assert page.status_code == 200 and "desk-api.js" in page.text
        assert c.get("/static/js/desk.js").status_code == 200
        assert c.get("/static/js/desk-api.js").status_code == 200


def test_no_token_needed_by_default(env, journal):
    with client_for(journal) as c:
        assert c.get("/api/desk/auth").json() == {"auth_required": False}
        cfg = c.get("/api/desk/config").json()
        assert cfg["auth_required"] is False and cfg["watchlist"][0] == "SPY"
        assert cfg["newsbot"]["execution"] == "off"


def test_desk_token_is_enforced_on_api_and_socket(env, journal):
    env.setenv("DESK_TOKEN", "s3cret")
    with client_for(journal) as c:
        assert c.get("/api/desk/auth").json() == {"auth_required": True}
        assert c.get("/api/desk/config").status_code == 401
        assert c.get("/api/desk/config", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.get("/api/desk/config", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        with pytest.raises(Exception):
            with c.websocket_connect("/ws/desk"):
                pass
        with c.websocket_connect("/ws/desk?token=s3cret") as ws:
            assert ws.receive_json()["type"] == "hello"


def test_main_refuses_public_bind_without_token(env, monkeypatch):
    monkeypatch.setattr(desk.config, "load_env", lambda *a, **k: None)
    env.setenv("DESK_HOST", "0.0.0.0")
    assert desk.main() == 2


# --- agent activity from the journal ----------------------------------------------


def test_bot_signals_orders_and_decisions_become_events(env, journal):
    bot_signal(journal, signal="none")
    bot_signal(journal, symbol="AMD", blocked="daily cap reached")
    bot_signal(journal, symbol="NVDA", order=True)
    journal.log_event(newsbot.FLATTEN_EVENT, tool="newsbot", detail="closed 1 positions", now=NOW)
    journal.log_event("order_blocked", symbol="TSLA", side="buy", detail="Review-only mode", now=NOW)
    journal.add_decision("NVDA", "buy", 120.0, confidence=0.91, thesis="[newsbot] product: Nvidia wins",
                         mode="newsbot-paper", now=NOW)
    journal.add_decision("AAPL", "pass", 200.0, thesis="No catalyst", mode="review", now=NOW)

    events = desk.recent_events(journal)
    by_type = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)
    assert set(by_type) == {"news_skip", "news_signal", "order_submitted", "flatten", "guard", "decision"}
    order = by_type["order_submitted"][0]
    assert order["symbol"] == "NVDA" and "BOT BUY 4 @ 120.0" in order["message"] and "Jev 280ms" in order["message"]
    signal = order["data"]["signal"]
    assert signal["call"] == "BUY" and signal["stop"] == 118.8 and signal["target"] == 122.4
    assert signal["order_id"] == "abc12345-order"
    blocked = by_type["news_signal"][0]
    assert blocked["level"] == "warning" and "daily cap reached" in blocked["message"]
    sources = sorted(e["message"].split()[0] for e in by_type["decision"])
    assert sources == ["BOT", "CLAUDE"]


def test_newsbot_status_counts_today(env, journal):
    env.setenv("NEWSBOT_SYMBOLS", "auto")
    env.setenv("NEWSBOT_EXECUTION", "paper")
    bot_signal(journal, signal="none")
    bot_signal(journal, order=True)
    status = desk.newsbot_status(journal)
    assert status["execution"] == "paper" and status["universe"] == "auto"
    assert status["median_jev_ms"] == 280 and status["last_activity"]
    assert status["max_trades_per_day"] == 5


def test_hub_streams_only_new_journal_rows(env, journal):
    import asyncio

    bot_signal(journal, signal="none")
    hub = desk.Hub(journal)
    sent = []

    async def run():
        async def fake_broadcast(message):
            sent.append(message)

        hub.broadcast = fake_broadcast
        hub.last_event_id = journal.events(1)[0]["id"]
        assert await hub.poll_journal() == 0
        bot_signal(journal, order=True)
        journal.add_decision("NVDA", "buy", 120.0, thesis="[newsbot] x", mode="newsbot-paper", now=NOW)
        assert await hub.poll_journal() == 2
        assert await hub.poll_journal() == 0

    asyncio.run(run())
    assert [m["data"]["type"] for m in sent] == ["order_submitted", "decision"]


def test_websocket_hello_and_watch(env, journal, monkeypatch):
    bot_signal(journal, order=True)
    app = desk.create_app(journal=journal, background=False)
    app.state.hub.last_quotes["NVDA"] = {"last": 121.0}
    with TestClient(app) as c, c.websocket_connect("/ws/desk") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert hello["data"]["events"][0]["type"] == "order_submitted"
        assert hello["data"]["newsbot"]["orders_today"] >= 0
        ws.send_text(json.dumps({"type": "watch", "symbols": ["nvda", "bad sym"]}))
        assert ws.receive_json() == {"type": "quotes", "data": {"NVDA": {"last": 121.0}}}
        ws.send_text(json.dumps({"type": "ping"}))
        assert ws.receive_json() == {"type": "pong"}


# --- JUDGE ----------------------------------------------------------------------------


def test_judge_applies_the_bot_rules_to_recent_headlines(env, journal, monkeypatch):
    env.setenv("TYPESAFE_API_KEY", "t")
    articles = [
        {"title": "Nvidia wins $5B contract", "publishedAt": "2026-09-28T10:00:00Z", "url": "u1"},
        {"title": "10 stocks to watch", "publishedAt": "2026-09-28T09:00:00Z", "url": "u2"},
    ]
    judgments = {
        "Nvidia wins $5B contract": {"relevance": 0.97, "materiality": 0.8, "p_bullish": 0.9,
                                     "p_bearish": 0.02, "event_type": "product"},
        "10 stocks to watch": {"relevance": 0.2, "materiality": 0.1, "p_bullish": 0.5,
                               "p_bearish": 0.1, "event_type": "none"},
    }
    monkeypatch.setattr(desk, "get_news_articles", lambda s: articles)
    monkeypatch.setattr(desk.jev_news, "judge_headline", lambda s, a: judgments[a["title"]])
    monkeypatch.setattr(desk.market_feed, "get_quotes", lambda syms: {"NVDA": {"last": 100.0}})
    with client_for(journal) as c:
        res = c.post("/api/desk/judge/nvda").json()
    assert res["call"] == "BUY" and res["probability"] == 0.9
    assert res["price"] == 100.0 and res["stop"] == 99.0 and res["target"] == 102.0
    assert [h["action"] for h in res["headlines"]] == ["buy", "none"]
    assert "relevance" in res["headlines"][1]["reason"]


def test_judge_needs_jev(env, journal):
    with client_for(journal) as c:
        res = c.post("/api/desk/judge/NVDA")
    assert res.status_code == 400 and "TYPESAFE_API_KEY" in res.json()["detail"]


# --- manual paper ticket ----------------------------------------------------------------


def test_manual_order_is_paper_only_and_journaled(env, journal, monkeypatch):
    env.setenv("ALPACA_API_KEY", "k")
    env.setenv("ALPACA_SECRET_KEY", "s")
    sent = []
    monkeypatch.setattr(desk.alpaca, "submit_order", lambda p: sent.append(p) or {"id": "o-123456789", "status": "accepted"})
    with client_for(journal) as c:
        res = c.post("/api/desk/orders", json={"symbol": "nvda", "side": "BUY", "qty": 3}).json()
        assert res["submitted"] is True
        assert sent[0]["symbol"] == "NVDA" and sent[0]["side"] == "buy" and sent[0]["qty"] == "3"
        assert sent[0]["client_order_id"].startswith("desk-")
        event = journal.events(1)[0]
        assert event["kind"] == desk.DESK_ORDER_EVENT and event["quantity"] == 3
        assert desk.journal_event(event)["message"].startswith("MANUAL BUY 3")

        env.setenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets")
        refused = c.post("/api/desk/orders", json={"symbol": "NVDA", "side": "buy", "qty": 1})
        assert refused.status_code == 400 and "paper" in refused.json()["detail"]
        assert len(sent) == 1


def test_manual_order_rejection_is_reported_and_journaled(env, journal, monkeypatch):
    env.setenv("ALPACA_API_KEY", "k")
    env.setenv("ALPACA_SECRET_KEY", "s")

    def reject(payload):
        raise RuntimeError("insufficient buying power")

    monkeypatch.setattr(desk.alpaca, "submit_order", reject)
    with client_for(journal) as c:
        res = c.post("/api/desk/orders", json={"symbol": "NVDA", "side": "sell", "qty": 1}).json()
    assert res == {"submitted": False, "skipped_reason": "insufficient buying power"}
    assert desk.journal_event(journal.events(1)[0])["type"] == "order_failed"


@pytest.mark.parametrize("body", [
    {"symbol": "NVDA", "side": "hold", "qty": 1},
    {"symbol": "NVDA", "side": "buy", "qty": 0},
    {"symbol": "", "side": "buy", "qty": 1},
])
def test_manual_order_validation(env, journal, body):
    with client_for(journal) as c:
        assert c.post("/api/desk/orders", json=body).status_code == 422


# --- account ------------------------------------------------------------------------------


def test_account_snapshot_marks_order_origin(env, monkeypatch):
    env.setenv("ALPACA_API_KEY", "k")
    env.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setattr(desk.alpaca, "account", lambda: {"equity": "10100", "last_equity": "10000",
                                                         "buying_power": "20000", "cash": "5000"})
    monkeypatch.setattr(desk.alpaca, "positions", lambda: [{"symbol": "NVDA", "qty": "4", "unrealized_plpc": "0.012"}])
    monkeypatch.setattr(desk.alpaca, "orders", lambda limit: [
        {"id": "1", "client_order_id": "newsbot-x", "symbol": "NVDA", "status": "filled"},
        {"id": "2", "client_order_id": "desk-y", "symbol": "AMD", "status": "new"},
        {"id": "3", "client_order_id": "autopilot-z", "symbol": "TSLA", "status": "new"},
    ])
    monkeypatch.setattr(desk.alpaca, "market_clock", lambda: {"is_open": True})
    snap = desk.account_snapshot()
    assert snap["connected"] and snap["account"]["day_pl"] == 100.0
    assert snap["positions"][0]["unrealized_plpc"] == pytest.approx(1.2)
    assert [o["origin"] for o in snap["orders"]] == ["bot", "desk", "autopilot"]


def test_account_snapshot_refuses_live_endpoint(env):
    env.setenv("ALPACA_API_KEY", "k")
    env.setenv("ALPACA_SECRET_KEY", "s")
    env.setenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets")
    snap = desk.account_snapshot()
    assert snap["connected"] is False and "paper" in snap["message"]


# --- analysis, autopilot, portfolios, pause ----------------------------------------------


def fake_analysis(symbol="NVDA", rec="BUY", qty=5):
    return {"symbol": symbol, "timestamp": NOW.isoformat(), "recommendation": rec, "quantity": qty,
            "confidence": 0.7, "primary_strategy": "technical", "current_price": 100.0,
            "all_signals": {}, "risk_parameters": {"stop_loss": 96.0, "take_profit": 106.0}}


def test_analyze_uses_portfolio_capital_and_streams_a_decision(env, journal, monkeypatch):
    calls = []

    def analyze(symbol, capital, held, capital_source):
        calls.append((symbol, capital, held, capital_source))
        return fake_analysis(symbol)

    monkeypatch.setattr(desk.analysis, "analyze", analyze)
    app = desk.create_app(journal=journal, background=False)
    with TestClient(app) as c:
        c.post("/api/desk/portfolios", json={"name": "fast", "cash": 2000})
        res = c.post("/api/desk/analyze/nvda?portfolio=fast").json()
        assert res["recommendation"] == "BUY"
        assert calls == [("NVDA", 2000, 0.0, "portfolio fast")]
        assert c.post("/api/desk/analyze/nvda?portfolio=nope").status_code == 404
        assert c.post("/api/desk/analyze/nvda?portfolio=alpaca").status_code == 400
        assert c.post("/api/desk/analyze/bad sym").status_code == 400
        with c.websocket_connect("/ws/desk") as ws:
            hello = ws.receive_json()["data"]
            assert hello["events"][0]["type"] == "decision" and hello["events"][0]["data"]["quantity"] == 5
            assert hello["portfolios"] == ["default", "fast"]
            assert hello["autopilot"]["running"] is False


def test_portfolio_crud_and_manual_fills(env, journal):
    app = desk.create_app(journal=journal, background=False)
    app.state.hub.last_quotes["NVDA"] = {"last": 100.0}
    with TestClient(app) as c:
        assert [p["name"] for p in c.get("/api/desk/portfolios").json()] == ["default"]
        assert c.post("/api/desk/portfolios", json={"name": "swing", "cash": 1000}).json()["equity"] == 1000
        assert c.post("/api/desk/portfolios", json={"name": "swing", "cash": 1000}).status_code == 400
        res = c.post("/api/desk/orders", json={"symbol": "NVDA", "side": "buy", "qty": 4,
                                               "destination": "swing"}).json()
        assert res["submitted"] and res["fill"]["cash"] == 600
        view = c.get("/api/desk/portfolios/swing").json()
        assert view["positions"][0]["qty"] == 4 and view["fills"][0]["source"] == "desk"
        too_much = c.post("/api/desk/orders", json={"symbol": "NVDA", "side": "buy", "qty": 40,
                                                    "destination": "swing"}).json()
        assert too_much["submitted"] is False and "Not enough cash" in too_much["skipped_reason"]
        assert journal.events(1)[0]["kind"] == desk.DESK_ORDER_EVENT
        assert c.delete("/api/desk/portfolios/swing").json() == {"deleted": "swing"}
        assert c.get("/api/desk/portfolios/swing").status_code == 404


def test_autopilot_start_status_stop(env, journal, monkeypatch):
    monkeypatch.setattr(desk.analysis, "analyze", lambda *a, **k: fake_analysis(rec="HOLD", qty=0))
    with client_for(journal) as c:
        bad = c.post("/api/desk/autopilot", json={"symbols": ["NVDA"], "interval_seconds": 10, "mode": "signals"})
        assert bad.status_code == 422
        live = c.post("/api/desk/autopilot", json={"symbols": ["NVDA"], "interval_seconds": 60, "mode": "live"})
        assert live.status_code == 422
        missing = c.post("/api/desk/autopilot", json={"symbols": ["NVDA"], "interval_seconds": 60,
                                                      "mode": "record", "portfolio": "nope"})
        assert missing.status_code == 400
        started = c.post("/api/desk/autopilot", json={"symbols": ["nvda"], "interval_seconds": 300,
                                                      "mode": "signals"}).json()
        assert started["running"] and started["config"]["symbols"] == ["NVDA"]
        assert c.get("/api/desk/autopilot").json()["running"]
        assert c.delete("/api/desk/autopilot").json()["running"] is False


def test_newsbot_pause_from_the_desk(env, journal, tmp_path):
    env.setenv("TRADER_JOURNAL_PATH", str(tmp_path / "desk.db"))
    with client_for(journal) as c:
        assert c.post("/api/desk/newsbot/pause", json={"paused": True}).json()["paused"] is True
        assert newsbot.is_paused()
        assert c.get("/api/desk/newsbot").json()["paused"] is True
        assert c.post("/api/desk/newsbot/pause", json={"paused": False}).json()["paused"] is False
        assert not newsbot.is_paused()


def test_autopilot_rows_are_not_streamed_twice(env, journal):
    import asyncio

    hub = desk.Hub(journal)
    sent = []

    async def run():
        async def fake_broadcast(message):
            sent.append(message)

        hub.broadcast = fake_broadcast
        journal.log_event(desk.autotrader.ORDER_EVENT, tool="autopilot", symbol="NVDA", detail="x", now=NOW)
        journal.add_decision("NVDA", "buy", 100.0, thesis="[autopilot] t", mode="autopilot-paper", now=NOW)
        assert await hub.poll_journal() == 0

    asyncio.run(run())
    assert sent == []
    row = journal.events(1)[0]
    assert desk.journal_event(row)["message"].startswith("AUTO")
    assert desk.decision_event(journal.decisions(1)[0])["message"].startswith("AUTO BUY")
