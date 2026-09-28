"""
Live trading desk: a browser window onto the agents, plus a manual paper ticket.

    python -m trader.desk            # http://127.0.0.1:8000/desk

What it shows, live over /ws/desk:
  * quotes and chart bars (trader.market_feed: Alpaca, optional Yahoo, or DESK_DEMO_MODE)
  * the agents' activity from the journal: every news bot signal with Jev's numbers
    and latency, its orders and end-of-day flatten, Claude's decisions and the order
    guard's reviews, plus manual desk orders
  * the Alpaca paper account: equity, positions, orders, market clock
  * the news bot's status: mode, universe, today's trades against the cap, latency

ANALYZE runs the multi-strategy analysis (technical + Jev sentiment + dividend) with
sizing, stop/target and risk. JUDGE runs Jev on a symbol's recent headlines with the
news bot's own rules. The AUTOPILOT panel starts and stops the desk autopilot
(trader.autotrader: signals, record to a local portfolio, or Alpaca paper orders),
and the NEWS BOT tab pauses or resumes the news bot's trading. Local paper
portfolios (trader.portfolios) can be created, viewed and traded from the desk.
The manual ticket sends Alpaca paper market orders or records a fill in a local
portfolio; manual orders are journaled as desk orders.

The news bot itself runs on its own (python -m trader.newsbot run): the desk can
pause and resume its trading but does not start it or change its rules. Access: binds to 127.0.0.1 by default. Set DESK_TOKEN to
require a token (needed if you bind to another interface with DESK_HOST).
"""

from __future__ import annotations

import asyncio
import dataclasses
import hmac
import json
import logging
import os
import statistics
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from trader import alpaca, analysis, autotrader, config, jev_news, market_feed, newsbot
from trader.journal import Journal
from trader.news_sources import alpaca_configured, get_news_articles
from trader.portfolios import PortfolioError, Portfolios

logger = logging.getLogger("trader.desk")

STATIC_DIR = config.PROJECT_ROOT / "static"
DEFAULT_WATCHLIST = "SPY,QQQ,AAPL,MSFT,NVDA,TSLA,AMZN,META,GOOGL,AMD"
MAX_WATCH = 40
HISTORY_EVENTS = 200
JUDGE_MAX_ARTICLES = 10
JUDGE_MAX_AGE_HOURS = 24
JOURNAL_POLL_SECONDS = 2.0
NEWSBOT_STATUS_SECONDS = 15.0
DESK_ORDER_EVENT = "desk_order"
DESK_CANCEL_EVENT = "desk_cancel"
GUARD_EVENTS = {"review", "order_allowed", "order_blocked", "order_submitted"}


# --- settings -------------------------------------------------------------------


def desk_token() -> Optional[str]:
    return config.secret("DESK_TOKEN")


def token_ok(token: Optional[str]) -> bool:
    expected = desk_token()
    return expected is None or (token is not None and hmac.compare_digest(token, expected))


def default_watchlist() -> List[str]:
    raw = os.getenv("DESK_WATCHLIST") or os.getenv("TRADER_WATCHLIST") or DEFAULT_WATCHLIST
    return clean_symbols(raw.split(","))


def clean_symbols(symbols: Any) -> List[str]:
    cleaned: List[str] = []
    for symbol in symbols or []:
        symbol = str(symbol).strip().upper()
        if symbol and len(symbol) <= 10 and symbol.replace(".", "").isalpha() and symbol not in cleaned:
            cleaned.append(symbol)
    return cleaned[:MAX_WATCH]


def quote_interval_seconds() -> float:
    configured = os.getenv("DESK_QUOTE_INTERVAL")
    if configured:
        return max(1.0, float(configured))
    if market_feed.demo_mode_enabled():
        return 1.0
    return 2.0 if alpaca_configured() else 10.0


def account_interval_seconds() -> float:
    return max(3.0, float(os.getenv("DESK_ACCOUNT_INTERVAL", "10")))


def bot_settings() -> Optional[newsbot.Settings]:
    try:
        return newsbot.load_settings()
    except ValueError as exc:
        logger.warning(f"News bot settings invalid: {exc}")
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def desk_config() -> Dict[str, Any]:
    settings = bot_settings()
    return {
        "demo_mode": market_feed.demo_mode_enabled(),
        "alpaca_configured": alpaca_configured(),
        "paper": alpaca.is_paper(),
        "jev_configured": jev_news.is_configured(),
        "auth_required": desk_token() is not None,
        "watchlist": default_watchlist(),
        "timeframes": list(market_feed.TIMEFRAMES),
        "default_timeframe": market_feed.DEFAULT_TIMEFRAME,
        "quote_interval_seconds": quote_interval_seconds(),
        "newsbot": _jsonable(dataclasses.asdict(settings)) if settings else None,
    }


# --- journal rows -> desk events ------------------------------------------------


def _payload(row: Any) -> Dict[str, Any]:
    try:
        return json.loads(row["payload"]) if row["payload"] else {}
    except (TypeError, ValueError):
        return {}


def _exits(settings: Optional[newsbot.Settings], price: Optional[float]) -> Dict[str, Optional[float]]:
    if not settings or not price:
        return {"stop": None, "target": None}
    return {"stop": round(price * (1 - settings.stop_loss_pct / 100), 2),
            "target": round(price * (1 + settings.take_profit_pct / 100), 2)}


def signal_from_record(symbol: str, record: Dict[str, Any], timestamp: str, source: str,
                       settings: Optional[newsbot.Settings]) -> Dict[str, Any]:
    """The signal panel's view of one judged headline."""
    action = record.get("signal") or record.get("action") or "none"
    price = record.get("price")
    headline = {k: record.get(k) for k in (
        "headline", "published_at", "url", "relevance", "materiality", "p_bullish", "p_bearish",
        "event_type", "reason", "blocked")}
    headline["action"] = action
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "source": source,
        "call": {"buy": "BUY", "bearish": "BEARISH"}.get(action, "NONE"),
        "probability": record.get("probability"),
        "reason": record.get("blocked") or record.get("reason") or "",
        "headline": record.get("headline"),
        "price": price,
        **(_exits(settings, price) if action == "buy" else {"stop": None, "target": None}),
        "execution": record.get("execution"),
        "order_id": record.get("order_id"),
        "jev_ms": record.get("jev_ms"),
        "headlines": [headline],
    }


def journal_event(row: Any, settings: Optional[newsbot.Settings] = None) -> Optional[Dict[str, Any]]:
    kind, symbol = row["kind"], row["symbol"]
    payload = _payload(row)
    event: Dict[str, Any] = {"id": f"e{row['id']}", "timestamp": row["ts"], "symbol": symbol,
                             "level": "info", "data": payload}
    if kind in (newsbot.SIGNAL_EVENT, newsbot.ORDER_EVENT):
        action = payload.get("signal", "none")
        prob = payload.get("probability")
        head = (payload.get("headline") or "")[:110]
        timing = f" · Jev {payload['jev_ms']}ms" if payload.get("jev_ms") is not None else ""
        if kind == newsbot.ORDER_EVENT:
            event.update(type="order_submitted", level="success", message=(
                f"BOT BUY {payload.get('qty')} @ {payload.get('price')} bracket"
                f" · p={prob:.2f}{timing} · {head}" if prob is not None else f"BOT BUY {head}"))
        elif action == "none":
            event.update(type="news_skip", level="debug", message=f"{payload.get('reason', '')}{timing} · {head}")
        else:
            blocked = payload.get("blocked")
            event.update(type="news_signal", level="warning" if blocked else "info", message=(
                f"{action.upper()} p={prob:.2f}{timing}" if prob is not None else action.upper())
                + (f" · {blocked}" if blocked else "") + f" · {head}")
        event["data"] = {"record": payload,
                         "signal": signal_from_record(symbol, payload, row["ts"], "newsbot", settings)}
        return event
    if kind == newsbot.FLATTEN_EVENT:
        event.update(type="flatten", level="system", message=f"Flatten: {row['detail']}")
        return event
    if kind == autotrader.ORDER_EVENT:
        event.update(type="order_submitted", level="success", message=f"AUTO {row['detail']}")
        return event
    if kind == DESK_ORDER_EVENT:
        ok = payload.get("submitted")
        event.update(type="order_submitted" if ok else "order_failed", level="success" if ok else "error",
                     message=f"MANUAL {(row['side'] or '').upper()} {row['quantity'] or ''} · {row['detail']}")
        return event
    if kind == DESK_CANCEL_EVENT:
        event.update(type="order_cancelled", level="system", message=f"MANUAL cancel · {row['detail']}")
        return event
    if kind in GUARD_EVENTS:
        event.update(type="guard", level="error" if kind == "order_blocked" else "info",
                     message=f"Guard {kind.replace('_', ' ')} · {(row['side'] or '').upper()} "
                             f"{row['detail'] or ''}".strip())
        return event
    return None


def decision_event(row: Any) -> Dict[str, Any]:
    thesis = row["thesis"] or ""
    source = ("BOT" if thesis.startswith("[newsbot]") else
              "AUTO" if thesis.startswith(autotrader.THESIS_PREFIX) else "CLAUDE")
    conf = f" conf {row['confidence']:.2f}" if row["confidence"] is not None else ""
    return {
        "id": f"d{row['id']}", "type": "decision", "timestamp": row["ts"], "symbol": row["symbol"],
        "level": "info",
        "message": f"{source} {row['action'].upper()} @ {row['price']}{conf} · "
                   f"{thesis.replace('[newsbot] ', '').replace(autotrader.THESIS_PREFIX + ' ', '')[:110]}",
        "data": {k: row[k] for k in row.keys()},
    }


def recent_events(journal: Journal, limit: int = HISTORY_EVENTS) -> List[Dict[str, Any]]:
    settings = bot_settings()
    events = [e for e in (journal_event(r, settings) for r in journal.events(limit)) if e]
    events += [decision_event(r) for r in journal.decisions(limit)]
    events.sort(key=lambda e: e["timestamp"], reverse=True)
    return events[:limit]


# --- news bot status --------------------------------------------------------------


def newsbot_status(journal: Journal) -> Dict[str, Any]:
    settings = bot_settings()
    rows = [r for r in journal.events(300)
            if r["kind"] in (newsbot.SIGNAL_EVENT, newsbot.ORDER_EVENT, newsbot.FLATTEN_EVENT)]
    jev_ms = [p["jev_ms"] for p in (_payload(r) for r in rows[:100]) if isinstance(p.get("jev_ms"), (int, float))]
    return {
        "settings": _jsonable(dataclasses.asdict(settings)) if settings else None,
        "execution": settings.execution if settings else None,
        "universe": ("auto" if settings.auto_symbols else sorted(settings.symbols) or "all") if settings else None,
        "orders_today": journal.count_events_today(newsbot.ORDER_EVENT),
        "signals_today": journal.count_events_today(newsbot.SIGNAL_EVENT),
        "max_trades_per_day": settings.max_trades_per_day if settings else None,
        "last_activity": rows[0]["ts"] if rows else None,
        "last_detail": rows[0]["detail"] if rows else None,
        "median_jev_ms": round(statistics.median(jev_ms)) if jev_ms else None,
        "max_jev_ms": max(jev_ms) if jev_ms else None,
        "paused": newsbot.is_paused(),
    }


# --- JUDGE: Jev on recent headlines, with the bot's rules --------------------------


def judge_symbol(symbol: str) -> Dict[str, Any]:
    if not jev_news.is_configured():
        raise HTTPException(400, "TYPESAFE_API_KEY is not set: Jev is needed to judge headlines.")
    settings = bot_settings() or newsbot.Settings()
    relaxed = dataclasses.replace(settings, max_headline_age_seconds=JUDGE_MAX_AGE_HOURS * 3600)
    now = datetime.now(timezone.utc)
    articles = [a for a in get_news_articles(symbol) if a.get("title")][:JUDGE_MAX_ARTICLES]
    headlines = []
    for article in articles:
        judgment = jev_news.judge_headline(symbol, article)
        signal = newsbot.decide(symbol, article, judgment, relaxed, now)
        headlines.append({
            "headline": article.get("title"), "published_at": article.get("publishedAt"),
            "url": article.get("url"), "action": signal.action, "reason": signal.reason,
            "probability": signal.probability,
            **{k: (judgment or {}).get(k) for k in (
                "relevance", "materiality", "p_bullish", "p_bearish", "event_type")},
        })
    buys = sorted((h for h in headlines if h["action"] == "buy"), key=lambda h: -(h["probability"] or 0))
    bears = sorted((h for h in headlines if h["action"] == "bearish"), key=lambda h: -(h["probability"] or 0))
    best = (buys or bears or [None])[0]
    quote = market_feed.get_quotes([symbol]).get(symbol) or {}
    price = quote.get("last")
    call = {"buy": "BUY", "bearish": "BEARISH"}.get(best["action"], "NONE") if best else "NONE"
    return {
        "symbol": symbol,
        "timestamp": now.isoformat(timespec="seconds"),
        "source": "judge",
        "call": call,
        "probability": best["probability"] if best else None,
        "reason": best["reason"] if best else (
            "no headlines in the last 24h" if not headlines else "no headline passes the bot's rules"),
        "headline": best["headline"] if best else None,
        "price": price,
        **(_exits(settings, price) if call == "BUY" else {"stop": None, "target": None}),
        "execution": settings.execution,
        "note": (f"Headlines up to {JUDGE_MAX_AGE_HOURS}h old are judged here; the bot only acts on "
                 f"headlines under {settings.max_headline_age_seconds}s old."),
        "headlines": headlines,
    }


# --- Alpaca paper account ------------------------------------------------------------


def _f(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _origin(client_order_id: Any) -> str:
    cid = str(client_order_id or "")
    for prefix, origin in (("newsbot-", "bot"), ("autopilot-", "autopilot"), ("desk-", "desk")):
        if cid.startswith(prefix):
            return origin
    return "other"


def account_snapshot() -> Dict[str, Any]:
    if market_feed.demo_mode_enabled():
        return {"connected": False, "message": "Demo mode"}
    if not alpaca_configured():
        return {"connected": False, "message": "Alpaca keys not configured"}
    if not alpaca.is_paper():
        return {"connected": False, "message": "The desk shows the Alpaca paper account only"}
    try:
        acct, positions, orders, clock = alpaca.account(), alpaca.positions(), alpaca.orders(25), alpaca.market_clock()
    except Exception as exc:
        return {"connected": False, "message": f"Alpaca unavailable: {exc}"}
    equity, last_equity = _f(acct.get("equity")), _f(acct.get("last_equity"))
    return {
        "connected": True,
        "account": {
            "status": acct.get("status"),
            "equity": equity,
            "last_equity": last_equity,
            "day_pl": (equity - last_equity) if equity is not None and last_equity else None,
            "day_pl_pct": ((equity - last_equity) / last_equity * 100) if equity is not None and last_equity else None,
            "cash": _f(acct.get("cash")),
            "buying_power": _f(acct.get("buying_power")),
            "daytrade_count": acct.get("daytrade_count"),
        },
        "positions": [{
            "symbol": p.get("symbol"), "qty": _f(p.get("qty")), "side": p.get("side"),
            "avg_entry_price": _f(p.get("avg_entry_price")), "current_price": _f(p.get("current_price")),
            "market_value": _f(p.get("market_value")), "unrealized_pl": _f(p.get("unrealized_pl")),
            "unrealized_plpc": (_f(p.get("unrealized_plpc")) or 0) * 100,
        } for p in positions],
        "orders": [{
            "id": o.get("id"), "client_order_id": o.get("client_order_id"), "symbol": o.get("symbol"),
            "side": o.get("side"), "qty": _f(o.get("qty")), "filled_qty": _f(o.get("filled_qty")),
            "filled_avg_price": _f(o.get("filled_avg_price")), "limit_price": _f(o.get("limit_price")),
            "type": o.get("type"), "order_class": o.get("order_class"), "status": o.get("status"),
            "submitted_at": o.get("submitted_at"),
            "origin": _origin(o.get("client_order_id")),
        } for o in orders],
        "clock": clock,
    }


# --- live hub ------------------------------------------------------------------------


class DeskClient:
    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.symbols: Set[str] = set()
        self._lock = asyncio.Lock()

    async def send(self, message: Dict[str, Any]) -> bool:
        try:
            async with self._lock:
                await self.websocket.send_text(json.dumps(message, default=str))
            return True
        except Exception:
            return False


class Hub:
    def __init__(self, journal: Journal) -> None:
        self.journal = journal
        self.clients: Set[DeskClient] = set()
        self.last_quotes: Dict[str, Dict[str, Any]] = {}
        self.last_account: Optional[Dict[str, Any]] = None
        self.last_event_id = 0
        self.last_decision_id = 0
        self.tasks: List[asyncio.Task] = []
        self.live_events: deque = deque(maxlen=HISTORY_EVENTS)
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def publish(self, event: Dict[str, Any]) -> None:
        """Broadcast a live (non-journal) event; safe to call from worker threads."""
        self.live_events.appendleft(event)
        if self.loop is None or not self.loop.is_running():
            return
        message = {"type": "event", "data": event}
        try:
            if asyncio.get_running_loop() is self.loop:
                self.loop.create_task(self.broadcast(message))
                return
        except RuntimeError:
            pass
        asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)

    def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        events, decisions = self.journal.events(1), self.journal.decisions(1)
        self.last_event_id = events[0]["id"] if events else 0
        self.last_decision_id = decisions[0]["id"] if decisions else 0
        for loop in (self._quote_loop, self._account_loop, self._journal_loop, self._newsbot_loop):
            self.tasks.append(asyncio.create_task(loop()))

    async def stop(self) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()

    async def broadcast(self, message: Dict[str, Any]) -> None:
        dead = [c for c in list(self.clients) if not await c.send(message)]
        for client in dead:
            self.clients.discard(client)

    def watched(self) -> List[str]:
        return sorted(set().union(*(c.symbols for c in self.clients))) if self.clients else []

    async def poll_journal(self) -> int:
        settings = bot_settings()
        rows = await run_in_threadpool(self.journal.events_after, self.last_event_id)
        decisions = await run_in_threadpool(self.journal.decisions_after, self.last_decision_id)
        sent = 0
        for row in rows:
            self.last_event_id = row["id"]
            if row["kind"] == autotrader.ORDER_EVENT:
                continue  # already streamed live by the autopilot
            event = journal_event(row, settings)
            if event:
                await self.broadcast({"type": "event", "data": event})
                sent += 1
        for row in decisions:
            self.last_decision_id = row["id"]
            if (row["thesis"] or "").startswith(autotrader.THESIS_PREFIX):
                continue  # the autopilot streamed its decision live
            await self.broadcast({"type": "event", "data": decision_event(row)})
            sent += 1
        return sent

    async def _journal_loop(self) -> None:
        while True:
            try:
                if await self.poll_journal():
                    await self.broadcast({"type": "newsbot",
                                          "data": await run_in_threadpool(newsbot_status, self.journal)})
            except Exception:
                logger.exception("Journal poll failed")
            await asyncio.sleep(JOURNAL_POLL_SECONDS)

    async def _newsbot_loop(self) -> None:
        while True:
            await asyncio.sleep(NEWSBOT_STATUS_SECONDS)
            if self.clients:
                try:
                    await self.broadcast({"type": "newsbot",
                                          "data": await run_in_threadpool(newsbot_status, self.journal)})
                except Exception:
                    logger.exception("News bot status failed")

    async def _quote_loop(self) -> None:
        while True:
            symbols = self.watched()
            if symbols:
                try:
                    quotes = await run_in_threadpool(market_feed.get_quotes, symbols)
                    self.last_quotes.update(quotes)
                    for client in list(self.clients):
                        mine = {s: q for s, q in quotes.items() if s in client.symbols}
                        if mine and not await client.send({"type": "quotes", "data": mine}):
                            self.clients.discard(client)
                except Exception:
                    logger.exception("Quote poll failed")
            await asyncio.sleep(quote_interval_seconds())

    async def _account_loop(self) -> None:
        while True:
            if self.clients:
                try:
                    self.last_account = await run_in_threadpool(account_snapshot)
                    await self.broadcast({"type": "account", "data": self.last_account})
                except Exception:
                    logger.exception("Account poll failed")
            await asyncio.sleep(account_interval_seconds())


# --- app ---------------------------------------------------------------------------------


class OrderRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=10)
    side: str = Field(pattern="^(buy|sell|BUY|SELL)$")
    qty: int = Field(ge=1, le=100000)
    # "alpaca" (Alpaca paper account) or the name of a local portfolio
    destination: str = Field(default="alpaca", min_length=1, max_length=40)


class PortfolioRequest(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    cash: float = Field(gt=0, le=1e9)


class AutopilotRequest(BaseModel):
    symbols: List[str] = Field(min_length=1, max_length=40)
    interval_seconds: int = Field(ge=autotrader.MIN_INTERVAL, le=autotrader.MAX_INTERVAL)
    mode: str = Field(pattern="^(signals|record|paper)$")
    portfolio: str = Field(default="default", max_length=40)


class PauseRequest(BaseModel):
    paused: bool


def create_app(journal: Optional[Journal] = None, background: bool = True) -> FastAPI:
    journal = journal or Journal()
    hub = Hub(journal)
    portfolios = Portfolios(journal.path)
    portfolios.ensure_default()
    pilot = autotrader.AutoTrader(journal, portfolios, publish=hub.publish)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        if background:
            hub.start()
        yield
        await pilot.stop(announce=False)
        await hub.stop()

    app = FastAPI(title="ADT Desk", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.hub = hub
    app.state.autopilot = pilot
    app.state.portfolios = portfolios

    def quotes_for(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        missing = [s for s in symbols if s not in hub.last_quotes]
        fresh = market_feed.get_quotes(missing) if missing else {}
        hub.last_quotes.update(fresh)
        return {s: hub.last_quotes[s] for s in symbols if s in hub.last_quotes}

    def portfolio_view(name: str) -> Dict[str, Any]:
        p = portfolios.get(name)
        return {**portfolios.valuation(name, quotes_for([h["symbol"] for h in p["holdings"]])),
                "fills": portfolios.fills(name, 25)}

    def capital_for(source: str, symbol: str) -> "tuple[float, float, str]":
        if source == "alpaca":
            if not (alpaca_configured() and alpaca.is_paper()):
                raise HTTPException(400, "Alpaca paper account not configured")
            acct = alpaca.account()
            held = next((float(p.get("qty") or 0) for p in alpaca.positions() if p.get("symbol") == symbol), 0.0)
            return float(acct.get("equity") or 0), held, "alpaca paper equity"
        try:
            view = portfolio_view(source)
        except PortfolioError as exc:
            raise HTTPException(404, str(exc))
        return view["equity"], portfolios.held(source, symbol), f"portfolio {source}"

    def require_token(request: Request) -> None:
        header = request.headers.get("authorization", "")
        token = header[7:] if header.lower().startswith("bearer ") else request.query_params.get("token")
        if not token_ok(token):
            raise HTTPException(401, "Desk token required")

    auth = [Depends(require_token)]

    @app.get("/", include_in_schema=False)
    @app.get("/desk", include_in_schema=False)
    async def desk_page():
        return FileResponse(STATIC_DIR / "desk.html")

    @app.get("/api/desk/auth")
    async def auth_info():
        return {"auth_required": desk_token() is not None}

    @app.get("/api/desk/config", dependencies=auth)
    async def get_config():
        return desk_config()

    @app.get("/api/desk/quotes", dependencies=auth)
    async def get_quotes(symbols: str = Query(...)):
        return await run_in_threadpool(market_feed.get_quotes, clean_symbols(symbols.split(",")))

    @app.get("/api/desk/bars/{symbol}", dependencies=auth)
    async def get_bars(symbol: str, timeframe: str = market_feed.DEFAULT_TIMEFRAME, limit: int = 300):
        try:
            return await run_in_threadpool(market_feed.get_bars, symbol, timeframe, limit)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/desk/news/{symbol}", dependencies=auth)
    async def get_news(symbol: str):
        articles = await run_in_threadpool(get_news_articles, symbol.upper())
        return [{
            "title": a.get("title"), "url": a.get("url"), "published_at": a.get("publishedAt"),
            "source": (a.get("source") or {}).get("name") if isinstance(a.get("source"), dict) else a.get("source"),
            "symbols": a.get("symbols") or [],
        } for a in articles[:30]]

    @app.get("/api/desk/events", dependencies=auth)
    async def get_events(limit: int = HISTORY_EVENTS):
        return await run_in_threadpool(recent_events, journal, max(1, min(limit, 500)))

    @app.get("/api/desk/newsbot", dependencies=auth)
    async def get_newsbot():
        return await run_in_threadpool(newsbot_status, journal)

    @app.get("/api/desk/account", dependencies=auth)
    async def get_account():
        return await run_in_threadpool(account_snapshot)

    @app.post("/api/desk/judge/{symbol}", dependencies=auth)
    async def judge(symbol: str):
        symbol = (clean_symbols([symbol]) or [None])[0]
        if not symbol:
            raise HTTPException(400, "Invalid symbol")
        result = await run_in_threadpool(judge_symbol, symbol)
        await hub.broadcast({"type": "event", "data": {
            "id": f"j{datetime.now(timezone.utc).timestamp()}", "type": "judgment", "level": "info",
            "timestamp": result["timestamp"], "symbol": symbol,
            "message": f"JUDGE {result['call']}" + (f" p={result['probability']:.2f}" if result["probability"] else "")
                       + f" · {len(result['headlines'])} headlines · {result['reason']}",
            "data": {"signal": result},
        }})
        return result

    @app.post("/api/desk/analyze/{symbol}", dependencies=auth)
    async def analyze_symbol(symbol: str, portfolio: str = "default"):
        symbol = (clean_symbols([symbol]) or [None])[0]
        if not symbol:
            raise HTTPException(400, "Invalid symbol")
        capital, held, source = await run_in_threadpool(capital_for, portfolio, symbol)
        result = await run_in_threadpool(lambda: analysis.analyze(symbol, capital, held, capital_source=source))
        hub.publish({
            "id": f"an{datetime.now(timezone.utc).timestamp()}", "type": "decision", "level": "info",
            "timestamp": result["timestamp"], "symbol": symbol,
            "message": f"ANALYZE {symbol}: {result['recommendation']} {result['quantity']} "
                       f"({result['confidence']:.0%} conf) via {result['primary_strategy']}",
            "data": {**result, "source": "analyze"},
        })
        return result

    @app.get("/api/desk/autopilot", dependencies=auth)
    async def autopilot_status():
        return await run_in_threadpool(pilot.status)

    @app.post("/api/desk/autopilot", dependencies=auth)
    async def autopilot_start(req: AutopilotRequest):
        try:
            status = await pilot.start(autotrader.RunConfig(req.symbols, req.interval_seconds, req.mode, req.portfolio))
        except (ValueError, PortfolioError) as exc:
            raise HTTPException(400, str(exc))
        await hub.broadcast({"type": "autopilot", "data": status})
        return status

    @app.delete("/api/desk/autopilot", dependencies=auth)
    async def autopilot_stop():
        status = await pilot.stop()
        await hub.broadcast({"type": "autopilot", "data": status})
        return status

    @app.post("/api/desk/newsbot/pause", dependencies=auth)
    async def newsbot_pause(req: PauseRequest):
        newsbot.set_paused(req.paused)
        status = await run_in_threadpool(newsbot_status, journal)
        hub.publish({"id": f"np{datetime.now(timezone.utc).timestamp()}", "type": "flatten", "level": "system",
                     "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "symbol": None,
                     "message": "News bot trading PAUSED from the desk" if req.paused
                     else "News bot trading RESUMED from the desk", "data": {}})
        await hub.broadcast({"type": "newsbot", "data": status})
        return status

    @app.get("/api/desk/portfolios", dependencies=auth)
    async def list_portfolios():
        return await run_in_threadpool(lambda: [portfolio_view(n) for n in portfolios.names()])

    @app.get("/api/desk/portfolios/{name}", dependencies=auth)
    async def get_portfolio(name: str):
        try:
            return await run_in_threadpool(portfolio_view, name)
        except PortfolioError as exc:
            raise HTTPException(404, str(exc))

    @app.post("/api/desk/portfolios", dependencies=auth)
    async def create_portfolio(req: PortfolioRequest):
        try:
            await run_in_threadpool(portfolios.create, req.name, req.cash)
        except PortfolioError as exc:
            raise HTTPException(400, str(exc))
        return await run_in_threadpool(portfolio_view, req.name.strip())

    @app.delete("/api/desk/portfolios/{name}", dependencies=auth)
    async def delete_portfolio(name: str):
        if pilot.running and pilot.config and pilot.config.mode == "record" and pilot.config.portfolio == name:
            raise HTTPException(400, "The autopilot is recording into this portfolio; stop it first")
        try:
            await run_in_threadpool(portfolios.delete, name)
        except PortfolioError as exc:
            raise HTTPException(404, str(exc))
        return {"deleted": name}

    @app.post("/api/desk/orders", dependencies=auth)
    async def submit_order(req: OrderRequest):
        symbol = (clean_symbols([req.symbol]) or [None])[0]
        if not symbol:
            raise HTTPException(400, "Invalid symbol")
        if req.destination != "alpaca":
            price = (await run_in_threadpool(quotes_for, [symbol])).get(symbol, {}).get("last")
            if not price:
                raise HTTPException(400, f"No live price for {symbol}")
            try:
                fill = await run_in_threadpool(lambda: portfolios.record_fill(
                    req.destination, symbol, req.side, req.qty, price, source="desk", note="manual"))
                result = {"submitted": True, "fill": fill, "destination": req.destination}
                detail = f"{symbol} {req.qty} @ {price} recorded in {req.destination}"
            except PortfolioError as exc:
                result = {"submitted": False, "skipped_reason": str(exc), "destination": req.destination}
                detail = f"{symbol} not recorded in {req.destination}: {exc}"
            await run_in_threadpool(lambda: journal.log_event(
                DESK_ORDER_EVENT, tool="desk", symbol=symbol, side=req.side.lower(), quantity=req.qty,
                price=price, detail=detail, payload=result))
            return result
        if not alpaca_configured():
            raise HTTPException(400, "Alpaca keys not configured")
        if not alpaca.is_paper():
            raise HTTPException(400, "The desk sends Alpaca paper orders only")
        payload = alpaca.market_order_payload(symbol, req.side, req.qty)
        quote = hub.last_quotes.get(symbol) or {}
        try:
            order = await run_in_threadpool(alpaca.submit_order, payload)
            result = {"submitted": True, "order": order}
            detail = f"{symbol} market order {order.get('status', 'submitted')} {str(order.get('id', ''))[:8]}"
        except Exception as exc:
            result = {"submitted": False, "skipped_reason": str(exc)}
            detail = f"{symbol} rejected: {exc}"
        await run_in_threadpool(lambda: journal.log_event(
            DESK_ORDER_EVENT, tool="desk", symbol=symbol, side=req.side.lower(), quantity=req.qty,
            price=quote.get("last"), detail=detail,
            payload={**result, "request": payload}))
        return result

    @app.post("/api/desk/orders/{order_id}/cancel", dependencies=auth)
    async def cancel_order(order_id: str):
        try:
            await run_in_threadpool(alpaca.cancel_order, order_id)
        except Exception as exc:
            raise HTTPException(400, f"Cancel failed: {exc}")
        await run_in_threadpool(lambda: journal.log_event(
            DESK_CANCEL_EVENT, tool="desk", detail=f"order {order_id[:8]}", payload={"order_id": order_id}))
        return {"cancelled": order_id}

    @app.websocket("/ws/desk")
    async def desk_socket(websocket: WebSocket):
        if not token_ok(websocket.query_params.get("token")):
            await websocket.close(code=4401)
            return
        await websocket.accept()
        client = DeskClient(websocket)
        hub.clients.add(client)
        try:
            events = await run_in_threadpool(recent_events, journal)
            events = sorted(list(hub.live_events) + events, key=lambda e: e["timestamp"], reverse=True)
            hello = {
                "events": events[:HISTORY_EVENTS],
                "newsbot": await run_in_threadpool(newsbot_status, journal),
                "autopilot": await run_in_threadpool(pilot.status),
                "portfolios": portfolios.names(),
                "account": hub.last_account,
            }
            await client.send({"type": "hello", "data": hello})
            while True:
                message = json.loads(await websocket.receive_text())
                if message.get("type") == "watch":
                    client.symbols = set(clean_symbols(message.get("symbols")))
                    cached = {s: hub.last_quotes[s] for s in client.symbols if s in hub.last_quotes}
                    if cached:
                        await client.send({"type": "quotes", "data": cached})
                elif message.get("type") == "ping":
                    await client.send({"type": "pong"})
        except (WebSocketDisconnect, ValueError):
            pass
        finally:
            hub.clients.discard(client)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def main() -> int:
    import uvicorn

    config.load_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    host = os.getenv("DESK_HOST", "127.0.0.1")
    port = int(os.getenv("DESK_PORT", "8000"))
    if host not in ("127.0.0.1", "localhost", "::1") and desk_token() is None:
        logger.error("DESK_HOST exposes the desk beyond this machine: set DESK_TOKEN first.")
        return 2
    print(f"ADT Desk on http://{host}:{port}/desk")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
