#!/usr/bin/env python3
"""
Live trading desk API.

REST endpoints under ``/api/desk`` serve the desk's initial state (quotes,
chart bars, news, Alpaca account, agent event history, autopilot control),
and the ``/ws/desk`` WebSocket streams everything that changes afterwards:

    server -> client
        hello       desk config, autopilot status and recent agent events
        quotes      {symbol: quote} for the client's watched symbols
        event       one agent event (analysis step, decision, order, ...)
        account     Alpaca account, positions, orders and market clock
        autopilot   autopilot status change
        pong        reply to ping

    client -> server
        watch       {"symbols": [...]} replaces the client's quote subscription
        ping
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import jwt
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.concurrency import run_in_threadpool
from jwt.exceptions import PyJWTError
from pydantic import BaseModel, Field, field_validator

from config.api.auth import JWT_ALGORITHM, JWT_SECRET_KEY, User, get_current_active_user, get_user
from config.api.dependencies import get_portfolio_manager
from core import market_feed
from core.event_bus import event_bus, publish_event
from core.portfolio_manager import PortfolioManager
from core.trading_workflow import TradingWorkflow

logger = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_WATCHLIST = "SPY,QQQ,AAPL,MSFT,NVDA,TSLA,AMZN,META,GOOGL,AMD"
MAX_WATCH_SYMBOLS = 40
MIN_AUTOPILOT_INTERVAL = 60
SYMBOL_PATTERN = "^[A-Z][A-Z.]{0,9}$"


def default_watchlist() -> List[str]:
    raw = os.getenv("DESK_WATCHLIST", DEFAULT_WATCHLIST)
    return [s.strip().upper() for s in raw.split(",") if s.strip()][:MAX_WATCH_SYMBOLS]


def alpaca_configured() -> bool:
    return bool(
        (os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY_ID"))
        and (os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET"))
    )


def quote_interval_seconds() -> float:
    """Poll fast when quotes come from one batched call, slower on per-symbol Yahoo."""
    configured = os.getenv("DESK_QUOTE_INTERVAL")
    if configured:
        return max(1.0, float(configured))
    if market_feed.demo_mode_enabled():
        return 1.0
    return 2.0 if alpaca_configured() else 10.0


def account_interval_seconds() -> float:
    return max(3.0, float(os.getenv("DESK_ACCOUNT_INTERVAL", "10")))


def _clean_symbols(symbols: List[str]) -> List[str]:
    cleaned = []
    for symbol in symbols or []:
        symbol = str(symbol).strip().upper()
        if symbol and len(symbol) <= 10 and symbol.replace(".", "").isalpha() and symbol not in cleaned:
            cleaned.append(symbol)
    return cleaned[:MAX_WATCH_SYMBOLS]


def desk_config() -> Dict[str, Any]:
    return {
        "demo_mode": market_feed.demo_mode_enabled(),
        "alpaca_configured": alpaca_configured(),
        "watchlist": default_watchlist(),
        "timeframes": list(market_feed.TIMEFRAMES),
        "default_timeframe": market_feed.DEFAULT_TIMEFRAME,
        "quote_interval_seconds": quote_interval_seconds(),
        "min_autopilot_interval_seconds": MIN_AUTOPILOT_INTERVAL,
    }


# ----------------------------------------------------------------------
# Alpaca account snapshot
# ----------------------------------------------------------------------

def _f(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def account_snapshot() -> Dict[str, Any]:
    """Collect account, positions, recent orders and clock from Alpaca paper trading."""
    if not alpaca_configured():
        return {"connected": False, "message": "Alpaca keys not configured"}

    from core.alpaca_executor_provider import get_alpaca_executor

    try:
        executor = get_alpaca_executor()
        account = executor.get_account()
        positions = executor.get_positions()
        orders = executor.get_orders(limit=25)
        clock = executor.get_clock()
    except Exception as exc:
        return {"connected": False, "message": f"Alpaca unavailable: {exc}"}

    equity = _f(account.get("equity"))
    last_equity = _f(account.get("last_equity"))
    return {
        "connected": True,
        "account": {
            "status": account.get("status"),
            "equity": equity,
            "last_equity": last_equity,
            "day_pl": (equity - last_equity) if equity is not None and last_equity else None,
            "day_pl_pct": ((equity - last_equity) / last_equity * 100) if equity is not None and last_equity else None,
            "cash": _f(account.get("cash")),
            "buying_power": _f(account.get("buying_power")),
            "long_market_value": _f(account.get("long_market_value")),
            "daytrade_count": account.get("daytrade_count"),
        },
        "positions": [
            {
                "symbol": p.get("symbol"),
                "qty": _f(p.get("qty")),
                "side": p.get("side"),
                "avg_entry_price": _f(p.get("avg_entry_price")),
                "current_price": _f(p.get("current_price")),
                "market_value": _f(p.get("market_value")),
                "unrealized_pl": _f(p.get("unrealized_pl")),
                "unrealized_plpc": (_f(p.get("unrealized_plpc")) or 0) * 100,
                "unrealized_intraday_pl": _f(p.get("unrealized_intraday_pl")),
                "change_today": (_f(p.get("change_today")) or 0) * 100,
            }
            for p in positions
        ],
        "orders": [
            {
                "id": o.get("id"),
                "symbol": o.get("symbol"),
                "side": o.get("side"),
                "type": o.get("type"),
                "qty": _f(o.get("qty")),
                "filled_qty": _f(o.get("filled_qty")),
                "filled_avg_price": _f(o.get("filled_avg_price")),
                "limit_price": _f(o.get("limit_price")),
                "stop_price": _f(o.get("stop_price")),
                "status": o.get("status"),
                "submitted_at": o.get("submitted_at"),
                "filled_at": o.get("filled_at"),
            }
            for o in orders
        ],
        "clock": {
            "is_open": clock.get("is_open"),
            "next_open": clock.get("next_open"),
            "next_close": clock.get("next_close"),
            "timestamp": clock.get("timestamp"),
        },
    }


# ----------------------------------------------------------------------
# Autopilot
# ----------------------------------------------------------------------

class AutopilotConfig(BaseModel):
    symbols: List[str] = Field(..., min_length=1, max_length=MAX_WATCH_SYMBOLS)
    interval_seconds: int = Field(300, ge=MIN_AUTOPILOT_INTERVAL, le=86400)
    portfolio_name: str = Field("default", min_length=1, max_length=50)
    record_local_trade: bool = Field(False, description="Record actionable signals in the local portfolio")
    submit_paper_order: bool = Field(False, description="Submit actionable signals to Alpaca paper trading")

    @field_validator("symbols", mode="before")
    @classmethod
    def clean(cls, value: List[str]) -> List[str]:
        symbols = _clean_symbols(value)
        if not symbols:
            raise ValueError("At least one valid symbol is required")
        return symbols


@dataclass
class AutopilotRun:
    user_id: int
    config: AutopilotConfig
    task: Optional[asyncio.Task] = None
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cycles: int = 0
    current_symbol: Optional[str] = None
    last_cycle_at: Optional[str] = None
    next_cycle_at: Optional[str] = None

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.task is not None and not self.task.done(),
            "config": self.config.model_dump(),
            "started_at": self.started_at,
            "cycles": self.cycles,
            "current_symbol": self.current_symbol,
            "last_cycle_at": self.last_cycle_at,
            "next_cycle_at": self.next_cycle_at,
        }


class AutopilotManager:
    """One scanning loop per user: analyze each symbol, optionally paper trade, sleep, repeat."""

    def __init__(self) -> None:
        self.runs: Dict[int, AutopilotRun] = {}

    def status(self, user_id: int) -> Dict[str, Any]:
        run = self.runs.get(user_id)
        if not run:
            return {"running": False, "config": None, "cycles": 0}
        return run.status()

    async def start(self, user_id: int, config: AutopilotConfig, db: PortfolioManager) -> Dict[str, Any]:
        await self.stop(user_id, announce=False)
        run = AutopilotRun(user_id=user_id, config=config)
        self.runs[user_id] = run
        run.task = asyncio.create_task(self._loop(run, db))
        mode = "paper orders" if config.submit_paper_order else (
            "local paper records" if config.record_local_trade else "signals only")
        publish_event(
            "autopilot",
            f"Autopilot ON: {len(config.symbols)} symbols every {config.interval_seconds}s ({mode})",
            level="system", user_id=user_id, data=run.status(),
        )
        return run.status()

    async def stop(self, user_id: int, announce: bool = True) -> Dict[str, Any]:
        run = self.runs.get(user_id)
        if run and run.task and not run.task.done():
            run.task.cancel()
            try:
                await run.task
            except (asyncio.CancelledError, Exception):
                pass
            if announce:
                publish_event("autopilot", "Autopilot OFF", level="system", user_id=user_id,
                              data=run.status())
        return self.status(user_id)

    async def stop_all(self) -> None:
        for user_id in list(self.runs):
            await self.stop(user_id, announce=False)

    async def _loop(self, run: AutopilotRun, db: PortfolioManager) -> None:
        config = run.config
        workflow = TradingWorkflow(db)
        while True:
            run.cycles += 1
            publish_event("autopilot", f"Autopilot cycle {run.cycles}: scanning {', '.join(config.symbols)}",
                          level="system", user_id=run.user_id, data=run.status())
            for symbol in config.symbols:
                run.current_symbol = symbol
                try:
                    result = await run_in_threadpool(
                        workflow.run,
                        symbol,
                        config.portfolio_name,
                        record_paper_trade=config.record_local_trade or config.submit_paper_order,
                        submit_alpaca_paper_order=config.submit_paper_order,
                        user_id=run.user_id,
                    )
                    if result.skipped_reason and result.analysis.get("error"):
                        publish_event("autopilot", f"{symbol}: {result.skipped_reason}", level="warning",
                                      symbol=symbol, user_id=run.user_id)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception(f"Autopilot error on {symbol}")
                    publish_event("autopilot", f"{symbol}: autopilot error - {exc}", level="error",
                                  symbol=symbol, user_id=run.user_id)
            run.current_symbol = None
            now = datetime.now(timezone.utc)
            run.last_cycle_at = now.isoformat()
            run.next_cycle_at = datetime.fromtimestamp(
                now.timestamp() + config.interval_seconds, timezone.utc).isoformat()
            publish_event("autopilot", f"Autopilot cycle {run.cycles} complete; next in {config.interval_seconds}s",
                          level="system", user_id=run.user_id, data=run.status())
            await asyncio.sleep(config.interval_seconds)


autopilot = AutopilotManager()


# ----------------------------------------------------------------------
# WebSocket hub
# ----------------------------------------------------------------------

@dataclass(eq=False)
class DeskClient:
    websocket: WebSocket
    user_id: int
    symbols: Set[str] = field(default_factory=set)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def send(self, message: Dict[str, Any]) -> bool:
        try:
            async with self.send_lock:
                await self.websocket.send_json(message)
            return True
        except Exception:
            return False


class DeskHub:
    """Shares one quote poller and one account poller across every open desk."""

    def __init__(self) -> None:
        self.clients: Set[DeskClient] = set()
        self._tasks: List[asyncio.Task] = []
        self.last_quotes: Dict[str, Dict[str, Any]] = {}
        self.last_account: Optional[Dict[str, Any]] = None

    def watched_symbols(self) -> List[str]:
        symbols: Set[str] = set()
        for client in self.clients:
            symbols |= client.symbols
        return sorted(symbols)

    def add(self, client: DeskClient) -> None:
        self.clients.add(client)
        if not self._tasks:
            self._tasks = [
                asyncio.create_task(self._quote_loop()),
                asyncio.create_task(self._account_loop()),
            ]

    async def remove(self, client: DeskClient) -> None:
        self.clients.discard(client)
        if not self.clients:
            await self.shutdown()

    async def shutdown(self) -> None:
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def push_quotes(self, client: DeskClient) -> None:
        """Send cached quotes immediately so a new watch list is not blank until the next poll."""
        cached = {s: self.last_quotes[s] for s in client.symbols if s in self.last_quotes}
        if cached:
            await client.send({"type": "quotes", "data": cached})

    async def _quote_loop(self) -> None:
        while True:
            symbols = self.watched_symbols()
            if symbols:
                try:
                    quotes = await run_in_threadpool(market_feed.get_quotes, symbols)
                    self.last_quotes.update(quotes)
                    for client in list(self.clients):
                        mine = {s: q for s, q in quotes.items() if s in client.symbols}
                        if mine:
                            await client.send({"type": "quotes", "data": mine})
                except Exception as exc:
                    logger.warning(f"Desk quote poll failed: {exc}")
            await asyncio.sleep(quote_interval_seconds())

    async def _account_loop(self) -> None:
        while True:
            if alpaca_configured() and not market_feed.demo_mode_enabled():
                try:
                    snapshot = await run_in_threadpool(account_snapshot)
                    self.last_account = snapshot
                    for client in list(self.clients):
                        await client.send({"type": "account", "data": snapshot})
                except Exception as exc:
                    logger.warning(f"Desk account poll failed: {exc}")
            await asyncio.sleep(account_interval_seconds())


hub = DeskHub()


async def _authenticate_ws(websocket: WebSocket, token: Optional[str]) -> Optional[User]:
    if not token:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except PyJWTError:
        return None
    username = payload.get("sub")
    if not username or payload.get("type") != "access":
        return None
    db = get_portfolio_manager()
    jti = payload.get("jti")
    if jti and await run_in_threadpool(db.is_token_blacklisted, jti):
        return None
    user = await run_in_threadpool(get_user, username, db)
    if not user or not user.is_active:
        return None
    return user


async def _forward_events(client: DeskClient) -> None:
    queue = event_bus.subscribe()
    try:
        while True:
            event = await queue.get()
            if event["user_id"] is None or event["user_id"] == client.user_id:
                if event["type"] == "autopilot" and event["user_id"] == client.user_id:
                    await client.send({"type": "autopilot", "data": autopilot.status(client.user_id)})
                await client.send({"type": "event", "data": event})
    finally:
        event_bus.unsubscribe(queue)


async def desk_websocket(websocket: WebSocket, token: Optional[str] = Query(None)):
    """Live desk stream. Connect with ``/ws/desk?token=<access token>``."""
    user = await _authenticate_ws(websocket, token)
    if not user:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    client = DeskClient(websocket=websocket, user_id=user.id, symbols=set(default_watchlist()))
    await client.send({
        "type": "hello",
        "data": {
            "user": user.username,
            "config": desk_config(),
            "autopilot": autopilot.status(user.id),
            "events": event_bus.history(limit=200, user_id=user.id),
            "account": hub.last_account,
        },
    })
    hub.add(client)
    await hub.push_quotes(client)
    forwarder = asyncio.create_task(_forward_events(client))

    try:
        while True:
            message = await websocket.receive_json()
            kind = message.get("type") if isinstance(message, dict) else None
            if kind == "watch":
                client.symbols = set(_clean_symbols(message.get("symbols") or []))
                await hub.push_quotes(client)
            elif kind == "ping":
                await client.send({"type": "pong", "timestamp": datetime.now(timezone.utc).isoformat()})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning(f"Desk websocket error for {user.username}: {exc}")
    finally:
        forwarder.cancel()
        await hub.remove(client)


# ----------------------------------------------------------------------
# REST endpoints
# ----------------------------------------------------------------------

@router.get("/config")
async def get_desk_config(current_user: User = Depends(get_current_active_user)):
    return desk_config()


@router.get("/quotes")
async def get_desk_quotes(
    symbols: str = Query(..., description="Comma-separated symbols"),
    current_user: User = Depends(get_current_active_user),
):
    cleaned = _clean_symbols(symbols.split(","))
    if not cleaned:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid symbols")
    return await run_in_threadpool(market_feed.get_quotes, cleaned)


@router.get("/bars/{symbol}")
async def get_desk_bars(
    symbol: str,
    timeframe: str = Query(market_feed.DEFAULT_TIMEFRAME),
    limit: int = Query(300, ge=10, le=1000),
    current_user: User = Depends(get_current_active_user),
):
    cleaned = _clean_symbols([symbol])
    if not cleaned:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid symbol")
    try:
        return await run_in_threadpool(market_feed.get_bars, cleaned[0], timeframe, limit)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/news/{symbol}")
async def get_desk_news(symbol: str, current_user: User = Depends(get_current_active_user)):
    cleaned = _clean_symbols([symbol])
    if not cleaned:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid symbol")
    from core.news_fetcher import get_news_articles

    try:
        articles = await run_in_threadpool(get_news_articles, cleaned[0])
    except Exception as exc:
        logger.warning(f"News fetch failed for {cleaned[0]}: {exc}")
        articles = []
    return [
        {
            "title": a.get("title"),
            "summary": a.get("description"),
            "url": a.get("url"),
            "published_at": a.get("publishedAt"),
            "source": (a.get("source") or {}).get("name"),
            "symbols": a.get("symbols") or [],
        }
        for a in articles[:30]
        if a.get("title")
    ]


@router.get("/events")
async def get_desk_events(
    limit: int = Query(200, ge=1, le=500),
    current_user: User = Depends(get_current_active_user),
):
    return event_bus.history(limit=limit, user_id=current_user.id)


@router.get("/account")
async def get_desk_account(current_user: User = Depends(get_current_active_user)):
    return await run_in_threadpool(account_snapshot)


@router.post("/orders/{order_id}/cancel")
async def cancel_desk_order(order_id: str, current_user: User = Depends(get_current_active_user)):
    if not alpaca_configured():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Alpaca keys not configured")
    from core.alpaca_executor_provider import get_alpaca_executor

    executor = await run_in_threadpool(get_alpaca_executor)
    cancelled = await run_in_threadpool(executor.cancel_order, order_id)
    publish_event(
        "order_cancelled" if cancelled else "order_failed",
        f"Cancel {'accepted' if cancelled else 'rejected'} for order {order_id[:8]}",
        level="warning" if cancelled else "error",
        data={"order_id": order_id},
    )
    return {"cancelled": cancelled}


@router.get("/autopilot")
async def get_autopilot(current_user: User = Depends(get_current_active_user)):
    return autopilot.status(current_user.id)


@router.post("/autopilot")
async def start_autopilot(
    config: AutopilotConfig,
    current_user: User = Depends(get_current_active_user),
    db: PortfolioManager = Depends(get_portfolio_manager),
):
    if config.record_local_trade or config.submit_paper_order:
        portfolio = await run_in_threadpool(db.get_portfolio, config.portfolio_name, current_user.id)
        if not portfolio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Portfolio '{config.portfolio_name}' not found; trading modes need an existing portfolio",
            )
    return await autopilot.start(current_user.id, config, db)


@router.delete("/autopilot")
async def stop_autopilot(current_user: User = Depends(get_current_active_user)):
    return await autopilot.stop(current_user.id)
