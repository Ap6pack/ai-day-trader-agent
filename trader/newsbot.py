"""
Autonomous news trading on Alpaca paper (docs/PLAN.md, Phase 3).

Alpaca's real-time news stream feeds each headline to Jev (one request per
headline and symbol); plain rules in code turn the judgment into a signal, and
a bullish signal that passes the hard limits becomes a paper bracket order
(market entry, take-profit limit, stop-loss). There is no approval step.

Safety:
  * NEWSBOT_EXECUTION=off (the default) journals signals only; `paper` places
    orders. Order functions in trader.alpaca refuse any non-paper base URL.
  * Long only: bearish signals are journaled as `sell` decisions, never shorted.
  * Limits from .env: size per trade, trades per day, per-symbol cooldown,
    minimum price, no new entries close to the close, bracket on every order.
    `run` flattens every position NEWSBOT_FLATTEN_MINUTES before the close from
    Alpaca's clock (so early-close days are covered); the cron `flatten` is a backup.

    python -m trader.newsbot run                   # stream and trade
    python -m trader.newsbot replay AAPL --hours 24  # apply the rules to recent headlines
    python -m trader.newsbot flatten               # close all paper positions
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional

from trader import alpaca, config, jev_news
from trader.journal import Journal
from trader.news_sources import _normalize_alpaca, alpaca_configured, get_alpaca_news
from trader.scan import SYMBOL_RE

logger = logging.getLogger(__name__)

STREAM_URL = "wss://stream.data.alpaca.markets/v1beta1/news"
EXECUTION_MODES = ("off", "paper")
ORDER_EVENT = "newsbot_order"
SIGNAL_EVENT = "newsbot_signal"
FLATTEN_EVENT = "newsbot_flatten"
CLOCK_CACHE_SECONDS = 60
SEEN_IDS_MAX = 10000
HANDLER_WORKERS = 8
FLATTEN_CHECK_SECONDS = 30


@dataclass(frozen=True)
class Settings:
    execution: str = "off"
    symbols: FrozenSet[str] = field(default_factory=frozenset)
    min_relevance: float = 0.8
    min_materiality: float = 0.6
    min_probability: float = 0.75
    max_headline_age_seconds: int = 120
    max_symbols_per_headline: int = 2
    order_usd: float = 500.0
    take_profit_pct: float = 2.0
    stop_loss_pct: float = 1.0
    max_trades_per_day: int = 5
    cooldown_minutes: int = 30
    min_price: float = 5.0
    entry_cutoff_minutes: int = 15
    flatten_minutes: int = 10


def load_settings(env: Optional[Dict[str, str]] = None) -> Settings:
    env = dict(os.environ) if env is None else env
    execution = config._get(env, "NEWSBOT_EXECUTION", "off").lower()
    if execution not in EXECUTION_MODES:
        raise ValueError(f"NEWSBOT_EXECUTION must be one of {EXECUTION_MODES}, got {execution!r}")
    symbols = config._get(env, "NEWSBOT_SYMBOLS", "")
    settings = Settings(
        execution=execution,
        symbols=frozenset(s.strip().upper() for s in symbols.split(",") if s.strip()),
        min_relevance=config._float(env, "NEWSBOT_MIN_RELEVANCE", 0.8),
        min_materiality=config._float(env, "NEWSBOT_MIN_MATERIALITY", 0.6),
        min_probability=config._float(env, "NEWSBOT_MIN_PROBABILITY", 0.75),
        max_headline_age_seconds=config._int(env, "NEWSBOT_MAX_HEADLINE_AGE_SECONDS", 120),
        max_symbols_per_headline=config._int(env, "NEWSBOT_MAX_SYMBOLS_PER_HEADLINE", 2),
        order_usd=config._float(env, "NEWSBOT_ORDER_USD", 500.0),
        take_profit_pct=config._float(env, "NEWSBOT_TAKE_PROFIT_PCT", 2.0),
        stop_loss_pct=config._float(env, "NEWSBOT_STOP_LOSS_PCT", 1.0),
        max_trades_per_day=config._int(env, "NEWSBOT_MAX_TRADES_PER_DAY", 5),
        cooldown_minutes=config._int(env, "NEWSBOT_COOLDOWN_MINUTES", 30),
        min_price=config._float(env, "NEWSBOT_MIN_PRICE", 5.0),
        entry_cutoff_minutes=config._int(env, "NEWSBOT_ENTRY_CUTOFF_MINUTES", 15),
        flatten_minutes=config._int(env, "NEWSBOT_FLATTEN_MINUTES", 10),
    )
    if not 0 < settings.min_probability <= 1:
        raise ValueError("NEWSBOT_MIN_PROBABILITY must be in (0, 1]")
    if settings.take_profit_pct <= 0 or settings.stop_loss_pct <= 0:
        raise ValueError("NEWSBOT_TAKE_PROFIT_PCT and NEWSBOT_STOP_LOSS_PCT must be positive")
    if settings.order_usd <= 0 or settings.max_trades_per_day < 0 or settings.cooldown_minutes < 0:
        raise ValueError("NEWSBOT_ORDER_USD must be positive; trade cap and cooldown must not be negative")
    if not 0 < settings.flatten_minutes < settings.entry_cutoff_minutes:
        raise ValueError("NEWSBOT_FLATTEN_MINUTES must be positive and less than NEWSBOT_ENTRY_CUTOFF_MINUTES, "
                         "so no entry can open after the flatten")
    return settings


# --- rules ------------------------------------------------------------------


@dataclass
class Signal:
    action: str  # "buy", "bearish" or "none"
    reason: str
    probability: Optional[float] = None


def _parse_time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def decide(symbol: str, article: Dict[str, Any], judgment: Optional[Dict[str, Any]],
           settings: Settings, now: datetime) -> Signal:
    """Turn one Jev judgment into a signal. Pure: no I/O."""
    if judgment is None:
        return Signal("none", "no Jev judgment")
    published = _parse_time(article.get("publishedAt"))
    if published is None:
        return Signal("none", "headline has no timestamp")
    age = (now - published).total_seconds()
    if age > settings.max_headline_age_seconds:
        return Signal("none", f"headline is {age:.0f}s old (max {settings.max_headline_age_seconds}s)")
    relevance = judgment.get("relevance") or 0.0
    if relevance < settings.min_relevance:
        return Signal("none", f"relevance {relevance:.2f} < {settings.min_relevance}")
    materiality = judgment.get("materiality") or 0.0
    if materiality < settings.min_materiality:
        return Signal("none", f"materiality {materiality:.2f} < {settings.min_materiality}")
    p_bullish = judgment.get("p_bullish") or 0.0
    p_bearish = judgment.get("p_bearish") or 0.0
    if p_bullish >= settings.min_probability:
        return Signal("buy", f"p_bullish {p_bullish:.2f} >= {settings.min_probability}", p_bullish)
    if p_bearish >= settings.min_probability:
        return Signal("bearish", f"p_bearish {p_bearish:.2f} >= {settings.min_probability}", p_bearish)
    return Signal("none", (
        f"no clear direction (p_bullish {p_bullish:.2f}, p_bearish {p_bearish:.2f}, "
        f"need {settings.min_probability})"
    ))


def symbols_to_judge(item_symbols: Iterable[str], settings: Settings) -> List[str]:
    """Tagged symbols worth a Jev call: none for multi-stock roundups, allowlist applied."""
    symbols = list(dict.fromkeys(str(s).upper() for s in item_symbols or [] if s))
    if not symbols or len(symbols) > settings.max_symbols_per_headline:
        return []
    return [
        s for s in symbols
        if SYMBOL_RE.match(s) and (not settings.symbols or s in settings.symbols)
    ]


# --- bot --------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


class NewsBot:
    def __init__(
        self,
        settings: Settings,
        journal: Optional[Journal] = None,
        judge: Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]] = jev_news.judge_headline,
        broker: Any = alpaca,
        now: Callable[[], datetime] = _now,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.journal = journal or Journal()
        self.judge = judge
        self.broker = broker
        self.now = now
        self.monotonic = monotonic
        self._clock: Optional[Dict[str, Any]] = None
        self._clock_at = -math.inf
        self._clock_lock = threading.Lock()
        # Serializes the limit checks and the order so two headlines at once
        # cannot both pass the daily cap or the cooldown.
        self._order_lock = threading.Lock()
        self._flattened_for: Optional[str] = None

    def market_clock(self) -> Dict[str, Any]:
        with self._clock_lock:
            if self._clock is None or self.monotonic() - self._clock_at >= CLOCK_CACHE_SECONDS:
                self._clock = self.broker.market_clock()
                self._clock_at = self.monotonic()
            return self._clock

    def handle(self, symbol: str, article: Dict[str, Any], received_at: datetime,
               news_id: Any = None) -> Dict[str, Any]:
        """Judge one headline for one symbol, journal it and trade it if it passes every rule."""
        symbol = symbol.upper()
        started = time.perf_counter()
        try:
            judgment = self.judge(symbol, article)
        except Exception as exc:
            logger.warning(f"Jev failed for {symbol}: {exc}")
            judgment = None
        jev_ms = round((time.perf_counter() - started) * 1000)
        signal = decide(symbol, article, judgment, self.settings, self.now())

        record: Dict[str, Any] = {
            "news_id": news_id,
            "headline": article.get("title"),
            "published_at": article.get("publishedAt"),
            "signal": signal.action,
            "reason": signal.reason,
            "probability": signal.probability,
            "execution": self.settings.execution,
            "jev_ms": jev_ms,
        }
        if judgment:
            record.update({k: judgment.get(k) for k in (
                "relevance", "materiality", "p_bullish", "p_bearish", "event_type", "direction")})

        if signal.action == "none":
            return self._log(SIGNAL_EVENT, symbol, record, received_at)

        try:
            price = self.broker.latest_price(symbol)
        except Exception as exc:
            record["blocked"] = f"latest price unavailable: {exc}"
            return self._log(SIGNAL_EVENT, symbol, record, received_at)
        record["price"] = price

        side = "buy" if signal.action == "buy" else "sell"
        record["decision_id"] = self.journal.add_decision(
            symbol, side, price,
            confidence=signal.probability,
            thesis=f"[newsbot] {record.get('event_type') or 'news'}: {article.get('title') or ''}"[:500],
            mode=f"newsbot-{self.settings.execution}",
            now=self.now(),
        )
        if signal.action == "bearish":
            record["blocked"] = "long only: bearish signals are journaled, not shorted"
            return self._log(SIGNAL_EVENT, symbol, record, received_at, side=side)

        with self._order_lock:
            blocked, qty = self._check_limits(symbol, price)
            if blocked:
                record["blocked"] = blocked
                return self._log(SIGNAL_EVENT, symbol, record, received_at, side=side)
            key = f"{news_id}:{symbol}" if news_id is not None else None
            payload = self.broker.bracket_order_payload(
                symbol, qty, price, self.settings.take_profit_pct, self.settings.stop_loss_pct, key=key)
            record["order_request"] = payload
            try:
                order = self.broker.submit_order(payload)
            except Exception as exc:
                record["blocked"] = f"order rejected: {exc}"
                return self._log(SIGNAL_EVENT, symbol, record, received_at, side=side)
            record["order_id"] = (order or {}).get("id")
            record["order_status"] = (order or {}).get("status")
            record["qty"] = qty
            return self._log(ORDER_EVENT, symbol, record, received_at, side=side, qty=qty)

    def maybe_flatten(self) -> Optional[Dict[str, Any]]:
        """Close every paper position once per session, NEWSBOT_FLATTEN_MINUTES before the close.

        Bracket legs are day orders: a position still open at the close would be
        held overnight with no stop. Failures are retried on the next check."""
        if self.settings.execution != "paper":
            return None
        try:
            clock = self.market_clock()
        except Exception as exc:
            logger.warning(f"Flatten check skipped, market clock unavailable: {exc}")
            return None
        next_close = _parse_time(clock.get("next_close"))
        if not clock.get("is_open") or next_close is None or clock["next_close"] == self._flattened_for:
            return None
        if next_close - self.now() > timedelta(minutes=self.settings.flatten_minutes):
            return None
        with self._order_lock:
            closed = self.broker.close_all_positions()
            self._flattened_for = clock["next_close"]
        detail = f"closed {len(closed)} positions before the {clock['next_close']} close"
        self.journal.log_event(FLATTEN_EVENT, tool="newsbot", detail=detail,
                               payload={"response": closed}, now=self.now())
        logger.info(f"{FLATTEN_EVENT}: {detail}")
        return {"closed": len(closed), "next_close": clock["next_close"]}

    def _check_limits(self, symbol: str, price: float) -> "tuple[Optional[str], int]":
        s = self.settings
        if s.execution != "paper":
            return "execution off (NEWSBOT_EXECUTION=off): signal journaled only", 0
        try:
            clock = self.market_clock()
        except Exception as exc:
            return f"market clock unavailable: {exc}", 0
        if not clock.get("is_open"):
            return "market closed", 0
        next_close = _parse_time(clock.get("next_close"))
        now = self.now()
        if next_close is None:
            return "market clock has no next_close", 0
        if next_close - now < timedelta(minutes=s.entry_cutoff_minutes):
            return f"within {s.entry_cutoff_minutes} minutes of the close (NEWSBOT_ENTRY_CUTOFF_MINUTES)", 0
        placed = self.journal.count_events_today(ORDER_EVENT, now=now)
        if placed >= s.max_trades_per_day:
            return f"daily trade limit reached ({placed}/{s.max_trades_per_day})", 0
        last = self.journal.last_event_for_symbol(ORDER_EVENT, symbol)
        if last is not None:
            since = now - _parse_time(last["ts"])
            if since < timedelta(minutes=s.cooldown_minutes):
                return (f"cooldown: last {symbol} order {since.total_seconds() / 60:.0f} min ago "
                        f"(NEWSBOT_COOLDOWN_MINUTES={s.cooldown_minutes})"), 0
        if price < s.min_price:
            return f"price ${price:.2f} below NEWSBOT_MIN_PRICE (${s.min_price:.2f})", 0
        qty = math.floor(s.order_usd / price)
        if qty < 1:
            return f"NEWSBOT_ORDER_USD ${s.order_usd:.2f} buys less than one share at ${price:.2f}", 0
        return None, qty

    def _log(self, kind: str, symbol: str, record: Dict[str, Any], received_at: datetime,
             side: Optional[str] = None, qty: Optional[int] = None) -> Dict[str, Any]:
        now = self.now()
        record["total_ms"] = round((now - received_at).total_seconds() * 1000)
        price = record.get("price")
        self.journal.log_event(
            kind, tool="newsbot", symbol=symbol, side=side, quantity=qty, price=price,
            notional=qty * price if qty and price else None,
            detail=record.get("blocked") or record["reason"], payload=record, now=now,
        )
        record["kind"] = kind
        record["symbol"] = symbol
        logger.info(f"{kind} {symbol} {record['signal']}: {record.get('blocked') or record['reason']} "
                    f"| {record.get('headline')!r} (jev {record['jev_ms']} ms, total {record['total_ms']} ms)")
        return record


# --- stream -----------------------------------------------------------------


class StreamError(RuntimeError):
    """The stream refused us (bad keys, connection limit, no subscription): do not retry."""


class SeenIds:
    """Bounded set of news ids already dispatched; Alpaca re-sends items when they are updated."""

    def __init__(self, max_size: int = SEEN_IDS_MAX) -> None:
        self._ids: "OrderedDict[Any, None]" = OrderedDict()
        self._max_size = max_size

    def add(self, news_id: Any) -> bool:
        """Record the id; False if it was already seen."""
        if news_id in self._ids:
            return False
        self._ids[news_id] = None
        while len(self._ids) > self._max_size:
            self._ids.popitem(last=False)
        return True


def parse_messages(raw: Any) -> List[Dict[str, Any]]:
    data = json.loads(raw)
    return [m for m in (data if isinstance(data, list) else [data]) if isinstance(m, dict)]


def check_control(message: Dict[str, Any]) -> Optional[str]:
    """Returns the control message ("connected", "authenticated", "subscription"), raises on error."""
    kind = message.get("T")
    if kind == "error":
        raise StreamError(f"Alpaca news stream error {message.get('code')}: {message.get('msg')}")
    if kind == "success":
        return message.get("msg")
    if kind == "subscription":
        return "subscription"
    return None


def news_tasks(messages: List[Dict[str, Any]], seen: SeenIds,
               settings: Settings) -> List["tuple[str, Dict[str, Any], Any]"]:
    """(symbol, article, news_id) for each new news item, one per symbol worth judging."""
    tasks = []
    for message in messages:
        if message.get("T") != "n" or not message.get("headline"):
            continue
        if not seen.add(message.get("id")):
            continue
        article = _normalize_alpaca(message)
        for symbol in symbols_to_judge(message.get("symbols"), settings):
            tasks.append((symbol, article, message.get("id")))
    return tasks


async def _authenticate(ws: Any, settings: Settings) -> None:
    await ws.send(json.dumps({
        "action": "auth", "key": config.secret("ALPACA_API_KEY"), "secret": config.secret("ALPACA_SECRET_KEY"),
    }))
    while True:
        for message in parse_messages(await ws.recv()):
            if check_control(message) == "authenticated":
                subscription = sorted(settings.symbols) or ["*"]
                await ws.send(json.dumps({"action": "subscribe", "news": subscription}))
                return


async def consume(ws: Any, bot: NewsBot, executor: Any, seen: SeenIds) -> None:
    """Dispatch news from an authenticated stream. Errors after auth are logged, not fatal:
    if the server gives up on us it closes the socket and the caller reconnects."""

    def run_handle(symbol: str, article: Dict[str, Any], received_at: datetime, news_id: Any) -> None:
        try:
            bot.handle(symbol, article, received_at, news_id=news_id)
        except Exception:
            logger.exception(f"newsbot handler failed for {symbol} news {news_id}")

    async for raw in ws:
        received_at = _now()
        try:
            messages = parse_messages(raw)
        except ValueError as exc:
            logger.warning(f"Ignoring unparseable stream frame: {exc}")
            continue
        for message in messages:
            try:
                if check_control(message) == "subscription":
                    logger.info(f"Subscribed to news: {message.get('news')}")
            except StreamError as exc:
                logger.error(str(exc))
        for symbol, article, news_id in news_tasks(messages, seen, bot.settings):
            executor.submit(run_handle, symbol, article, received_at, news_id)


async def flatten_loop(bot: NewsBot, interval: float = FLATTEN_CHECK_SECONDS) -> None:
    while True:
        try:
            await asyncio.to_thread(bot.maybe_flatten)
        except Exception:
            logger.exception("End-of-day flatten failed; retrying")
        await asyncio.sleep(interval)


async def stream(bot: NewsBot, executor: ThreadPoolExecutor) -> None:
    from websockets.asyncio.client import connect
    from websockets.exceptions import ConnectionClosed

    seen = SeenIds()
    flattener = asyncio.create_task(flatten_loop(bot))
    try:
        async for ws in connect(STREAM_URL):  # reconnects with backoff on network errors
            try:
                await _authenticate(ws, bot.settings)
                logger.info(f"Connected to Alpaca news stream (execution={bot.settings.execution}, "
                            f"symbols={sorted(bot.settings.symbols) or 'all'})")
                await consume(ws, bot, executor, seen)
            except ConnectionClosed as exc:
                logger.warning(f"News stream closed ({exc}); reconnecting")
                continue
    finally:
        flattener.cancel()


# --- CLI --------------------------------------------------------------------


def run(settings: Settings) -> int:
    if not alpaca_configured():
        print("ALPACA_API_KEY and ALPACA_SECRET_KEY are required for the news stream.", file=sys.stderr)
        return 2
    if not jev_news.is_configured():
        print("TYPESAFE_API_KEY is required: Jev judges every headline.", file=sys.stderr)
        return 2
    if settings.execution == "paper" and not alpaca.is_paper():
        print(f"NEWSBOT_EXECUTION=paper but ALPACA_TRADING_BASE_URL is {alpaca.trading_base_url()!r}; "
              f"the news bot trades {alpaca.DEFAULT_TRADING_URL} only.", file=sys.stderr)
        return 2
    bot = NewsBot(settings)
    with ThreadPoolExecutor(max_workers=HANDLER_WORKERS) as executor:
        try:
            asyncio.run(stream(bot, executor))
        except StreamError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            return 0
    return 0


def replay(symbol: str, hours: float, settings: Settings) -> int:
    """Apply the rules to recent headlines as if each arrived when published. Never trades or journals."""
    symbol = symbol.upper()
    articles = get_alpaca_news(symbol, since=_now() - timedelta(hours=hours))
    for article in reversed(articles):
        published = _parse_time(article.get("publishedAt")) or _now()
        if symbol not in symbols_to_judge(article.get("symbols"), settings):
            signal = Signal("none", f"skipped: tagged {article.get('symbols')} "
                                    f"(roundup or not in NEWSBOT_SYMBOLS)")
            judgment = None
        else:
            judgment = jev_news.judge_headline(symbol, article)
            signal = decide(symbol, article, judgment, settings, now=published)
        print(json.dumps({
            "published_at": article.get("publishedAt"),
            "headline": article.get("title"),
            "tagged": article.get("symbols"),
            **asdict(signal),
            **({k: judgment.get(k) for k in ("relevance", "materiality", "p_bullish", "p_bearish", "event_type")}
               if judgment else {}),
        }))
    return 0


def flatten() -> int:
    closed = alpaca.close_all_positions()
    Journal().log_event(FLATTEN_EVENT, tool="newsbot", detail=f"closed {len(closed)} positions",
                        payload={"response": closed})
    print(json.dumps({"closed": len(closed), "response": closed}, indent=2, default=str))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    config.load_env()
    parser = argparse.ArgumentParser(prog="python -m trader.newsbot")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="Stream Alpaca news and trade it (paper) per NEWSBOT_* settings")
    rp = sub.add_parser("replay", help="Apply the rules to recent headlines; never trades or journals")
    rp.add_argument("symbol")
    rp.add_argument("--hours", type=float, default=24)
    sub.add_parser("flatten", help="Cancel open orders and close all Alpaca paper positions")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr)
    try:
        if args.command == "flatten":
            return flatten()
        settings = load_settings()
        if args.command == "replay":
            return replay(args.symbol, args.hours, settings)
        return run(settings)
    except (alpaca.NotPaperError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
