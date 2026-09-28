"""
Desk autopilot: runs the multi-strategy analysis on a symbol list every interval
and acts on it, started and stopped from the live desk.

Modes:
  signals  analyze and report only
  record   trade a local paper portfolio (trader.portfolios) at the live price
  paper    send Alpaca paper orders: bracket buys at the analysis's stop/target,
           market sells of a held position
  live     the same orders with REAL MONEY on the Alpaca live account (needs
           ALPACA_LIVE_TRADING=true and the live host; the desk asks you to type LIVE)

Hard limits, checked before every trade (AUTOPILOT_* in .env): minimum confidence,
daily trade cap, per-symbol cooldown, minimum price, market open and no new buys
within the entry cutoff before the close. Long only: SELL reduces a held position.
The mode must match the configured Alpaca account (trader.alpaca.require_mode).
In paper and live modes the autopilot closes the positions it opened
AUTOPILOT_FLATTEN_MINUTES before the close (bracket legs are day orders). On the
live account it only ever sells shares it bought itself, never your holdings,
and every buy also passes the ALPACA_LIVE_* limits in trader.alpaca.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from trader import alpaca, analysis, market_feed
from trader.journal import Journal
from trader.portfolios import Portfolios

logger = logging.getLogger("trader.autotrader")

MODES = ("signals", "record", "paper", "live")
BROKER_MODES = ("paper", "live")
ORDER_PREFIX = "autopilot-"
FLATTEN_CHECK_SECONDS = 30
MIN_INTERVAL, MAX_INTERVAL = 60, 3600
ORDER_EVENT = "autopilot_order"
THESIS_PREFIX = "[autopilot]"


def _env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Limits:
    min_confidence: float = 0.5
    max_trades_per_day: int = 5
    cooldown_minutes: int = 30
    min_price: float = 5.0
    entry_cutoff_minutes: int = 15
    flatten_minutes: int = 10

    @classmethod
    def from_env(cls) -> "Limits":
        return cls(
            min_confidence=_env("AUTOPILOT_MIN_CONFIDENCE", 0.5),
            max_trades_per_day=int(_env("AUTOPILOT_MAX_TRADES_PER_DAY", 5)),
            cooldown_minutes=int(_env("AUTOPILOT_COOLDOWN_MINUTES", 30)),
            min_price=_env("AUTOPILOT_MIN_PRICE", 5.0),
            entry_cutoff_minutes=int(_env("AUTOPILOT_ENTRY_CUTOFF_MINUTES", 15)),
            flatten_minutes=int(_env("AUTOPILOT_FLATTEN_MINUTES", 10)),
        )


@dataclass
class RunConfig:
    symbols: List[str]
    interval_seconds: int = 300
    mode: str = "signals"
    portfolio: str = "default"

    def validate(self) -> "RunConfig":
        self.symbols = [s.strip().upper() for s in self.symbols if s and s.strip()][:40]
        if not self.symbols:
            raise ValueError("Give at least one symbol")
        if not all(s.replace(".", "").isalpha() and len(s) <= 10 for s in self.symbols):
            raise ValueError("Symbols must be letters (and '.')")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if not MIN_INTERVAL <= int(self.interval_seconds) <= MAX_INTERVAL:
            raise ValueError(f"interval must be {MIN_INTERVAL}-{MAX_INTERVAL} seconds")
        self.interval_seconds = int(self.interval_seconds)
        return self


def _parse(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    try:
        value = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class AutoTrader:
    def __init__(
        self,
        journal: Journal,
        portfolios: Portfolios,
        publish: Callable[[Dict[str, Any]], None] = lambda event: None,
        analyze: Callable[..., Dict[str, Any]] = analysis.analyze,
        broker: Any = alpaca,
        quotes: Callable[[List[str]], Dict[str, Dict[str, Any]]] = market_feed.get_quotes,
        limits: Optional[Limits] = None,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.journal = journal
        self.portfolios = portfolios
        self.publish = publish
        self.analyze = analyze
        self.broker = broker
        self.quotes = quotes
        self.limits = limits or Limits.from_env()
        self.now = now
        self.config: Optional[RunConfig] = None
        self.cycles = 0
        self.current_symbol: Optional[str] = None
        self.next_cycle_at: Optional[str] = None
        self.last_error: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._trade_lock = threading.Lock()
        self._flattened_for: Optional[str] = None

    # --- status / control ---------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "config": asdict(self.config) if self.config else None,
            "cycles": self.cycles,
            "current_symbol": self.current_symbol,
            "next_cycle_at": self.next_cycle_at,
            "last_error": self.last_error,
            "trades_today": self.journal.count_events_today(ORDER_EVENT, now=self.now()),
            "limits": asdict(self.limits),
        }

    async def start(self, config: RunConfig) -> Dict[str, Any]:
        config.validate()
        if config.mode == "record":
            self.portfolios.ensure_default()
            self.portfolios.get(config.portfolio)  # raises if missing
        if config.mode in BROKER_MODES:
            try:
                self.broker.require_mode(config.mode, who=f"Autopilot {config.mode} mode")
            except Exception as exc:
                raise ValueError(str(exc)) from exc
        await self.stop(announce=False)
        self.limits = Limits.from_env()
        self.config, self.cycles, self.last_error = config, 0, None
        self._task = asyncio.create_task(self._loop())
        self._event("autopilot", None, "system",
                    f"Autopilot ON · {config.mode.upper()} · {','.join(config.symbols)} every {config.interval_seconds}s"
                    + (f" · portfolio {config.portfolio}" if config.mode == "record" else ""))
        return self.status()

    async def stop(self, announce: bool = True) -> Dict[str, Any]:
        task, self._task = self._task, None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self.current_symbol = self.next_cycle_at = None
        if announce:
            self._event("autopilot", None, "system", "Autopilot OFF")
        return self.status()

    async def _loop(self) -> None:
        while True:
            config = self.config
            try:
                await asyncio.to_thread(self.run_cycle, config)
                self.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Autopilot cycle failed")
                self.last_error = str(exc)
                self._event("autopilot", None, "error", f"Autopilot cycle error: {exc}")
            self.next_cycle_at = (self.now() + timedelta(seconds=config.interval_seconds)).isoformat()
            self._event("autopilot", None, "system",
                        f"Autopilot cycle {self.cycles} complete; next in {config.interval_seconds}s")
            waited = 0
            while waited < config.interval_seconds:  # short naps so the pre-close flatten is not missed
                step = min(FLATTEN_CHECK_SECONDS, config.interval_seconds - waited)
                await asyncio.sleep(step)
                waited += step
                await asyncio.to_thread(self.maybe_flatten, config)
            self.next_cycle_at = None

    def maybe_flatten(self, config: RunConfig) -> Optional[int]:
        """Close the autopilot's own positions once per session, flatten_minutes before the close."""
        if config.mode not in BROKER_MODES:
            return None
        try:
            clock = self.broker.market_clock()
            next_close = _parse(clock.get("next_close"))
            if (not clock.get("is_open") or next_close is None or clock["next_close"] == self._flattened_for
                    or next_close - self.now() > timedelta(minutes=self.limits.flatten_minutes)):
                return None
            with self._trade_lock:
                closed = self.broker.flatten_owned(ORDER_PREFIX)
                self._flattened_for = clock["next_close"]
        except Exception as exc:
            self._event("order_failed", None, "error", f"Autopilot pre-close flatten failed: {exc}")
            return None
        detail = f"closed {len(closed)} autopilot positions before the {clock['next_close']} close"
        self.journal.log_event("flatten", tool="autopilot", detail=detail, payload={"mode": config.mode},
                               now=self.now())
        self._event("flatten", None, "system", f"AUTO {detail}")
        return len(closed)

    # --- one cycle --------------------------------------------------------------------

    def _event(self, etype: str, symbol: Optional[str], level: str, message: str,
               data: Optional[Dict[str, Any]] = None) -> None:
        self.publish({
            "id": f"a{self.now().timestamp()}-{etype}-{symbol}",
            "type": etype, "symbol": symbol, "level": level, "message": message,
            "timestamp": self.now().isoformat(timespec="seconds"), "data": data or {},
        })

    def _capital_and_held(self, config: RunConfig, symbol: str) -> "tuple[float, float, str]":
        if config.mode == "live":
            # Only shares the autopilot bought itself count as held: it never sells your holdings.
            account = self.broker.account()
            return float(account.get("equity") or 0), self.broker.owned_qty(ORDER_PREFIX, symbol), \
                "alpaca LIVE equity"
        if config.mode == "paper":
            account = self.broker.account()
            held = next((float(p.get("qty") or 0) for p in self.broker.positions()
                         if p.get("symbol") == symbol and p.get("side", "long") == "long"), 0.0)
            return float(account.get("equity") or 0), held, "alpaca paper equity"
        self.portfolios.ensure_default()
        name = config.portfolio if config.portfolio in self.portfolios.names() else "default"
        p = self.portfolios.get(name)
        quotes = self.quotes([h["symbol"] for h in p["holdings"]]) if p["holdings"] else {}
        return self.portfolios.valuation(name, quotes)["equity"], self.portfolios.held(name, symbol), \
            f"portfolio {name}"

    def run_cycle(self, config: RunConfig) -> List[Dict[str, Any]]:
        self.cycles += 1
        results = []
        for symbol in config.symbols:
            self.current_symbol = symbol
            self._event("analysis_started", symbol, "info", f"Analysis started for {symbol}")
            try:
                capital, held, source = self._capital_and_held(config, symbol)
                result = self.analyze(symbol, capital, held, capital_source=source)
            except Exception as exc:
                self._event("analysis_error", symbol, "error", f"{symbol}: analysis failed - {exc}")
                continue
            for name, sig in (result.get("all_signals") or {}).items():
                self._event("strategy_signal", symbol, "info",
                            f"{name.capitalize()}: {sig.get('signal')} ({(sig.get('strength') or 0):.0%}) - "
                            f"{sig.get('reason', '')}")
            rec = result["recommendation"]
            self._event("decision", symbol, "info",
                        f"{symbol}: {rec} {result['quantity']} ({result['confidence']:.0%} conf) via "
                        f"{result['primary_strategy']}", data=result)
            if rec in ("BUY", "SELL") and result.get("current_price"):
                self.journal.add_decision(
                    symbol, rec.lower(), result["current_price"], confidence=result["confidence"],
                    thesis=f"{THESIS_PREFIX} {result['primary_strategy']}: {result['reason']}"[:500],
                    mode=f"autopilot-{config.mode}", now=self.now())
            results.append(self.act(result, config))
        self.current_symbol = None
        return results

    # --- acting on a decision -------------------------------------------------------------

    def _blocked(self, result: Dict[str, Any], config: RunConfig) -> Optional[str]:
        rec, symbol, price = result["recommendation"], result["symbol"], result.get("current_price") or 0
        lim = self.limits
        if config.mode == "signals":
            return "signals only"
        if rec == "HOLD":
            return "HOLD"
        if result["quantity"] < 1:
            return "quantity 0" + (" (nothing held to sell)" if rec == "SELL" else "")
        if result["confidence"] < lim.min_confidence:
            return f"confidence {result['confidence']:.0%} < {lim.min_confidence:.0%}"
        if price < lim.min_price:
            return f"price ${price:.2f} < ${lim.min_price:.2f}"
        clock = self.broker.market_clock()
        if not clock.get("is_open"):
            return "market closed"
        next_close = _parse(clock.get("next_close"))
        if rec == "BUY" and next_close and next_close - self.now() < timedelta(minutes=lim.entry_cutoff_minutes):
            return f"within {lim.entry_cutoff_minutes} min of the close"
        placed = self.journal.count_events_today(ORDER_EVENT, now=self.now())
        if placed >= lim.max_trades_per_day:
            return f"daily trade limit reached ({placed}/{lim.max_trades_per_day})"
        last = self.journal.last_event_for_symbol(ORDER_EVENT, symbol)
        if last is not None and (self.now() - _parse(last["ts"])) < timedelta(minutes=lim.cooldown_minutes):
            return f"cooldown: traded {symbol} within {lim.cooldown_minutes} min"
        return None

    def act(self, result: Dict[str, Any], config: RunConfig) -> Dict[str, Any]:
        symbol, rec = result["symbol"], result["recommendation"]
        with self._trade_lock:
            try:
                blocked = self._blocked(result, config)
            except Exception as exc:
                blocked = f"limit check failed: {exc}"
            if blocked:
                if blocked not in ("HOLD", "signals only"):
                    self._event("order_skipped", symbol, "warning", f"{symbol}: {rec} not placed - {blocked}")
                return {"symbol": symbol, "action": "none", "reason": blocked}
            qty, price = result["quantity"], result["current_price"]
            if config.mode == "live" and rec == "BUY":
                qty = min(qty, self.broker.max_live_qty(price))
                if qty < 1:
                    reason = f"one share at ${price:.2f} exceeds ALPACA_LIVE_MAX_ORDER_USD"
                    self._event("order_skipped", symbol, "warning", f"{symbol}: {rec} not placed - {reason}")
                    return {"symbol": symbol, "action": "none", "reason": reason}
                result = {**result, "quantity": qty}
            try:
                if config.mode == "record":
                    fill = self.portfolios.record_fill(config.portfolio, symbol, rec.lower(), qty, price,
                                                       source="autopilot", note=result["reason"][:200])
                    detail, payload = f"recorded {rec} {qty} {symbol} @ {price} in {config.portfolio}", {"fill": fill}
                else:
                    payload = self._paper_order(result)
                    order = self.broker.submit_order(payload)
                    label = "LIVE" if config.mode == "live" else "paper"
                    detail = f"{label} {rec} {qty} {symbol} order {str(order.get('id', ''))[:8]}"
                    payload = {"request": payload, "order_id": order.get("id"), "status": order.get("status")}
            except Exception as exc:
                self._event("order_failed", symbol, "error", f"{symbol}: {rec} {qty} failed - {exc}")
                return {"symbol": symbol, "action": "failed", "reason": str(exc)}
            self.journal.log_event(ORDER_EVENT, tool="autopilot", symbol=symbol, side=rec.lower(), quantity=qty,
                                   price=price, notional=round(qty * price, 2), detail=detail,
                                   payload={"mode": config.mode, **payload,
                                            "confidence": result["confidence"],
                                            "risk": result.get("risk_parameters")},
                                   now=self.now())
            self._event("order_submitted", symbol, "success", f"AUTO {detail}")
            return {"symbol": symbol, "action": rec.lower(), "qty": qty, "detail": detail}

    def _paper_order(self, result: Dict[str, Any]) -> Dict[str, Any]:
        symbol, qty, price = result["symbol"], result["quantity"], result["current_price"]
        if result["recommendation"] == "SELL":
            payload = self.broker.market_order_payload(symbol, "sell", qty)
        else:
            risk = result.get("risk_parameters") or {}
            stop, target = risk.get("stop_loss"), risk.get("take_profit")
            sl_pct = (1 - stop / price) * 100 if stop and stop < price else 1.0
            tp_pct = (target / price - 1) * 100 if target and target > price else 2.0
            payload = self.broker.bracket_order_payload(symbol, qty, price, tp_pct, sl_pct)
        # Tagged so the desk's blotter can tell autopilot orders from the news bot's and manual ones.
        payload["client_order_id"] = f"{ORDER_PREFIX}{uuid.uuid4()}"
        return payload
