#!/usr/bin/env python3
"""
Market data feed for the live trading desk.

Quotes come from Alpaca's multi-symbol snapshot endpoint (one request for the
whole watchlist, with bid/ask, day OHLC and previous close), falling back to
Yahoo Finance when the optional yfinance package is installed. Chart bars come
from Alpaca bars, with the same fallback.

``DESK_DEMO_MODE=true`` swaps both for a seeded random-walk simulator so the
desk can be explored without API keys. Every demo payload carries
``source="demo"`` and the UI labels it as simulated data.
"""

from __future__ import annotations

import logging
import math
import os
import random
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

import requests

logger = logging.getLogger(__name__)

TIMEFRAMES = {
    # desk timeframe: (alpaca timeframe, seconds per bar, lookback days, yahoo period, yahoo interval)
    "1Min": ("1Min", 60, 3, "5d", "1m"),
    "5Min": ("5Min", 300, 10, "1mo", "5m"),
    "15Min": ("15Min", 900, 30, "1mo", "15m"),
    "1Hour": ("1Hour", 3600, 120, "6mo", "1h"),
    "1Day": ("1Day", 86400, 730, "2y", "1d"),
}
DEFAULT_TIMEFRAME = "5Min"


def demo_mode_enabled() -> bool:
    return os.getenv("DESK_DEMO_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}


def _alpaca_credentials() -> Optional[Dict[str, str]]:
    key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_KEY_ID")
    secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET")
    if not key or not secret:
        return None
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _alpaca_data_url() -> str:
    return os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets").rstrip("/")


def _alpaca_feed() -> str:
    return os.getenv("ALPACA_DATA_FEED", "iex")


def _to_epoch(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, datetime):
        return int(value.timestamp())
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _quote(symbol: str, source: str, *, last: Optional[float], prev_close: Optional[float] = None,
           bid: Optional[float] = None, ask: Optional[float] = None,
           bid_size: Optional[float] = None, ask_size: Optional[float] = None,
           open_: Optional[float] = None, high: Optional[float] = None,
           low: Optional[float] = None, volume: Optional[float] = None,
           timestamp: Any = None) -> Dict[str, Any]:
    change = change_pct = None
    if last is not None and prev_close:
        change = last - prev_close
        change_pct = change / prev_close * 100
    return {
        "symbol": symbol,
        "last": last,
        "bid": bid,
        "ask": ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "open": open_,
        "high": high,
        "low": low,
        "prev_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "volume": volume,
        "timestamp": _to_epoch(timestamp) or int(time.time()),
        "source": source,
    }


# ----------------------------------------------------------------------
# Alpaca
# ----------------------------------------------------------------------

def _alpaca_snapshots(symbols: List[str], headers: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    resp = requests.get(
        f"{_alpaca_data_url()}/v2/stocks/snapshots",
        headers=headers,
        params={"symbols": ",".join(symbols), "feed": _alpaca_feed()},
        timeout=10,
    )
    resp.raise_for_status()
    payload = resp.json() or {}
    # The endpoint returns {SYMBOL: snapshot}; older versions nest under "snapshots".
    snapshots = payload.get("snapshots", payload) if isinstance(payload, dict) else {}

    quotes = {}
    for symbol in symbols:
        snap = snapshots.get(symbol)
        if not snap:
            continue
        trade = snap.get("latestTrade") or {}
        nbbo = snap.get("latestQuote") or {}
        daily = snap.get("dailyBar") or {}
        prev = snap.get("prevDailyBar") or {}
        minute = snap.get("minuteBar") or {}
        last = trade.get("p") or minute.get("c") or daily.get("c")
        if last is None:
            continue
        quotes[symbol] = _quote(
            symbol, "alpaca",
            last=float(last),
            prev_close=float(prev["c"]) if prev.get("c") else None,
            bid=nbbo.get("bp") or None,
            ask=nbbo.get("ap") or None,
            bid_size=nbbo.get("bs"),
            ask_size=nbbo.get("as"),
            open_=daily.get("o"),
            high=daily.get("h"),
            low=daily.get("l"),
            volume=daily.get("v"),
            timestamp=trade.get("t") or minute.get("t"),
        )
    return quotes


def _alpaca_bars(symbol: str, timeframe: str, limit: int, headers: Dict[str, str]) -> List[Dict[str, Any]]:
    alpaca_tf, _, lookback_days, _, _ = TIMEFRAMES[timeframe]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    resp = requests.get(
        f"{_alpaca_data_url()}/v2/stocks/{symbol}/bars",
        headers=headers,
        params={
            "timeframe": alpaca_tf,
            "start": start.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "end": end.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "limit": min(limit, 10000),
            "adjustment": "raw",
            "feed": _alpaca_feed(),
            "sort": "desc",
        },
        timeout=10,
    )
    resp.raise_for_status()
    bars = (resp.json() or {}).get("bars") or []
    out = [
        {
            "time": _to_epoch(bar.get("t")),
            "open": float(bar["o"]),
            "high": float(bar["h"]),
            "low": float(bar["l"]),
            "close": float(bar["c"]),
            "volume": float(bar.get("v") or 0),
        }
        for bar in bars
        if bar.get("t") and bar.get("c") is not None
    ]
    out.reverse()
    return out


# ----------------------------------------------------------------------
# Yahoo Finance (optional fallback: only when the yfinance package is installed)
# ----------------------------------------------------------------------

def yahoo_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("yfinance") is not None


def _yahoo_quote(symbol: str) -> Optional[Dict[str, Any]]:
    import yfinance as yf

    info = yf.Ticker(symbol).fast_info
    last = getattr(info, "last_price", None)
    if not last:
        return None
    return _quote(
        symbol, "yahoo_finance",
        last=float(last),
        prev_close=float(getattr(info, "previous_close", 0) or 0) or None,
        open_=getattr(info, "open", None),
        high=getattr(info, "day_high", None),
        low=getattr(info, "day_low", None),
        volume=getattr(info, "last_volume", None),
    )


def _yahoo_bars(symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
    import yfinance as yf

    _, _, _, period, interval = TIMEFRAMES[timeframe]
    hist = yf.Ticker(symbol).history(period=period, interval=interval)
    bars = [
        {
            "time": int(idx.timestamp()),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"] or 0),
        }
        for idx, row in hist.iterrows()
        if row["Close"] == row["Close"]  # drop NaN rows
    ]
    return bars[-limit:]


# ----------------------------------------------------------------------
# Demo simulator
# ----------------------------------------------------------------------

class DemoMarket:
    """Seeded random-walk market so the desk runs without credentials."""

    BASE_PRICES = {
        "AAPL": 228.0, "MSFT": 431.0, "NVDA": 121.0, "TSLA": 251.0, "AMZN": 187.0,
        "GOOGL": 164.0, "META": 566.0, "SPY": 571.0, "QQQ": 487.0, "AMD": 158.0,
        "JPM": 211.0, "KO": 71.0, "T": 22.0, "O": 63.0, "VZ": 44.0,
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Dict[str, Dict[str, float]] = {}

    def _init_symbol(self, symbol: str) -> Dict[str, float]:
        rng = random.Random(symbol)
        base = self.BASE_PRICES.get(symbol, rng.uniform(20, 400))
        prev_close = base * (1 + rng.uniform(-0.01, 0.01))
        state = {
            "prev_close": prev_close,
            "last": prev_close * (1 + rng.uniform(-0.015, 0.015)),
            "open": prev_close * (1 + rng.uniform(-0.004, 0.004)),
            "volume": rng.uniform(2e6, 3e7),
            "vol": rng.uniform(0.0006, 0.0015),
        }
        state["high"] = max(state["open"], state["last"])
        state["low"] = min(state["open"], state["last"])
        return state

    def quotes(self, symbols: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        out = {}
        with self._lock:
            for symbol in symbols:
                state = self._state.get(symbol) or self._init_symbol(symbol)
                self._state[symbol] = state
                drift = (state["prev_close"] - state["last"]) * 0.002
                state["last"] = max(0.5, state["last"] + drift + random.gauss(0, state["vol"]) * state["last"])
                state["high"] = max(state["high"], state["last"])
                state["low"] = min(state["low"], state["last"])
                state["volume"] += random.uniform(500, 20000)
                spread = max(0.01, state["last"] * 0.0002)
                out[symbol] = _quote(
                    symbol, "demo",
                    last=round(state["last"], 2),
                    prev_close=round(state["prev_close"], 2),
                    bid=round(state["last"] - spread / 2, 2),
                    ask=round(state["last"] + spread / 2, 2),
                    bid_size=random.randint(1, 20) * 100,
                    ask_size=random.randint(1, 20) * 100,
                    open_=round(state["open"], 2),
                    high=round(state["high"], 2),
                    low=round(state["low"], 2),
                    volume=int(state["volume"]),
                )
        return out

    def bars(self, symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
        seconds = TIMEFRAMES[timeframe][1]
        last = self.quotes([symbol])[symbol]["last"]
        rng = random.Random(f"{symbol}:{timeframe}")
        now = int(time.time()) // seconds * seconds
        vol = 0.0012 * math.sqrt(seconds / 60)
        # Walk backwards from the live price so the chart meets the quote.
        closes = [last]
        for _ in range(limit - 1):
            closes.append(closes[-1] / (1 + rng.gauss(0.00005, vol)))
        closes.reverse()
        bars = []
        prev = closes[0]
        for i, close in enumerate(closes):
            open_ = prev
            wick = abs(rng.gauss(0, vol / 2)) * close
            bars.append({
                "time": now - (limit - 1 - i) * seconds,
                "open": round(open_, 2),
                "high": round(max(open_, close) + wick, 2),
                "low": round(min(open_, close) - wick, 2),
                "close": round(close, 2),
                "volume": int(rng.uniform(0.3, 1.7) * 50000 * math.sqrt(seconds / 60)),
            })
            prev = close
        return bars


demo_market = DemoMarket()


def demo_candles(symbol: str, intervals: Iterable[str], outputsize: int) -> Dict[str, List[Dict[str, Any]]]:
    """Simulated candles in the pipeline's fetcher format (newest first)."""
    interval_map = {"1min": "1Min", "15min": "15Min", "1h": "1Hour"}
    data = {}
    for interval in intervals:
        timeframe = interval_map.get(interval)
        if not timeframe:
            data[interval] = []
            continue
        bars = demo_market.bars(symbol.upper(), timeframe, max(10, min(outputsize, 1000)))
        data[interval] = [
            {
                "datetime": datetime.fromtimestamp(bar["time"], timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": bar["volume"],
            }
            for bar in reversed(bars)
        ]
    return data


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def get_quotes(symbols: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Return the latest quote for each symbol that any provider could price."""
    symbols = [s.upper() for s in dict.fromkeys(symbols) if s]
    if not symbols:
        return {}
    if demo_mode_enabled():
        return demo_market.quotes(symbols)

    quotes: Dict[str, Dict[str, Any]] = {}
    headers = _alpaca_credentials()
    if headers:
        try:
            quotes.update(_alpaca_snapshots(symbols, headers))
        except Exception as exc:
            logger.warning(f"Alpaca snapshot request failed: {exc}")

    if not yahoo_available():
        return quotes
    for symbol in symbols:
        if symbol in quotes:
            continue
        try:
            quote = _yahoo_quote(symbol)
            if quote:
                quotes[symbol] = quote
        except Exception as exc:
            logger.warning(f"Yahoo quote failed for {symbol}: {exc}")
    return quotes


def get_bars(symbol: str, timeframe: str = DEFAULT_TIMEFRAME, limit: int = 300) -> Dict[str, Any]:
    """Return chart bars (oldest first) and the provider that served them."""
    symbol = symbol.upper()
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Use one of: {', '.join(TIMEFRAMES)}")
    limit = max(10, min(int(limit), 1000))

    if demo_mode_enabled():
        return {"symbol": symbol, "timeframe": timeframe, "source": "demo",
                "bars": demo_market.bars(symbol, timeframe, limit)}

    errors = []
    headers = _alpaca_credentials()
    if headers:
        try:
            bars = _alpaca_bars(symbol, timeframe, limit, headers)
            if bars:
                return {"symbol": symbol, "timeframe": timeframe, "source": "alpaca", "bars": bars}
            errors.append("alpaca: no bars")
        except Exception as exc:
            errors.append(f"alpaca: {exc}")
    if not yahoo_available():
        errors.append("yahoo_finance: not installed (pip install yfinance)")
        return {"symbol": symbol, "timeframe": timeframe, "source": None, "bars": [], "errors": errors}
    try:
        bars = _yahoo_bars(symbol, timeframe, limit)
        if bars:
            return {"symbol": symbol, "timeframe": timeframe, "source": "yahoo_finance", "bars": bars}
        errors.append("yahoo_finance: no bars")
    except Exception as exc:
        errors.append(f"yahoo_finance: {exc}")

    return {"symbol": symbol, "timeframe": timeframe, "source": None, "bars": [], "errors": errors}
