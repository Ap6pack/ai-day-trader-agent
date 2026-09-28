"""
Dividend signal from Alpaca's corporate-actions data (cash dividends).

Trailing 12-month yield from the dividends paid, and whether an ex-dividend
date falls within DIVIDEND_CAPTURE_WINDOW_DAYS. A qualifying upcoming ex-date
(yield >= MIN_DIVIDEND_YIELD percent) is a BUY for dividend capture; anything
else is HOLD with the yield and next ex-date as context. Results are cached for
a few hours: dividend data changes slowly. DIVIDEND_STRATEGY_ENABLED=false
turns it off.

Alpaca notes there can be delays before announced dividends appear, so an
upcoming ex-date may be missing; the signal then says no ex-date is known.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from trader.news_sources import alpaca_configured, alpaca_data_url, alpaca_headers

logger = logging.getLogger(__name__)

CORPORATE_ACTIONS_PATH = "/v1/corporate-actions"
CACHE_SECONDS = 6 * 3600
_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_cache_lock = threading.Lock()


def enabled() -> bool:
    return os.getenv("DIVIDEND_STRATEGY_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


def cash_dividends(symbol: str, today: Optional[date] = None) -> List[Dict[str, Any]]:
    """Cash dividends with ex-dates from a year back to 60 days ahead (cached)."""
    symbol = symbol.upper()
    with _cache_lock:
        hit = _cache.get(symbol)
        if hit and time.monotonic() - hit[0] < CACHE_SECONDS:
            return hit[1]
    today = today or datetime.now(timezone.utc).date()
    params = {
        "symbols": symbol,
        "types": "cash_dividend",
        "start": (today - timedelta(days=380)).isoformat(),
        "end": (today + timedelta(days=60)).isoformat(),
        "limit": 1000,
    }
    resp = requests.get(alpaca_data_url(CORPORATE_ACTIONS_PATH), headers=alpaca_headers(), params=params, timeout=10)
    resp.raise_for_status()
    rows = (resp.json().get("corporate_actions") or {}).get("cash_dividends") or []
    rows = [r for r in rows if r.get("ex_date") and r.get("rate") is not None]
    with _cache_lock:
        _cache[symbol] = (time.monotonic(), rows)
    return rows


def dividend_signal(symbol: str, price: Optional[float], today: Optional[date] = None,
                    rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if not enabled():
        return {"signal": "HOLD", "strength": 0.0, "reason": "Dividend strategy off (DIVIDEND_STRATEGY_ENABLED=false)"}
    if rows is None:
        if not alpaca_configured():
            return {"signal": "HOLD", "strength": 0.0, "reason": "Dividend: Alpaca keys needed for dividend data"}
        try:
            rows = cash_dividends(symbol, today)
        except Exception as exc:
            logger.warning(f"Dividend data failed for {symbol}: {exc}")
            return {"signal": "HOLD", "strength": 0.0, "reason": f"Dividend data unavailable: {exc}"}

    today = today or datetime.now(timezone.utc).date()
    window = int(_float_env("DIVIDEND_CAPTURE_WINDOW_DAYS", 7))
    min_yield = _float_env("MIN_DIVIDEND_YIELD", 6.0)

    def ex_date(row: Dict[str, Any]) -> date:
        return date.fromisoformat(str(row["ex_date"])[:10])

    past = [r for r in rows if today - timedelta(days=365) < ex_date(r) <= today and not r.get("special")]
    upcoming = sorted((r for r in rows if ex_date(r) > today), key=ex_date)
    annual = sum(float(r["rate"]) for r in past)
    yield_pct = annual / price * 100 if price and annual else 0.0
    info: Dict[str, Any] = {
        "trailing_annual_dividend": round(annual, 4),
        "yield_pct": round(yield_pct, 2),
        "next_ex_date": ex_date(upcoming[0]).isoformat() if upcoming else None,
        "next_amount": float(upcoming[0]["rate"]) if upcoming else None,
    }
    if not rows:
        return {"signal": "HOLD", "strength": 0.0, **info, "reason": "Dividend: pays no cash dividend"}
    if not upcoming:
        return {"signal": "HOLD", "strength": 0.0, **info,
                "reason": f"Dividend: {yield_pct:.1f}% trailing yield, no upcoming ex-date known"}
    days = (ex_date(upcoming[0]) - today).days
    info["days_to_ex_dividend"] = days
    if days <= window and yield_pct >= min_yield:
        strength = round(0.3 + 0.4 * (1 - (days - 1) / max(window, 1)), 3)
        return {"signal": "BUY", "strength": max(0.3, min(strength, 0.7)), **info,
                "reason": f"Dividend: ex-date in {days}d, ${info['next_amount']:.4g}/sh, {yield_pct:.1f}% yield"}
    why = f"ex-date in {days}d" + (f" (window {window}d)" if days > window else "")
    if yield_pct < min_yield:
        why += f", yield {yield_pct:.1f}% < {min_yield:.1f}%"
    return {"signal": "HOLD", "strength": 0.0, **info, "reason": f"Dividend: {why}"}


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
