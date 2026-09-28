"""
Multi-strategy analysis: technical + Jev news sentiment + dividend, fused into
one BUY / SELL / HOLD with quantity, stop, target, risk/reward, position value
and risk. Used by the desk's ANALYZE button and its autopilot.

    python -m trader.analysis NVDA [--capital 10000]

Fusion follows the original desk: the highest-priority signal leads (an upcoming
dividend capture, then a non-HOLD technical signal, then news sentiment), agreeing
strategies raise confidence and conflicting ones lower it. Stops and targets are
2 and 3 ATRs from the price (STOP_LOSS_PCT if ATR is unavailable). Size scales
with capital, confidence and volatility, within MIN/MAX_POSITION_PERCENTAGE.
Long only: SELL means reduce or exit a position you hold.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from trader import config, dividends, jev_news, market_feed, technicals
from trader.news_sources import get_news_articles

logger = logging.getLogger(__name__)

PRIORITY = {"dividend": 3, "technical": 2, "sentiment": 1}
SENTIMENT_ARTICLES = 10


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except ValueError:
        return default


def timeframe() -> str:
    tf = os.getenv("ANALYSIS_TIMEFRAME") or "15Min"
    return tf if tf in market_feed.TIMEFRAMES else "15Min"


def sentiment_signal(symbol: str, articles: Optional[List[Dict[str, Any]]] = None,
                     judge: Optional[Callable[..., Dict[str, Any]]] = None) -> Dict[str, Any]:
    if judge is None:
        if not jev_news.is_configured():
            return {"signal": "HOLD", "strength": 0.0, "reason": "Sentiment: Jev not configured (TYPESAFE_API_KEY)"}
        judge = jev_news.analyze_news
    try:
        articles = get_news_articles(symbol) if articles is None else articles
        if not articles:
            return {"signal": "HOLD", "strength": 0.0, "score": 0.0, "reason": "Sentiment: no recent headlines"}
        result = judge(symbol, articles, max_articles=SENTIMENT_ARTICLES)
    except Exception as exc:
        logger.warning(f"Sentiment failed for {symbol}: {exc}")
        return {"signal": "HOLD", "strength": 0.0, "reason": f"Sentiment unavailable: {exc}"}
    score = float(result.get("score") or 0.0)
    threshold = _env_float("SENTIMENT_THRESHOLD", 0.3)
    signal = "BUY" if score >= threshold else "SELL" if score <= -threshold else "HOLD"
    return {
        "signal": signal,
        "strength": round(min(abs(score), 1.0), 3),
        "score": round(score, 4),
        "evidence_weight": result.get("evidence_weight"),
        "headlines": len(result.get("articles") or []),
        "reason": f"Sentiment: Jev {score:+.2f} on {len(result.get('articles') or [])} headlines"
                  + (f" ({result['rationale'][:120]})" if result.get("rationale") else ""),
    }


def fuse(signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    def rank(item):
        name, s = item
        active = s.get("signal") != "HOLD"
        return (PRIORITY[name] if active else 0, s.get("strength") or 0.0)

    ordered = sorted(signals.items(), key=rank, reverse=True)
    primary_name, primary = ordered[0]
    confirming = sum(1 for _, s in ordered[1:] if s["signal"] == primary["signal"] and primary["signal"] != "HOLD")
    conflicting = sum(1 for _, s in ordered[1:] if s["signal"] not in ("HOLD", primary["signal"]))
    base = primary.get("strength") or 0.0
    if primary["signal"] == "HOLD":
        confidence = base
    elif confirming:
        confidence = min(base + 0.1 * confirming, 0.95)
    else:
        confidence = max(base - 0.15 * conflicting, 0.2)
    return {"signal": primary["signal"], "primary_strategy": primary_name, "primary_reason": primary["reason"],
            "confidence": round(confidence, 3), "confirming": confirming, "conflicting": conflicting}


def position_size(signal: str, confidence: float, price: float, capital: float, vol: Optional[float],
                  bar_seconds: int, held: float = 0) -> int:
    if signal == "HOLD" or price <= 0:
        return 0
    if signal == "SELL":
        return int(held) if held > 0 else 0
    min_pct = _env_float("MIN_POSITION_PERCENTAGE", 0.02)
    max_pct = _env_float("MAX_POSITION_PERCENTAGE", 0.10)
    base = max_pct if confidence >= 0.7 else (min_pct + max_pct) / 2 if confidence >= 0.5 else min_pct
    if vol:
        bars_per_year = 252 * max(1, 23400 // max(bar_seconds, 1)) if bar_seconds < 86400 else 252
        vol_factor = max(0.5, 1 / (1 + vol * math.sqrt(bars_per_year)))
    else:
        vol_factor = 1.0
    value = capital * base * confidence * vol_factor
    qty = int(value / price)
    if qty == 0 and value >= price * 0.5:
        qty = 1
    return qty


def risk_parameters(signal: str, qty: int, price: float, atr: Optional[float], capital: float) -> Dict[str, Any]:
    if atr and atr > 0:
        stop_dist, target_dist, method = 2 * atr, 3 * atr, "ATR"
    else:
        stop_dist = price * _env_float("STOP_LOSS_PCT", 3.0) / 100
        target_dist, method = stop_dist * 1.5, "STOP_LOSS_PCT"
    if signal == "SELL":
        stop, target = price + stop_dist, price - target_dist
    else:
        stop, target = price - stop_dist, price + target_dist
    risk_per_share = abs(price - stop)
    total_risk = risk_per_share * qty
    return {
        "stop_loss": round(stop, 2),
        "take_profit": round(target, 2),
        "risk_reward_ratio": round(target_dist / stop_dist, 2) if stop_dist else None,
        "position_value": round(qty * price, 2),
        "total_risk": round(total_risk, 2),
        "risk_percentage": round(total_risk / capital * 100, 2) if capital else None,
        "method": method,
    }


def default_capital() -> float:
    return _env_float("TRADING_CAPITAL", 5000.0)


def analyze(symbol: str, capital: Optional[float] = None, held: float = 0, *,
            bars: Optional[List[Dict[str, Any]]] = None, articles: Optional[List[Dict[str, Any]]] = None,
            dividend_rows: Optional[List[Dict[str, Any]]] = None, judge: Optional[Callable] = None,
            capital_source: str = "TRADING_CAPITAL") -> Dict[str, Any]:
    symbol = symbol.upper()
    tf = timeframe()
    if bars is None:
        bars = market_feed.get_bars(symbol, tf, 300).get("bars") or []
    technical = technicals.technical_signal(bars)
    price = technical.get("current_price")
    if not price:
        quote = market_feed.get_quotes([symbol]).get(symbol) or {}
        price = quote.get("last")
    sentiment = sentiment_signal(symbol, articles, judge)
    dividend = dividends.dividend_signal(symbol, price, rows=dividend_rows)
    signals = {"technical": technical, "sentiment": sentiment, "dividend": dividend}
    fused = fuse(signals)
    capital = capital if capital is not None else default_capital()
    ind = technical.get("indicators") or {}
    qty = position_size(fused["signal"], fused["confidence"], price or 0, capital, ind.get("volatility"),
                        market_feed.TIMEFRAMES[tf][1], held) if price else 0
    risk = risk_parameters(fused["signal"], qty, price, ind.get("atr"), capital) if price else {}
    reason = fused["primary_reason"]
    if fused["signal"] == "SELL" and qty == 0:
        reason += " · long only: nothing held to sell"
    return {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recommendation": fused["signal"],
        "signal": fused["signal"],
        "quantity": qty,
        "confidence": fused["confidence"],
        "primary_strategy": fused["primary_strategy"],
        "primary_reason": reason,
        "reason": reason,
        "confirming_strategies": fused["confirming"],
        "conflicting_strategies": fused["conflicting"],
        "all_signals": signals,
        "technical_indicators": ind,
        "current_price": price,
        "risk_parameters": risk,
        "capital": capital,
        "capital_source": capital_source,
        "held": held,
        "timeframe": tf,
    }


def main(argv: Optional[List[str]] = None) -> int:
    config.load_env()
    parser = argparse.ArgumentParser(prog="python -m trader.analysis")
    parser.add_argument("symbol")
    parser.add_argument("--capital", type=float, help="capital to size against (default TRADING_CAPITAL)")
    parser.add_argument("--held", type=float, default=0, help="shares currently held")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    print(json.dumps(analyze(args.symbol, args.capital, args.held), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
