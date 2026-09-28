"""
Technical indicators and the technical signal, from OHLCV bars (oldest first).

Plain Python, no pandas. The signal rules match the original desk's analysis:
RSI (oversold/overbought bands), MACD against its signal line, and price against
the 20-period SMA and EMA.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


def sma(values: Sequence[float], n: int) -> Optional[float]:
    return sum(values[-n:]) / n if len(values) >= n else None


def ema_series(values: Sequence[float], n: int) -> List[float]:
    if len(values) < n:
        return []
    k = 2 / (n + 1)
    out = [sum(values[:n]) / n]
    for v in values[n:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def ema(values: Sequence[float], n: int) -> Optional[float]:
    series = ema_series(values, n)
    return series[-1] if series else None


def rsi(closes: Sequence[float], n: int = 14) -> Optional[float]:
    """Wilder's RSI."""
    if len(closes) <= n:
        return None
    gains = [max(closes[i] - closes[i - 1], 0.0) for i in range(1, len(closes))]
    losses = [max(closes[i - 1] - closes[i], 0.0) for i in range(1, len(closes))]
    avg_gain = sum(gains[:n]) / n
    avg_loss = sum(losses[:n]) / n
    for g, l in zip(gains[n:], losses[n:]):
        avg_gain = (avg_gain * (n - 1) + g) / n
        avg_loss = (avg_loss * (n - 1) + l) / n
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100 - 100 / (1 + avg_gain / avg_loss)


def macd(closes: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Optional[float]]:
    fast_s, slow_s = ema_series(closes, fast), ema_series(closes, slow)
    if not slow_s:
        return {"macd": None, "macd_signal": None, "macd_hist": None}
    offset = len(fast_s) - len(slow_s)
    line = [f - s for f, s in zip(fast_s[offset:], slow_s)]
    sig = ema_series(line, signal)
    if not sig:
        return {"macd": line[-1], "macd_signal": None, "macd_hist": None}
    return {"macd": line[-1], "macd_signal": sig[-1], "macd_hist": line[-1] - sig[-1]}


def atr(bars: Sequence[Dict[str, Any]], n: int = 14) -> Optional[float]:
    if len(bars) <= n:
        return None
    trs = []
    for prev, bar in zip(bars, bars[1:]):
        trs.append(max(bar["high"] - bar["low"], abs(bar["high"] - prev["close"]), abs(bar["low"] - prev["close"])))
    value = sum(trs[:n]) / n
    for tr in trs[n:]:
        value = (value * (n - 1) + tr) / n
    return value


def volatility(closes: Sequence[float]) -> Optional[float]:
    """Standard deviation of bar-to-bar returns (not annualized)."""
    if len(closes) < 6:
        return None
    returns = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes)) if closes[i - 1]]
    mean = sum(returns) / len(returns)
    return math.sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1))


def indicators(bars: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    closes = [float(b["close"]) for b in bars]
    return {
        "current_price": closes[-1] if closes else None,
        "rsi": rsi(closes),
        **macd(closes),
        "sma_20": sma(closes, 20),
        "sma_50": sma(closes, 50),
        "ema_9": ema(closes, 9),
        "ema_20": ema(closes, 20),
        "atr": atr(bars),
        "volatility": volatility(closes),
    }


def technical_signal(bars: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """BUY / SELL / HOLD with strength 0..1 from RSI, MACD and moving averages."""
    ind = indicators(bars)
    price = ind["current_price"]
    if price is None or ind["rsi"] is None:
        return {"signal": "HOLD", "strength": 0.0, "indicators": ind, "current_price": price,
                "reason": "Technical: not enough price history"}
    score, net, notes = 0.0, 0, []

    r = ind["rsi"]
    if r < 45:
        net, score = net + 1, score + 0.4
        notes.append(f"RSI {r:.0f} low")
        if r < 30:
            net, score = net + 1, score + 0.3
    elif r > 55:
        net, score = net - 1, score - 0.4
        notes.append(f"RSI {r:.0f} high")
        if r > 70:
            net, score = net - 1, score - 0.3

    if ind["macd"] is not None and ind["macd_signal"] is not None:
        if ind["macd"] > ind["macd_signal"]:
            net, score = net + 1, score + 0.2
            notes.append("MACD above signal")
        elif ind["macd"] < ind["macd_signal"]:
            net, score = net - 1, score - 0.2
            notes.append("MACD below signal")

    if ind["sma_20"] is not None and ind["ema_20"] is not None:
        if price > ind["sma_20"] and price > ind["ema_20"]:
            net, score = net + 1, score + 0.2
            notes.append("above SMA/EMA20")
        elif price < ind["sma_20"] and price < ind["ema_20"]:
            net, score = net - 1, score - 0.2
            notes.append("below SMA/EMA20")

    if net >= 1 and score > 0.2:
        signal = "BUY"
    elif net <= -1 and score < -0.2:
        signal = "SELL"
    else:
        signal = "HOLD"
    return {
        "signal": signal,
        "strength": round(min(abs(score), 1.0), 3),
        "indicators": ind,
        "current_price": price,
        "reason": f"Technical: {net:+d} net indicators ({', '.join(notes) or 'neutral'})",
    }
