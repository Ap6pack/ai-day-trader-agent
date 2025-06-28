#!/usr/bin/env python3
import pandas as pd
import ta

def compute_indicators(candles_dict):
    """
    Compute technical indicators (RSI, MACD, SMA, EMA) for each interval.
    candles_dict: {interval: [candles]}
    Returns: {interval: {indicator_name: value, ...}}
    """
    indicators = {}
    for interval, candles in candles_dict.items():
        if not candles:
            indicators[interval] = {}
            continue
        df = pd.DataFrame(candles)
        # Ensure correct types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("datetime")  # Oldest first
        # Compute indicators
        result = {}
        try:
            result["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi().iloc[-1]
        except Exception:
            result["rsi"] = None
        try:
            macd = ta.trend.MACD(df["close"])
            result["macd"] = macd.macd().iloc[-1]
            result["macd_signal"] = macd.macd_signal().iloc[-1]
        except Exception:
            result["macd"] = None
            result["macd_signal"] = None
        try:
            result["sma_20"] = df["close"].rolling(window=20).mean().iloc[-1]
        except Exception:
            result["sma_20"] = None
        try:
            result["ema_20"] = df["close"].ewm(span=20, adjust=False).mean().iloc[-1]
        except Exception:
            result["ema_20"] = None
        indicators[interval] = result
    return indicators
