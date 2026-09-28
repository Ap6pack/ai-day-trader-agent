#!/usr/bin/env python3
import pandas as pd
import ta

CANDLE_FIELDS = ("open", "high", "low", "close", "volume", "datetime")


def sort_candlesticks(candlesticks):
    """
    Return a copy of column-oriented candlesticks in chronological (oldest-first) order.

    Providers disagree on order: Alpaca (sort=desc), Yahoo Finance and Twelve Data
    all return newest-first, while every indicator reads ``iloc[-1]`` as the latest
    bar. Rows are sorted by ``datetime`` when every timestamp parses; otherwise the
    input order is kept unchanged.
    """
    datetimes = candlesticks.get("datetime")
    if not datetimes or len(datetimes) < 2:
        return dict(candlesticks)
    if any(len(candlesticks.get(f, [])) != len(datetimes) for f in CANDLE_FIELDS if f in candlesticks):
        return dict(candlesticks)

    # utc=True accepts both naive ("2026-01-02 15:00:00") and aware ("...Z") stamps.
    parsed = pd.to_datetime(pd.Series(datetimes), errors="coerce", utc=True)
    if parsed.isna().any():
        return dict(candlesticks)

    order = parsed.sort_values(kind="stable").index.tolist()
    if order == list(range(len(order))):
        return dict(candlesticks)
    return {
        field: ([values[i] for i in order] if field in CANDLE_FIELDS else values)
        for field, values in candlesticks.items()
    }


def compute_indicators(market_data):
    """
    Compute technical indicators (RSI, MACD, SMA, EMA) from market data.
    market_data: {'candlesticks': {'open': [...], 'high': [...], 'low': [...], 'close': [...],
                                   'volume': [...], 'datetime': [...]}}
    Candles are sorted by ``datetime`` when present, so provider order does not matter;
    without datetimes the lists must already be oldest-first.
    Returns: {indicator_name: value, ...} computed at the most recent bar.
    """
    # Handle the new format from the pipeline
    if 'candlesticks' in market_data:
        candlesticks = sort_candlesticks(market_data['candlesticks'])
        
        # Create DataFrame from the candlesticks data
        df = pd.DataFrame({
            'open': candlesticks.get('open', []),
            'high': candlesticks.get('high', []),
            'low': candlesticks.get('low', []),
            'close': candlesticks.get('close', []),
            'volume': candlesticks.get('volume', [])
        })
        
        # Ensure correct types
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Compute indicators
        result = {}
        try:
            if len(df) >= 14:  # Need at least 14 periods for RSI
                result["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi().iloc[-1]
            else:
                result["rsi"] = None
        except Exception:
            result["rsi"] = None
            
        try:
            if len(df) >= 26:  # Need at least 26 periods for MACD
                macd = ta.trend.MACD(df["close"])
                result["macd"] = macd.macd().iloc[-1]
                result["macd_signal"] = macd.macd_signal().iloc[-1]
            else:
                result["macd"] = None
                result["macd_signal"] = None
        except Exception:
            result["macd"] = None
            result["macd_signal"] = None
            
        try:
            if len(df) >= 20:  # Need at least 20 periods for SMA/EMA
                result["sma_20"] = df["close"].rolling(window=20).mean().iloc[-1]
                result["ema_20"] = df["close"].ewm(span=20, adjust=False).mean().iloc[-1]
            else:
                result["sma_20"] = None
                result["ema_20"] = None
        except Exception:
            result["sma_20"] = None
            result["ema_20"] = None
            
        return result
    
    # Fallback for old format
    else:
        indicators = {}
        for interval, candles in market_data.items():
            if not candles:
                indicators[interval] = {}
                continue
            df = pd.DataFrame(candles)
            # Ensure correct types
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Sort by datetime if available, otherwise by index
            if 'datetime' in df.columns:
                df = df.sort_values("datetime")
            else:
                df = df.sort_index()
                
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
