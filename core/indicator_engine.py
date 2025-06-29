#!/usr/bin/env python3
import pandas as pd
import ta

def compute_indicators(market_data):
    """
    Compute technical indicators (RSI, MACD, SMA, EMA) from market data.
    market_data: {'candlesticks': {'open': [...], 'high': [...], 'low': [...], 'close': [...], 'volume': [...]}}
    Returns: {indicator_name: value, ...}
    """
    # Handle the new format from the pipeline
    if 'candlesticks' in market_data:
        candlesticks = market_data['candlesticks']
        
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
        
        # Sort by index (oldest first) - no datetime column needed
        df = df.sort_index()
        
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
