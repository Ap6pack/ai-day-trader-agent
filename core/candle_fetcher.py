#!/usr/bin/env python3
import requests
from config import settings

def fetch_twelvedata_candles(symbol, intervals=("1min", "15min", "1h"), outputsize=100):
    """
    Fetch candlestick data for the given symbol and intervals from Twelve Data API.
    Returns a dict: {interval: [candles]}
    """
    base_url = "https://api.twelvedata.com/time_series"
    api_key = settings.TWELVE_DATA_API_KEY
    data = {}

    for interval in intervals:
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": api_key
        }
        try:
            resp = requests.get(base_url, params=params, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            if "values" in result:
                data[interval] = result["values"]
            else:
                data[interval] = []
        except Exception:
            data[interval] = []
    return data

def fetch_alphavantage_candles(symbol, intervals=("1min", "15min", "1h"), outputsize=100):
    """
    Fetch candlestick data for the given symbol and intervals from Alpha Vantage API.
    Returns a dict: {interval: [candles]}
    """
    base_url = "https://www.alphavantage.co/query"
    api_key = settings.ALPHA_VANTAGE_API_KEY
    interval_map = {"1min": "1min", "15min": "15min", "1h": "60min"}
    data = {}

    for interval in intervals:
        av_interval = interval_map.get(interval, interval)
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": av_interval,
            "outputsize": "compact" if outputsize <= 100 else "full",
            "apikey": api_key
        }
        try:
            resp = requests.get(base_url, params=params, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            key = f"Time Series ({av_interval})"
            if key in result:
                # Convert Alpha Vantage format to list of dicts, sorted by datetime descending
                candles = []
                for dt, values in sorted(result[key].items(), reverse=True):
                    candles.append({
                        "datetime": dt,
                        "open": values.get("1. open"),
                        "high": values.get("2. high"),
                        "low": values.get("3. low"),
                        "close": values.get("4. close"),
                        "volume": values.get("5. volume")
                    })
                data[interval] = candles[:outputsize]
            else:
                data[interval] = []
        except Exception:
            data[interval] = []
    return data

def get_candlestick_data(symbol, intervals=("1min", "15min", "1h"), outputsize=100):
    """
    Fetch candlestick data from both Twelve Data and Alpha Vantage in parallel.
    Returns a dict:
    {
        "twelvedata": {interval: [candles]},
        "alphavantage": {interval: [candles]}
    }
    """
    td_data = fetch_twelvedata_candles(symbol, intervals, outputsize)
    av_data = fetch_alphavantage_candles(symbol, intervals, outputsize)
    return {
        "twelvedata": td_data,
        "alphavantage": av_data
    }
