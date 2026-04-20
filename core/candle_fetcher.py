#!/usr/bin/env python3
"""
Enhanced candlestick data fetcher with robust multi-source failover.
Implements a professional-grade data acquisition system with proper error handling.
"""

import requests
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
import yfinance as yf
import random
import os
from collections import deque
from threading import Lock

from config import settings

logger = logging.getLogger(__name__)


class DataSource:
    """Enum-like class for data source priorities."""
    ALPACA = "alpaca"
    TWELVE_DATA = "twelve_data"
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO_FINANCE = "yahoo_finance"


class RateLimitConfig:
    """Configuration for rate limiting from environment variables."""
    ALPACA_RATE_LIMIT_WAIT = int(os.getenv('ALPACA_RATE_LIMIT_WAIT', '60'))
    ALPACA_MAX_RETRIES = int(os.getenv('ALPACA_MAX_RETRIES', '3'))
    ALPACA_CALLS_PER_MINUTE = int(os.getenv('ALPACA_CALLS_PER_MINUTE', '180'))

    TWELVE_DATA_RATE_LIMIT_WAIT = int(os.getenv('TWELVE_DATA_RATE_LIMIT_WAIT', '60'))
    TWELVE_DATA_MAX_RETRIES = int(os.getenv('TWELVE_DATA_MAX_RETRIES', '3'))
    TWELVE_DATA_CALLS_PER_MINUTE = int(os.getenv('TWELVE_DATA_CALLS_PER_MINUTE', '8'))
    
    ALPHA_VANTAGE_RATE_LIMIT_WAIT = int(os.getenv('ALPHA_VANTAGE_RATE_LIMIT_WAIT', '60'))
    ALPHA_VANTAGE_MAX_RETRIES = int(os.getenv('ALPHA_VANTAGE_MAX_RETRIES', '3'))
    ALPHA_VANTAGE_CALLS_PER_MINUTE = int(os.getenv('ALPHA_VANTAGE_CALLS_PER_MINUTE', '5'))
    
    # Exponential backoff settings
    BACKOFF_FACTOR = float(os.getenv('API_BACKOFF_FACTOR', '2.0'))
    MAX_BACKOFF_SECONDS = int(os.getenv('API_MAX_BACKOFF_SECONDS', '300'))
    JITTER_ENABLED = os.getenv('API_JITTER_ENABLED', 'true').lower() == 'true'


class RateLimiter:
    """
    Professional rate limiter with request tracking and exponential backoff.
    Implements a sliding window approach for API call tracking.
    """
    
    def __init__(self, calls_per_minute: int, source_name: str):
        self.calls_per_minute = calls_per_minute
        self.source_name = source_name
        self.request_times = deque()
        self.lock = Lock()
        self.rate_limit_until = None
        self.consecutive_rate_limits = 0
        
    def can_make_request(self) -> bool:
        """Check if we can make a request without hitting rate limit."""
        with self.lock:
            now = datetime.now()
            
            # Check if we're in a rate limit cooldown period
            if self.rate_limit_until and now < self.rate_limit_until:
                return False
            
            # Remove requests older than 1 minute
            cutoff_time = now - timedelta(minutes=1)
            while self.request_times and self.request_times[0] < cutoff_time:
                self.request_times.popleft()
            
            # Check if we can make another request
            return len(self.request_times) < self.calls_per_minute
    
    def record_request(self):
        """Record that a request was made."""
        with self.lock:
            self.request_times.append(datetime.now())
            logger.debug(f"{self.source_name}: Recorded request. {len(self.request_times)}/{self.calls_per_minute} in last minute")
    
    def record_rate_limit(self, wait_seconds: Optional[int] = None):
        """Record that we hit a rate limit."""
        with self.lock:
            self.consecutive_rate_limits += 1
            
            # Calculate backoff time
            if wait_seconds:
                backoff_time = wait_seconds
            else:
                # Exponential backoff with jitter
                base_wait = RateLimitConfig.TWELVE_DATA_RATE_LIMIT_WAIT
                backoff_time = min(
                    base_wait * (RateLimitConfig.BACKOFF_FACTOR ** (self.consecutive_rate_limits - 1)),
                    RateLimitConfig.MAX_BACKOFF_SECONDS
                )
                
                # Add jitter to prevent thundering herd
                if RateLimitConfig.JITTER_ENABLED:
                    jitter = random.uniform(0, backoff_time * 0.1)
                    backoff_time += jitter
            
            self.rate_limit_until = datetime.now() + timedelta(seconds=backoff_time)
            logger.warning(f"{self.source_name}: Rate limited for {backoff_time:.1f}s until {self.rate_limit_until}")
    
    def reset_consecutive_limits(self):
        """Reset consecutive rate limit counter after successful request."""
        with self.lock:
            self.consecutive_rate_limits = 0
    
    def get_wait_time(self) -> Optional[float]:
        """Get seconds to wait before next request, or None if ready."""
        with self.lock:
            if self.rate_limit_until:
                now = datetime.now()
                if now < self.rate_limit_until:
                    return (self.rate_limit_until - now).total_seconds()
            return None

    def reserve_request_or_raise(self) -> None:
        """Record a request slot or fail fast when the source is cooling down."""
        wait_time = self.get_wait_time()
        if wait_time:
            raise ValueError(
                f"{self.source_name}: Rate limited for {wait_time:.1f}s; skipping provider"
            )

        if not self.can_make_request():
            self.record_rate_limit(60)
            raise ValueError(f"{self.source_name}: Local rate limit reached; skipping provider")

        self.record_request()


class CandlestickDataFetcher:
    """
    Professional candlestick data fetcher with multi-source failover.
    Implements intelligent retry logic and caching for reliability.
    """
    
    def __init__(self):
        self.alpaca_api_key = settings.ALPACA_API_KEY
        self.alpaca_secret_key = settings.ALPACA_SECRET_KEY
        self.alpaca_data_base_url = settings.ALPACA_DATA_BASE_URL.rstrip("/")
        self.alpaca_data_feed = settings.ALPACA_DATA_FEED
        self.twelve_data_key = settings.TWELVE_DATA_API_KEY
        self.alpha_vantage_key = settings.ALPHA_VANTAGE_API_KEY
        self.source_priority = self._load_source_priority()
        self._rate_limit_cache = {}
        
        # Initialize rate limiters
        self.rate_limiters = {
            DataSource.ALPACA: RateLimiter(
                RateLimitConfig.ALPACA_CALLS_PER_MINUTE,
                "Alpaca"
            ),
            DataSource.TWELVE_DATA: RateLimiter(
                RateLimitConfig.TWELVE_DATA_CALLS_PER_MINUTE,
                "TwelveData"
            ),
            DataSource.ALPHA_VANTAGE: RateLimiter(
                RateLimitConfig.ALPHA_VANTAGE_CALLS_PER_MINUTE,
                "AlphaVantage"
            )
        }

    def _load_source_priority(self) -> List[str]:
        """Load provider order from env, defaulting to Alpaca then fallbacks."""
        default_priority = [
            DataSource.ALPACA,
            DataSource.YAHOO_FINANCE,
            DataSource.TWELVE_DATA,
            DataSource.ALPHA_VANTAGE,
        ]
        raw_priority = os.getenv('MARKET_DATA_PROVIDERS')
        if not raw_priority:
            return default_priority

        aliases = {
            'alpaca': DataSource.ALPACA,
            'yahoo': DataSource.YAHOO_FINANCE,
            'yahoo_finance': DataSource.YAHOO_FINANCE,
            'twelvedata': DataSource.TWELVE_DATA,
            'twelve_data': DataSource.TWELVE_DATA,
            'alpha': DataSource.ALPHA_VANTAGE,
            'alphavantage': DataSource.ALPHA_VANTAGE,
            'alpha_vantage': DataSource.ALPHA_VANTAGE,
        }
        configured_priority = []
        for source_name in raw_priority.split(','):
            source = aliases.get(source_name.strip().lower())
            if source and source not in configured_priority:
                configured_priority.append(source)

        return configured_priority or default_priority

    def _source_configured(self, source: str) -> bool:
        """Return whether a source can be called with current configuration."""
        if source == DataSource.ALPACA:
            return bool(self.alpaca_api_key and self.alpaca_secret_key)
        if source == DataSource.TWELVE_DATA:
            return bool(self.twelve_data_key)
        if source == DataSource.ALPHA_VANTAGE:
            return bool(self.alpha_vantage_key)
        if source == DataSource.YAHOO_FINANCE:
            return True
        return False
        
    def fetch_with_failover(self, symbol: str, intervals: Tuple[str, ...] = ("1min", "15min", "1h"), 
                           outputsize: int = 500) -> Dict[str, any]:
        """
        Fetch candlestick data with automatic failover between sources.
        
        Returns:
            Dict with structure:
            {
                'success': bool,
                'source': str,  # Which API provided the data
                'data': {source_name: {interval: [candles]}},
                'errors': List[str]  # Any errors encountered
            }
        """
        errors = []
        successful_source = None
        result_data = {
            'alpaca': {},
            'yahoo_finance': {},
            'twelvedata': {},
            'alphavantage': {}
        }
        
        for source in self.source_priority:
            try:
                if not self._source_configured(source):
                    errors.append(f"{source}: Not configured")
                    continue

                if self._is_rate_limited(source):
                    errors.append(f"{source}: Rate limited (cached)")
                    continue
                
                logger.info(f"Attempting to fetch data from {source} for {symbol}")
                
                if source == DataSource.ALPACA:
                    data = self._fetch_alpaca_candles(symbol, intervals, outputsize)
                    if self._validate_data(data):
                        result_data['alpaca'] = data
                        successful_source = source
                        break
                    else:
                        errors.append(f"{source}: Invalid or empty data")

                elif source == DataSource.TWELVE_DATA:
                    data = self._fetch_twelvedata_candles(symbol, intervals, outputsize)
                    if self._validate_data(data):
                        result_data['twelvedata'] = data
                        successful_source = source
                        break
                    else:
                        errors.append(f"{source}: Invalid or empty data")
                        
                elif source == DataSource.ALPHA_VANTAGE:
                    data = self._fetch_alphavantage_candles(symbol, intervals, outputsize)
                    if self._validate_data(data):
                        result_data['alphavantage'] = data
                        successful_source = source
                        break
                    else:
                        errors.append(f"{source}: Invalid or empty data")
                        
                elif source == DataSource.YAHOO_FINANCE:
                    data = self._fetch_yahoo_candles(symbol, intervals, outputsize)
                    if self._validate_data(data):
                        result_data['yahoo_finance'] = data
                        successful_source = source
                        break
                    else:
                        errors.append(f"{source}: Invalid or empty data")
                        
            except Exception as e:
                error_msg = f"{source}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                
                # Cache rate limit errors
                if "rate limit" in str(e).lower() or "api message" in str(e).lower():
                    self._cache_rate_limit(source)
        
        return {
            'success': successful_source is not None,
            'source': successful_source,
            'data': result_data,
            'errors': errors
        }
    
    def _validate_data(self, data: Dict) -> bool:
        """Validate that we have meaningful data, not empty or dummy values."""
        if not data:
            return False
        
        # Check each interval
        for interval, candles in data.items():
            if not candles or not isinstance(candles, list):
                continue
                
            # Need at least some data points
            if len(candles) < 10:
                continue
                
            # Check if data looks valid (not all zeros or same values)
            closes = [float(c.get('close', 0)) for c in candles if c.get('close')]
            if len(closes) > 0 and len(set(closes)) > 1 and all(c > 0 for c in closes):
                return True
                
        return False
    
    def _is_rate_limited(self, source: str) -> bool:
        """Check if a source is currently rate limited."""
        if source not in self._rate_limit_cache:
            return False
            
        cached_time = self._rate_limit_cache[source]
        # Consider rate limited for 1 hour
        if (datetime.now() - cached_time).seconds < 3600:
            return True
            
        # Clear expired cache
        del self._rate_limit_cache[source]
        return False
    
    def _cache_rate_limit(self, source: str):
        """Cache that a source is rate limited."""
        self._rate_limit_cache[source] = datetime.now()
        logger.warning(f"Caching rate limit for {source} until {datetime.now() + timedelta(hours=1)}")

    def _fetch_alpaca_candles(self, symbol: str, intervals: Tuple[str, ...],
                              outputsize: int) -> Dict[str, List[Dict]]:
        """Fetch candlestick data from Alpaca Market Data API."""
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            raise ValueError("Alpaca API credentials not configured")

        rate_limiter = self.rate_limiters[DataSource.ALPACA]
        interval_map = {
            "1min": ("1Min", 1),
            "15min": ("15Min", 5),
            "1h": ("1Hour", 30),
        }
        headers = {
            "APCA-API-KEY-ID": self.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.alpaca_secret_key,
        }
        data = {}

        for interval in intervals:
            timeframe, lookback_days = interval_map.get(interval, (interval, 5))
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=lookback_days)
            params = {
                "timeframe": timeframe,
                "start": start.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "end": end.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "limit": min(outputsize, 10000),
                "adjustment": "raw",
                "feed": self.alpaca_data_feed,
                "sort": "desc",
            }
            url = f"{self.alpaca_data_base_url}/v2/stocks/{symbol}/bars"

            try:
                rate_limiter.reserve_request_or_raise()
                response = requests.get(url, headers=headers, params=params, timeout=10)

                if response.status_code == 429:
                    rate_limiter.record_rate_limit(self._parse_retry_after(response))
                    raise ValueError("API rate limit: Alpaca returned 429")

                response.raise_for_status()
                result = response.json()
                bars = result.get("bars", [])
                data[interval] = [
                    {
                        "datetime": bar.get("t"),
                        "open": bar.get("o"),
                        "high": bar.get("h"),
                        "low": bar.get("l"),
                        "close": bar.get("c"),
                        "volume": bar.get("v"),
                    }
                    for bar in bars[:outputsize]
                ]
                rate_limiter.reset_consecutive_limits()

            except requests.exceptions.RequestException as e:
                logger.error(f"Network error fetching {interval} from Alpaca: {e}")
                data[interval] = []
            except ValueError:
                raise
            except Exception as e:
                logger.error(f"Unexpected error fetching {interval} from Alpaca: {e}")
                raise

        return data

    def _parse_retry_after(self, response) -> int:
        """Read retry wait seconds from Alpaca rate-limit headers when available."""
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return int(float(retry_after))
            except ValueError:
                pass
        return RateLimitConfig.ALPACA_RATE_LIMIT_WAIT
    
    def _fetch_twelvedata_candles(self, symbol: str, intervals: Tuple[str, ...], 
                                 outputsize: int) -> Dict[str, List[Dict]]:
        """Fetch candlestick data from Twelve Data API with rate limiting and retry logic."""
        if not self.twelve_data_key:
            raise ValueError("Twelve Data API key not configured")
            
        rate_limiter = self.rate_limiters[DataSource.TWELVE_DATA]
        base_url = "https://api.twelvedata.com/time_series"
        data = {}
        
        for interval in intervals:
            params = {
                "symbol": symbol,
                "interval": interval,
                "outputsize": min(outputsize, 5000),  # Twelve Data max
                "apikey": self.twelve_data_key
            }
            
            try:
                rate_limiter.reserve_request_or_raise()
                resp = requests.get(base_url, params=params, timeout=10)
                resp.raise_for_status()
                result = resp.json()

                if result.get('status') == 'error':
                    error_msg = result.get('message', 'Unknown error')
                    if 'api message' in error_msg.lower() or 'api credits' in error_msg.lower():
                        wait_seconds = 60 if 'wait for the next minute' in error_msg.lower() else None
                        rate_limiter.record_rate_limit(wait_seconds)
                        raise ValueError(f"API rate limit: {error_msg}")
                    raise ValueError(f"API error: {error_msg}")

                rate_limiter.reset_consecutive_limits()
                data[interval] = result.get("values") or []

            except requests.exceptions.RequestException as e:
                logger.error(f"Network error fetching {interval} from Twelve Data: {e}")
                data[interval] = []
            except ValueError as e:
                logger.error(f"Error fetching {interval} from Twelve Data: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error fetching {interval} from Twelve Data: {e}")
                raise
                
        return data
    
    def _fetch_alphavantage_candles(self, symbol: str, intervals: Tuple[str, ...], 
                                   outputsize: int) -> Dict[str, List[Dict]]:
        """Fetch candlestick data from Alpha Vantage API with rate limiting and retry logic."""
        if not self.alpha_vantage_key:
            raise ValueError("Alpha Vantage API key not configured")
            
        rate_limiter = self.rate_limiters[DataSource.ALPHA_VANTAGE]
        base_url = "https://www.alphavantage.co/query"
        interval_map = {"1min": "1min", "15min": "15min", "1h": "60min"}
        data = {}
        
        for interval in intervals:
            av_interval = interval_map.get(interval, interval)
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": av_interval,
                "outputsize": "full" if outputsize > 100 else "compact",
                "apikey": self.alpha_vantage_key
            }
            
            try:
                rate_limiter.reserve_request_or_raise()
                resp = requests.get(base_url, params=params, timeout=10)
                resp.raise_for_status()
                result = resp.json()

                if 'Information' in result and 'api message' in result['Information'].lower():
                    rate_limiter.record_rate_limit(RateLimitConfig.ALPHA_VANTAGE_RATE_LIMIT_WAIT)
                    raise ValueError(f"API rate limit: {result['Information']}")

                if 'Error Message' in result:
                    raise ValueError(f"API error: {result['Error Message']}")

                rate_limiter.reset_consecutive_limits()
                key = f"Time Series ({av_interval})"
                if key in result:
                    candles = []
                    for dt, values in sorted(result[key].items(), reverse=True)[:outputsize]:
                        candles.append({
                            "datetime": dt,
                            "open": values.get("1. open"),
                            "high": values.get("2. high"),
                            "low": values.get("3. low"),
                            "close": values.get("4. close"),
                            "volume": values.get("5. volume")
                        })
                    data[interval] = candles
                else:
                    data[interval] = []

            except requests.exceptions.RequestException as e:
                logger.error(f"Network error fetching {interval} from Alpha Vantage: {e}")
                data[interval] = []
            except ValueError as e:
                logger.error(f"Error fetching {interval} from Alpha Vantage: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error fetching {interval} from Alpha Vantage: {e}")
                raise
                
        return data
    
    def _fetch_yahoo_candles(self, symbol: str, intervals: Tuple[str, ...], 
                           outputsize: int) -> Dict[str, List[Dict]]:
        """Fetch candlestick data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            data = {}
            
            # Map our intervals to Yahoo Finance periods and intervals
            interval_map = {
                "1min": ("1d", "1m"),      # 1 day of 1-minute data
                "15min": ("5d", "15m"),    # 5 days of 15-minute data
                "1h": ("1mo", "1h")        # 1 month of hourly data
            }
            
            for our_interval in intervals:
                if our_interval not in interval_map:
                    data[our_interval] = []
                    continue
                    
                period, yf_interval = interval_map[our_interval]
                
                try:
                    # Fetch data from Yahoo Finance
                    hist = ticker.history(period=period, interval=yf_interval)
                    
                    if hist.empty:
                        data[our_interval] = []
                        continue
                    
                    # Convert to our standard format
                    candles = []
                    for idx, row in hist.iterrows():
                        # Convert timestamp to string format
                        dt_str = idx.strftime('%Y-%m-%d %H:%M:%S')
                        
                        candles.append({
                            "datetime": dt_str,
                            "open": str(row['Open']),
                            "high": str(row['High']),
                            "low": str(row['Low']),
                            "close": str(row['Close']),
                            "volume": str(int(row['Volume']))
                        })
                    
                    # Reverse to have most recent first and limit to outputsize
                    candles.reverse()
                    data[our_interval] = candles[:outputsize]
                    
                except Exception as e:
                    logger.error(f"Error fetching {our_interval} from Yahoo Finance: {e}")
                    data[our_interval] = []
                    
            return data
            
        except Exception as e:
            logger.error(f"Error initializing Yahoo Finance for {symbol}: {e}")
            raise ValueError(f"Failed to fetch data from Yahoo Finance: {str(e)}")
    
    def fetch_realtime_quote(self, symbol: str) -> Dict[str, float]:
        """
        Fetch current market quote with failover.
        
        Returns:
            Dict with 'current_price', 'previous_close', 'change_percent'
        """
        if self.alpaca_api_key and self.alpaca_secret_key and not self._is_rate_limited(DataSource.ALPACA):
            try:
                return self._fetch_alpaca_quote(symbol)
            except Exception as e:
                logger.error(f"Alpaca quote error: {e}")

        # Try Alpha Vantage only if it remains configured as a fallback.
        if self.alpha_vantage_key and not self._is_rate_limited(DataSource.ALPHA_VANTAGE):
            try:
                return self._fetch_alpha_vantage_quote(symbol)
            except Exception as e:
                logger.error(f"Alpha Vantage quote error: {e}")
        
        # Fallback to Yahoo Finance
        try:
            return self._fetch_yahoo_quote(symbol)
        except Exception as e:
            logger.error(f"Yahoo Finance quote error: {e}")
            
        # Return empty dict if all fail
        return {}

    def _fetch_alpaca_quote(self, symbol: str) -> Dict[str, float]:
        """Fetch a latest minute-bar quote from Alpaca Market Data API."""
        headers = {
            "APCA-API-KEY-ID": self.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.alpaca_secret_key,
        }
        params = {"feed": self.alpaca_data_feed}
        url = f"{self.alpaca_data_base_url}/v2/stocks/{symbol}/bars/latest"

        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 429:
            raise ValueError("API rate limit: Alpaca returned 429")

        response.raise_for_status()
        payload = response.json()
        bar = payload.get("bar") or {}
        close_price = bar.get("c")

        if close_price:
            return {
                "current_price": float(close_price),
                "previous_close": 0.0,
                "change_percent": "0%",
            }

        raise ValueError("No latest bar data in Alpaca response")
    
    def _fetch_alpha_vantage_quote(self, symbol: str) -> Dict[str, float]:
        """Fetch quote from Alpha Vantage."""
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.alpha_vantage_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if 'Global Quote' in data and data['Global Quote']:
            quote = data['Global Quote']
            return {
                'current_price': float(quote.get('05. price', 0)),
                'previous_close': float(quote.get('08. previous close', 0)),
                'change_percent': quote.get('10. change percent', '0%')
            }
        
        raise ValueError("No quote data in response")
    
    def _fetch_yahoo_quote(self, symbol: str) -> Dict[str, float]:
        """Fetch quote from Yahoo Finance."""
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        previous_close = info.get('previousClose', 0)
        
        if current_price and previous_close:
            change_pct = ((current_price - previous_close) / previous_close) * 100
            return {
                'current_price': float(current_price),
                'previous_close': float(previous_close),
                'change_percent': f"{change_pct:.2f}%"
            }
        
        # Try fast_info as backup
        fast_info = ticker.fast_info
        if hasattr(fast_info, 'last_price'):
            return {
                'current_price': float(fast_info.last_price),
                'previous_close': float(fast_info.previous_close) if hasattr(fast_info, 'previous_close') else 0,
                'change_percent': '0%'
            }
        
        raise ValueError("No quote data available")


def get_candlestick_data(symbol: str, intervals: Tuple[str, ...] = ("1min", "15min", "1h"), 
                        outputsize: int = 500) -> Dict:
    """
    Main entry point for fetching candlestick data.
    Maintains backward compatibility while adding failover.
    """
    from core.candle_fetcher_provider import get_candlestick_fetcher

    result = get_candlestick_fetcher().fetch_with_failover(symbol, intervals, outputsize)
    
    if result['success']:
        logger.info(f"Successfully fetched data for {symbol} from {result['source']}")
        return result['data']
    else:
        logger.error(f"Failed to fetch data for {symbol}. Errors: {result['errors']}")
        # Return structure expected by pipeline
        return {
            'alpaca': {interval: [] for interval in intervals},
            'yahoo_finance': {interval: [] for interval in intervals},
            'twelvedata': {interval: [] for interval in intervals},
            'alphavantage': {interval: [] for interval in intervals}
        }


def fetch_realtime_quote(symbol: str, api_keys: Dict[str, str]) -> Dict:
    """Fetch current market quote with failover."""
    from core.candle_fetcher_provider import get_candlestick_fetcher

    return get_candlestick_fetcher().fetch_realtime_quote(symbol)


# Legacy functions for backward compatibility
def fetch_twelvedata_candles(symbol, intervals=("1min", "15min", "1h"), outputsize=100):
    """Legacy function - use get_candlestick_data instead."""
    logger.warning("fetch_twelvedata_candles is deprecated. Use get_candlestick_data instead.")
    fetcher = CandlestickDataFetcher()
    try:
        return fetcher._fetch_twelvedata_candles(symbol, intervals, outputsize)
    except Exception:
        return {interval: [] for interval in intervals}


def fetch_alphavantage_candles(symbol, intervals=("1min", "15min", "1h"), outputsize=100):
    """Legacy function - use get_candlestick_data instead."""
    logger.warning("fetch_alphavantage_candles is deprecated. Use get_candlestick_data instead.")
    fetcher = CandlestickDataFetcher()
    try:
        return fetcher._fetch_alphavantage_candles(symbol, intervals, outputsize)
    except Exception:
        return {interval: [] for interval in intervals}
