#!/usr/bin/env python3
"""
Enhanced candlestick data fetcher with robust multi-source failover.
Implements a professional-grade data acquisition system with proper error handling.
"""

import requests
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
import time
import random
import os
from collections import deque
from threading import Lock

from config import settings

logger = logging.getLogger(__name__)


class DataSource:
    """Enum-like class for data source priorities."""
    TWELVE_DATA = "twelve_data"
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO_FINANCE = "yahoo_finance"


class RateLimitConfig:
    """Configuration for rate limiting from environment variables."""
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
            logger.warning(f"{self.source_name}: Rate limited. Waiting {backoff_time:.1f}s until {self.rate_limit_until}")
    
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


class CandlestickDataFetcher:
    """
    Professional candlestick data fetcher with multi-source failover.
    Implements intelligent retry logic and caching for reliability.
    """
    
    def __init__(self):
        self.twelve_data_key = settings.TWELVE_DATA_API_KEY
        self.alpha_vantage_key = settings.ALPHA_VANTAGE_API_KEY
        self.source_priority = [
            DataSource.TWELVE_DATA,
            DataSource.ALPHA_VANTAGE,
            DataSource.YAHOO_FINANCE
        ]
        self._rate_limit_cache = {}
        
        # Initialize rate limiters
        self.rate_limiters = {
            DataSource.TWELVE_DATA: RateLimiter(
                RateLimitConfig.TWELVE_DATA_CALLS_PER_MINUTE,
                "TwelveData"
            ),
            DataSource.ALPHA_VANTAGE: RateLimiter(
                RateLimitConfig.ALPHA_VANTAGE_CALLS_PER_MINUTE,
                "AlphaVantage"
            )
        }
        
    def fetch_with_failover(self, symbol: str, intervals: Tuple[str, ...] = ("1min", "15min", "1h"), 
                           outputsize: int = 500) -> Dict[str, any]:
        """
        Fetch candlestick data with automatic failover between sources.
        
        Returns:
            Dict with structure:
            {
                'success': bool,
                'source': str,  # Which API provided the data
                'data': {
                    'twelvedata': {interval: [candles]},
                    'alphavantage': {interval: [candles]}
                },
                'errors': List[str]  # Any errors encountered
            }
        """
        errors = []
        successful_source = None
        result_data = {
            'twelvedata': {},
            'alphavantage': {}
        }
        
        for source in self.source_priority:
            try:
                if self._is_rate_limited(source):
                    errors.append(f"{source}: Rate limited (cached)")
                    continue
                
                logger.info(f"Attempting to fetch data from {source} for {symbol}")
                
                if source == DataSource.TWELVE_DATA:
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
                        # Convert to both formats for compatibility
                        result_data['twelvedata'] = data
                        result_data['alphavantage'] = data
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
            
            # Retry logic with rate limiting
            for attempt in range(RateLimitConfig.TWELVE_DATA_MAX_RETRIES):
                try:
                    # Check if we need to wait due to rate limiting
                    wait_time = rate_limiter.get_wait_time()
                    if wait_time:
                        logger.info(f"TwelveData: Waiting {wait_time:.1f}s due to rate limit...")
                        time.sleep(wait_time)
                    
                    # Check if we can make a request
                    if not rate_limiter.can_make_request():
                        logger.warning(f"TwelveData: Approaching rate limit, waiting 60s...")
                        time.sleep(60)
                    
                    # Record the request
                    rate_limiter.record_request()
                    
                    # Make the API request
                    resp = requests.get(base_url, params=params, timeout=10)
                    resp.raise_for_status()
                    result = resp.json()
                    
                    # Check for API errors
                    if result.get('status') == 'error':
                        error_msg = result.get('message', 'Unknown error')
                        
                        # Check if it's a rate limit error
                        if 'api message' in error_msg.lower() or 'api credits' in error_msg.lower():
                            # Extract wait time if provided
                            wait_seconds = None
                            if 'wait for the next minute' in error_msg.lower():
                                wait_seconds = 60
                            
                            rate_limiter.record_rate_limit(wait_seconds)
                            
                            if attempt < RateLimitConfig.TWELVE_DATA_MAX_RETRIES - 1:
                                wait_time = rate_limiter.get_wait_time()
                                logger.warning(f"TwelveData rate limit hit: {error_msg}. Waiting {wait_time:.1f}s before retry...")
                                time.sleep(wait_time)
                                continue
                            else:
                                raise ValueError(f"API rate limit: {error_msg}")
                        else:
                            raise ValueError(f"API error: {error_msg}")
                    
                    # Success - reset consecutive limits counter
                    rate_limiter.reset_consecutive_limits()
                    
                    if "values" in result and result["values"]:
                        data[interval] = result["values"]
                    else:
                        data[interval] = []
                    
                    break  # Success, exit retry loop
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"Network error fetching {interval} from Twelve Data: {e}")
                    if attempt < RateLimitConfig.TWELVE_DATA_MAX_RETRIES - 1:
                        time.sleep(5)  # Brief wait before network retry
                        continue
                    data[interval] = []
                    break
                except ValueError as e:
                    # Re-raise ValueError (API errors) after logging
                    logger.error(f"Error fetching {interval} from Twelve Data: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error fetching {interval} from Twelve Data: {e}")
                    if attempt < RateLimitConfig.TWELVE_DATA_MAX_RETRIES - 1:
                        time.sleep(5)
                        continue
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
            
            # Retry logic with rate limiting
            for attempt in range(RateLimitConfig.ALPHA_VANTAGE_MAX_RETRIES):
                try:
                    # Check if we need to wait due to rate limiting
                    wait_time = rate_limiter.get_wait_time()
                    if wait_time:
                        logger.info(f"AlphaVantage: Waiting {wait_time:.1f}s due to rate limit...")
                        time.sleep(wait_time)
                    
                    # Check if we can make a request
                    if not rate_limiter.can_make_request():
                        logger.warning(f"AlphaVantage: Approaching rate limit, waiting 60s...")
                        time.sleep(60)
                    
                    # Record the request
                    rate_limiter.record_request()
                    
                    # Make the API request
                    resp = requests.get(base_url, params=params, timeout=10)
                    resp.raise_for_status()
                    result = resp.json()
                    
                    # Check for rate limiting
                    if 'Information' in result and 'api message' in result['Information'].lower():
                        rate_limiter.record_rate_limit(RateLimitConfig.ALPHA_VANTAGE_RATE_LIMIT_WAIT)
                        
                        if attempt < RateLimitConfig.ALPHA_VANTAGE_MAX_RETRIES - 1:
                            wait_time = rate_limiter.get_wait_time()
                            logger.warning(f"AlphaVantage rate limit hit: {result['Information']}. Waiting {wait_time:.1f}s before retry...")
                            time.sleep(wait_time)
                            continue
                        else:
                            raise ValueError(f"API rate limit: {result['Information']}")
                    
                    # Check for errors
                    if 'Error Message' in result:
                        raise ValueError(f"API error: {result['Error Message']}")
                    
                    # Success - reset consecutive limits counter
                    rate_limiter.reset_consecutive_limits()
                    
                    key = f"Time Series ({av_interval})"
                    if key in result:
                        # Convert to standard format
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
                    
                    break  # Success, exit retry loop
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"Network error fetching {interval} from Alpha Vantage: {e}")
                    if attempt < RateLimitConfig.ALPHA_VANTAGE_MAX_RETRIES - 1:
                        time.sleep(5)  # Brief wait before network retry
                        continue
                    data[interval] = []
                    break
                except ValueError as e:
                    # Re-raise ValueError (API errors) after logging
                    logger.error(f"Error fetching {interval} from Alpha Vantage: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error fetching {interval} from Alpha Vantage: {e}")
                    if attempt < RateLimitConfig.ALPHA_VANTAGE_MAX_RETRIES - 1:
                        time.sleep(5)
                        continue
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
        # Try Alpha Vantage first
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


# Global fetcher instance
_fetcher = CandlestickDataFetcher()


def get_candlestick_data(symbol: str, intervals: Tuple[str, ...] = ("1min", "15min", "1h"), 
                        outputsize: int = 500) -> Dict:
    """
    Main entry point for fetching candlestick data.
    Maintains backward compatibility while adding failover.
    """
    result = _fetcher.fetch_with_failover(symbol, intervals, outputsize)
    
    if result['success']:
        logger.info(f"Successfully fetched data for {symbol} from {result['source']}")
        return result['data']
    else:
        logger.error(f"Failed to fetch data for {symbol}. Errors: {result['errors']}")
        # Return structure expected by pipeline
        return {
            'twelvedata': {interval: [] for interval in intervals},
            'alphavantage': {interval: [] for interval in intervals}
        }


def fetch_realtime_quote(symbol: str, api_keys: Dict[str, str]) -> Dict:
    """Fetch current market quote with failover."""
    return _fetcher.fetch_realtime_quote(symbol)


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
