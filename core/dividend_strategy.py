#!/usr/bin/env python3
"""
Dividend Capture Strategy Module for APAM and other dividend stocks.

This module implements a sophisticated approach to trading around dividend events,
learning from historical patterns to optimize entry and exit timing. Since you maintain
a core position, we'll use a "core and explore" strategy - keeping your base 100 shares
while trading additional shares for dividend capture.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
from dataclasses import dataclass
import logging
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

logger = logging.getLogger(__name__)


@dataclass
class DividendEvent:
    """Represents a single dividend event with all relevant dates and amounts."""
    symbol: str
    declaration_date: Optional[datetime]
    ex_dividend_date: datetime
    record_date: datetime
    payment_date: datetime
    dividend_amount: float
    dividend_yield: float
    frequency: str  # 'quarterly', 'monthly', 'annual'
    
    @property
    def days_until_ex_dividend(self) -> int:
        """Calculate days remaining until ex-dividend date."""
        return (self.ex_dividend_date - datetime.now()).days
    
    @property
    def capture_window(self) -> Tuple[datetime, datetime]:
        """Define the optimal trading window for dividend capture."""
        # Start looking 7 days before ex-dividend
        start_date = self.ex_dividend_date - timedelta(days=7)
        # Exit window extends 3 days after ex-dividend
        end_date = self.ex_dividend_date + timedelta(days=3)
        return start_date, end_date


class DividendPatternAnalyzer:
    """
    Analyzes historical price patterns around dividend events to identify
    optimal entry and exit points for dividend capture strategies.
    """
    
    def __init__(self, symbol: str, lookback_years: int = 3):
        self.symbol = symbol
        self.lookback_years = lookback_years
        self.patterns = {}
        self.ml_model = None
        self.scaler = StandardScaler()
        self.model_path = f"models/dividend_{symbol}_model.pkl"
        self.scaler_path = f"models/dividend_{symbol}_scaler.pkl"
        
    def analyze_historical_patterns(self, price_data: pd.DataFrame, 
                                  dividend_history: List[DividendEvent]) -> Dict:
        """
        Analyze price movements around historical dividend events.
        
        This function examines how the stock price behaved before, during,
        and after each dividend event to identify repeatable patterns.
        """
        patterns = {
            'avg_pre_dividend_runup': [],
            'avg_ex_dividend_drop': [],
            'avg_recovery_time': [],
            'optimal_entry_days': [],
            'optimal_exit_days': [],
            'success_rate': 0,
            'market_condition_impact': {}
        }
        
        for dividend in dividend_history:
            # Extract price data around the dividend event
            window_start = dividend.ex_dividend_date - timedelta(days=15)
            window_end = dividend.ex_dividend_date + timedelta(days=10)
            
            event_prices = price_data[
                (price_data.index >= window_start) & 
                (price_data.index <= window_end)
            ].copy()
            
            if len(event_prices) < 20:
                continue  # Skip if insufficient data
                
            # Calculate key metrics for this dividend event
            pre_div_prices = event_prices[event_prices.index < dividend.ex_dividend_date]
            ex_div_price = event_prices[event_prices.index == dividend.ex_dividend_date]['close'].iloc[0]
            post_div_prices = event_prices[event_prices.index > dividend.ex_dividend_date]
            
            # Measure pre-dividend run-up (percentage increase in 7 days before ex-div)
            if len(pre_div_prices) >= 7:
                seven_days_before = pre_div_prices['close'].iloc[-7]
                day_before = pre_div_prices['close'].iloc[-1]
                runup = ((day_before - seven_days_before) / seven_days_before) * 100
                patterns['avg_pre_dividend_runup'].append(runup)
                
            # Measure ex-dividend drop
            day_before_ex = pre_div_prices['close'].iloc[-1]
            actual_drop = ((ex_div_price - day_before_ex) / day_before_ex) * 100
            theoretical_drop = (dividend.dividend_amount / day_before_ex) * 100
            excess_drop = actual_drop - (-theoretical_drop)  # Negative because it's a drop
            patterns['avg_ex_dividend_drop'].append(excess_drop)
            
            # Find optimal entry point (lowest price in pre-dividend window)
            if len(pre_div_prices) >= 7:
                last_seven = pre_div_prices.tail(7)
                min_price_day = (dividend.ex_dividend_date - last_seven['close'].idxmin()).days
                patterns['optimal_entry_days'].append(min_price_day)
                
            # Analyze recovery pattern
            if len(post_div_prices) >= 5:
                recovery_threshold = day_before_ex - dividend.dividend_amount
                recovery_days = 0
                for i, (date, row) in enumerate(post_div_prices.iterrows()):
                    if row['close'] >= recovery_threshold:
                        recovery_days = i + 1
                        break
                patterns['avg_recovery_time'].append(recovery_days if recovery_days > 0 else 5)
        
        # Calculate summary statistics
        patterns['avg_pre_dividend_runup'] = np.mean(patterns['avg_pre_dividend_runup'])
        patterns['avg_ex_dividend_drop'] = np.mean(patterns['avg_ex_dividend_drop'])
        patterns['avg_recovery_time'] = np.mean(patterns['avg_recovery_time'])
        patterns['optimal_entry_days'] = int(np.median(patterns['optimal_entry_days']))
        patterns['optimal_exit_days'] = int(np.median(patterns['avg_recovery_time']))
        
        # Calculate success rate (profitable dividend captures)
        profitable_captures = sum(1 for drop in patterns['avg_ex_dividend_drop'] if drop < 0)
        patterns['success_rate'] = profitable_captures / len(patterns['avg_ex_dividend_drop'])
        
        self.patterns = patterns
        return patterns
    
    def train_ml_model(self, price_data: pd.DataFrame, 
                      dividend_history: List[DividendEvent],
                      market_data: pd.DataFrame) -> None:
        """
        Train a machine learning model to predict optimal entry/exit points
        based on market conditions and historical patterns.
        """
        features_list = []
        targets_list = []
        
        for dividend in dividend_history:
            # Create feature vector for each dividend event
            features = self._extract_features(dividend, price_data, market_data)
            if features is not None:
                # Target is the best profit percentage achievable in the capture window
                target = self._calculate_optimal_profit(dividend, price_data)
                features_list.append(features)
                targets_list.append(target)
        
        if len(features_list) < 10:
            logger.warning(f"Insufficient data to train ML model for {self.symbol}")
            return
            
        X = np.array(features_list)
        y = np.array(targets_list)
        
        # Scale features for better model performance
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest model - good for capturing non-linear patterns
        self.ml_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.ml_model.fit(X_scaled, y)
        
        # Save model and scaler for future use
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.ml_model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        logger.info(f"Trained dividend capture model for {self.symbol} with score: "
                   f"{self.ml_model.score(X_scaled, y):.3f}")
    
    def _extract_features(self, dividend: DividendEvent, 
                         price_data: pd.DataFrame,
                         market_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features for ML model from dividend event and market conditions."""
        try:
            ex_date = dividend.ex_dividend_date
            
            # Price-based features
            recent_prices = price_data[price_data.index < ex_date].tail(20)
            if len(recent_prices) < 20:
                return None
                
            price_features = [
                recent_prices['close'].pct_change().mean(),  # Recent momentum
                recent_prices['close'].std() / recent_prices['close'].mean(),  # Volatility
                (recent_prices['close'].iloc[-1] - recent_prices['close'].iloc[0]) / recent_prices['close'].iloc[0],  # 20-day return
                recent_prices['volume'].mean(),  # Average volume
                dividend.dividend_yield,  # Current yield
                dividend.dividend_amount / recent_prices['close'].iloc[-1],  # Dividend as % of price
            ]
            
            # Market condition features
            market_features = []
            if market_data is not None and len(market_data) > 0:
                recent_market = market_data[market_data.index < ex_date].tail(20)
                if len(recent_market) >= 20:
                    market_features = [
                        recent_market['close'].pct_change().mean(),  # Market momentum
                        recent_market['close'].std() / recent_market['close'].mean(),  # Market volatility
                    ]
                else:
                    market_features = [0, 0]  # Default values
            else:
                market_features = [0, 0]
                
            # Calendar features
            calendar_features = [
                ex_date.weekday(),  # Day of week (0=Monday, 4=Friday)
                ex_date.month,  # Month of year
                1 if ex_date.weekday() == 0 else 0,  # Is Monday (special behavior)
                1 if ex_date.weekday() == 4 else 0,  # Is Friday (special behavior)
            ]
            
            return np.array(price_features + market_features + calendar_features)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _calculate_optimal_profit(self, dividend: DividendEvent, 
                                 price_data: pd.DataFrame) -> float:
        """
        Calculate the maximum profit achievable for a dividend capture trade.
        This serves as the target variable for our ML model.
        """
        window_start, window_end = dividend.capture_window
        window_prices = price_data[
            (price_data.index >= window_start) & 
            (price_data.index <= window_end)
        ]
        
        if len(window_prices) < 5:
            return 0.0
            
        # Find best entry price (lowest in pre-dividend window)
        pre_div_prices = window_prices[window_prices.index < dividend.ex_dividend_date]
        if len(pre_div_prices) == 0:
            return 0.0
            
        best_entry_price = pre_div_prices['close'].min()
        
        # Find best exit price (highest in post-dividend window)
        post_div_prices = window_prices[window_prices.index >= dividend.ex_dividend_date]
        if len(post_div_prices) == 0:
            return 0.0
            
        best_exit_price = post_div_prices['close'].max()
        
        # Calculate profit including dividend
        profit_per_share = (best_exit_price - best_entry_price) + dividend.dividend_amount
        profit_percentage = (profit_per_share / best_entry_price) * 100
        
        return profit_percentage


class DividendCaptureStrategy:
    """
    Main strategy class that generates trading signals for dividend capture.
    Integrates pattern analysis with real-time decision making.
    """
    
    def __init__(self, symbol: str, core_position: int = 100):
        self.symbol = symbol
        self.core_position = core_position  # Shares always held
        self.capture_position = 0  # Additional shares for dividend capture
        self.analyzer = DividendPatternAnalyzer(symbol)
        self.upcoming_dividends = []
        self.active_capture = None
        
    def update_dividend_calendar(self, dividends: List[DividendEvent]) -> None:
        """Update the list of upcoming dividends."""
        self.upcoming_dividends = [
            d for d in dividends 
            if d.ex_dividend_date > datetime.now()
        ]
        self.upcoming_dividends.sort(key=lambda x: x.ex_dividend_date)
        
    def generate_signals(self, current_price: float, 
                        price_history: pd.DataFrame,
                        market_data: pd.DataFrame) -> Dict[str, any]:
        """
        Generate trading signals based on dividend events and market conditions.
        
        Returns a dictionary with:
        - signal: 'BUY', 'SELL', 'HOLD'
        - quantity: Number of shares to trade
        - reason: Explanation for the signal
        - confidence: 0-1 score indicating signal strength
        """
        if not self.upcoming_dividends:
            return {
                'signal': 'HOLD',
                'quantity': 0,
                'reason': 'No upcoming dividends',
                'confidence': 0.0
            }
        
        next_dividend = self.upcoming_dividends[0]
        days_to_ex_div = next_dividend.days_until_ex_dividend
        
        # Check if we're in the capture window
        window_start, window_end = next_dividend.capture_window
        current_date = datetime.now()
        
        if window_start <= current_date <= window_end:
            # We're in the dividend capture window
            if current_date < next_dividend.ex_dividend_date:
                # Pre-dividend: Consider buying additional shares
                return self._generate_entry_signal(
                    next_dividend, current_price, price_history, market_data
                )
            else:
                # Post-dividend: Consider selling capture position
                return self._generate_exit_signal(
                    next_dividend, current_price, price_history, market_data
                )
        else:
            # Outside capture window
            return {
                'signal': 'HOLD',
                'quantity': 0,
                'reason': f'Outside capture window. Next dividend in {days_to_ex_div} days',
                'confidence': 0.0
            }
    
    def _generate_entry_signal(self, dividend: DividendEvent,
                              current_price: float,
                              price_history: pd.DataFrame,
                              market_data: pd.DataFrame) -> Dict[str, any]:
        """Generate entry signal for dividend capture position."""
        # Use ML model if available, otherwise use rule-based approach
        if self.analyzer.ml_model is not None:
            features = self.analyzer._extract_features(dividend, price_history, market_data)
            if features is not None:
                features_scaled = self.analyzer.scaler.transform([features])
                predicted_profit = self.analyzer.ml_model.predict(features_scaled)[0]
                
                # Generate signal based on predicted profit
                if predicted_profit > 1.0:  # Expecting >1% profit
                    confidence = min(predicted_profit / 3.0, 1.0)  # Scale confidence
                    quantity = self._calculate_capture_size(current_price, confidence)
                    
                    return {
                        'signal': 'BUY',
                        'quantity': quantity,
                        'reason': f'ML model predicts {predicted_profit:.2f}% profit from dividend capture',
                        'confidence': confidence,
                        'entry_type': 'DIVIDEND_CAPTURE',
                        'expected_profit': predicted_profit
                    }
        
        # Fallback to rule-based approach
        days_to_ex = dividend.days_until_ex_dividend
        
        if days_to_ex <= self.analyzer.patterns.get('optimal_entry_days', 5):
            # Check if price is attractive relative to recent average
            avg_price = price_history['close'].tail(20).mean()
            if current_price < avg_price * 0.99:  # 1% below average
                confidence = 0.7
                quantity = self._calculate_capture_size(current_price, confidence)
                
                return {
                    'signal': 'BUY',
                    'quantity': quantity,
                    'reason': f'Approaching optimal entry window ({days_to_ex} days to ex-dividend)',
                    'confidence': confidence,
                    'entry_type': 'DIVIDEND_CAPTURE'
                }
        
        return {
            'signal': 'HOLD',
            'quantity': 0,
            'reason': f'Waiting for better entry point ({days_to_ex} days to ex-dividend)',
            'confidence': 0.3
        }
    
    def _generate_exit_signal(self, dividend: DividendEvent,
                             current_price: float,
                             price_history: pd.DataFrame,
                             market_data: pd.DataFrame) -> Dict[str, any]:
        """Generate exit signal for dividend capture position."""
        if self.capture_position == 0:
            return {
                'signal': 'HOLD',
                'quantity': 0,
                'reason': 'No capture position to exit',
                'confidence': 0.0
            }
        
        days_since_ex = (datetime.now() - dividend.ex_dividend_date).days
        
        # Check if we've reached optimal exit time
        optimal_exit_days = self.analyzer.patterns.get('optimal_exit_days', 2)
        
        if days_since_ex >= optimal_exit_days:
            return {
                'signal': 'SELL',
                'quantity': self.capture_position,
                'reason': f'Reached optimal exit window ({days_since_ex} days post ex-dividend)',
                'confidence': 0.8,
                'exit_type': 'DIVIDEND_CAPTURE_COMPLETE'
            }
        
        # Check for early exit if price has recovered
        ex_div_price = price_history[
            price_history.index == dividend.ex_dividend_date
        ]['close'].iloc[0]
        recovery_target = ex_div_price + (dividend.dividend_amount * 0.5)  # 50% recovery
        
        if current_price >= recovery_target:
            return {
                'signal': 'SELL',
                'quantity': self.capture_position,
                'reason': 'Price recovered to target level',
                'confidence': 0.9,
                'exit_type': 'DIVIDEND_CAPTURE_EARLY'
            }
        
        return {
            'signal': 'HOLD',
            'quantity': 0,
            'reason': f'Waiting for price recovery (day {days_since_ex} post ex-dividend)',
            'confidence': 0.4
        }
    
    def _calculate_capture_size(self, current_price: float, 
                               confidence: float) -> int:
        """
        Calculate how many additional shares to buy for dividend capture.
        This is separate from your core position of 100 shares.
        """
        # Base capture size on confidence and available capital
        max_capture_shares = 50  # Maximum additional shares for capture
        
        # Scale by confidence
        capture_shares = int(max_capture_shares * confidence)
        
        # Ensure we're buying in round lots if preferred
        capture_shares = (capture_shares // 10) * 10  # Round to nearest 10
        
        return max(capture_shares, 10)  # Minimum 10 shares for capture
    
    def update_positions(self, signal: Dict[str, any]) -> None:
        """Update internal position tracking based on executed signals."""
        if signal['signal'] == 'BUY' and signal.get('entry_type') == 'DIVIDEND_CAPTURE':
            self.capture_position += signal['quantity']
            self.active_capture = {
                'entry_date': datetime.now(),
                'entry_price': signal.get('execution_price'),
                'quantity': signal['quantity']
            }
        elif signal['signal'] == 'SELL' and 'DIVIDEND_CAPTURE' in signal.get('exit_type', ''):
            self.capture_position = 0
            if self.active_capture:
                # Log the completed capture for analysis
                self.active_capture['exit_date'] = datetime.now()
                self.active_capture['exit_price'] = signal.get('execution_price')
                self.active_capture = None


class DividendDataFetcher:
    """
    Fetches dividend data from various sources and maintains a local cache.
    This ensures we always have up-to-date dividend information.
    """
    
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.cache_file = 'data/dividend_cache.json'
        self.cache = self._load_cache()
        self.twelve_data_premium = None  # Will be auto-detected
        self._api_tier_detected = False
        self._detect_api_tier()
        
    def _detect_api_tier(self) -> None:
        """Detect Twelve Data API tier to avoid premium endpoint errors."""
        try:
            from config.settings import trading_config
            
            # Check if user has manually configured premium status
            if not trading_config.TWELVE_DATA_AUTO_DETECT:
                self.twelve_data_premium = trading_config.TWELVE_DATA_PREMIUM
                self._api_tier_detected = True
                logger.info(f"Twelve Data API tier manually configured: {'Premium' if self.twelve_data_premium else 'Free'}")
                return
            
            # Check cache for previous detection
            if 'api_tier_detection' in self.cache:
                cached_detection = self.cache['api_tier_detection']
                cache_date = datetime.fromisoformat(cached_detection['timestamp'])
                # Cache API tier detection for 24 hours
                time_diff = datetime.now() - cache_date
                if time_diff.total_seconds() < 86400:  # 24 hours in seconds
                    self.twelve_data_premium = cached_detection['premium']
                    self._api_tier_detected = True
                    logger.info(f"Using cached Twelve Data API tier: {'Premium' if self.twelve_data_premium else 'Free'}")
                    return
            
            # Auto-detect by trying a simple endpoint
            self._auto_detect_twelve_data_tier()
            
        except Exception as e:
            logger.warning(f"Error detecting API tier: {e}")
            # Default to free tier to avoid errors
            self.twelve_data_premium = False
            self._api_tier_detected = True
    
    def _auto_detect_twelve_data_tier(self) -> None:
        """Auto-detect Twelve Data API tier by testing endpoint access."""
        try:
            # Try a simple quote endpoint first to test basic API access
            url = 'https://api.twelvedata.com/quote'
            params = {
                'symbol': 'AAPL',  # Use a common stock for testing
                'apikey': self.api_keys.get('TWELVE_DATA_API_KEY')
            }
            
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            # Check if basic API access works
            if 'status' in data and data['status'] == 'error':
                if 'api key' in data.get('message', '').lower():
                    logger.warning("Twelve Data API key invalid or missing")
                    self.twelve_data_premium = False
                else:
                    # API key works, assume free tier (we'll skip premium endpoints)
                    self.twelve_data_premium = False
            else:
                # Basic API works, assume free tier for safety
                # We don't test premium endpoints to avoid wasting API calls
                self.twelve_data_premium = False
            
            # Cache the detection result
            self.cache['api_tier_detection'] = {
                'timestamp': datetime.now().isoformat(),
                'premium': self.twelve_data_premium
            }
            self._save_cache()
            self._api_tier_detected = True
            
            logger.info(f"Auto-detected Twelve Data API tier: {'Premium' if self.twelve_data_premium else 'Free'}")
            
        except Exception as e:
            logger.warning(f"Error auto-detecting Twelve Data API tier: {e}")
            # Default to free tier
            self.twelve_data_premium = False
            self._api_tier_detected = True
    
    def _load_cache(self) -> Dict:
        """Load dividend data cache from disk."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self) -> None:
        """Save dividend data cache to disk."""
        os.makedirs('data', exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2, default=str)
    
    def fetch_dividend_history(self, symbol: str, years: int = 3) -> List[DividendEvent]:
        """
        Fetch historical dividend data for the specified symbol.
        Uses dual-API system: Twelve Data (primary) -> Alpha Vantage (fallback) with caching.
        """
        cache_key = f"{symbol}_history_{years}y"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            cache_date = datetime.fromisoformat(cached_data['timestamp'])
            if (datetime.now() - cache_date).days < 7:  # Cache for 7 days
                return self._parse_cached_dividends(cached_data['data'])
        
        # Smart API selection based on detected tier
        if self.twelve_data_premium:
            # Try Twelve Data premium endpoints
            dividends = self._fetch_from_twelve_data(symbol, years)
            if dividends:
                logger.info(f"Successfully fetched dividend data from Twelve Data Premium for {symbol}")
                # Cache the results
                self.cache[cache_key] = {
                    'timestamp': datetime.now().isoformat(),
                    'data': [self._dividend_to_dict(d) for d in dividends],
                    'source': 'twelve_data'
                }
                self._save_cache()
                return dividends
        else:
            # Skip Twelve Data premium endpoints for free users
            logger.info(f"Using free data sources for dividend information on {symbol}")
        
        # Try Alpha Vantage (free tier)
        dividends = self._fetch_from_alpha_vantage(symbol, years)
        if dividends:
            logger.info(f"Successfully fetched dividend data from Alpha Vantage for {symbol}")
            # Cache the results
            self.cache[cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'data': [self._dividend_to_dict(d) for d in dividends],
                'source': 'alpha_vantage'
            }
            self._save_cache()
            return dividends
        
        # Final fallback to Yahoo Finance (free, unlimited)
        dividends = self._fetch_from_yahoo_finance(symbol, years)
        if dividends:
            logger.info(f"Successfully retrieved dividend information from Yahoo Finance for {symbol}")
            # Cache the results
            self.cache[cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'data': [self._dividend_to_dict(d) for d in dividends],
                'source': 'yahoo_finance'
            }
            self._save_cache()
            return dividends
        
        logger.warning(f"No dividend data available for {symbol}")
        return []
    
    def _fetch_from_twelve_data(self, symbol: str, years: int) -> List[DividendEvent]:
        """Fetch dividend data from Twelve Data API."""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            url = 'https://api.twelvedata.com/dividends'
            params = {
                'symbol': symbol,
                'apikey': self.api_keys.get('TWELVE_DATA_API_KEY'),
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            # Check for API errors
            if 'status' in data and data['status'] == 'error':
                logger.error(f"Twelve Data API error: {data.get('message', 'Unknown error')}")
                return []
            
            if 'dividends' in data and data['dividends']:
                return self._parse_twelve_data_dividends(data['dividends'], symbol)
            else:
                logger.warning(f"No dividend data found in Twelve Data response for {symbol}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching from Twelve Data: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching dividend data from Twelve Data: {e}")
            return []
    
    def _fetch_from_alpha_vantage(self, symbol: str, years: int) -> List[DividendEvent]:
        """Fetch dividend data from Alpha Vantage API (fallback)."""
        try:
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'TIME_SERIES_MONTHLY_ADJUSTED',
                'symbol': symbol,
                'apikey': self.api_keys.get('ALPHA_VANTAGE_API_KEY')
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            # Check for rate limiting or errors
            if 'Information' in data:
                logger.error(f"Alpha Vantage API message: {data['Information']}")
                return []
            
            if 'Monthly Adjusted Time Series' in data:
                return self._parse_dividend_data(data, symbol)
            else:
                logger.error(f"Failed to fetch dividend data from Alpha Vantage: {data}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching from Alpha Vantage: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching dividend data from Alpha Vantage: {e}")
            return []
    
    def _fetch_from_yahoo_finance(self, symbol: str, years: int) -> List[DividendEvent]:
        """Fetch dividend data from Yahoo Finance (free fallback)."""
        try:
            import yfinance as yf
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Get dividend data
            dividends_df = ticker.dividends
            
            if dividends_df.empty:
                logger.warning(f"No dividend data found in Yahoo Finance for {symbol}")
                return []
            
            # Filter to requested years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            # Convert to timezone-naive datetime for comparison
            if dividends_df.index.tz is not None:
                # If the index is timezone-aware, convert to UTC then remove timezone
                dividends_df.index = dividends_df.index.tz_convert('UTC').tz_localize(None)
            
            # Filter dividends within date range
            recent_dividends = dividends_df[dividends_df.index >= start_date]
            
            if recent_dividends.empty:
                logger.warning(f"No recent dividend data found in Yahoo Finance for {symbol}")
                return []
            
            return self._parse_yahoo_finance_dividends(recent_dividends, symbol)
            
        except ImportError:
            logger.error("yfinance library not installed. Please install with: pip install yfinance")
            return []
        except Exception as e:
            logger.error(f"Error fetching dividend data from Yahoo Finance: {e}")
            return []
    
    def _parse_yahoo_finance_dividends(self, dividends_df: pd.DataFrame, symbol: str) -> List[DividendEvent]:
        """Parse dividend data from Yahoo Finance response."""
        dividends = []
        
        # Get stock info for additional context
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 100))
        except:
            current_price = 100  # Fallback price
        
        for date, dividend_amount in dividends_df.items():
            try:
                # Convert pandas timestamp to datetime
                ex_date = date.to_pydatetime()
                
                # Estimate frequency based on dividend history
                frequency = self._estimate_dividend_frequency(dividends_df)
                
                # Calculate annualized yield
                annual_dividend = dividend_amount * self._get_frequency_multiplier(frequency)
                dividend_yield = (annual_dividend / current_price) * 100 if current_price > 0 else 0
                
                dividend_event = DividendEvent(
                    symbol=symbol,
                    declaration_date=None,  # Yahoo Finance doesn't provide declaration date
                    ex_dividend_date=ex_date,
                    record_date=ex_date + timedelta(days=1),  # Approximation
                    payment_date=ex_date + timedelta(days=30),  # Approximation
                    dividend_amount=float(dividend_amount),
                    dividend_yield=dividend_yield,
                    frequency=frequency
                )
                dividends.append(dividend_event)
                
            except Exception as e:
                logger.warning(f"Error parsing Yahoo Finance dividend data: {e}")
                continue
        
        return dividends
    
    def _estimate_dividend_frequency(self, dividends_df: pd.DataFrame) -> str:
        """Estimate dividend frequency based on historical data."""
        if len(dividends_df) < 2:
            return 'quarterly'  # Default assumption
        
        # Calculate average days between dividends
        dates = dividends_df.index.sort_values()
        intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        avg_interval = sum(intervals) / len(intervals)
        
        # Classify based on average interval
        if avg_interval < 45:
            return 'monthly'
        elif avg_interval < 120:
            return 'quarterly'
        elif avg_interval < 200:
            return 'semi-annual'
        else:
            return 'annual'
    
    def _get_frequency_multiplier(self, frequency: str) -> int:
        """Get multiplier to calculate annual dividend from single payment."""
        frequency_map = {
            'monthly': 12,
            'quarterly': 4,
            'semi-annual': 2,
            'annual': 1
        }
        return frequency_map.get(frequency, 4)  # Default to quarterly
    
    def _parse_twelve_data_dividends(self, dividends_data: List[Dict], symbol: str) -> List[DividendEvent]:
        """Parse dividend data from Twelve Data API response."""
        dividends = []
        
        for dividend_info in dividends_data:
            try:
                # Parse dates
                ex_date = datetime.strptime(dividend_info['ex_date'], '%Y-%m-%d')
                
                # Twelve Data provides more accurate dividend information
                dividend_event = DividendEvent(
                    symbol=symbol,
                    declaration_date=datetime.strptime(dividend_info['declaration_date'], '%Y-%m-%d') if dividend_info.get('declaration_date') else None,
                    ex_dividend_date=ex_date,
                    record_date=datetime.strptime(dividend_info['record_date'], '%Y-%m-%d') if dividend_info.get('record_date') else ex_date + timedelta(days=1),
                    payment_date=datetime.strptime(dividend_info['payment_date'], '%Y-%m-%d') if dividend_info.get('payment_date') else ex_date + timedelta(days=30),
                    dividend_amount=float(dividend_info['amount']),
                    dividend_yield=float(dividend_info.get('yield', 0)) if dividend_info.get('yield') else 0,
                    frequency=dividend_info.get('frequency', 'quarterly').lower()
                )
                dividends.append(dividend_event)
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Error parsing dividend data from Twelve Data: {e}")
                continue
        
        return dividends
    
    def _parse_dividend_data(self, data: Dict, symbol: str) -> List[DividendEvent]:
        """Parse dividend data from Alpha Vantage response."""
        dividends = []
        monthly_data = data.get('Monthly Adjusted Time Series', {})
        
        # Extract dividend amounts from adjusted close vs close differences
        for date_str, values in monthly_data.items():
            date = datetime.strptime(date_str, '%Y-%m-%d')
            close = float(values['4. close'])
            adjusted_close = float(values['5. adjusted close'])
            dividend = float(values['7. dividend amount'])
            
            if dividend > 0:
                # Create dividend event
                # Note: Alpha Vantage provides limited dividend date info
                # In production, you'd want to use a more comprehensive data source
                dividend_event = DividendEvent(
                    symbol=symbol,
                    declaration_date=None,  # Not provided by this API
                    ex_dividend_date=date,  # Approximation
                    record_date=date + timedelta(days=1),  # Approximation
                    payment_date=date + timedelta(days=30),  # Approximation
                    dividend_amount=dividend,
                    dividend_yield=(dividend / close) * 4 * 100,  # Annualized yield
                    frequency='quarterly'
                )
                dividends.append(dividend_event)
        
        return dividends
    
    def _dividend_to_dict(self, dividend: DividendEvent) -> Dict:
        """Convert DividendEvent to dictionary for caching."""
        return {
            'symbol': dividend.symbol,
            'declaration_date': dividend.declaration_date.isoformat() if dividend.declaration_date else None,
            'ex_dividend_date': dividend.ex_dividend_date.isoformat(),
            'record_date': dividend.record_date.isoformat(),
            'payment_date': dividend.payment_date.isoformat(),
            'dividend_amount': dividend.dividend_amount,
            'dividend_yield': dividend.dividend_yield,
            'frequency': dividend.frequency
        }
    
    def _parse_cached_dividends(self, cached_data: List[Dict]) -> List[DividendEvent]:
        """Parse cached dividend data back into DividendEvent objects."""
        dividends = []
        for item in cached_data:
            dividend = DividendEvent(
                symbol=item['symbol'],
                declaration_date=datetime.fromisoformat(item['declaration_date']) if item['declaration_date'] else None,
                ex_dividend_date=datetime.fromisoformat(item['ex_dividend_date']),
                record_date=datetime.fromisoformat(item['record_date']),
                payment_date=datetime.fromisoformat(item['payment_date']),
                dividend_amount=item['dividend_amount'],
                dividend_yield=item['dividend_yield'],
                frequency=item['frequency']
            )
            dividends.append(dividend)
        return dividends
    
    def fetch_next_dividend(self, symbol: str) -> Optional[DividendEvent]:
        """
        Fetch information about the next upcoming dividend.
        Uses multiple strategies to find upcoming dividend information.
        """
        # Try to get historical data to estimate next dividend
        history = self.fetch_dividend_history(symbol, years=2)  # Look back 2 years for better pattern
        
        if history:
            # Find the most recent dividend
            history.sort(key=lambda x: x.ex_dividend_date, reverse=True)
            last_dividend = history[0]
            
            # Calculate average interval between dividends for better estimation
            if len(history) >= 2:
                intervals = []
                for i in range(len(history) - 1):
                    interval = (history[i].ex_dividend_date - history[i+1].ex_dividend_date).days
                    intervals.append(interval)
                avg_interval = sum(intervals) / len(intervals)
            else:
                # Fallback to frequency-based estimation
                if last_dividend.frequency == 'quarterly':
                    avg_interval = 91
                elif last_dividend.frequency == 'monthly':
                    avg_interval = 30
                elif last_dividend.frequency == 'semi-annual':
                    avg_interval = 182
                else:  # annual
                    avg_interval = 365
            
            # Estimate next dividend date
            next_ex_date = last_dividend.ex_dividend_date + timedelta(days=int(avg_interval))
            
            # Only return if the estimated date is in the future
            if next_ex_date > datetime.now():
                logger.info(f"Estimated next dividend for {symbol}: {next_ex_date.strftime('%Y-%m-%d')} (${last_dividend.dividend_amount:.2f})")
                return DividendEvent(
                    symbol=symbol,
                    declaration_date=None,
                    ex_dividend_date=next_ex_date,
                    record_date=next_ex_date + timedelta(days=1),
                    payment_date=next_ex_date + timedelta(days=30),
                    dividend_amount=last_dividend.dividend_amount,
                    dividend_yield=last_dividend.dividend_yield,
                    frequency=last_dividend.frequency
                )
            else:
                # If estimated date is in the past, try adding another interval
                next_ex_date = next_ex_date + timedelta(days=int(avg_interval))
                if next_ex_date > datetime.now():
                    logger.info(f"Estimated next dividend for {symbol} (adjusted): {next_ex_date.strftime('%Y-%m-%d')} (${last_dividend.dividend_amount:.2f})")
                    return DividendEvent(
                        symbol=symbol,
                        declaration_date=None,
                        ex_dividend_date=next_ex_date,
                        record_date=next_ex_date + timedelta(days=1),
                        payment_date=next_ex_date + timedelta(days=30),
                        dividend_amount=last_dividend.dividend_amount,
                        dividend_yield=last_dividend.dividend_yield,
                        frequency=last_dividend.frequency
                    )
        
        # If no historical data or estimation failed, try to get current dividend info from Yahoo Finance
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if Yahoo Finance has dividend information
            dividend_rate = info.get('dividendRate', 0)
            dividend_yield = info.get('dividendYield', 0)
            ex_dividend_date = info.get('exDividendDate')
            
            if dividend_rate and dividend_rate > 0:
                # If we have a dividend rate but no ex-dividend date, estimate quarterly
                if not ex_dividend_date:
                    # Assume quarterly dividends, estimate next date
                    next_ex_date = datetime.now() + timedelta(days=45)  # Rough estimate
                else:
                    # Convert timestamp to datetime if needed
                    if isinstance(ex_dividend_date, (int, float)):
                        next_ex_date = datetime.fromtimestamp(ex_dividend_date)
                    else:
                        next_ex_date = datetime.now() + timedelta(days=45)
                
                # Calculate quarterly dividend amount
                quarterly_dividend = dividend_rate / 4 if dividend_rate else 0.5
                
                logger.info(f"Using Yahoo Finance dividend info for {symbol}: estimated ${quarterly_dividend:.2f} quarterly")
                return DividendEvent(
                    symbol=symbol,
                    declaration_date=None,
                    ex_dividend_date=next_ex_date,
                    record_date=next_ex_date + timedelta(days=1),
                    payment_date=next_ex_date + timedelta(days=30),
                    dividend_amount=quarterly_dividend,
                    dividend_yield=dividend_yield * 100 if dividend_yield else 0,
                    frequency='quarterly'
                )
        except Exception as e:
            logger.warning(f"Could not get dividend info from Yahoo Finance for {symbol}: {e}")
        
        logger.warning(f"No dividend information found for {symbol}")
        return None


# Integration function for the main pipeline
def integrate_dividend_strategy(trade_pipeline, symbol: str = 'APAM'):
    """
    Integration function to add dividend capture capabilities to the existing pipeline.
    This function modifies the pipeline to consider dividend events in trading decisions.
    """
    from config.env_loader import load_env_variables
    
    # Load API keys
    api_keys = load_env_variables()
    
    # Initialize components
    fetcher = DividendDataFetcher(api_keys)
    strategy = DividendCaptureStrategy(symbol)
    
    # Fetch dividend data
    dividend_history = fetcher.fetch_dividend_history(symbol)
    next_dividend = fetcher.fetch_next_dividend(symbol)
    
    if next_dividend:
        strategy.update_dividend_calendar([next_dividend])
    
    # Train ML model if we have enough history
    if len(dividend_history) >= 10:
        # You'll need to pass price_data and market_data from your main pipeline
        # strategy.analyzer.train_ml_model(price_data, dividend_history, market_data)
        pass
    
    return strategy


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the dividend strategy
    strategy = DividendCaptureStrategy('APAM')
    
    # Simulate some dividend events
    test_dividend = DividendEvent(
        symbol='APAM',
        declaration_date=datetime.now() - timedelta(days=30),
        ex_dividend_date=datetime.now() + timedelta(days=5),
        record_date=datetime.now() + timedelta(days=6),
        payment_date=datetime.now() + timedelta(days=35),
        dividend_amount=0.88,
        dividend_yield=7.5,
        frequency='quarterly'
    )
    
    strategy.update_dividend_calendar([test_dividend])
    
    # Generate a signal (you'd pass real price data in production)
    signal = strategy.generate_signals(
        current_price=45.0,
        price_history=pd.DataFrame({'close': [44, 44.5, 45]}, 
                                 index=pd.date_range(end=datetime.now(), periods=3)),
        market_data=None
    )
    
    print(f"Generated signal: {signal}")
