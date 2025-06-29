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
        Uses Alpha Vantage API with caching to minimize API calls.
        """
        cache_key = f"{symbol}_history_{years}y"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            cache_date = datetime.fromisoformat(cached_data['timestamp'])
            if (datetime.now() - cache_date).days < 7:  # Cache for 7 days
                return self._parse_cached_dividends(cached_data['data'])
        
        # Fetch from API
        try:
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'TIME_SERIES_MONTHLY_ADJUSTED',
                'symbol': symbol,
                'apikey': self.api_keys.get('ALPHA_VANTAGE_API_KEY')
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'Monthly Adjusted Time Series' in data:
                dividends = self._parse_dividend_data(data, symbol)
                
                # Cache the results
                self.cache[cache_key] = {
                    'timestamp': datetime.now().isoformat(),
                    'data': [self._dividend_to_dict(d) for d in dividends]
                }
                self._save_cache()
                
                return dividends
            else:
                logger.error(f"Failed to fetch dividend data: {data}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching dividend history: {e}")
            return []
    
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
        This would ideally use a real-time data source.
        """
        # For now, estimate based on historical pattern
        history = self.fetch_dividend_history(symbol, years=1)
        if not history:
            return None
        
        # Find the most recent dividend
        history.sort(key=lambda x: x.ex_dividend_date, reverse=True)
        last_dividend = history[0]
        
        # Estimate next dividend (quarterly = ~91 days)
        if last_dividend.frequency == 'quarterly':
            days_between = 91
        elif last_dividend.frequency == 'monthly':
            days_between = 30
        else:  # annual
            days_between = 365
        
        next_ex_date = last_dividend.ex_dividend_date + timedelta(days=days_between)
        
        # Only return if the estimated date is in the future
        if next_ex_date > datetime.now():
            return DividendEvent(
                symbol=symbol,
                declaration_date=None,
                ex_dividend_date=next_ex_date,
                record_date=next_ex_date + timedelta(days=1),
                payment_date=next_ex_date + timedelta(days=30),
                dividend_amount=last_dividend.dividend_amount,  # Assume same amount
                dividend_yield=last_dividend.dividend_yield,
                frequency=last_dividend.frequency
            )
        
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
