#!/usr/bin/env python3
"""
Enhanced trading pipeline that integrates dividend capture strategy with existing
technical and sentiment analysis. This creates a multi-strategy approach where
dividend events can override or complement other trading signals.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
import logging

from core.dividend_strategy import (
    DividendCaptureStrategy, 
    DividendDataFetcher,
    DividendEvent
)
from core.candle_fetcher import get_candlestick_data
from core.indicator_engine import compute_indicators
from core.sentiment_analyzer import analyze_sentiment
from core.news_fetcher import get_news_articles

def _configure_sqlite_adapters():
    """Configure SQLite datetime adapters for Python 3.12+ compatibility."""
    import sqlite3
    from datetime import datetime
    
    # Register datetime adapter and converter
    sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
    sqlite3.register_converter("TIMESTAMP", lambda b: datetime.fromisoformat(b.decode()))

logger = logging.getLogger(__name__)


class SignalPriority:
    """Define priority levels for different signal types."""
    DIVIDEND_CAPTURE = 3  # Highest priority - time-sensitive
    TECHNICAL_STRONG = 2  # Strong technical signals
    SENTIMENT_BASED = 1   # News-driven signals
    TECHNICAL_WEAK = 0    # Weak technical signals


class EnhancedTradingPipeline:
    """
    Enhanced pipeline that coordinates multiple trading strategies.
    
    The key innovation here is the signal fusion approach - we don't just
    pick one strategy, but intelligently combine signals based on their
    strength, timing, and market context.
    """

    def __init__(self, symbol: str, core_position: int = 100):
        self.symbol = symbol
        self.core_position = core_position
        self.dividend_strategy = DividendCaptureStrategy(symbol, core_position)
        self.position_tracker = PositionTracker(symbol, core_position)
        self.signal_history = []
    
    def _prepare_market_data_for_analysis(self, raw_api_data: Dict) -> Dict:
        """Convert dual-API response to format expected by analysis methods."""
        # Try TwelveData first, then Alpha Vantage
        for api_source in ['twelvedata', 'alphavantage']:
            if api_source in raw_api_data:
                api_data = raw_api_data[api_source]
                
                # Try different intervals, prefer longer timeframes
                for interval in ['1h', '15min', '1min']:
                    if interval in api_data and api_data[interval]:
                        candles = api_data[interval]
                        
                        if candles and isinstance(candles, list) and len(candles) > 0:
                            # Convert to expected format
                            candlesticks = {
                                'open': [],
                                'high': [],
                                'low': [],
                                'close': [],
                                'volume': [],
                                'datetime': []
                            }
                            
                            for candle in candles:
                                if isinstance(candle, dict):
                                    # Validate that we have the required price data
                                    try:
                                        # Handle both string and numeric values from APIs
                                        open_val = candle.get('open', 0)
                                        high_val = candle.get('high', 0)
                                        low_val = candle.get('low', 0)
                                        close_val = candle.get('close', 0)
                                        volume_val = candle.get('volume', 0)
                                        
                                        # Convert to float/int, handling string values
                                        open_price = float(str(open_val).replace(',', '')) if open_val else 0
                                        high_price = float(str(high_val).replace(',', '')) if high_val else 0
                                        low_price = float(str(low_val).replace(',', '')) if low_val else 0
                                        close_price = float(str(close_val).replace(',', '')) if close_val else 0
                                        volume = int(float(str(volume_val).replace(',', ''))) if volume_val else 0
                                        
                                        # Only add if we have valid price data
                                        if close_price > 0:
                                            candlesticks['open'].append(open_price)
                                            candlesticks['high'].append(high_price)
                                            candlesticks['low'].append(low_price)
                                            candlesticks['close'].append(close_price)
                                            candlesticks['volume'].append(volume)
                                            candlesticks['datetime'].append(candle.get('datetime', ''))
                                    except (ValueError, TypeError):
                                        continue  # Skip invalid data
                            
                            # Only return if we have at least some data
                            if len(candlesticks['close']) > 0:
                                return {'candlesticks': candlesticks}
        
        # Fallback: return empty structure with at least one dummy data point to prevent index errors
        return {
            'candlesticks': {
                'open': [100.0],
                'high': [100.0],
                'low': [100.0],
                'close': [100.0],
                'volume': [1000],
                'datetime': ['2024-01-01']
            }
        }
        
    def run_analysis(self, api_keys: Dict[str, str]) -> Dict[str, any]:
        """
        Run complete analysis combining all strategies.
        
        This is where the magic happens - we gather signals from all sources
        and use sophisticated logic to determine the best action.
        """
        logger.info(f"Running enhanced analysis for {self.symbol}")
        
        # Fetch all necessary data
        raw_market_data = get_candlestick_data(self.symbol)
        market_data = self._prepare_market_data_for_analysis(raw_market_data)
        price_history = self._prepare_price_history(market_data)
        
        # Run traditional technical analysis
        technical_signals = self._run_technical_analysis(market_data)
        
        # Run sentiment analysis
        sentiment_signals = self._run_sentiment_analyzer(self.symbol, api_keys)
        
        # Run dividend capture analysis
        dividend_signals = self._run_dividend_analysis(price_history, market_data, api_keys)
        
        # Combine all signals intelligently
        final_decision = self._fuse_signals(
            technical_signals, 
            sentiment_signals, 
            dividend_signals,
            price_history
        )
        
        # Track the decision for future analysis
        self._record_decision(final_decision)
        
        return final_decision
    
    def _run_technical_analysis(self, market_data: Dict) -> Dict[str, any]:
        """Run traditional technical indicator analysis."""
        try:
            indicators = compute_indicators(market_data)
            
            # Analyze indicator strength
            signal_strength = 0
            bullish_indicators = 0
            
            # RSI analysis - more sensitive thresholds
            rsi = indicators.get('rsi', 50)
            if rsi is not None:
                if rsi < 45:  # More sensitive oversold
                    bullish_indicators += 1
                    signal_strength += 0.4
                    if rsi < 30:  # Very oversold
                        bullish_indicators += 1
                        signal_strength += 0.3
                elif rsi > 55:  # More sensitive overbought
                    bullish_indicators -= 1
                    signal_strength -= 0.4
                    if rsi > 70:  # Very overbought
                        bullish_indicators -= 1
                        signal_strength -= 0.3
            
            # MACD analysis
            macd = indicators.get('macd')
            macd_signal = indicators.get('macd_signal')
            if macd is not None and macd_signal is not None:
                if macd > macd_signal:
                    bullish_indicators += 1
                    signal_strength += 0.2
                elif macd < macd_signal:
                    bullish_indicators -= 1
                    signal_strength -= 0.2
            
            # Moving average analysis
            price = market_data['candlesticks']['close'][-1]
            sma_20 = indicators.get('sma_20', price)
            ema_20 = indicators.get('ema_20', price)
            
            if sma_20 is not None and ema_20 is not None:
                if price > sma_20 and price > ema_20:
                    bullish_indicators += 1
                    signal_strength += 0.2
                elif price < sma_20 and price < ema_20:
                    bullish_indicators -= 1
                    signal_strength -= 0.2
            
            # Determine signal with more sensitive thresholds
            if bullish_indicators >= 1 and signal_strength > 0.2:
                signal = 'BUY'
                priority = SignalPriority.TECHNICAL_STRONG
            elif bullish_indicators <= -1 and signal_strength < -0.2:
                signal = 'SELL'
                priority = SignalPriority.TECHNICAL_STRONG
            else:
                signal = 'HOLD'
                priority = SignalPriority.TECHNICAL_WEAK
            
            return {
                'signal': signal,
                'strength': abs(signal_strength),
                'priority': priority,
                'indicators': indicators,
                'current_price': price,
                'reason': f"Technical: {bullish_indicators} bullish indicators"
            }
            
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            return {
                'signal': 'HOLD',
                'strength': 0,
                'priority': SignalPriority.TECHNICAL_WEAK,
                'reason': 'Technical analysis failed'
            }
    
    def _run_sentiment_analyzer(self, symbol: str, api_keys: Dict[str, str]) -> Dict[str, any]:
        """Run news sentiment analysis."""
        try:
            # Fetch news articles and analyze sentiment
            articles = get_news_articles(symbol)
            if not articles:
                sentiment_result = {
                    "category": "neutral",
                    "score": 0,
                    "rationale": "No news articles found",
                    "overall_sentiment": 0
                }
            else:
                sentiment_result = analyze_sentiment(articles)
                sentiment_result['overall_sentiment'] = sentiment_result.get('score', 0)
            
            # Convert sentiment to trading signal
            sentiment_score = sentiment_result.get('overall_sentiment', 0)
            
            if sentiment_score > 0.6:
                signal = 'BUY'
                strength = sentiment_score
            elif sentiment_score < -0.6:
                signal = 'SELL'
                strength = abs(sentiment_score)
            else:
                signal = 'HOLD'
                strength = 0.3
            
            return {
                'signal': signal,
                'strength': strength,
                'priority': SignalPriority.SENTIMENT_BASED,
                'sentiment_score': sentiment_score,
                'reason': f"Sentiment score: {sentiment_score:.2f}"
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {
                'signal': 'HOLD',
                'strength': 0,
                'priority': SignalPriority.SENTIMENT_BASED,
                'reason': 'Sentiment analysis failed'
            }
    
    def _run_dividend_analysis(self, price_history: pd.DataFrame, 
                              market_data: Dict,
                              api_keys: Dict[str, str]) -> Dict[str, any]:
        """Run dividend capture analysis."""
        try:
            # Initialize dividend data fetcher
            fetcher = DividendDataFetcher(api_keys)
            
            # Fetch dividend information
            next_dividend = fetcher.fetch_next_dividend(self.symbol)
            
            if next_dividend:
                self.dividend_strategy.update_dividend_calendar([next_dividend])
                
                # Get current price
                current_price = market_data['candlesticks']['close'][-1]
                
                # Generate dividend capture signal
                signal = self.dividend_strategy.generate_signals(
                    current_price,
                    price_history,
                    None  # Market data for correlation
                )
                
                # Add priority based on dividend timing
                days_to_ex = next_dividend.days_until_ex_dividend
                if 0 <= days_to_ex <= 7:
                    signal['priority'] = SignalPriority.DIVIDEND_CAPTURE
                else:
                    signal['priority'] = SignalPriority.TECHNICAL_WEAK
                
                return signal
            else:
                return {
                    'signal': 'HOLD',
                    'quantity': 0,
                    'strength': 0,
                    'priority': SignalPriority.TECHNICAL_WEAK,
                    'reason': 'No upcoming dividends found'
                }
                
        except Exception as e:
            logger.error(f"Dividend analysis error: {e}")
            return {
                'signal': 'HOLD',
                'quantity': 0,
                'strength': 0,
                'priority': SignalPriority.TECHNICAL_WEAK,
                'reason': 'Dividend analysis failed'
            }
    
    def _fuse_signals(self, technical: Dict, sentiment: Dict, 
                     dividend: Dict, price_history: pd.DataFrame) -> Dict[str, any]:
        """
        Intelligently combine signals from all strategies.
        
        This is the heart of the multi-strategy approach. We don't just pick
        the strongest signal - we consider context, timing, and risk.
        """
        signals = [
            ('technical', technical),
            ('sentiment', sentiment),
            ('dividend', dividend)
        ]
        
        # Sort by priority and strength
        signals.sort(key=lambda x: (x[1].get('priority', 0), x[1].get('strength', 0)), reverse=True)
        
        # Get the highest priority signal
        primary_strategy, primary_signal = signals[0]
        
        # Check for confirming signals
        confirming_signals = 0
        conflicting_signals = 0
        
        for strategy, signal in signals[1:]:
            if signal['signal'] == primary_signal['signal']:
                confirming_signals += 1
            elif signal['signal'] != 'HOLD' and signal['signal'] != primary_signal['signal']:
                conflicting_signals += 1
        
        # Adjust confidence based on confirmation
        base_confidence = primary_signal.get('confidence', primary_signal.get('strength', 0.5))
        
        if confirming_signals > 0:
            confidence_boost = 0.1 * confirming_signals
            final_confidence = min(base_confidence + confidence_boost, 0.95)
        else:
            confidence_penalty = 0.15 * conflicting_signals
            final_confidence = max(base_confidence - confidence_penalty, 0.2)
        
        # Determine position size
        if primary_strategy == 'dividend' and 'quantity' in primary_signal:
            # Use dividend strategy's calculated quantity
            quantity = primary_signal['quantity']
        else:
            # Calculate quantity based on confidence and signal type
            quantity = self._calculate_position_size(
                primary_signal['signal'],
                final_confidence,
                price_history
            )
        
        # Build final decision
        decision = {
            'timestamp': datetime.now(),
            'symbol': self.symbol,
            'signal': primary_signal['signal'],
            'quantity': quantity,
            'confidence': final_confidence,
            'primary_strategy': primary_strategy,
            'primary_reason': primary_signal['reason'],
            'confirming_strategies': confirming_signals,
            'conflicting_strategies': conflicting_signals,
            'all_signals': {
                'technical': technical,
                'sentiment': sentiment,
                'dividend': dividend
            },
            'position_type': primary_signal.get('entry_type', 'STANDARD'),
            'risk_parameters': self._calculate_risk_parameters(
                primary_signal['signal'],
                quantity,
                price_history
            )
        }
        
        return decision
    
    def _calculate_position_size(self, signal: str, confidence: float, 
                               price_history: pd.DataFrame) -> int:
        """
        Calculate position size based on signal strength and risk management.
        
        This implements a sophisticated position sizing algorithm that considers:
        - Current positions
        - Market volatility
        - Signal confidence
        - Risk limits
        """
        if signal == 'HOLD':
            return 0
        
        # Get current positions
        current_total = self.position_tracker.get_total_position()
        
        # Calculate maximum position based on risk management
        volatility = price_history['close'].pct_change().std() * np.sqrt(252)  # Annualized
        
        # High volatility = smaller positions
        volatility_factor = 1.0 / (1.0 + volatility * 2)
        
        # Base position size (as percentage of core position)
        if signal == 'BUY':
            base_size = 0.3  # 30% of core position
        else:  # SELL
            # Can only sell what we have above core position
            excess_position = max(0, current_total - self.core_position)
            base_size = min(0.5, excess_position / self.core_position)
        
        # Adjust by confidence and volatility
        position_percentage = base_size * confidence * volatility_factor
        
        # Convert to shares
        position_shares = int(self.core_position * position_percentage)
        
        # Round to nearest 10 shares
        position_shares = (position_shares // 10) * 10
        
        # Ensure minimum tradeable size
        if position_shares > 0 and position_shares < 10:
            position_shares = 10
            
        return position_shares
    
    def _calculate_risk_parameters(self, signal: str, quantity: int,
                                 price_history: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk management parameters for the trade."""
        if signal == 'HOLD' or quantity == 0:
            return {}
        
        current_price = price_history['close'].iloc[-1]
        atr = self._calculate_atr(price_history)  # Average True Range
        
        if signal == 'BUY':
            # Stop loss at 2 ATR below entry
            stop_loss = current_price - (2 * atr)
            # Take profit at 3 ATR above entry
            take_profit = current_price + (3 * atr)
        else:  # SELL
            stop_loss = current_price + (2 * atr)
            take_profit = current_price - (3 * atr)
        
        # Calculate position risk
        position_value = quantity * current_price
        risk_per_share = abs(current_price - stop_loss)
        total_risk = quantity * risk_per_share
        risk_percentage = (total_risk / position_value) * 100 if position_value > 0 else 0
        
        return {
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'position_value': round(position_value, 2),
            'total_risk': round(total_risk, 2),
            'risk_percentage': round(risk_percentage, 2),
            'risk_reward_ratio': round(abs(take_profit - current_price) / risk_per_share, 2)
        }
    
    def _calculate_atr(self, price_history: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range for volatility measurement."""
        high = price_history['high'].iloc[-period:]
        low = price_history['low'].iloc[-period:]
        close = price_history['close'].iloc[-period:]
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def _prepare_price_history(self, market_data: Dict) -> pd.DataFrame:
        """Convert market data to DataFrame for analysis."""
        candlesticks = market_data.get('candlesticks', {})
        
        if not candlesticks or not candlesticks.get('close'):
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        # Create DataFrame from candlestick data
        df = pd.DataFrame({
            'open': candlesticks.get('open', []),
            'high': candlesticks.get('high', []),
            'low': candlesticks.get('low', []),
            'close': candlesticks.get('close', []),
            'volume': candlesticks.get('volume', [])
        })
        
        # Add datetime index if available
        if 'datetime' in candlesticks:
            df.index = pd.to_datetime(candlesticks['datetime'])
        
        return df
    
    def _record_decision(self, decision: Dict[str, any]) -> None:
        """Record trading decision for analysis and learning."""
        self.signal_history.append(decision)
        
        # Keep only recent history to prevent memory bloat
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
        
        # Log the decision
        logger.info(f"Trading decision for {self.symbol}: "
                   f"{decision['signal']} {decision['quantity']} shares "
                   f"(confidence: {decision['confidence']:.2%})")


class PositionTracker:
    """
    Tracks current positions and provides position management utilities.
    
    This is crucial for risk management - we always need to know exactly
    what positions we hold and why.
    """
    
    def __init__(self, symbol: str, core_position: int):
        self.symbol = symbol
        self.core_position = core_position
        self.trading_position = 0  # Additional shares beyond core
        self.position_history = []
        
    def get_total_position(self) -> int:
        """Get total current position (core + trading)."""
        return self.core_position + self.trading_position
    
    def update_position(self, trade: Dict[str, any]) -> None:
        """Update position based on executed trade."""
        if trade['signal'] == 'BUY':
            self.trading_position += trade['quantity']
        elif trade['signal'] == 'SELL':
            self.trading_position -= trade['quantity']
        
        # Record the trade
        self.position_history.append({
            'timestamp': trade['timestamp'],
            'action': trade['signal'],
            'quantity': trade['quantity'],
            'position_after': self.get_total_position(),
            'strategy': trade.get('primary_strategy', 'unknown')
        })
    
    def get_position_summary(self) -> Dict[str, any]:
        """Get summary of current positions."""
        return {
            'symbol': self.symbol,
            'core_position': self.core_position,
            'trading_position': self.trading_position,
            'total_position': self.get_total_position(),
            'recent_trades': self.position_history[-5:] if self.position_history else []
        }


# Enhanced main execution function
def run_enhanced_analysis(symbol: str, api_keys: Dict[str, str]) -> Dict[str, any]:
    """
    Main entry point for enhanced trading analysis.
    
    This replaces the original run_analysis function with our multi-strategy approach.
    """
    pipeline = EnhancedTradingPipeline(symbol)
    result = pipeline.run_analysis(api_keys)
    
    # Format result for display
    formatted_result = {
        'recommendation': result['signal'],
        'quantity': result['quantity'],
        'confidence': f"{result['confidence']:.1%}",
        'primary_strategy': result['primary_strategy'],
        'reason': result['primary_reason'],
        'risk_parameters': result.get('risk_parameters', {}),
        'timestamp': result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add technical indicators to the result
    if 'technical' in result['all_signals'] and 'indicators' in result['all_signals']['technical']:
        formatted_result['technical_indicators'] = result['all_signals']['technical']['indicators']
    
    # Add all signal details for debugging
    formatted_result['all_signals'] = {
        'technical': result['all_signals']['technical'],
        'sentiment': result['all_signals']['sentiment'], 
        'dividend': result['all_signals']['dividend']
    }
    
    # Add strategy-specific information
    if result['primary_strategy'] == 'dividend':
        dividend_info = result['all_signals']['dividend']
        if 'days_to_ex_dividend' in dividend_info:
            formatted_result['dividend_info'] = {
                'days_to_ex_dividend': dividend_info['days_to_ex_dividend'],
                'expected_dividend': dividend_info.get('dividend_amount', 'Unknown')
            }
    
    return formatted_result


if __name__ == "__main__":
    # Test the enhanced pipeline
    from config.env_loader import load_env_variables
    
    api_keys = load_env_variables()
    result = run_enhanced_analysis('APAM', api_keys)
    
    print("\nEnhanced Trading Analysis Result:")
    print("=" * 50)
    for key, value in result.items():
        print(f"{key}: {value}")
