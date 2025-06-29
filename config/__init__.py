#!/usr/bin/env python3
"""
Central configuration management for AI Day Trader Agent.
Provides trading parameters, API settings, and system configuration.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json

# Import environment variables
from .env_loader import load_env_variables

# Load environment variables once
_env = load_env_variables()

# Extract API keys
DISCORD_BOT_TOKEN = _env.get('DISCORD_BOT_TOKEN')
DISCORD_GUILD_ID = _env.get('DISCORD_GUILD_ID')
DISCORD_CHANNEL_ID = _env.get('DISCORD_CHANNEL_ID')
ALPHA_VANTAGE_API_KEY = _env.get('ALPHA_VANTAGE_API_KEY')
TWELVE_DATA_API_KEY = _env.get('TWELVE_DATA_API_KEY')
NEWS_API_KEY = _env.get('NEWS_API_KEY')
OPENAI_API_KEY = _env.get('OPENAI_API_KEY')

# Database path
DATABASE_PATH = os.getenv('DATABASE_PATH', 'data/dividend_trading.db')

@dataclass
class TradingConfig:
    """Centralized trading configuration with environment variable support."""
    
    # Position Management
    CORE_POSITION_SIZE: int = int(os.getenv('CORE_POSITION_SIZE', '100'))
    MAX_CAPTURE_POSITION: int = int(os.getenv('MAX_CAPTURE_POSITION', '50'))
    MAX_PORTFOLIO_ALLOCATION: float = float(os.getenv('MAX_PORTFOLIO_ALLOCATION', '0.25'))
    
    # Dividend Capture Settings
    MIN_DIVIDEND_YIELD_ANNUAL: float = float(os.getenv('MIN_DIVIDEND_YIELD', '6.0'))
    DIVIDEND_CAPTURE_WINDOW_DAYS: int = int(os.getenv('DIVIDEND_CAPTURE_WINDOW_DAYS', '7'))
    DIVIDEND_EXIT_WINDOW_DAYS: int = int(os.getenv('DIVIDEND_EXIT_WINDOW_DAYS', '5'))
    MIN_DAYS_AFTER_EX: int = int(os.getenv('MIN_DAYS_AFTER_EX', '1'))
    MAX_DAYS_AFTER_EX: int = int(os.getenv('MAX_DAYS_AFTER_EX', '5'))
    
    # Risk Management
    STOP_LOSS_PERCENTAGE: float = float(os.getenv('STOP_LOSS_PCT', '3.0'))
    POSITION_SIZING_METHOD: str = os.getenv('POSITION_SIZING_METHOD', 'fixed')
    MAX_DAILY_TRADES: int = int(os.getenv('MAX_DAILY_TRADES', '3'))
    MIN_PRICE: float = float(os.getenv('MIN_PRICE', '5.0'))
    USE_FRACTIONAL_SHARES: bool = os.getenv('USE_FRACTIONAL_SHARES', 'false').lower() == 'true'
    
    # Technical Indicators
    RSI_OVERSOLD: float = float(os.getenv('RSI_OVERSOLD', '30.0'))
    RSI_OVERBOUGHT: float = float(os.getenv('RSI_OVERBOUGHT', '70.0'))
    MACD_SIGNAL_THRESHOLD: float = float(os.getenv('MACD_SIGNAL_THRESHOLD', '0.0'))
    SMA_PERIOD: int = int(os.getenv('SMA_PERIOD', '20'))
    EMA_PERIOD: int = int(os.getenv('EMA_PERIOD', '20'))
    
    # Sentiment Analysis
    SENTIMENT_THRESHOLD_BULLISH: float = float(os.getenv('SENTIMENT_THRESHOLD_BULLISH', '0.6'))
    SENTIMENT_THRESHOLD_BEARISH: float = float(os.getenv('SENTIMENT_THRESHOLD_BEARISH', '-0.6'))
    NEWS_LOOKBACK_DAYS: int = int(os.getenv('NEWS_LOOKBACK_DAYS', '7'))
    
    # Backtesting
    BACKTEST_COMMISSION_PER_SHARE: float = float(os.getenv('BACKTEST_COMMISSION_PER_SHARE', '0.005'))
    BACKTEST_SLIPPAGE_BPS: int = int(os.getenv('BACKTEST_SLIPPAGE_BPS', '5'))
    BACKTEST_INITIAL_CAPITAL: float = float(os.getenv('BACKTEST_INITIAL_CAPITAL', '100000'))
    BACKTEST_REINVEST_DIVIDENDS: bool = os.getenv('BACKTEST_REINVEST_DIVIDENDS', 'true').lower() == 'true'
    BACKTEST_TAX_RATE: float = float(os.getenv('BACKTEST_TAX_RATE', '0.15'))
    
    # API Settings
    API_CALLS_PER_MINUTE: int = int(os.getenv('API_CALLS_PER_MINUTE', '5'))
    CACHE_EXPIRY_MINUTES: int = int(os.getenv('CACHE_EXPIRY_MINUTES', '15'))
    CANDLESTICK_LIMIT: int = int(os.getenv('CANDLESTICK_LIMIT', '500'))
    API_TIMEOUT_SECONDS: int = int(os.getenv('API_TIMEOUT_SECONDS', '30'))
    MAX_API_RETRIES: int = int(os.getenv('MAX_API_RETRIES', '3'))
    
    # Data Settings
    LOOKBACK_DAYS: int = int(os.getenv('LOOKBACK_DAYS', '365'))
    MIN_DATA_POINTS: int = int(os.getenv('MIN_DATA_POINTS', '100'))
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: str = os.getenv('LOG_FILE', 'logs/trading.log')
    LOG_MAX_BYTES: int = int(os.getenv('LOG_MAX_BYTES', '10485760'))  # 10MB
    LOG_BACKUP_COUNT: int = int(os.getenv('LOG_BACKUP_COUNT', '5'))
    
    # Signal Priorities (from SignalPriority class)
    SIGNAL_PRIORITY_DIVIDEND: int = 3
    SIGNAL_PRIORITY_TECHNICAL_STRONG: int = 2
    SIGNAL_PRIORITY_SENTIMENT: int = 1
    SIGNAL_PRIORITY_TECHNICAL_WEAK: int = 0
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        validations = [
            self.CORE_POSITION_SIZE > 0,
            self.MAX_CAPTURE_POSITION >= 0,
            0 < self.MAX_PORTFOLIO_ALLOCATION <= 1,
            self.MIN_DIVIDEND_YIELD_ANNUAL > 0,
            self.STOP_LOSS_PERCENTAGE > 0,
            self.POSITION_SIZING_METHOD in ['fixed', 'volatility', 'kelly'],
            self.RSI_OVERSOLD < self.RSI_OVERBOUGHT,
            self.SENTIMENT_THRESHOLD_BEARISH < self.SENTIMENT_THRESHOLD_BULLISH,
            self.MIN_DAYS_AFTER_EX <= self.MAX_DAYS_AFTER_EX,
            self.API_CALLS_PER_MINUTE > 0,
            self.CANDLESTICK_LIMIT > 0,
        ]
        return all(validations)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging."""
        return {
            'position_settings': {
                'core_position': self.CORE_POSITION_SIZE,
                'max_capture': self.MAX_CAPTURE_POSITION,
                'max_allocation': f"{self.MAX_PORTFOLIO_ALLOCATION:.1%}"
            },
            'dividend_settings': {
                'min_yield': f"{self.MIN_DIVIDEND_YIELD_ANNUAL:.1f}%",
                'capture_window': f"{self.DIVIDEND_CAPTURE_WINDOW_DAYS} days",
                'exit_window': f"{self.MIN_DAYS_AFTER_EX}-{self.MAX_DAYS_AFTER_EX} days"
            },
            'risk_settings': {
                'stop_loss': f"{self.STOP_LOSS_PERCENTAGE:.1f}%",
                'sizing_method': self.POSITION_SIZING_METHOD,
                'max_daily_trades': self.MAX_DAILY_TRADES,
                'min_price': f"${self.MIN_PRICE}"
            },
            'technical_settings': {
                'rsi_oversold': self.RSI_OVERSOLD,
                'rsi_overbought': self.RSI_OVERBOUGHT,
                'sma_period': self.SMA_PERIOD,
                'ema_period': self.EMA_PERIOD
            },
            'api_settings': {
                'calls_per_minute': self.API_CALLS_PER_MINUTE,
                'cache_expiry': f"{self.CACHE_EXPIRY_MINUTES} min",
                'candlestick_limit': self.CANDLESTICK_LIMIT
            }
        }
    
    def save_to_file(self, filepath: str = 'config/trading_config.json'):
        """Save configuration to JSON file."""
        config_dict = {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str = 'config/trading_config.json') -> 'TradingConfig':
        """Load configuration from JSON file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            # Create instance with loaded values
            instance = cls()
            for key, value in config_dict.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            return instance
        else:
            # Return default configuration
            return cls()
    
    def get_backtest_config(self):
        """Convert to BacktestConfig for dividend backtester."""
        from core.dividend_backtester import BacktestConfig
        
        return BacktestConfig(
            initial_capital=self.BACKTEST_INITIAL_CAPITAL,
            commission_per_share=self.BACKTEST_COMMISSION_PER_SHARE,
            slippage_bps=self.BACKTEST_SLIPPAGE_BPS,
            min_price=self.MIN_PRICE,
            max_position_pct=self.MAX_PORTFOLIO_ALLOCATION,
            use_fractional_shares=self.USE_FRACTIONAL_SHARES,
            reinvest_dividends=self.BACKTEST_REINVEST_DIVIDENDS,
            tax_rate=self.BACKTEST_TAX_RATE,
            min_dividend_yield=self.MIN_DIVIDEND_YIELD_ANNUAL / 4,  # Convert annual to quarterly
            max_days_before_ex=self.DIVIDEND_CAPTURE_WINDOW_DAYS,
            min_days_after_ex=self.MIN_DAYS_AFTER_EX,
            max_days_after_ex=self.MAX_DAYS_AFTER_EX,
            stop_loss_pct=self.STOP_LOSS_PERCENTAGE / 100,  # Convert percentage to decimal
            position_sizing_method=self.POSITION_SIZING_METHOD
        )

# Global configuration instance
trading_config = TradingConfig()

# Validate on import
if not trading_config.validate():
    import warnings
    warnings.warn(
        "Invalid trading configuration detected. Please check your settings.",
        UserWarning
    )

# Log configuration summary on import (if in debug mode)
if os.getenv('LOG_LEVEL') == 'DEBUG':
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Trading configuration loaded: {json.dumps(trading_config.get_summary(), indent=2)}")