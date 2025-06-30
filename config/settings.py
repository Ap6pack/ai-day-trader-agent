#!/usr/bin/env python3
"""
Central configuration management for AI Day Trader Agent.
Provides easy access to all trading parameters and settings.
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DISCORD_GUILD_ID = os.getenv("DISCORD_GUILD_ID")
DISCORD_CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
# COIN_GECKO_API_KEY = os.getenv("COIN_GECKO_API_KEY")
# COIN_MARKET_CAP_API_KEY = os.getenv("COIN_MARKET_CAP_API_KEY")
# CRYPTO_COMPARE_API_KEY = os.getenv("CRYPTO_COMPARE_API_KEY")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@dataclass
class TradingConfig:
    """Centralized trading configuration with environment variable support."""
    
    # Position Management
    TRADING_CAPITAL: float = float(os.getenv('TRADING_CAPITAL', '5000.0'))
    MIN_POSITION_PERCENTAGE: float = float(os.getenv('MIN_POSITION_PERCENTAGE', '0.02'))
    MAX_POSITION_PERCENTAGE: float = float(os.getenv('MAX_POSITION_PERCENTAGE', '0.10'))
    MAX_PORTFOLIO_ALLOCATION: float = float(os.getenv('MAX_PORTFOLIO_ALLOCATION', '0.25'))
    
    # Dividend Capture Settings
    MIN_DIVIDEND_YIELD_ANNUAL: float = float(os.getenv('MIN_DIVIDEND_YIELD', '6.0'))
    DIVIDEND_CAPTURE_WINDOW_DAYS: int = int(os.getenv('DIVIDEND_CAPTURE_WINDOW_DAYS', '7'))
    DIVIDEND_EXIT_WINDOW_DAYS: int = int(os.getenv('DIVIDEND_EXIT_WINDOW_DAYS', '5'))
    
    # Risk Management
    STOP_LOSS_PERCENTAGE: float = float(os.getenv('STOP_LOSS_PCT', '3.0'))
    POSITION_SIZING_METHOD: str = os.getenv('POSITION_SIZING_METHOD', 'fixed')
    MAX_DAILY_TRADES: int = int(os.getenv('MAX_DAILY_TRADES', '3'))
    
    # Technical Indicators
    RSI_OVERSOLD: float = float(os.getenv('RSI_OVERSOLD', '30.0'))
    RSI_OVERBOUGHT: float = float(os.getenv('RSI_OVERBOUGHT', '70.0'))
    MACD_SIGNAL_THRESHOLD: float = float(os.getenv('MACD_SIGNAL_THRESHOLD', '0.0'))
    
    # Sentiment Analysis
    SENTIMENT_THRESHOLD_BULLISH: float = float(os.getenv('SENTIMENT_THRESHOLD_BULLISH', '0.6'))
    SENTIMENT_THRESHOLD_BEARISH: float = float(os.getenv('SENTIMENT_THRESHOLD_BEARISH', '-0.6'))
    
    # Backtesting
    BACKTEST_COMMISSION_PER_SHARE: float = float(os.getenv('BACKTEST_COMMISSION_PER_SHARE', '0.005'))
    BACKTEST_SLIPPAGE_BPS: int = int(os.getenv('BACKTEST_SLIPPAGE_BPS', '5'))
    
    # API Settings
    API_CALLS_PER_MINUTE: int = int(os.getenv('API_CALLS_PER_MINUTE', '5'))
    CACHE_EXPIRY_MINUTES: int = int(os.getenv('CACHE_EXPIRY_MINUTES', '15'))
    CANDLESTICK_LIMIT: int = int(os.getenv('CANDLESTICK_LIMIT', '500'))
    
    # Twelve Data API Tier Detection
    TWELVE_DATA_PREMIUM: bool = os.getenv('TWELVE_DATA_PREMIUM', 'auto').lower() in ['true', '1', 'yes']
    TWELVE_DATA_AUTO_DETECT: bool = os.getenv('TWELVE_DATA_PREMIUM', 'auto').lower() == 'auto'
    
    # Database
    DATABASE_PATH: str = os.getenv('DATABASE_PATH', 'data/dividend_trading.db')
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: str = os.getenv('LOG_FILE', 'logs/trading.log')
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        validations = [
            self.TRADING_CAPITAL > 0,
            0 < self.MIN_POSITION_PERCENTAGE <= 1,
            0 < self.MAX_POSITION_PERCENTAGE <= 1,
            self.MIN_POSITION_PERCENTAGE <= self.MAX_POSITION_PERCENTAGE,
            0 < self.MAX_PORTFOLIO_ALLOCATION <= 1,
            self.MIN_DIVIDEND_YIELD_ANNUAL > 0,
            self.STOP_LOSS_PERCENTAGE > 0,
            self.POSITION_SIZING_METHOD in ['fixed', 'volatility', 'kelly'],
        ]
        return all(validations)
    
    def get_summary(self) -> dict:
        """Get configuration summary for logging."""
        return {
            'position_settings': {
                'trading_capital': f"${self.TRADING_CAPITAL:,.2f}",
                'min_position': f"{self.MIN_POSITION_PERCENTAGE:.1%}",
                'max_position': f"{self.MAX_POSITION_PERCENTAGE:.1%}",
                'max_allocation': f"{self.MAX_PORTFOLIO_ALLOCATION:.1%}"
            },
            'dividend_settings': {
                'min_yield': f"{self.MIN_DIVIDEND_YIELD_ANNUAL:.1f}%",
                'capture_window': f"{self.DIVIDEND_CAPTURE_WINDOW_DAYS} days",
                'exit_window': f"{self.DIVIDEND_EXIT_WINDOW_DAYS} days"
            },
            'risk_settings': {
                'stop_loss': f"{self.STOP_LOSS_PERCENTAGE:.1f}%",
                'sizing_method': self.POSITION_SIZING_METHOD,
                'max_daily_trades': self.MAX_DAILY_TRADES
            }
        }

# Global configuration instance
trading_config = TradingConfig()

# Validate on import
if not trading_config.validate():
    raise ValueError("Invalid trading configuration. Please check your settings.")
