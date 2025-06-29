#!/usr/bin/env python3
import os
from dotenv import load_dotenv

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

"""
Configuration settings for the enhanced AI Day Trader Agent.
"""

from dataclasses import dataclass
from typing import Optional

DATABASE_PATH = os.getenv('DATABASE_PATH', str("data/dividend_trading.db"))

@dataclass
class TradingConfig:
    """Main configuration for trading strategies."""
    
    # Position management
    core_position_size: int = 100  # Your base APAM position
    max_capture_position: int = 50  # Additional shares for dividend capture
    max_portfolio_allocation: float = 0.25  # Max 25% in one stock
    
    # Dividend capture settings
    min_dividend_yield_annual: float = 6.0  # Minimum 6% annual yield
    dividend_capture_window_days: int = 7  # Start looking 7 days before ex-div
    dividend_exit_window_days: int = 5  # Exit within 5 days after ex-div
    
    # Risk management
    stop_loss_percentage: float = 3.0  # 3% stop loss
    position_sizing_method: str = 'fixed'  # or 'volatility', 'kelly'
    max_daily_trades: int = 3  # Prevent overtrading
    
    # Technical indicators
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    macd_signal_threshold: float = 0.0
    
    # Sentiment analysis
    sentiment_threshold_bullish: float = 0.6
    sentiment_threshold_bearish: float = -0.6
    
    # Backtesting
    backtest_commission_per_share: float = 0.005
    backtest_slippage_bps: int = 5
    
    # API rate limits
    api_calls_per_minute: int = 5
    cache_expiry_minutes: int = 15


# Load configuration
trading_config = TradingConfig()
