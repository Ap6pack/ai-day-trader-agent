#!/usr/bin/env python3
"""
Configuration module for AI Day Trader Agent.

This module provides centralized access to all configuration settings,
environment variables, and trading parameters.
"""

from . import settings
from .settings import (
    # API Keys
    DISCORD_BOT_TOKEN,
    DISCORD_GUILD_ID, 
    DISCORD_CHANNEL_ID,
    ALPHA_VANTAGE_API_KEY,
    TWELVE_DATA_API_KEY,
    NEWS_API_KEY,
    OPENAI_API_KEY,
    
    # Database
    DATABASE_PATH,
    
    # Trading Configuration
    TradingConfig,
    trading_config
)

__all__ = [
    'settings',
    'DISCORD_BOT_TOKEN',
    'DISCORD_GUILD_ID',
    'DISCORD_CHANNEL_ID', 
    'ALPHA_VANTAGE_API_KEY',
    'TWELVE_DATA_API_KEY',
    'NEWS_API_KEY',
    'OPENAI_API_KEY',
    'DATABASE_PATH',
    'TradingConfig',
    'trading_config'
]
