#!/usr/bin/env python3
"""
Environment variable loader for the AI Day Trader Agent.
Provides a centralized way to load and validate environment variables.
"""

import os
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def load_env_variables() -> Dict[str, Optional[str]]:
    """
    Load and return application environment variables.
    
    Returns:
        Dict containing all environment variables needed by the application
        
    Market data credentials are optional because providers are tried in order
    and Yahoo Finance remains available as a no-key fallback.
    """
    env_vars = {
        'DISCORD_BOT_TOKEN': os.getenv('DISCORD_BOT_TOKEN'),
        'DISCORD_GUILD_ID': os.getenv('DISCORD_GUILD_ID'),
        'DISCORD_CHANNEL_ID': os.getenv('DISCORD_CHANNEL_ID'),
        'ALPACA_API_KEY': os.getenv('ALPACA_API_KEY') or os.getenv('ALPACA_KEY_ID'),
        'ALPACA_SECRET_KEY': os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_SECRET'),
        'ALPHA_VANTAGE_API_KEY': os.getenv('ALPHA_VANTAGE_API_KEY'),
        'TWELVE_DATA_API_KEY': os.getenv('TWELVE_DATA_API_KEY'),
        'NEWS_API_KEY': os.getenv('NEWS_API_KEY'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    }

    return env_vars


def get_api_key(service: str) -> Optional[str]:
    """
    Get a specific API key by service name.
    
    Args:
        service: Name of the service (e.g., 'alpha_vantage', 'twelve_data')
        
    Returns:
        API key string or None if not found
    """
    service_map = {
        'alpaca': 'ALPACA_API_KEY',
        'alpaca_secret': 'ALPACA_SECRET_KEY',
        'alpha_vantage': 'ALPHA_VANTAGE_API_KEY',
        'twelve_data': 'TWELVE_DATA_API_KEY',
        'news': 'NEWS_API_KEY',
        'openai': 'OPENAI_API_KEY',
        'discord': 'DISCORD_BOT_TOKEN'
    }
    
    env_var_name = service_map.get(service.lower())
    if not env_var_name:
        return None

    return load_env_variables().get(env_var_name)


def validate_environment() -> bool:
    """
    Validate that environment variables can be loaded.
    
    Returns:
        True if environment variables can be loaded.
    """
    load_env_variables()
    return True
