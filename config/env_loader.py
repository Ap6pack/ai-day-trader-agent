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
    Load and return all required environment variables.
    
    Returns:
        Dict containing all environment variables needed by the application
        
    Raises:
        ValueError: If required environment variables are missing
    """
    env_vars = {
        'DISCORD_BOT_TOKEN': os.getenv('DISCORD_BOT_TOKEN'),
        'DISCORD_GUILD_ID': os.getenv('DISCORD_GUILD_ID'),
        'DISCORD_CHANNEL_ID': os.getenv('DISCORD_CHANNEL_ID'),
        'ALPHA_VANTAGE_API_KEY': os.getenv('ALPHA_VANTAGE_API_KEY'),
        'TWELVE_DATA_API_KEY': os.getenv('TWELVE_DATA_API_KEY'),
        'NEWS_API_KEY': os.getenv('NEWS_API_KEY'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    }
    
    # Check for critical missing variables
    critical_vars = ['ALPHA_VANTAGE_API_KEY', 'TWELVE_DATA_API_KEY']
    missing_critical = [var for var in critical_vars if not env_vars.get(var)]
    
    if missing_critical:
        raise ValueError(f"Missing critical environment variables: {', '.join(missing_critical)}")
    
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
        'alpha_vantage': 'ALPHA_VANTAGE_API_KEY',
        'twelve_data': 'TWELVE_DATA_API_KEY',
        'news': 'NEWS_API_KEY',
        'openai': 'OPENAI_API_KEY',
        'discord': 'DISCORD_BOT_TOKEN'
    }
    
    env_var_name = service_map.get(service.lower())
    if not env_var_name:
        return None
        
    return os.getenv(env_var_name)


def validate_environment() -> bool:
    """
    Validate that all required environment variables are set.
    
    Returns:
        True if all required variables are present, False otherwise
    """
    try:
        load_env_variables()
        return True
    except ValueError:
        return False
