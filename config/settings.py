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
# CRYPTO_COMPARE_API_KEY = os.getenv("CRYPTO_COMPARE_API_KEY")  # Uncomment if needed
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
