#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from datetime import datetime, timezone , timedelta
from config import settings

def get_news_articles(symbol):
    """
    Fetch news articles for the given symbol from NewsAPI for the last 24 hours.
    Returns a list of articles.
    """
    api_key = settings.NEWS_API_KEY
    base_url = "https://newsapi.org/v2/everything"
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
    params = {
        "q": symbol,
        "from": yesterday,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": api_key
    }
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        return result.get("articles", [])
    except Exception:
        return []
