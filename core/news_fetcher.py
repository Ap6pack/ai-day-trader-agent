#!/usr/bin/env python3
"""
News fetcher module for retrieving financial news articles.

Alpaca's news API (Benzinga-sourced, real-time, ticker-tagged) is the primary
source; NewsAPI is used as a fallback when Alpaca is not configured or returns
nothing. Articles are normalized to the NewsAPI shape
({title, description, url, publishedAt, source: {name}}) plus a `symbols` list
when the provider tags tickers.
"""

import re
from datetime import datetime, timezone, timedelta
from html import unescape

import requests

from config import settings
from utils.logger import get_logger

logger = get_logger("news_fetcher")

ALPACA_NEWS_PATH = "/v1beta1/news"
ALPACA_MAX_LIMIT = 50
LOOKBACK_HOURS = 24
NEWSAPI_LOOKBACK_DAYS = 3

_TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(text):
    return " ".join(unescape(_TAG_RE.sub(" ", text or "")).split())


def _alpaca_configured():
    return bool(settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY)


def _normalize_alpaca(item):
    return {
        "title": _clean_text(item.get("headline")),
        "description": _clean_text(item.get("summary")),
        "url": item.get("url"),
        "publishedAt": item.get("created_at"),
        "source": {"name": item.get("source") or "alpaca"},
        "symbols": item.get("symbols") or [],
        "provider": "alpaca",
    }


def get_alpaca_news(symbol, since=None, limit=ALPACA_MAX_LIMIT):
    """
    Fetch recent news for `symbol` from Alpaca, newest first.
    `since` is a timezone-aware datetime; defaults to the last 24 hours.
    Raises on HTTP errors so callers can fall back.
    """
    since = since or datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    url = settings.ALPACA_DATA_BASE_URL.rstrip("/") + ALPACA_NEWS_PATH
    headers = {
        "APCA-API-KEY-ID": settings.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY,
    }
    params = {
        "symbols": symbol,
        "start": since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": min(limit, ALPACA_MAX_LIMIT),
        "sort": "desc",
    }
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    articles = [_normalize_alpaca(item) for item in resp.json().get("news", []) if item.get("headline")]
    logger.info(f"Alpaca news returned {len(articles)} articles for {symbol} since {params['start']}")
    return articles


def get_newsapi_articles(symbol):
    """
    Fetch news articles for the given symbol from NewsAPI for the last few days.
    Returns a list of articles.
    """
    api_key = settings.NEWS_API_KEY
    if not api_key:
        logger.info("NEWS_API_KEY is not set; skipping NewsAPI")
        return []
    base_url = "https://newsapi.org/v2/everything"
    # NewsAPI's free plan delays articles by about a day, so a 24-hour window
    # is often empty; look back further and let recency weighting discount it.
    since = (datetime.now(timezone.utc) - timedelta(days=NEWSAPI_LOOKBACK_DAYS)).strftime('%Y-%m-%dT%H:%M:%S')
    params = {
        "q": symbol,
        "from": since,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": api_key
    }
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("articles", [])
    except Exception as e:
        logger.warning(f"NewsAPI request failed for {symbol}: {e}")
        return []


def get_news_articles(symbol):
    """
    Fetch recent news for `symbol`: Alpaca first, NewsAPI as fallback.
    Returns a list of NewsAPI-shaped articles (empty if no source works).
    """
    if _alpaca_configured():
        try:
            articles = get_alpaca_news(symbol)
            if articles:
                return articles
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            hint = ""
            if status in (401, 403):
                hint = (" Alpaca rejected the credentials: ALPACA_API_KEY and ALPACA_SECRET_KEY must be "
                        "a Trading API key pair generated together (paper keys usually start with PK).")
            logger.warning(f"Alpaca news request failed for {symbol} (HTTP {status}), falling back to NewsAPI.{hint}")
        except Exception as e:
            logger.warning(f"Alpaca news request failed for {symbol}, falling back to NewsAPI: {e}")
    else:
        logger.info("ALPACA_API_KEY/ALPACA_SECRET_KEY are not set; skipping Alpaca news")
    articles = get_newsapi_articles(symbol)
    if settings.NEWS_API_KEY:
        logger.info(f"NewsAPI returned {len(articles)} articles for {symbol}")
    return articles
