#!/usr/bin/env python3
"""
Live check of the Jev news analyzer.

Usage:
    python scripts/jev_smoke_test.py AAPL

Needs TYPESAFE_API_KEY in the environment or .env. Headlines come from
Alpaca news (ALPACA_API_KEY/ALPACA_SECRET_KEY), falling back to NewsAPI;
if neither returns anything, two sample headlines are used.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402  (loads .env)
from core import jev_news_analyzer  # noqa: E402
from core.news_fetcher import get_news_articles  # noqa: E402

SAMPLE_ARTICLES = [
    {"title": "{sym} beats quarterly earnings estimates and raises full-year guidance",
     "url": "sample://1"},
    {"title": "10 stocks to watch this week, including {sym}", "url": "sample://2"},
]


def main() -> int:
    symbol = (sys.argv[1] if len(sys.argv) > 1 else "AAPL").upper()
    if not jev_news_analyzer.is_configured():
        print("TYPESAFE_API_KEY is not set.")
        return 1

    def is_set(name):
        return "set" if getattr(settings, name, None) else "MISSING"

    print("Keys: " + ", ".join(
        f"{name}={is_set(name)}"
        for name in ("TYPESAFE_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "NEWS_API_KEY")
    ))
    articles = get_news_articles(symbol)
    if articles:
        print(f"Fetched {len(articles)} articles from {articles[0].get('provider', 'newsapi')}.")
    else:
        print("No news articles found; using sample headlines.")
        articles = [dict(a, title=a["title"].format(sym=symbol)) for a in SAMPLE_ARTICLES]

    result = jev_news_analyzer.analyze_news(symbol, articles)
    print(f"{symbol}: {result['category']} score={result['score']:+.3f} "
          f"evidence={result['evidence_weight']:.2f}")
    print(f"Rationale: {result['rationale']}\n")
    for a in sorted(result["articles"], key=lambda j: j["weight"], reverse=True):
        print(json.dumps({k: a[k] for k in ("headline", "relevance", "direction",
                                            "materiality", "event_type", "weight")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
