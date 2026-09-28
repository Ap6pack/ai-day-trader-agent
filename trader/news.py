"""
Score recent news for a symbol with Jev.

Usage:
    python -m trader.news AAPL            # human-readable
    python -m trader.news AAPL --json     # machine-readable, for Claude
    python -m trader.news AAPL --sample   # made-up headlines, to check Jev itself

Needs TYPESAFE_API_KEY. Headlines come from Alpaca news (ALPACA_API_KEY and
ALPACA_SECRET_KEY), falling back to NewsAPI (NEWS_API_KEY). If neither returns
anything, nothing is scored.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from trader import config, jev_news
from trader.news_sources import get_news_articles

SAMPLE_ARTICLES = [
    {"title": "{sym} beats quarterly earnings estimates and raises full-year guidance",
     "url": "sample://1"},
    {"title": "10 stocks to watch this week, including {sym}", "url": "sample://2"},
]
ARTICLE_FIELDS = ("headline", "published_at", "relevance", "direction", "materiality",
                  "event_type", "weight")


def main(argv: Optional[List[str]] = None) -> int:
    config.load_env()
    parser = argparse.ArgumentParser(prog="python -m trader.news", description=__doc__.split("\n")[1])
    parser.add_argument("symbol")
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    parser.add_argument("--sample", action="store_true",
                        help="score made-up sample headlines instead of fetching news")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.json else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    symbol = args.symbol.upper()

    if not jev_news.is_configured():
        print("TYPESAFE_API_KEY is not set.", file=sys.stderr)
        return 1

    if args.sample:
        articles = [dict(a, title=a["title"].format(sym=symbol)) for a in SAMPLE_ARTICLES]
        source = "sample"
    else:
        articles = get_news_articles(symbol)
        source = articles[0].get("provider", "newsapi") if articles else None

    if not articles:
        result = {"symbol": symbol, "source": None, "category": None, "score": None,
                  "rationale": "No news articles found; nothing to score.", "articles": []}
    else:
        result = jev_news.analyze_news(symbol, articles)
        result = {
            "symbol": symbol,
            "source": source,
            "category": result["category"],
            "score": result["score"],
            "evidence_weight": result["evidence_weight"],
            "rationale": result["rationale"],
            "articles": [
                {k: a.get(k) for k in ARTICLE_FIELDS}
                for a in sorted(result["articles"], key=lambda j: j["weight"], reverse=True)
            ],
        }

    if args.json:
        print(json.dumps(result))
        return 0

    if source == "sample":
        print("Using made-up sample headlines (--sample); this score is not real news.")
    if not articles:
        print(f"{symbol}: no news articles found; nothing to score. "
              "Use --sample to check Jev with made-up headlines.")
        return 0
    print(f"Fetched {len(articles)} articles from {source}.")
    print(f"{symbol}: {result['category']} score={result['score']:+.3f} "
          f"evidence={result['evidence_weight']:.2f}")
    print(f"Rationale: {result['rationale']}\n")
    for article in result["articles"]:
        print(json.dumps(article))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
