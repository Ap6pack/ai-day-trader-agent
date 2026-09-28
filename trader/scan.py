"""
Find today's day-trade candidates without being asked.

1. Universe: your TRADER_WATCHLIST plus Alpaca's most-active stocks and top
   gainers and losers (real-time screener, same Alpaca keys as the news feed).
2. Triage: Jev judges each candidate's recent headlines.
3. Rank: symbols with fresh material news first, then by size of today's
   move, then by volume. The top TRADER_SHORTLIST_SIZE go to Claude.

The ranking decides only what Claude looks at first; Claude still does the
analysis and the decision.

    python -m trader.scan            # shortlist as JSON
    python -m trader.scan --all      # every candidate considered
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

from trader import config, jev_news
from trader.news_sources import alpaca_configured, alpaca_data_url, alpaca_headers, get_news_articles

logger = logging.getLogger(__name__)

MOST_ACTIVES_PATH = "/v1beta1/screener/stocks/most-actives"
MOVERS_PATH = "/v1beta1/screener/stocks/movers"
SYMBOL_RE = re.compile(r"^[A-Z]{1,5}$")  # skips warrants, units, preferreds
SCAN_ARTICLES_PER_SYMBOL = 10
MATERIAL_RELEVANCE = 0.7
MATERIAL_MATERIALITY = 0.5
MATERIAL_MAX_AGE_HOURS = 24


@dataclass
class Candidate:
    symbol: str
    sources: List[str] = field(default_factory=list)
    price: Optional[float] = None
    percent_change: Optional[float] = None
    volume: Optional[float] = None
    news_score: Optional[float] = None
    news_evidence: Optional[float] = None
    material_news: bool = False
    top_headline: Optional[str] = None
    top_event_type: Optional[str] = None
    news_error: Optional[str] = None


def _settings() -> Dict[str, Any]:
    env = os.environ
    watchlist = [s.strip().upper() for s in env.get("TRADER_WATCHLIST", "").split(",") if s.strip()]
    return {
        "watchlist": watchlist,
        "screener_top": int(env.get("TRADER_SCREENER_TOP") or 20),
        "min_price": float(env.get("TRADER_MIN_PRICE") or 5),
        "max_symbols": int(env.get("TRADER_SCAN_MAX_SYMBOLS") or 25),
        "shortlist_size": int(env.get("TRADER_SHORTLIST_SIZE") or 5),
    }


def _get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.get(url, headers=alpaca_headers(), params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def build_universe(settings: Dict[str, Any]) -> Dict[str, Candidate]:
    candidates: Dict[str, Candidate] = {}

    def add(symbol: Any, source: str, **fields: Any) -> None:
        symbol = str(symbol or "").upper()
        if not SYMBOL_RE.match(symbol):
            return
        cand = candidates.setdefault(symbol, Candidate(symbol))
        if source not in cand.sources:
            cand.sources.append(source)
        for key, value in fields.items():
            if value is not None and getattr(cand, key) is None:
                setattr(cand, key, float(value))

    for symbol in settings["watchlist"]:
        add(symbol, "watchlist")

    if not alpaca_configured():
        logger.warning("Alpaca keys not set: scanning the watchlist only")
        return candidates

    top = settings["screener_top"]
    try:
        actives = _get(alpaca_data_url(MOST_ACTIVES_PATH), {"by": "volume", "top": top})
        for row in actives.get("most_actives", []):
            add(row.get("symbol"), "most_active", volume=row.get("volume"))
    except Exception as exc:
        logger.warning(f"Alpaca most-actives screener failed: {exc}")
    try:
        movers = _get(alpaca_data_url(MOVERS_PATH), {"top": top})
        for kind in ("gainers", "losers"):
            for row in movers.get(kind, []):
                add(row.get("symbol"), kind[:-1], price=row.get("price"),
                    percent_change=row.get("percent_change"))
    except Exception as exc:
        logger.warning(f"Alpaca movers screener failed: {exc}")

    min_price = settings["min_price"]
    return {
        s: c for s, c in candidates.items()
        if "watchlist" in c.sources or c.price is None or c.price >= min_price
    }


def _is_material(article: Dict[str, Any], now: datetime) -> bool:
    if (article.get("relevance") or 0) < MATERIAL_RELEVANCE:
        return False
    if (article.get("materiality") or 0) < MATERIAL_MATERIALITY:
        return False
    published = article.get("published_at")
    if not published:
        return True
    try:
        ts = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
    except ValueError:
        return True
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return now - ts <= timedelta(hours=MATERIAL_MAX_AGE_HOURS)


def add_news(candidate: Candidate, now: datetime) -> Candidate:
    try:
        articles = get_news_articles(candidate.symbol)
        if not articles:
            return candidate
        result = jev_news.analyze_news(candidate.symbol, articles, now=now,
                                       max_articles=SCAN_ARTICLES_PER_SYMBOL)
    except Exception as exc:
        candidate.news_error = str(exc)
        return candidate
    candidate.news_score = result["score"]
    candidate.news_evidence = result["evidence_weight"]
    judged = sorted(result["articles"],
                    key=lambda a: (a.get("materiality") or 0) * (a.get("relevance") or 0), reverse=True)
    material = [a for a in judged if _is_material(a, now)]
    candidate.material_news = bool(material)
    lead = (material or judged or [None])[0]
    if lead:
        candidate.top_headline = lead.get("headline")
        candidate.top_event_type = lead.get("event_type")
    return candidate


def rank(candidates: List[Candidate]) -> List[Candidate]:
    return sorted(
        candidates,
        key=lambda c: (
            c.material_news,
            abs(c.percent_change or 0),
            c.volume or 0,
            "watchlist" in c.sources,
        ),
        reverse=True,
    )


def _prefilter(candidates: Dict[str, Candidate], limit: int) -> List[Candidate]:
    """Cap how many symbols get Jev news calls: watchlist first, then biggest moves, then volume."""
    ordered = sorted(
        candidates.values(),
        key=lambda c: ("watchlist" in c.sources, abs(c.percent_change or 0), c.volume or 0),
        reverse=True,
    )
    return ordered[:limit]


def scan(now: Optional[datetime] = None) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    settings = _settings()
    universe = build_universe(settings)
    pool = _prefilter(universe, settings["max_symbols"])
    news_enabled = jev_news.is_configured()
    if news_enabled:
        with ThreadPoolExecutor(max_workers=4) as executor:
            pool = list(executor.map(lambda c: add_news(c, now), pool))
    ranked = rank(pool)
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "universe_size": len(universe),
        "considered": len(pool),
        "news_enabled": news_enabled,
        "shortlist": [asdict(c) for c in ranked[: settings["shortlist_size"]]],
        "all": [asdict(c) for c in ranked],
    }


def main(argv: Optional[List[str]] = None) -> int:
    config.load_env()
    parser = argparse.ArgumentParser(prog="python -m trader.scan")
    parser.add_argument("--all", action="store_true", help="include every candidate considered")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr)
    result = scan()
    if not args.all:
        result.pop("all")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
