#!/usr/bin/env python3
"""
News analysis using TypeSafe's Jev System One model.

Instead of asking a generative model to write a sentiment JSON blob, each
article is judged independently with typed questions (relevance, direction,
materiality, event type). Jev returns calibrated probabilities, and the
aggregation policy (weights, recency decay, thresholds) stays in code.

Per-article judgments are cached by article identity, so a polling loop only
pays for headlines it has not seen before.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("jev_news_analyzer")

MAX_ARTICLES = 20
MAX_WORKERS = 8
REQUEST_TIMEOUT_SECONDS = 5.0
RECENCY_HALF_LIFE_HOURS = 6.0
CACHE_SIZE = 2000

EVENT_TYPES = {
    "earnings": "Reported results, earnings surprises or misses.",
    "guidance": "Changes to forecasts, outlook or targets issued by the company.",
    "analyst_rating": "Analyst upgrades, downgrades or price-target changes.",
    "mergers_acquisitions": "Acquisitions, mergers, divestitures or takeover interest.",
    "regulatory_legal": "Lawsuits, investigations, fines, approvals or rulings.",
    "product": "Product launches, recalls, contracts or partnerships.",
    "management": "Executive hires, departures or governance changes.",
    "capital_actions": "Dividends, buybacks, offerings or debt financing.",
    "macro_sector": "Market-wide, economic or industry news rather than company-specific news.",
    "none": "Not a news event about the company, or none of the other types fit.",
}

DIRECTION_LEVELS = [
    "Clearly negative for the stock: likely to push the price down over the next trading session.",
    "Mildly negative for the stock.",
    "Neutral or mixed: no clear effect on the price either way.",
    "Mildly positive for the stock.",
    "Clearly positive for the stock: likely to push the price up over the next trading session.",
]

MATERIALITY_LEVELS = [
    "Routine or no new information (recaps, listicles, price-move commentary).",
    "Minor new information unlikely to change how investors value the company.",
    "Notable new information that could move the price by a few percent.",
    "Major, company-changing news (earnings surprise, guidance change, M&A, "
    "regulatory action, executive departure).",
]


def _build_questions() -> Dict[str, Any]:
    from typesafe_sdk import Choice, Noul, Score

    return {
        "relevant": Noul(
            instructions=(
                "Is `article` materially about the publicly traded company whose "
                "ticker is `symbol`, rather than a passing mention, a list of many "
                "stocks, or a different entity with a similar name?"
            ),
        ),
        "direction": Score(
            instructions=(
                "From the point of view of a shareholder of `symbol`, how does the "
                "news in `article` affect the stock price over the next trading session?"
            ),
            criteria=DIRECTION_LEVELS,
        ),
        "materiality": Score(
            instructions=(
                "How much genuinely new, price-relevant information about `symbol` "
                "does `article` contain?"
            ),
            criteria=MATERIALITY_LEVELS,
        ),
        "event_type": Choice(
            instructions="What kind of event about `symbol` does `article` report?",
            criteria=EVENT_TYPES,
        ),
    }


class _LRUCache:
    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._data: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            value = self._data.get(key)
            if value is not None:
                self._data.move_to_end(key)
            return value

    def put(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_judgment_cache = _LRUCache(CACHE_SIZE)
_client = None
_client_lock = threading.Lock()


def is_configured() -> bool:
    from config import settings

    return bool(getattr(settings, "TYPESAFE_API_KEY", None))


def _get_client():
    global _client
    with _client_lock:
        if _client is None:
            from typesafe_sdk import TypeSafeClient
            from config import settings

            _client = TypeSafeClient(
                api_key=settings.TYPESAFE_API_KEY,
                model=settings.TYPESAFE_MODEL,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        return _client


def _article_state(symbol: str, article: Dict[str, Any]) -> Dict[str, Any]:
    source = article.get("source")
    if isinstance(source, dict):
        source = source.get("name")
    return {
        "symbol": symbol,
        "article": {
            "headline": article.get("title") or "",
            "summary": article.get("description") or "",
            "source": source or "",
            "published_at": article.get("publishedAt") or "",
        },
    }


def _cache_key(symbol: str, article: Dict[str, Any]) -> str:
    identity = article.get("url") or f"{article.get('title')}|{article.get('publishedAt')}"
    return f"{symbol}|{identity}"


def _judge_article(client, questions, symbol: str, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    key = _cache_key(symbol, article)
    cached = _judgment_cache.get(key)
    if cached is not None:
        return cached

    try:
        response = client.system_one(state=_article_state(symbol, article), questions=questions)
    except Exception as e:
        logger.warning(f"Jev judgment failed for {symbol} article {article.get('title')!r}: {e}")
        return None

    direction = response.scores["direction"]
    materiality = response.scores["materiality"]
    event = response.choices["event_type"]
    judgment = {
        "headline": article.get("title") or "",
        "url": article.get("url"),
        "published_at": article.get("publishedAt"),
        "relevance": response.nouls["relevant"].noul,
        # Rescale the 0..4 direction rubric to -1..1.
        "direction": (direction.score - 2.0) / 2.0,
        "direction_confidence": direction.confidence,
        # Rescale the 0..3 materiality rubric to 0..1.
        "materiality": materiality.score / 3.0,
        "event_type": event.choice,
        "event_confidence": event.confidence,
    }
    _judgment_cache.put(key, judgment)
    return judgment


def _recency_weight(published_at: Optional[str], now: datetime) -> float:
    if not published_at:
        return 0.5
    try:
        published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.5
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    age_hours = max((now - published).total_seconds() / 3600.0, 0.0)
    return math.pow(0.5, age_hours / RECENCY_HALF_LIFE_HOURS)


def aggregate_judgments(judgments: List[Dict[str, Any]], now: Optional[datetime] = None) -> Dict[str, Any]:
    """Combine per-article judgments into one sentiment score in [-1, 1]."""
    now = now or datetime.now(timezone.utc)
    weighted_sum = 0.0
    total_weight = 0.0
    for judgment in judgments:
        weight = (
            judgment["relevance"]
            * judgment["materiality"]
            * judgment["direction_confidence"]
            * _recency_weight(judgment.get("published_at"), now)
        )
        judgment["weight"] = weight
        weighted_sum += weight * judgment["direction"]
        total_weight += weight

    if total_weight <= 1e-6:
        score = 0.0
    else:
        # Shrink toward neutral when the evidence is thin: a single weak,
        # stale headline should not produce a strong signal.
        evidence = total_weight / (total_weight + 0.5)
        score = (weighted_sum / total_weight) * evidence

    score = max(-1.0, min(1.0, score))
    if score > 0.15:
        category = "positive"
    elif score < -0.15:
        category = "negative"
    else:
        category = "neutral"

    drivers = sorted(judgments, key=lambda j: j["weight"] * abs(j["direction"]), reverse=True)[:3]
    if drivers and drivers[0]["weight"] * abs(drivers[0]["direction"]) > 0:
        rationale = "; ".join(
            f"{d['event_type']} ({d['direction']:+.2f}): {d['headline'][:100]}" for d in drivers
        )
    else:
        rationale = "No material, relevant news."

    return {
        "category": category,
        "score": round(score, 4),
        "rationale": rationale,
        "evidence_weight": round(total_weight, 4),
        "articles": judgments,
    }


def analyze_news(
    symbol: str, articles: List[Dict[str, Any]], now: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Judge each article with Jev and aggregate into a sentiment result.

    Returns the same {category, score, rationale} shape as the OpenAI
    sentiment analyzer, plus per-article judgments and evidence weight.
    Raises if the TypeSafe client cannot be created.
    """
    client = _get_client()
    questions = _build_questions()
    articles = [a for a in articles if a.get("title")][:MAX_ARTICLES]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(lambda a: _judge_article(client, questions, symbol, a), articles))

    judgments = [r for r in results if r is not None]
    if articles and not judgments:
        raise RuntimeError("All Jev article judgments failed")

    result = aggregate_judgments([dict(j) for j in judgments], now=now)
    result["provider"] = "typesafe"
    return result


def clear_cache() -> None:
    _judgment_cache.clear()
