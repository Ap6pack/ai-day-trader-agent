from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx2
import pytest
from typesafe_sdk import TypeSafeClient

from config import settings
from core import jev_news_analyzer, sentiment_analyzer

NOW = datetime(2026, 9, 25, 15, 0, tzinfo=timezone.utc)


def _iso(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat().replace("+00:00", "Z")


def _answers(relevant: float, direction: float, materiality: float, event: str) -> dict:
    return {
        "relevant": {"type": "noul", "noul": relevant},
        "direction": {
            "type": "score",
            "score": direction,
            "confidence": 0.9,
            "legend": {str(i): level for i, level in enumerate(jev_news_analyzer.DIRECTION_LEVELS)},
            "probabilities": {"0": 0.0, "1": 0.0, "2": 0.1, "3": 0.2, "4": 0.7},
        },
        "materiality": {
            "type": "score",
            "score": materiality,
            "confidence": 0.8,
            "legend": {str(i): level for i, level in enumerate(jev_news_analyzer.MATERIALITY_LEVELS)},
            "probabilities": {"0": 0.0, "1": 0.1, "2": 0.2, "3": 0.7},
        },
        "event_type": {
            "type": "choice",
            "choice": event,
            "confidence": 0.85,
            "probabilities": {event: 0.85, "none": 0.15},
        },
    }


@pytest.fixture
def fake_jev(monkeypatch):
    """Serve canned Jev answers keyed by headline through the real SDK client."""
    answers_by_headline: dict[str, dict] = {}
    requests: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        requests.append(body)
        headline = body["state"]["article"]["headline"]
        return httpx2.Response(
            200,
            json={
                "model": "jev-latest",
                "usage": {"input_tokens": 10, "output_tokens": 4},
                "answers": answers_by_headline[headline],
            },
        )

    client = TypeSafeClient(api_key="test-key", transport=httpx2.MockTransport(handler))
    monkeypatch.setattr(jev_news_analyzer, "_client", client)
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "test-key")
    jev_news_analyzer.clear_cache()
    yield answers_by_headline, requests
    jev_news_analyzer.clear_cache()
    client.close()


def test_analyze_news_sends_typed_questions_and_aggregates(fake_jev):
    answers, requests = fake_jev
    answers["Acme beats earnings, raises guidance"] = _answers(0.97, 3.8, 2.9, "earnings")
    answers["10 stocks to watch this week"] = _answers(0.1, 2.0, 0.2, "none")
    articles = [
        {"title": "Acme beats earnings, raises guidance", "url": "https://x/1", "publishedAt": _iso(1)},
        {"title": "10 stocks to watch this week", "url": "https://x/2", "publishedAt": _iso(2)},
    ]

    result = jev_news_analyzer.analyze_news("ACME", articles, now=NOW)

    assert result["provider"] == "typesafe"
    assert result["category"] == "positive"
    assert 0.5 < result["score"] <= 1.0
    assert "earnings" in result["rationale"]
    assert len(result["articles"]) == 2

    sent = requests[0]
    assert sent["model"] == "jev-latest"
    assert sent["state"]["symbol"] == "ACME"
    assert {q["type"] for q in sent["questions"].values()} == {"noul", "score", "choice"}


def test_irrelevant_or_routine_news_stays_neutral(fake_jev):
    answers, _ = fake_jev
    answers["Market wrap: stocks drift"] = _answers(0.2, 3.0, 0.3, "macro_sector")
    result = jev_news_analyzer.analyze_news(
        "ACME", [{"title": "Market wrap: stocks drift", "url": "https://x/3"}]
    )
    assert result["category"] == "neutral"
    assert abs(result["score"]) < 0.15


def test_judgments_are_cached_across_polls(fake_jev):
    answers, requests = fake_jev
    answers["Acme recalls flagship product"] = _answers(0.95, 0.4, 2.5, "product")
    articles = [{"title": "Acme recalls flagship product", "url": "https://x/4"}]

    first = jev_news_analyzer.analyze_news("ACME", articles)
    second = jev_news_analyzer.analyze_news("ACME", articles)

    assert len(requests) == 1
    assert first["score"] == second["score"] < 0


def test_recency_decay_discounts_stale_news():
    fresh = {"relevance": 1.0, "materiality": 1.0, "direction_confidence": 1.0,
             "direction": 1.0, "event_type": "earnings", "headline": "fresh",
             "published_at": _iso(0)}
    stale = dict(fresh, headline="stale", direction=-1.0, published_at=_iso(24))
    result = jev_news_analyzer.aggregate_judgments([fresh, stale], now=NOW)
    assert result["score"] > 0.5


def test_sentiment_analyzer_falls_back_to_openai_when_jev_fails(monkeypatch):
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "test-key")

    def boom(symbol, articles):
        raise RuntimeError("service down")

    monkeypatch.setattr(jev_news_analyzer, "analyze_news", boom)

    def fake_openai(**kwargs):
        raise RuntimeError("no openai in tests")

    fake_module = SimpleNamespace(
        api_key=None,
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_openai)),
    )
    monkeypatch.setattr(sentiment_analyzer, "openai", fake_module)

    result = sentiment_analyzer.analyze_sentiment([{"title": "x"}], "ACME")
    assert result["category"] == "neutral"
    assert result["rationale"] == "Sentiment analysis failed."
