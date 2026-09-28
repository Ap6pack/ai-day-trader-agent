from __future__ import annotations

from datetime import datetime, timezone

import pytest
import requests

from config import settings
from core import news_fetcher


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


ALPACA_ITEM = {
    "id": 1,
    "headline": "Acme &amp; Co beats estimates",
    "summary": "<p>Revenue up 20%.</p>",
    "created_at": "2026-09-28T13:05:00Z",
    "url": "https://example.com/acme",
    "source": "benzinga",
    "symbols": ["ACME"],
}


@pytest.fixture
def alpaca_keys(monkeypatch):
    monkeypatch.setattr(settings, "ALPACA_API_KEY", "key-id")
    monkeypatch.setattr(settings, "ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr(settings, "ALPACA_DATA_BASE_URL", "https://data.alpaca.markets/")
    monkeypatch.setattr(settings, "NEWS_API_KEY", "newsapi-key")


def test_alpaca_news_request_and_normalization(alpaca_keys, monkeypatch):
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append((url, headers, params))
        return FakeResponse({"news": [ALPACA_ITEM], "next_page_token": None})

    monkeypatch.setattr(news_fetcher.requests, "get", fake_get)

    since = datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc)
    articles = news_fetcher.get_alpaca_news("ACME", since=since)

    url, headers, params = calls[0]
    assert url == "https://data.alpaca.markets/v1beta1/news"
    assert headers == {"APCA-API-KEY-ID": "key-id", "APCA-API-SECRET-KEY": "secret"}
    assert params == {"symbols": "ACME", "start": "2026-09-28T09:00:00Z", "limit": 50, "sort": "desc"}

    assert articles == [{
        "title": "Acme & Co beats estimates",
        "description": "Revenue up 20%.",
        "url": "https://example.com/acme",
        "publishedAt": "2026-09-28T13:05:00Z",
        "source": {"name": "benzinga"},
        "symbols": ["ACME"],
        "provider": "alpaca",
    }]


def test_get_news_articles_prefers_alpaca(alpaca_keys, monkeypatch):
    def fake_get(url, **kwargs):
        assert "alpaca" in url, "NewsAPI should not be called when Alpaca returns news"
        return FakeResponse({"news": [ALPACA_ITEM]})

    monkeypatch.setattr(news_fetcher.requests, "get", fake_get)
    articles = news_fetcher.get_news_articles("ACME")
    assert [a["provider"] for a in articles] == ["alpaca"]


@pytest.mark.parametrize("alpaca_response", [
    FakeResponse({"news": []}),
    FakeResponse({"message": "forbidden"}, status=403),
])
def test_get_news_articles_falls_back_to_newsapi(alpaca_keys, monkeypatch, alpaca_response):
    newsapi_article = {"title": "Acme headline", "url": "https://n/1"}

    def fake_get(url, **kwargs):
        if "alpaca" in url:
            return alpaca_response
        return FakeResponse({"articles": [newsapi_article]})

    monkeypatch.setattr(news_fetcher.requests, "get", fake_get)
    assert news_fetcher.get_news_articles("ACME") == [newsapi_article]


def test_get_news_articles_uses_newsapi_without_alpaca_keys(monkeypatch):
    monkeypatch.setattr(settings, "ALPACA_API_KEY", None)
    monkeypatch.setattr(settings, "NEWS_API_KEY", "newsapi-key")

    def fake_get(url, **kwargs):
        assert "newsapi.org" in url
        return FakeResponse({"articles": [{"title": "t"}]})

    monkeypatch.setattr(news_fetcher.requests, "get", fake_get)
    assert news_fetcher.get_news_articles("ACME") == [{"title": "t"}]
