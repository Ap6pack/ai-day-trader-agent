from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trader import scan

NOW = datetime(2026, 9, 28, 15, 0, tzinfo=timezone.utc)


def iso(hours_ago):
    return (NOW - timedelta(hours=hours_ago)).isoformat().replace("+00:00", "Z")


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("TYPESAFE_API_KEY", "t")
    monkeypatch.setenv("TRADER_WATCHLIST", "msft, aapl")
    monkeypatch.setenv("TRADER_SHORTLIST_SIZE", "3")
    monkeypatch.setenv("TRADER_MIN_PRICE", "5")


SCREENERS = {
    scan.MOST_ACTIVES_PATH: {"most_actives": [
        {"symbol": "NVDA", "volume": 9e7}, {"symbol": "AAPL", "volume": 5e7},
        {"symbol": "ABC.WS", "volume": 1e9},  # warrant, skipped
    ]},
    scan.MOVERS_PATH: {
        "gainers": [{"symbol": "PENNY", "price": 1.2, "percent_change": 80.0},
                    {"symbol": "SMCI", "price": 40.0, "percent_change": 12.5}],
        "losers": [{"symbol": "TSLA", "price": 250.0, "percent_change": -6.0}],
    },
}


def fake_get(url, params):
    for path, payload in SCREENERS.items():
        if url.endswith(path):
            return payload
    raise AssertionError(url)


def test_universe_merges_sources_and_filters(env, monkeypatch):
    monkeypatch.setattr(scan, "_get", fake_get)
    universe = scan.build_universe(scan._settings())
    assert set(universe) == {"MSFT", "AAPL", "NVDA", "SMCI", "TSLA"}
    assert universe["AAPL"].sources == ["watchlist", "most_active"]
    assert universe["TSLA"].percent_change == -6.0
    assert universe["SMCI"].price == 40.0


def test_universe_is_watchlist_only_without_alpaca(env, monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY")
    universe = scan.build_universe(scan._settings())
    assert set(universe) == {"MSFT", "AAPL"}


def test_screener_failure_keeps_other_sources(env, monkeypatch):
    def flaky(url, params):
        if url.endswith(scan.MOVERS_PATH):
            raise RuntimeError("503")
        return fake_get(url, params)

    monkeypatch.setattr(scan, "_get", flaky)
    assert set(scan.build_universe(scan._settings())) == {"MSFT", "AAPL", "NVDA"}


def jev_result(*articles):
    return {"score": -0.4, "evidence_weight": 0.3, "articles": list(articles)}


def article(headline, relevance, materiality, hours_ago, event="earnings"):
    return {"headline": headline, "relevance": relevance, "materiality": materiality,
            "published_at": iso(hours_ago), "event_type": event}


def test_material_news_requires_relevant_material_and_recent(env, monkeypatch):
    results = {
        "TSLA": jev_result(article("Tesla recalls 1M cars", 0.95, 0.8, 2, "product")),
        "NVDA": jev_result(article("NVDA in 10 stocks to watch", 0.2, 0.9, 1)),
        "SMCI": jev_result(article("SMCI misses estimates", 0.9, 0.9, 40)),
    }
    monkeypatch.setattr(scan, "get_news_articles", lambda s: [{"title": s}])
    monkeypatch.setattr(scan.jev_news, "analyze_news", lambda s, a, now, max_articles: results[s])

    tsla = scan.add_news(scan.Candidate("TSLA"), NOW)
    assert tsla.material_news and tsla.top_headline == "Tesla recalls 1M cars"
    assert tsla.top_event_type == "product"
    assert not scan.add_news(scan.Candidate("NVDA"), NOW).material_news   # not relevant
    assert not scan.add_news(scan.Candidate("SMCI"), NOW).material_news   # too old


def test_news_errors_do_not_stop_the_scan(env, monkeypatch):
    monkeypatch.setattr(scan, "get_news_articles", lambda s: [{"title": s}])

    def boom(*a, **k):
        raise RuntimeError("jev down")

    monkeypatch.setattr(scan.jev_news, "analyze_news", boom)
    cand = scan.add_news(scan.Candidate("AAPL"), NOW)
    assert cand.news_error == "jev down" and not cand.material_news


def test_rank_puts_material_news_first_then_move_size():
    cands = [
        scan.Candidate("BIGMOVE", percent_change=15.0),
        scan.Candidate("NEWS", percent_change=1.0, material_news=True),
        scan.Candidate("DOWN", percent_change=-8.0),
        scan.Candidate("FLAT", volume=1e9),
    ]
    assert [c.symbol for c in scan.rank(cands)] == ["NEWS", "BIGMOVE", "DOWN", "FLAT"]


def test_scan_end_to_end(env, monkeypatch):
    monkeypatch.setattr(scan, "_get", fake_get)
    monkeypatch.setattr(scan, "get_news_articles", lambda s: [{"title": s}] if s == "MSFT" else [])
    monkeypatch.setattr(
        scan.jev_news, "analyze_news",
        lambda s, a, now, max_articles: jev_result(article("Microsoft wins $10B contract", 0.97, 0.8, 1)),
    )
    result = scan.scan(now=NOW)
    assert result["universe_size"] == 5
    assert [c["symbol"] for c in result["shortlist"]] == ["MSFT", "SMCI", "TSLA"]
    assert result["shortlist"][0]["material_news"] is True
