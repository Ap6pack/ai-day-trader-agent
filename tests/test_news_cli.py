from __future__ import annotations

import json

from trader import news


def _setup(monkeypatch, articles):
    monkeypatch.setattr(news.config, "load_env", lambda *a, **k: None)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(news, "get_news_articles", lambda symbol: articles)

    def fake_analyze(symbol, arts):
        return {"category": "positive", "score": 0.5, "evidence_weight": 0.4, "rationale": "r",
                "articles": [{"headline": a["title"], "weight": 0.4, "relevance": 0.9,
                              "direction": 0.8, "materiality": 0.7, "event_type": "earnings",
                              "published_at": None} for a in arts]}

    monkeypatch.setattr(news.jev_news, "analyze_news", fake_analyze)


def test_json_output_for_real_articles(monkeypatch, capsys):
    _setup(monkeypatch, [{"title": "Acme beats", "provider": "alpaca"}])
    assert news.main(["acme", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["symbol"] == "ACME" and out["source"] == "alpaca"
    assert out["articles"][0]["headline"] == "Acme beats"


def test_no_articles_scores_nothing(monkeypatch, capsys):
    _setup(monkeypatch, [])
    assert news.main(["acme", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["score"] is None and out["articles"] == []


def test_sample_is_labelled(monkeypatch, capsys):
    _setup(monkeypatch, [])
    assert news.main(["acme", "--sample"]) == 0
    assert "not real news" in capsys.readouterr().out


def test_missing_key(monkeypatch, capsys):
    monkeypatch.setattr(news.config, "load_env", lambda *a, **k: None)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert news.main(["acme"]) == 1
