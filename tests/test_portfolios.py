from __future__ import annotations

import json

import pytest

from trader import portfolios as portfolios_module
from trader.portfolios import PortfolioError, Portfolios


@pytest.fixture
def store(tmp_path):
    return Portfolios(tmp_path / "p.db")


def test_create_list_and_default(store, monkeypatch):
    monkeypatch.setenv("TRADING_CAPITAL", "7500")
    store.ensure_default()
    store.ensure_default()
    store.create("aggressive", 10000)
    assert store.names() == ["default", "aggressive"]
    assert store.get("default")["cash"] == 7500
    with pytest.raises(PortfolioError, match="already exists"):
        store.create("aggressive", 1)
    for bad in ("", "../x", "a" * 41):
        with pytest.raises(PortfolioError):
            store.create(bad, 100)
    with pytest.raises(PortfolioError):
        store.create("zero", 0)


def test_buy_sell_cash_avg_price_and_realized(store):
    store.create("p", 10000)
    store.record_fill("p", "nvda", "buy", 10, 100.0)
    store.record_fill("p", "NVDA", "buy", 10, 110.0)
    p = store.get("p")
    assert p["cash"] == pytest.approx(7900)
    assert p["holdings"] == [{"symbol": "NVDA", "qty": 20, "avg_price": pytest.approx(105)}]
    fill = store.record_fill("p", "NVDA", "sell", 5, 120.0, source="autopilot")
    assert fill["realized_pl"] == 75.0 and fill["cash"] == pytest.approx(8500)
    assert store.held("p", "nvda") == 15
    store.record_fill("p", "NVDA", "sell", 15, 100.0)
    p = store.get("p")
    assert p["holdings"] == [] and p["realized_pl"] == pytest.approx(75 - 75)
    assert [f["side"] for f in store.fills("p")] == ["sell", "sell", "buy", "buy"]


def test_long_only_and_cash_checks(store):
    store.create("p", 1000)
    with pytest.raises(PortfolioError, match="Not enough cash"):
        store.record_fill("p", "NVDA", "buy", 11, 100.0)
    with pytest.raises(PortfolioError, match="long only"):
        store.record_fill("p", "NVDA", "sell", 1, 100.0)
    with pytest.raises(PortfolioError):
        store.record_fill("p", "NVDA", "hold", 1, 100.0)
    with pytest.raises(PortfolioError):
        store.record_fill("missing", "NVDA", "buy", 1, 100.0)
    assert store.get("p")["cash"] == 1000


def test_valuation_at_quotes(store):
    store.create("p", 10000)
    store.record_fill("p", "NVDA", "buy", 10, 100.0)
    store.record_fill("p", "AMD", "buy", 5, 50.0)
    v = store.valuation("p", {"NVDA": {"last": 110.0}})
    assert v["cash"] == 8750 and v["market_value"] == 1100 + 250
    assert v["equity"] == 10100 and v["unrealized_pl"] == 100
    assert v["total_return_pct"] == 1.0
    nvda = next(p for p in v["positions"] if p["symbol"] == "NVDA")
    assert nvda["unrealized_plpc"] == 10.0


def test_delete_removes_holdings_and_fills(store):
    store.create("p", 1000)
    store.record_fill("p", "NVDA", "buy", 1, 100.0)
    store.delete("p")
    assert store.names() == []
    store.create("p", 1000)
    assert store.get("p")["holdings"] == [] and store.fills("p") == []


def test_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(portfolios_module.config, "load_env", lambda *a, **k: None)
    monkeypatch.setenv("TRADER_JOURNAL_PATH", str(tmp_path / "cli.db"))
    assert portfolios_module.main(["create", "safe", "--cash", "2500"]) == 0
    assert json.loads(capsys.readouterr().out)["cash"] == 2500
    assert portfolios_module.main(["create", "safe"]) == 1
    assert "already exists" in json.loads(capsys.readouterr().out)["error"]
