from __future__ import annotations

from pathlib import Path

import pytest

from config.api.auth import User
from config.api.trading import (
    AnalyzeAndPaperTradeRequest,
    PaperOrderRequest,
    get_alpaca_account_status,
    get_alpaca_executor,
    get_provider_status,
    submit_paper_order,
    analyze_and_paper_trade,
)
from core.alpaca_executor_provider import clear_alpaca_executor_cache
from core.portfolio_manager import PortfolioManager


@pytest.fixture
def current_user() -> User:
    return User(
        id=1,
        username="api-user",
        email="api-user@example.com",
        is_active=True,
        is_admin=False,
        created_at="2026-01-01T00:00:00",
    )


@pytest.mark.asyncio
async def test_provider_status_does_not_expose_secret_values(monkeypatch, current_user):
    monkeypatch.setenv("ALPACA_API_KEY", "secret-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret-value")
    monkeypatch.setenv("MARKET_DATA_PROVIDERS", "alpaca,yahoo_finance")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets/v2")
    monkeypatch.setenv("DIVIDEND_STRATEGY_ENABLED", "true")
    monkeypatch.setenv("DIVIDEND_DATA_PROVIDERS", "alpha_vantage,yahoo_finance")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "alpha-secret")

    result = await get_provider_status(current_user=current_user)

    assert result["market_data_providers"] == ["alpaca", "yahoo_finance"]
    assert result["configured_providers"]["alpaca"] is True
    assert result["dividend_strategy_enabled"] is True
    assert result["dividend_data_providers"] == ["alpha_vantage", "yahoo_finance"]
    assert result["configured_dividend_providers"]["alpha_vantage"] is True
    assert result["active_dividend_providers"] == ["alpha_vantage", "yahoo_finance"]
    assert result["paper_trading_endpoint"] is True
    assert "secret-key" not in str(result)
    assert "secret-value" not in str(result)
    assert "alpha-secret" not in str(result)


@pytest.mark.asyncio
async def test_alpaca_account_status_uses_executor(current_user):
    class FakeExecutor:
        def get_account(self):
            return {
                "status": "ACTIVE",
                "currency": "USD",
                "buying_power": "100000",
                "cash": "100000",
                "equity": "100000",
            }

        def is_market_open(self):
            return True

    result = await get_alpaca_account_status(
        current_user=current_user,
        executor=FakeExecutor(),
    )

    assert result["connected"] is True
    assert result["paper_trading"] is True
    assert result["account_status"] == "ACTIVE"
    assert result["buying_power"] == 100000.0
    assert result["market_open"] is True


@pytest.mark.asyncio
async def test_submit_paper_order_uses_executor(current_user):
    submitted = []

    class FakeExecutor:
        def execute_signal(self, signal):
            submitted.append(signal)
            return {
                "id": "order-1",
                "status": "accepted",
                "symbol": signal["symbol"],
                "side": signal["recommendation"].lower(),
                "qty": str(signal["quantity"]),
            }

    result = await submit_paper_order(
        request=PaperOrderRequest(symbol="aapl", action="buy", quantity=5),
        current_user=current_user,
        executor=FakeExecutor(),
    )

    assert result["submitted"] is True
    assert result["order"]["id"] == "order-1"
    assert submitted == [{"symbol": "AAPL", "recommendation": "BUY", "quantity": 5}]


@pytest.mark.asyncio
async def test_analyze_and_paper_trade_returns_workflow_result(
    monkeypatch,
    portfolio_manager: PortfolioManager,
    current_user,
):
    class FakeWorkflowResult:
        symbol = "AAPL"
        portfolio_name = "paper"
        analysis = {"symbol": "AAPL", "recommendation": "BUY", "quantity": 1}
        alpaca_order = {"id": "order-1", "status": "accepted"}
        recorded_trade_id = 10
        skipped_reason = None

    class FakeWorkflow:
        def __init__(self, db):
            self.db = db

        def run(
            self,
            symbol,
            portfolio_name,
            *,
            record_paper_trade,
            submit_alpaca_paper_order,
            user_id=None,
        ):
            assert symbol == "AAPL"
            assert portfolio_name == "paper"
            assert record_paper_trade is True
            assert submit_alpaca_paper_order is True
            assert user_id == 1
            return FakeWorkflowResult()

    monkeypatch.setattr("config.api.trading.TradingWorkflow", FakeWorkflow)

    result = await analyze_and_paper_trade(
        request=AnalyzeAndPaperTradeRequest(
            symbol="aapl",
            portfolio_name="paper",
            submit_paper_order=True,
            record_local_trade=True,
        ),
        current_user=current_user,
        db=portfolio_manager,
    )

    assert result["symbol"] == "AAPL"
    assert result["alpaca_order"]["id"] == "order-1"
    assert result["recorded_trade_id"] == 10


def test_dashboard_uses_canonical_analyze_and_paper_trade_endpoint():
    api_js = (Path(__file__).resolve().parents[1] / "static" / "js" / "api.js").read_text()
    app_js = (Path(__file__).resolve().parents[1] / "static" / "js" / "app.js").read_text()

    assert "analyzeAndPaperTrade" in api_js
    assert "/trading/analyze-and-paper-trade" in api_js
    assert "API.analyzeAndPaperTrade" in app_js


def test_dashboard_exposes_portfolio_crud_controls():
    root = Path(__file__).resolve().parents[1]
    api_js = (root / "static" / "js" / "api.js").read_text()
    app_js = (root / "static" / "js" / "app.js").read_text()
    html = (root / "static" / "index.html").read_text()

    for method in ["updatePortfolio", "deletePortfolio", "saveHolding", "removeHolding", "recordTrade"]:
        assert method in api_js

    for handler in ["handleUpdatePortfolio", "handleDeletePortfolio", "handleSaveHolding", "handleRemoveHolding", "handleRecordTrade"]:
        assert handler in app_js

    for element_id in ["edit-portfolio-form", "holding-form", "trade-form"]:
        assert element_id in html


def test_trading_api_uses_cached_alpaca_executor_provider(monkeypatch):
    clear_alpaca_executor_cache()
    monkeypatch.setenv("ALPACA_API_KEY", "paper-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "paper-secret")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets/v2")

    assert get_alpaca_executor() is get_alpaca_executor()

    clear_alpaca_executor_cache()
