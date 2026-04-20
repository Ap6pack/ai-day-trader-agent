from __future__ import annotations

from typing import Optional

import pytest

from core.portfolio_manager import PortfolioManager
from core.trading_workflow import TradingWorkflow


def _api_keys() -> dict[str, Optional[str]]:
    return {
        "TWELVE_DATA_API_KEY": None,
        "ALPHA_VANTAGE_API_KEY": None,
        "NEWS_API_KEY": None,
        "OPENAI_API_KEY": None,
    }


def test_workflow_records_paper_buy_without_external_apis(
    portfolio_manager: PortfolioManager,
) -> None:
    portfolio_manager.create_portfolio("paper", 10_000)

    def fake_analysis(symbol, api_keys, portfolio_name):
        assert symbol == "AAPL"
        assert portfolio_name == "paper"
        assert api_keys == _api_keys()
        return {
            "symbol": symbol,
            "recommendation": "BUY",
            "quantity": 2,
            "confidence": "75.0%",
            "primary_strategy": "technical",
            "risk_parameters": {"position_value": 300.0},
            "all_signals": {"technical": {"current_price": 150.0}},
        }

    workflow = TradingWorkflow(
        portfolio_manager,
        analysis_runner=fake_analysis,
        api_key_loader=_api_keys,
    )

    result = workflow.run("aapl", "paper", record_paper_trade=True)

    assert result.recorded_trade_id is not None
    assert result.skipped_reason is None
    assert portfolio_manager.get_holdings("paper")[0]["symbol"] == "AAPL"
    assert portfolio_manager.get_holdings("paper")[0]["quantity"] == 2

    trade = portfolio_manager.get_trade_history("paper", 1)[0]
    assert trade["action"] == "BUY"
    assert trade["price"] == 150.0
    assert trade["confidence"] == 0.75


def test_workflow_skips_hold_recommendation(portfolio_manager: PortfolioManager) -> None:
    portfolio_manager.create_portfolio("paper", 10_000)

    workflow = TradingWorkflow(
        portfolio_manager,
        analysis_runner=lambda symbol, api_keys, portfolio_name: {
            "symbol": symbol,
            "recommendation": "HOLD",
            "quantity": 0,
        },
        api_key_loader=_api_keys,
    )

    result = workflow.run("msft", "paper", record_paper_trade=True)

    assert result.recorded_trade_id is None
    assert result.skipped_reason == "No actionable trade recommendation"
    assert portfolio_manager.get_trade_history("paper", 1) == []


def test_workflow_requires_existing_portfolio(portfolio_manager: PortfolioManager) -> None:
    workflow = TradingWorkflow(portfolio_manager, api_key_loader=_api_keys)

    with pytest.raises(ValueError, match="Portfolio 'missing' not found"):
        workflow.run("AAPL", "missing", record_paper_trade=True)


def test_workflow_can_analyze_without_portfolio_when_not_recording(
    portfolio_manager: PortfolioManager,
) -> None:
    workflow = TradingWorkflow(
        portfolio_manager,
        analysis_runner=lambda symbol, api_keys, portfolio_name: {
            "symbol": symbol,
            "recommendation": "HOLD",
            "quantity": 0,
        },
        api_key_loader=_api_keys,
    )

    result = workflow.run("AAPL", "missing", record_paper_trade=False)

    assert result.analysis["symbol"] == "AAPL"
    assert result.skipped_reason == "Paper trading disabled"


def test_workflow_submits_alpaca_paper_order_and_records_trade(
    portfolio_manager: PortfolioManager,
) -> None:
    portfolio_manager.create_portfolio("paper", 10_000)
    submitted_signals = []

    class FakePaperExecutor:
        def execute_signal(self, signal):
            submitted_signals.append(signal)
            return {
                "id": "alpaca-order-1",
                "status": "accepted",
                "symbol": signal["symbol"],
                "side": signal["recommendation"].lower(),
                "qty": str(signal["quantity"]),
            }

    workflow = TradingWorkflow(
        portfolio_manager,
        analysis_runner=lambda symbol, api_keys, portfolio_name: {
            "symbol": symbol,
            "recommendation": "BUY",
            "quantity": 3,
            "confidence": 0.8,
            "primary_strategy": "technical",
            "all_signals": {"technical": {"current_price": 125.0}},
        },
        api_key_loader=_api_keys,
        paper_order_executor_factory=FakePaperExecutor,
    )

    result = workflow.run("aapl", "paper", submit_alpaca_paper_order=True)

    assert result.alpaca_order is not None
    assert result.alpaca_order["id"] == "alpaca-order-1"
    assert result.recorded_trade_id is not None
    assert submitted_signals[0]["symbol"] == "AAPL"
    assert submitted_signals[0]["recommendation"] == "BUY"
    assert submitted_signals[0]["quantity"] == 3

    trade = portfolio_manager.get_trade_history("paper", 1)[0]
    assert trade["notes"] == "Submitted to Alpaca paper trading order_id=alpaca-order-1 status=accepted"


def test_workflow_does_not_submit_hold_to_alpaca(
    portfolio_manager: PortfolioManager,
) -> None:
    portfolio_manager.create_portfolio("paper", 10_000)

    class FailingPaperExecutor:
        def execute_signal(self, signal):
            raise AssertionError("HOLD recommendations must not submit orders")

    workflow = TradingWorkflow(
        portfolio_manager,
        analysis_runner=lambda symbol, api_keys, portfolio_name: {
            "symbol": symbol,
            "recommendation": "HOLD",
            "quantity": 0,
        },
        api_key_loader=_api_keys,
        paper_order_executor_factory=FailingPaperExecutor,
    )

    result = workflow.run("AAPL", "paper", submit_alpaca_paper_order=True)

    assert result.alpaca_order is None
    assert result.recorded_trade_id is None
    assert result.skipped_reason == "No actionable trade recommendation"
