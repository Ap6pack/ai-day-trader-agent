from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

from config.api.auth import User
from config.api.analysis import (
    AnalysisResponse,
    delete_analysis_job,
    get_analysis_result,
    get_analysis_status,
    _normalize_all_signals,
)
from core.portfolio_manager import PortfolioManager
from core.pipeline import EnhancedTradingPipeline


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


@pytest.fixture
def other_user() -> User:
    return User(
        id=2,
        username="other-user",
        email="other-user@example.com",
        is_active=True,
        is_admin=False,
        created_at="2026-01-01T00:00:00",
    )


def test_analysis_signals_normalize_strategy_specific_shapes() -> None:
    signals = _normalize_all_signals(
        {
            "technical": {
                "signal": "HOLD",
                "strength": 0.2,
                "priority": 0,
                "reason": "Technical: neutral",
                "indicators": {"rsi": 50},
            },
            "sentiment": {
                "signal": "HOLD",
                "strength": 0.3,
                "priority": 1,
                "sentiment_score": 0,
                "reason": "Sentiment score: 0.00",
            },
            "dividend": {
                "signal": "HOLD",
                "quantity": 0,
                "reason": "Outside capture window",
                "confidence": 0.0,
                "priority": 0,
            },
        }
    )

    response = AnalysisResponse(
        symbol="FIS",
        timestamp=datetime.now(timezone.utc),
        recommendation="HOLD",
        quantity=0,
        confidence=0.5,
        primary_strategy="sentiment",
        primary_reason="Sentiment score: 0.00",
        confirming_strategies=1,
        conflicting_strategies=0,
        all_signals=signals,
        portfolio_context={"portfolio_name": "paper"},
        formatted_output="HOLD FIS",
    )

    assert response.all_signals["sentiment"].indicators is None
    assert response.all_signals["dividend"].strength == 0
    assert response.all_signals["dividend"].indicators is None


def test_pipeline_symbol_validation_does_not_call_yahoo(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("PortfolioManager should not be needed for this test")

    monkeypatch.setattr(
        "core.portfolio_manager.PortfolioManager",
        fail_if_called,
    )

    pipeline = EnhancedTradingPipeline.__new__(EnhancedTradingPipeline)

    result = pipeline._validate_ticker_symbol("FIS")

    assert result["valid"] is True
    assert "format is valid" in result["message"]


@pytest.mark.parametrize("symbol", ["", "TOOLONG", "BRK.B", "123"])
def test_pipeline_symbol_validation_rejects_bad_format(symbol: str) -> None:
    pipeline = EnhancedTradingPipeline.__new__(EnhancedTradingPipeline)

    result = pipeline._validate_ticker_symbol(symbol)

    assert result["valid"] is False


@pytest.mark.asyncio
async def test_analysis_jobs_are_stored_in_database(
    portfolio_manager: PortfolioManager,
    current_user: User,
) -> None:
    job = portfolio_manager.create_analysis_job(
        job_id="job-1",
        user_id=current_user.id,
        portfolio_name="paper",
    )

    assert job["status"] == "pending"

    portfolio_manager.update_analysis_job(
        "job-1",
        current_user.id,
        status="completed",
        completed_at=datetime.now(timezone.utc),
        result={
            "portfolio_name": "paper",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_holdings": 1,
            "analyzed_holdings": 1,
            "failed_analyses": 0,
            "buy_recommendations": [],
            "sell_recommendations": [],
            "hold_positions": [],
            "portfolio_value": 1000.0,
            "cash_available": 1000.0,
            "analysis_duration_seconds": 0.1,
        },
    )

    status = await get_analysis_status(
        job_id="job-1",
        current_user=current_user,
        db=portfolio_manager,
    )
    result = await get_analysis_result(
        job_id="job-1",
        current_user=current_user,
        db=portfolio_manager,
    )

    assert status.status == "completed"
    assert result["portfolio_name"] == "paper"


@pytest.mark.asyncio
async def test_analysis_jobs_are_scoped_to_user(
    portfolio_manager: PortfolioManager,
    current_user: User,
    other_user: User,
) -> None:
    portfolio_manager.create_analysis_job(
        job_id="private-job",
        user_id=current_user.id,
        portfolio_name="paper",
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_analysis_status(
            job_id="private-job",
            current_user=other_user,
            db=portfolio_manager,
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_analysis_job_delete_is_persistent(
    portfolio_manager: PortfolioManager,
    current_user: User,
) -> None:
    portfolio_manager.create_analysis_job(
        job_id="delete-me",
        user_id=current_user.id,
        portfolio_name="paper",
    )

    await delete_analysis_job(
        job_id="delete-me",
        current_user=current_user,
        db=portfolio_manager,
    )

    assert portfolio_manager.get_analysis_job("delete-me", current_user.id) is None
