#!/usr/bin/env python3
"""Trading and provider status API endpoints."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from config.api.auth import User, get_current_active_user
from config.api.dependencies import get_portfolio_manager
from core.alpaca_executor import AlpacaExecutor
from core.alpaca_executor_provider import get_alpaca_executor
from core.candle_fetcher_provider import get_candlestick_fetcher
from core.dividend_provider_config import (
    active_dividend_providers,
    configured_dividend_provider_order,
    configured_dividend_providers,
    dividend_strategy_enabled,
)
from core.portfolio_manager import PortfolioManager
from core.trading_workflow import TradingWorkflow


router = APIRouter()


class ProviderStatusResponse(BaseModel):
    market_data_providers: list[str]
    configured_providers: Dict[str, bool]
    dividend_strategy_enabled: bool
    dividend_data_providers: list[str]
    configured_dividend_providers: Dict[str, bool]
    active_dividend_providers: list[str]
    alpaca_data_base_url: str
    alpaca_trading_base_url: str
    alpaca_data_feed: str
    paper_trading_endpoint: bool
    legacy_fallbacks_configured: Dict[str, bool]


class AlpacaAccountStatusResponse(BaseModel):
    connected: bool
    paper_trading: bool
    account_status: Optional[str] = None
    currency: Optional[str] = None
    buying_power: Optional[float] = None
    cash: Optional[float] = None
    equity: Optional[float] = None
    market_open: Optional[bool] = None
    message: Optional[str] = None


class PaperOrderRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10, pattern="^[A-Z]+$")
    action: str = Field(..., pattern="^(BUY|SELL)$")
    quantity: int = Field(..., gt=0)

    @field_validator("symbol", mode="before")
    @classmethod
    def uppercase_symbol(cls, value: str) -> str:
        return value.upper()

    @field_validator("action", mode="before")
    @classmethod
    def uppercase_action(cls, value: str) -> str:
        return value.upper()


class PaperOrderResponse(BaseModel):
    submitted: bool
    order: Optional[Dict[str, Any]] = None
    skipped_reason: Optional[str] = None


class AnalyzeAndPaperTradeRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10, pattern="^[A-Z]+$")
    portfolio_name: str = Field("default", min_length=1, max_length=50)
    submit_paper_order: bool = Field(False, description="Submit actionable BUY/SELL result to Alpaca paper trading")
    record_local_trade: bool = Field(True, description="Record actionable result in the local portfolio database")

    @field_validator("symbol", mode="before")
    @classmethod
    def uppercase_symbol(cls, value: str) -> str:
        return value.upper()


class AnalyzeAndPaperTradeResponse(BaseModel):
    symbol: str
    portfolio_name: str
    analysis: Dict[str, Any]
    alpaca_order: Optional[Dict[str, Any]] = None
    recorded_trade_id: Optional[int] = None
    skipped_reason: Optional[str] = None


@router.get("/provider-status", response_model=ProviderStatusResponse)
async def get_provider_status(
    current_user: User = Depends(get_current_active_user),
):
    """Return configured market data and Alpaca endpoint status without exposing secrets."""
    fetcher = await run_in_threadpool(get_candlestick_fetcher)
    trading_base_url = (
        os.getenv("ALPACA_TRADING_BASE_URL")
        or os.getenv("ALPACA_BASE_URL")
        or "https://paper-api.alpaca.markets/v2"
    )
    return {
        "market_data_providers": fetcher.source_priority,
        "configured_providers": {
            source: fetcher._source_configured(source) for source in fetcher.source_priority
        },
        "dividend_strategy_enabled": dividend_strategy_enabled(),
        "dividend_data_providers": configured_dividend_provider_order(),
        "configured_dividend_providers": configured_dividend_providers(),
        "active_dividend_providers": active_dividend_providers(),
        "alpaca_data_base_url": fetcher.alpaca_data_base_url,
        "alpaca_trading_base_url": trading_base_url,
        "alpaca_data_feed": fetcher.alpaca_data_feed,
        "paper_trading_endpoint": "paper-api.alpaca.markets" in trading_base_url,
        "legacy_fallbacks_configured": {
            "twelve_data": bool(fetcher.twelve_data_key),
            "alpha_vantage": bool(fetcher.alpha_vantage_key),
            "news": bool(os.getenv("NEWS_API_KEY")),
        },
    }


@router.get("/alpaca/account", response_model=AlpacaAccountStatusResponse)
async def get_alpaca_account_status(
    current_user: User = Depends(get_current_active_user),
    executor: AlpacaExecutor = Depends(get_alpaca_executor),
):
    """Return Alpaca paper account status."""
    try:
        account = await run_in_threadpool(executor.get_account)
        market_open = await run_in_threadpool(executor.is_market_open)
        return {
            "connected": True,
            "paper_trading": True,
            "account_status": account.get("status"),
            "currency": account.get("currency"),
            "buying_power": _optional_float(account.get("buying_power")),
            "cash": _optional_float(account.get("cash")),
            "equity": _optional_float(account.get("equity")),
            "market_open": market_open,
            "message": "Connected to Alpaca paper trading",
        }
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to connect to Alpaca paper trading: {exc}",
        ) from exc


@router.post("/paper-order", response_model=PaperOrderResponse)
async def submit_paper_order(
    request: PaperOrderRequest,
    current_user: User = Depends(get_current_active_user),
    executor: AlpacaExecutor = Depends(get_alpaca_executor),
):
    """Submit a direct BUY/SELL order to Alpaca paper trading."""
    order = await run_in_threadpool(
        executor.execute_signal,
        {
            "symbol": request.symbol,
            "recommendation": request.action,
            "quantity": request.quantity,
        },
    )
    if not order:
        return {
            "submitted": False,
            "order": None,
            "skipped_reason": "Order was not submitted",
        }

    return {
        "submitted": True,
        "order": order,
        "skipped_reason": None,
    }


@router.post("/analyze-and-paper-trade", response_model=AnalyzeAndPaperTradeResponse)
async def analyze_and_paper_trade(
    request: AnalyzeAndPaperTradeRequest,
    current_user: User = Depends(get_current_active_user),
    db: PortfolioManager = Depends(get_portfolio_manager),
):
    """Run analysis and optionally submit an actionable result to Alpaca paper trading."""
    workflow = TradingWorkflow(db)
    try:
        result = await run_in_threadpool(
            workflow.run,
            request.symbol,
            request.portfolio_name,
            record_paper_trade=request.record_local_trade,
            submit_alpaca_paper_order=request.submit_paper_order,
            user_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return {
        "symbol": result.symbol,
        "portfolio_name": result.portfolio_name,
        "analysis": result.analysis,
        "alpaca_order": result.alpaca_order,
        "recorded_trade_id": result.recorded_trade_id,
        "skipped_reason": result.skipped_reason,
    }


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)
