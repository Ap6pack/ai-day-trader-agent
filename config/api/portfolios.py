#!/usr/bin/env python3
"""
Portfolio management API endpoints for AI Day Trader Agent.
Provides secure REST endpoints for portfolio CRUD operations, holdings, and trades.
"""

from typing import List, Optional
from datetime import datetime, timedelta
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, field_validator
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.portfolio_manager import PortfolioManager
from config.api.auth import get_current_active_user, User
from config.api.dependencies import get_portfolio_manager

# Logging
logger = logging.getLogger(__name__)

# Router
router = APIRouter()

# Pydantic models for request/response validation
class PortfolioCreate(BaseModel):
    """Model for creating a new portfolio"""
    name: str = Field(..., min_length=1, max_length=50, description="Portfolio name")
    trading_capital: float = Field(..., gt=0, description="Initial trading capital")
    description: Optional[str] = Field(None, max_length=500, description="Portfolio description")


class PortfolioUpdate(BaseModel):
    """Model for updating portfolio"""
    trading_capital: Optional[float] = Field(None, gt=0, description="Updated trading capital")
    description: Optional[str] = Field(None, max_length=500, description="Updated description")


class PortfolioResponse(BaseModel):
    """Portfolio response model"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    trading_capital: float
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    total_value: float
    cash_available: float
    holdings_value: float


class HoldingCreate(BaseModel):
    """Model for adding/updating holdings"""
    symbol: str = Field(..., min_length=1, max_length=10, pattern="^[A-Z]+$")
    quantity: int = Field(..., gt=0, description="Number of shares")
    avg_cost: Optional[float] = Field(None, gt=0, description="Average cost per share")
    
    @field_validator('symbol', mode='before')
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.upper()


class HoldingResponse(BaseModel):
    """Holding response model"""
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


class TradeCreate(BaseModel):
    """Model for recording trades"""
    symbol: str = Field(..., min_length=1, max_length=10, pattern="^[A-Z]+$")
    action: str = Field(..., pattern="^(BUY|SELL)$", description="Trade action: BUY or SELL")
    quantity: int = Field(..., gt=0, description="Number of shares")
    price: float = Field(..., gt=0, description="Price per share")
    strategy: Optional[str] = Field(None, max_length=50, description="Trading strategy used")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence score (0-1)")
    notes: Optional[str] = Field(None, max_length=500, description="Trade notes")
    
    @field_validator('symbol', mode='before')
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.upper()
    
    @field_validator('action', mode='before')
    @classmethod
    def uppercase_action(cls, v: str) -> str:
        return v.upper()


class TradeResponse(BaseModel):
    """Trade response model"""
    model_config = ConfigDict(from_attributes=True)

    id: int
    portfolio_id: int
    symbol: str
    action: str
    quantity: int
    price: float
    total_value: float
    strategy: Optional[str]
    confidence: Optional[float]
    notes: Optional[str]
    timestamp: datetime


class PortfolioSummaryMetrics(BaseModel):
    """Stock portfolio summary metrics."""
    total_return: float
    total_return_pct: float
    realized_pnl: float
    unrealized_pnl: float
    total_trades: int
    buy_trades: int
    sell_trades: int
    total_fees: float


# Helper function to get user's portfolio
async def get_user_portfolio(
    portfolio_name: str,
    current_user: User = Depends(get_current_active_user),
    db: PortfolioManager = Depends(get_portfolio_manager),
) -> dict:
    """Get portfolio for authenticated user"""
    await run_in_threadpool(db.claim_unowned_portfolios, current_user.id)
    portfolio = await run_in_threadpool(
        db.get_portfolio,
        portfolio_name,
        current_user.id,
    )
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{portfolio_name}' not found"
        )
    return portfolio


# API Endpoints
@router.get("/", response_model=List[PortfolioResponse])
async def list_portfolios(
    current_user: User = Depends(get_current_active_user),
    db: PortfolioManager = Depends(get_portfolio_manager),
):
    """
    List all portfolios for the authenticated user.
    """
    try:
        await run_in_threadpool(db.claim_unowned_portfolios, current_user.id)
        portfolios = await run_in_threadpool(db.list_portfolios, current_user.id)
        
        # Enrich with current values
        enriched_portfolios = []
        for portfolio in portfolios:
            value_info = await run_in_threadpool(
                db.get_portfolio_value,
                portfolio['name'],
                None,
                current_user.id,
            )
            enriched_portfolio = {
                **portfolio,
                'total_value': value_info['total_value'],
                'cash_available': value_info['cash_available'],
                'holdings_value': value_info['holdings_value']
            }
            enriched_portfolios.append(enriched_portfolio)
        
        return enriched_portfolios
    except Exception as e:
        logger.error(f"Error listing portfolios: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve portfolios"
        )


@router.post("/", response_model=PortfolioResponse, status_code=status.HTTP_201_CREATED)
async def create_portfolio(
    portfolio_data: PortfolioCreate,
    current_user: User = Depends(get_current_active_user),
    db: PortfolioManager = Depends(get_portfolio_manager),
):
    """
    Create a new portfolio.
    """
    try:
        # Check if portfolio name already exists
        existing = await run_in_threadpool(
            db.get_portfolio,
            portfolio_data.name,
            current_user.id,
        )
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Portfolio '{portfolio_data.name}' already exists"
            )
        
        # Create portfolio
        portfolio = await run_in_threadpool(
            db.create_portfolio,
            name=portfolio_data.name,
            trading_capital=portfolio_data.trading_capital,
            user_id=current_user.id,
        )
        
        # Get value info
        value_info = await run_in_threadpool(
            db.get_portfolio_value,
            portfolio_data.name,
            None,
            current_user.id,
        )
        
        return {
            **portfolio,
            'description': portfolio_data.description,
            'total_value': value_info['total_value'],
            'cash_available': value_info['cash_available'],
            'holdings_value': value_info['holdings_value']
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating portfolio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create portfolio"
        )


@router.get("/{portfolio_name}", response_model=PortfolioResponse)
async def get_portfolio(
    portfolio_name: str,
    current_user: User = Depends(get_current_active_user),
    db: PortfolioManager = Depends(get_portfolio_manager),
):
    """
    Get portfolio details by name.
    """
    portfolio = await get_user_portfolio(portfolio_name, current_user, db)
    value_info = await run_in_threadpool(
        db.get_portfolio_value,
        portfolio_name,
        None,
        current_user.id,
    )
    
    return {
        **portfolio,
        'description': None,  # Add description field to database in production
        'total_value': value_info['total_value'],
        'cash_available': value_info['cash_available'],
        'holdings_value': value_info['holdings_value']
    }


@router.put("/{portfolio_name}", response_model=PortfolioResponse)
async def update_portfolio(
    portfolio_name: str,
    portfolio_update: PortfolioUpdate,
    current_user: User = Depends(get_current_active_user),
    db: PortfolioManager = Depends(get_portfolio_manager),
):
    """
    Update portfolio details.
    """
    await get_user_portfolio(portfolio_name, current_user, db)
    
    try:
        # Update trading capital if provided
        if portfolio_update.trading_capital is not None:
            success = await run_in_threadpool(
                db.update_trading_capital,
                portfolio_name,
                portfolio_update.trading_capital,
                current_user.id,
            )
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to update portfolio"
                )
        
        # Get updated portfolio
        updated_portfolio = await run_in_threadpool(
            db.get_portfolio,
            portfolio_name,
            current_user.id,
        )
        value_info = await run_in_threadpool(
            db.get_portfolio_value,
            portfolio_name,
            None,
            current_user.id,
        )
        
        return {
            **updated_portfolio,
            'description': portfolio_update.description,
            'total_value': value_info['total_value'],
            'cash_available': value_info['cash_available'],
            'holdings_value': value_info['holdings_value']
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating portfolio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update portfolio"
        )


@router.delete("/{portfolio_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_portfolio(
    portfolio_name: str,
    current_user: User = Depends(get_current_active_user),
    db: PortfolioManager = Depends(get_portfolio_manager),
):
    """
    Delete a portfolio.
    """
    await get_user_portfolio(portfolio_name, current_user, db)

    try:
        deleted = await run_in_threadpool(
            db.delete_portfolio,
            portfolio_name,
            current_user.id,
        )
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Portfolio '{portfolio_name}' not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting portfolio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete portfolio"
        )


# Holdings endpoints
@router.get("/{portfolio_name}/holdings", response_model=List[HoldingResponse])
async def get_holdings(
    portfolio_name: str,
    current_user: User = Depends(get_current_active_user),
    db: PortfolioManager = Depends(get_portfolio_manager),
):
    """
    Get all holdings for a portfolio.
    """
    await get_user_portfolio(portfolio_name, current_user, db)
    
    try:
        holdings = await run_in_threadpool(db.get_holdings, portfolio_name, current_user.id)
        value_info = await run_in_threadpool(
            db.get_portfolio_value,
            portfolio_name,
            None,
            current_user.id,
        )
        
        # Enrich holdings with current market data
        enriched_holdings = []
        for holding in holdings:
            # Find matching holding details
            holding_detail = next(
                (h for h in value_info['holdings_details'] if h['symbol'] == holding['symbol']),
                None
            )
            
            if holding_detail:
                enriched_holdings.append({
                    'symbol': holding['symbol'],
                    'quantity': holding['quantity'],
                    'avg_cost': holding['avg_cost'],
                    'current_price': holding_detail['current_price'],
                    'market_value': holding_detail['market_value'],
                    'unrealized_pnl': holding_detail['unrealized_pnl'],
                    'unrealized_pnl_pct': holding_detail['unrealized_pnl_pct']
                })
        
        return enriched_holdings
    except Exception as e:
        logger.error(f"Error getting holdings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve holdings"
        )


@router.post("/{portfolio_name}/holdings", response_model=HoldingResponse, status_code=status.HTTP_201_CREATED)
async def add_or_update_holding(
    portfolio_name: str,
    holding_data: HoldingCreate,
    current_user: User = Depends(get_current_active_user),
    db: PortfolioManager = Depends(get_portfolio_manager),
):
    """
    Add or update a holding in the portfolio.
    """
    await get_user_portfolio(portfolio_name, current_user, db)
    
    try:
        # Update holding
        await run_in_threadpool(
            db.update_holding,
            portfolio_name,
            holding_data.symbol,
            holding_data.quantity,
            holding_data.avg_cost,
            current_user.id,
        )
        
        # Get updated holding info
        holdings = await run_in_threadpool(db.get_holdings, portfolio_name, current_user.id)
        holding = next((h for h in holdings if h['symbol'] == holding_data.symbol), None)
        
        if not holding:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to add holding"
            )
        
        # Get current market data
        value_info = await run_in_threadpool(
            db.get_portfolio_value,
            portfolio_name,
            None,
            current_user.id,
        )
        holding_detail = next(
            (h for h in value_info['holdings_details'] if h['symbol'] == holding_data.symbol),
            None
        )
        
        return {
            'symbol': holding['symbol'],
            'quantity': holding['quantity'],
            'avg_cost': holding['avg_cost'],
            'current_price': holding_detail['current_price'] if holding_detail else 0,
            'market_value': holding_detail['market_value'] if holding_detail else 0,
            'unrealized_pnl': holding_detail['unrealized_pnl'] if holding_detail else 0,
            'unrealized_pnl_pct': holding_detail['unrealized_pnl_pct'] if holding_detail else 0
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding/updating holding: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add/update holding"
        )


@router.delete("/{portfolio_name}/holdings/{symbol}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_holding(
    portfolio_name: str,
    symbol: str,
    current_user: User = Depends(get_current_active_user),
    db: PortfolioManager = Depends(get_portfolio_manager),
):
    """
    Remove a holding from the portfolio.
    """
    await get_user_portfolio(portfolio_name, current_user, db)
    
    try:
        # Remove holding by setting quantity to 0
        await run_in_threadpool(
            db.update_holding,
            portfolio_name,
            symbol.upper(),
            0,
            None,
            current_user.id,
        )
    except Exception as e:
        logger.error(f"Error removing holding: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove holding"
        )


# Trade endpoints
@router.get("/{portfolio_name}/trades", response_model=List[TradeResponse])
async def get_trades(
    portfolio_name: str,
    days: int = Query(30, ge=1, le=365, description="Number of days of history"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    current_user: User = Depends(get_current_active_user),
    db: PortfolioManager = Depends(get_portfolio_manager),
):
    """
    Get trade history for a portfolio.
    """
    await get_user_portfolio(portfolio_name, current_user, db)
    
    try:
        trades = await run_in_threadpool(
            db.get_trade_history,
            portfolio_name,
            days,
            current_user.id,
        )
        
        # Filter by symbol if provided
        if symbol:
            trades = [t for t in trades if t['symbol'] == symbol.upper()]
        
        return trades
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve trades"
        )


@router.post("/{portfolio_name}/trades", response_model=TradeResponse, status_code=status.HTTP_201_CREATED)
async def record_trade(
    portfolio_name: str,
    trade_data: TradeCreate,
    current_user: User = Depends(get_current_active_user),
    db: PortfolioManager = Depends(get_portfolio_manager),
):
    """
    Record a new trade.
    """
    await get_user_portfolio(portfolio_name, current_user, db)
    
    try:
        # Record trade
        trade_id = await run_in_threadpool(
            db.record_trade,
            name=portfolio_name,
            symbol=trade_data.symbol,
            action=trade_data.action,
            quantity=trade_data.quantity,
            price=trade_data.price,
            strategy=trade_data.strategy,
            confidence=trade_data.confidence,
            notes=trade_data.notes,
            user_id=current_user.id,
        )
        
        # Get the recorded trade
        trades = await run_in_threadpool(
            db.get_trade_history,
            portfolio_name,
            1,
            current_user.id,
        )
        if not trades:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to record trade"
            )
        
        trade = trades[0]
        trade['id'] = trade_id
        
        return trade
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording trade: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record trade"
        )


# Portfolio summary endpoint
@router.get("/{portfolio_name}/performance", response_model=PortfolioSummaryMetrics)
async def get_performance(
    portfolio_name: str,
    current_user: User = Depends(get_current_active_user),
    db: PortfolioManager = Depends(get_portfolio_manager),
):
    """
    Get stock portfolio summary metrics.
    """
    await get_user_portfolio(portfolio_name, current_user, db)
    
    try:
        metrics = await run_in_threadpool(
            db.get_performance_metrics,
            portfolio_name,
            current_user.id,
        )
        
        if not metrics:
            # Return default metrics if none available
            return PortfolioSummaryMetrics(
                total_return=0,
                total_return_pct=0,
                realized_pnl=0,
                unrealized_pnl=0,
                total_trades=0,
                buy_trades=0,
                sell_trades=0,
                total_fees=0,
            )
        
        return metrics
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance metrics"
        )
