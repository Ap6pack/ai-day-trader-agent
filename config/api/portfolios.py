#!/usr/bin/env python3
"""
Portfolio management API endpoints for AI Day Trader Agent.
Provides secure REST endpoints for portfolio CRUD operations, holdings, and trades.
"""

from typing import List, Optional
from datetime import datetime, timedelta
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field, validator
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.portfolio_manager import PortfolioManager
from config.api.auth import get_current_active_user, User

# Logging
logger = logging.getLogger(__name__)

# Router
router = APIRouter()

# Initialize portfolio manager
portfolio_manager = PortfolioManager()


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
    id: int
    name: str
    trading_capital: float
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    total_value: float
    cash_available: float
    holdings_value: float
    
    class Config:
        orm_mode = True


class HoldingCreate(BaseModel):
    """Model for adding/updating holdings"""
    symbol: str = Field(..., min_length=1, max_length=10, regex="^[A-Z]+$")
    quantity: int = Field(..., gt=0, description="Number of shares")
    avg_cost: Optional[float] = Field(None, gt=0, description="Average cost per share")
    
    @validator('symbol')
    def uppercase_symbol(cls, v):
        return v.upper()


class HoldingResponse(BaseModel):
    """Holding response model"""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    
    class Config:
        orm_mode = True


class TradeCreate(BaseModel):
    """Model for recording trades"""
    symbol: str = Field(..., min_length=1, max_length=10, regex="^[A-Z]+$")
    action: str = Field(..., regex="^(BUY|SELL)$", description="Trade action: BUY or SELL")
    quantity: int = Field(..., gt=0, description="Number of shares")
    price: float = Field(..., gt=0, description="Price per share")
    strategy: Optional[str] = Field(None, max_length=50, description="Trading strategy used")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence score (0-1)")
    notes: Optional[str] = Field(None, max_length=500, description="Trade notes")
    
    @validator('symbol')
    def uppercase_symbol(cls, v):
        return v.upper()
    
    @validator('action')
    def uppercase_action(cls, v):
        return v.upper()


class TradeResponse(BaseModel):
    """Trade response model"""
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
    
    class Config:
        orm_mode = True


class PerformanceMetrics(BaseModel):
    """Portfolio performance metrics"""
    total_return: float
    total_return_pct: float
    realized_pnl: float
    unrealized_pnl: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]


# Helper function to get user's portfolio
async def get_user_portfolio(
    portfolio_name: str,
    current_user: User = Depends(get_current_active_user)
) -> dict:
    """Get portfolio for authenticated user"""
    # In production, filter by user ID
    portfolio = portfolio_manager.get_portfolio(portfolio_name)
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{portfolio_name}' not found"
        )
    return portfolio


# API Endpoints
@router.get("/", response_model=List[PortfolioResponse])
async def list_portfolios(
    current_user: User = Depends(get_current_active_user)
):
    """
    List all portfolios for the authenticated user.
    """
    try:
        # In production, filter by user ID
        portfolios = portfolio_manager.list_portfolios()
        
        # Enrich with current values
        enriched_portfolios = []
        for portfolio in portfolios:
            value_info = portfolio_manager.get_portfolio_value(portfolio['name'])
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
    current_user: User = Depends(get_current_active_user)
):
    """
    Create a new portfolio.
    """
    try:
        # Check if portfolio name already exists
        existing = portfolio_manager.get_portfolio(portfolio_data.name)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Portfolio '{portfolio_data.name}' already exists"
            )
        
        # Create portfolio
        portfolio = portfolio_manager.create_portfolio(
            name=portfolio_data.name,
            trading_capital=portfolio_data.trading_capital
        )
        
        # Get value info
        value_info = portfolio_manager.get_portfolio_value(portfolio_data.name)
        
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
    current_user: User = Depends(get_current_active_user)
):
    """
    Get portfolio details by name.
    """
    portfolio = await get_user_portfolio(portfolio_name, current_user)
    value_info = portfolio_manager.get_portfolio_value(portfolio_name)
    
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
    current_user: User = Depends(get_current_active_user)
):
    """
    Update portfolio details.
    """
    portfolio = await get_user_portfolio(portfolio_name, current_user)
    
    try:
        # Update trading capital if provided
        if portfolio_update.trading_capital is not None:
            success = portfolio_manager.update_trading_capital(
                portfolio_name,
                portfolio_update.trading_capital
            )
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to update portfolio"
                )
        
        # Get updated portfolio
        updated_portfolio = portfolio_manager.get_portfolio(portfolio_name)
        value_info = portfolio_manager.get_portfolio_value(portfolio_name)
        
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
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete a portfolio.
    """
    portfolio = await get_user_portfolio(portfolio_name, current_user)
    
    # In production, implement soft delete or archive
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Portfolio deletion not implemented for safety. Archive feature coming soon."
    )


# Holdings endpoints
@router.get("/{portfolio_name}/holdings", response_model=List[HoldingResponse])
async def get_holdings(
    portfolio_name: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get all holdings for a portfolio.
    """
    portfolio = await get_user_portfolio(portfolio_name, current_user)
    
    try:
        holdings = portfolio_manager.get_holdings(portfolio_name)
        value_info = portfolio_manager.get_portfolio_value(portfolio_name)
        
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
    current_user: User = Depends(get_current_active_user)
):
    """
    Add or update a holding in the portfolio.
    """
    portfolio = await get_user_portfolio(portfolio_name, current_user)
    
    try:
        # Update holding
        portfolio_manager.update_holding(
            portfolio_name,
            holding_data.symbol,
            holding_data.quantity,
            holding_data.avg_cost
        )
        
        # Get updated holding info
        holdings = portfolio_manager.get_holdings(portfolio_name)
        holding = next((h for h in holdings if h['symbol'] == holding_data.symbol), None)
        
        if not holding:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to add holding"
            )
        
        # Get current market data
        value_info = portfolio_manager.get_portfolio_value(portfolio_name)
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
    current_user: User = Depends(get_current_active_user)
):
    """
    Remove a holding from the portfolio.
    """
    portfolio = await get_user_portfolio(portfolio_name, current_user)
    
    try:
        # Remove holding by setting quantity to 0
        portfolio_manager.update_holding(portfolio_name, symbol.upper(), 0)
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
    current_user: User = Depends(get_current_active_user)
):
    """
    Get trade history for a portfolio.
    """
    portfolio = await get_user_portfolio(portfolio_name, current_user)
    
    try:
        trades = portfolio_manager.get_trade_history(portfolio_name, days)
        
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
    current_user: User = Depends(get_current_active_user)
):
    """
    Record a new trade.
    """
    portfolio = await get_user_portfolio(portfolio_name, current_user)
    
    try:
        # Record trade
        trade_id = portfolio_manager.record_trade(
            name=portfolio_name,
            symbol=trade_data.symbol,
            action=trade_data.action,
            quantity=trade_data.quantity,
            price=trade_data.price,
            strategy=trade_data.strategy,
            confidence=trade_data.confidence,
            notes=trade_data.notes
        )
        
        # Get the recorded trade
        trades = portfolio_manager.get_trade_history(portfolio_name, 1)
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


# Performance endpoints
@router.get("/{portfolio_name}/performance", response_model=PerformanceMetrics)
async def get_performance(
    portfolio_name: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get portfolio performance metrics.
    """
    portfolio = await get_user_portfolio(portfolio_name, current_user)
    
    try:
        metrics = portfolio_manager.get_performance_metrics(portfolio_name)
        
        if not metrics:
            # Return default metrics if none available
            return PerformanceMetrics(
                total_return=0,
                total_return_pct=0,
                realized_pnl=0,
                unrealized_pnl=0,
                win_rate=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_win=0,
                avg_loss=0,
                sharpe_ratio=None,
                max_drawdown=None
            )
        
        return metrics
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance metrics"
        )
