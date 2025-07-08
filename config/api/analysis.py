#!/usr/bin/env python3
"""
Trading analysis API endpoints for AI Day Trader Agent.
Provides secure REST endpoints for running trading analysis on symbols and portfolios.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import asyncio

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field, validator
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.pipeline import EnhancedTradingPipeline
from core.portfolio_manager import PortfolioManager
from config.env_loader import load_env_variables
from config.api.auth import get_current_active_user, User
from utils.formatter import format_analysis_result

# Logging
logger = logging.getLogger(__name__)

# Router
router = APIRouter()

# Initialize components
portfolio_manager = PortfolioManager()
api_keys = load_env_variables()


# Pydantic models for request/response validation
class AnalysisRequest(BaseModel):
    """Model for analysis request"""
    symbol: str = Field(..., min_length=1, max_length=10, regex="^[A-Z]+$", description="Stock symbol to analyze")
    portfolio_name: Optional[str] = Field("default", description="Portfolio context for analysis")
    override_capital: Optional[float] = Field(None, gt=0, description="Override trading capital")
    override_holdings: Optional[int] = Field(None, ge=0, description="Override current holdings")
    
    @validator('symbol')
    def uppercase_symbol(cls, v):
        return v.upper()


class BatchAnalysisRequest(BaseModel):
    """Model for batch analysis request"""
    symbols: List[str] = Field(..., min_items=1, max_items=50, description="List of symbols to analyze")
    portfolio_name: Optional[str] = Field("default", description="Portfolio context for analysis")
    
    @validator('symbols')
    def uppercase_symbols(cls, v):
        return [s.upper() for s in v]


class TechnicalIndicators(BaseModel):
    """Technical indicators model"""
    current_price: float
    rsi: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    sma_20: Optional[float]
    ema_20: Optional[float]
    volume: Optional[int]
    price_change_pct: Optional[float]


class SignalDetails(BaseModel):
    """Signal details model"""
    signal: str = Field(..., regex="^(BUY|SELL|HOLD)$")
    strength: float = Field(..., ge=0, le=1)
    priority: int
    reason: str
    indicators: Optional[Dict[str, Any]]


class RiskParameters(BaseModel):
    """Risk parameters model"""
    stop_loss: float
    take_profit: float
    position_value: float
    total_risk: float
    risk_percentage: float
    risk_reward_ratio: float


class AnalysisResponse(BaseModel):
    """Analysis response model"""
    symbol: str
    timestamp: datetime
    recommendation: str = Field(..., regex="^(BUY|SELL|HOLD)$")
    quantity: int = Field(..., ge=0)
    confidence: float = Field(..., ge=0, le=1)
    primary_strategy: str
    primary_reason: str
    confirming_strategies: int
    conflicting_strategies: int
    technical_indicators: Optional[TechnicalIndicators]
    all_signals: Dict[str, SignalDetails]
    risk_parameters: Optional[RiskParameters]
    portfolio_context: Dict[str, Any]
    formatted_output: str


class PortfolioAnalysisResponse(BaseModel):
    """Portfolio analysis response model"""
    portfolio_name: str
    timestamp: datetime
    total_holdings: int
    analyzed_holdings: int
    failed_analyses: int
    buy_recommendations: List[Dict[str, Any]]
    sell_recommendations: List[Dict[str, Any]]
    hold_positions: List[Dict[str, Any]]
    portfolio_value: float
    cash_available: float
    analysis_duration_seconds: float


class AnalysisStatus(BaseModel):
    """Analysis job status model"""
    job_id: str
    status: str = Field(..., regex="^(pending|running|completed|failed)$")
    created_at: datetime
    completed_at: Optional[datetime]
    result: Optional[Any]
    error: Optional[str]


# In-memory job storage (use Redis or database in production)
analysis_jobs: Dict[str, AnalysisStatus] = {}


async def run_symbol_analysis(
    symbol: str,
    portfolio_name: str = "default",
    override_capital: Optional[float] = None,
    override_holdings: Optional[int] = None
) -> Dict[str, Any]:
    """Run analysis for a single symbol"""
    try:
        # Initialize pipeline
        pipeline = EnhancedTradingPipeline(symbol, portfolio_name)
        
        # Override settings if provided
        if override_capital:
            pipeline.config.TRADING_CAPITAL = override_capital
        if override_holdings is not None:
            pipeline.position_tracker.current_position = override_holdings
        
        # Run analysis
        result = pipeline.run_analysis(api_keys)
        
        # Check for errors
        if result.get('error'):
            return {
                'error': True,
                'error_type': result.get('error_type'),
                'message': result.get('message'),
                'symbol': symbol
            }
        
        # Format result
        formatted_output = format_analysis_result(result)
        
        # Build response
        response = {
            'symbol': symbol,
            'timestamp': result['timestamp'],
            'recommendation': result['signal'],
            'quantity': result['quantity'],
            'confidence': result['confidence'] if isinstance(result['confidence'], float) else float(result['confidence'].rstrip('%')) / 100,
            'primary_strategy': result['primary_strategy'],
            'primary_reason': result['primary_reason'],
            'confirming_strategies': result.get('confirming_strategies', 0),
            'conflicting_strategies': result.get('conflicting_strategies', 0),
            'all_signals': result.get('all_signals', {}),
            'risk_parameters': result.get('risk_parameters'),
            'portfolio_context': result.get('portfolio_context', {}),
            'formatted_output': formatted_output
        }
        
        # Add technical indicators if available
        if 'technical' in result.get('all_signals', {}) and 'indicators' in result['all_signals']['technical']:
            indicators = result['all_signals']['technical']['indicators']
            response['technical_indicators'] = {
                'current_price': result['all_signals']['technical'].get('current_price', 0),
                'rsi': indicators.get('rsi'),
                'macd': indicators.get('macd'),
                'macd_signal': indicators.get('macd_signal'),
                'sma_20': indicators.get('sma_20'),
                'ema_20': indicators.get('ema_20'),
                'volume': indicators.get('volume'),
                'price_change_pct': None  # Calculate if needed
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return {
            'error': True,
            'error_type': 'analysis_error',
            'message': str(e),
            'symbol': symbol
        }


async def run_portfolio_analysis_task(job_id: str, portfolio_name: str):
    """Background task to run portfolio analysis"""
    try:
        # Update job status
        analysis_jobs[job_id].status = "running"
        start_time = datetime.utcnow()
        
        # Get portfolio and holdings
        portfolio = portfolio_manager.get_portfolio(portfolio_name)
        if not portfolio:
            raise ValueError(f"Portfolio '{portfolio_name}' not found")
        
        holdings = portfolio_manager.get_holdings(portfolio_name)
        if not holdings:
            raise ValueError(f"No holdings in portfolio '{portfolio_name}'")
        
        # Track results
        buy_recommendations = []
        sell_recommendations = []
        hold_positions = []
        failed_analyses = 0
        
        # Analyze each holding
        for holding in holdings:
            symbol = holding['symbol']
            quantity = holding['quantity']
            avg_cost = holding['avg_cost']
            
            # Run analysis
            result = await run_symbol_analysis(symbol, portfolio_name)
            
            if result.get('error'):
                failed_analyses += 1
                continue
            
            # Categorize recommendation
            recommendation_data = {
                'symbol': symbol,
                'current_holding': quantity,
                'avg_cost': avg_cost,
                'recommendation': result
            }
            
            if result['recommendation'] == 'BUY' and result['quantity'] > 0:
                buy_recommendations.append(recommendation_data)
            elif result['recommendation'] == 'SELL' and result['quantity'] > 0:
                sell_recommendations.append(recommendation_data)
            else:
                hold_positions.append(recommendation_data)
        
        # Get portfolio value
        value_info = portfolio_manager.get_portfolio_value(portfolio_name)
        
        # Calculate duration
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Build result
        result = {
            'portfolio_name': portfolio_name,
            'timestamp': datetime.utcnow(),
            'total_holdings': len(holdings),
            'analyzed_holdings': len(holdings) - failed_analyses,
            'failed_analyses': failed_analyses,
            'buy_recommendations': buy_recommendations,
            'sell_recommendations': sell_recommendations,
            'hold_positions': hold_positions,
            'portfolio_value': value_info['total_value'],
            'cash_available': value_info['cash_available'],
            'analysis_duration_seconds': duration
        }
        
        # Update job
        analysis_jobs[job_id].status = "completed"
        analysis_jobs[job_id].completed_at = datetime.utcnow()
        analysis_jobs[job_id].result = result
        
    except Exception as e:
        logger.error(f"Portfolio analysis failed: {e}")
        analysis_jobs[job_id].status = "failed"
        analysis_jobs[job_id].completed_at = datetime.utcnow()
        analysis_jobs[job_id].error = str(e)


# API Endpoints
@router.post("/symbol", response_model=AnalysisResponse)
async def analyze_symbol(
    request: AnalysisRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Run trading analysis for a single symbol.
    
    Analyzes the symbol using technical indicators, sentiment analysis,
    and dividend capture strategies to provide a trading recommendation.
    """
    # Verify portfolio exists if specified
    if request.portfolio_name and request.portfolio_name != "default":
        portfolio = portfolio_manager.get_portfolio(request.portfolio_name)
        if not portfolio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Portfolio '{request.portfolio_name}' not found"
            )
    
    # Run analysis
    result = await run_symbol_analysis(
        request.symbol,
        request.portfolio_name,
        request.override_capital,
        request.override_holdings
    )
    
    # Check for errors
    if result.get('error'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get('message', 'Analysis failed')
        )
    
    return result


@router.post("/batch", response_model=List[AnalysisResponse])
async def analyze_batch(
    request: BatchAnalysisRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Run trading analysis for multiple symbols.
    
    Analyzes each symbol in the list and returns results for all.
    Failed analyses will include error information.
    """
    # Verify portfolio exists if specified
    if request.portfolio_name and request.portfolio_name != "default":
        portfolio = portfolio_manager.get_portfolio(request.portfolio_name)
        if not portfolio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Portfolio '{request.portfolio_name}' not found"
            )
    
    # Run analyses concurrently
    tasks = [
        run_symbol_analysis(symbol, request.portfolio_name)
        for symbol in request.symbols
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Filter out errors if needed or include them
    return [r for r in results if not r.get('error')]


@router.post("/portfolio/{portfolio_name}", response_model=AnalysisStatus)
async def analyze_portfolio(
    portfolio_name: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """
    Run analysis for all holdings in a portfolio.
    
    This is an asynchronous operation that analyzes each holding
    in the portfolio and returns a job ID to check status.
    """
    # Verify portfolio exists
    portfolio = portfolio_manager.get_portfolio(portfolio_name)
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Portfolio '{portfolio_name}' not found"
        )
    
    # Check if portfolio has holdings
    holdings = portfolio_manager.get_holdings(portfolio_name)
    if not holdings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Portfolio '{portfolio_name}' has no holdings to analyze"
        )
    
    # Create job
    job_id = f"portfolio_{portfolio_name}_{datetime.utcnow().timestamp()}"
    job = AnalysisStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.utcnow(),
        completed_at=None,
        result=None,
        error=None
    )
    analysis_jobs[job_id] = job
    
    # Start background task
    background_tasks.add_task(run_portfolio_analysis_task, job_id, portfolio_name)
    
    return job


@router.get("/portfolio/status/{job_id}", response_model=AnalysisStatus)
async def get_analysis_status(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get the status of a portfolio analysis job.
    """
    if job_id not in analysis_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis job '{job_id}' not found"
        )
    
    return analysis_jobs[job_id]


@router.get("/portfolio/result/{job_id}", response_model=PortfolioAnalysisResponse)
async def get_analysis_result(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get the result of a completed portfolio analysis job.
    """
    if job_id not in analysis_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis job '{job_id}' not found"
        )
    
    job = analysis_jobs[job_id]
    
    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Analysis job is {job.status}, not completed"
        )
    
    if job.error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {job.error}"
        )
    
    return job.result


@router.delete("/portfolio/job/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete a completed analysis job.
    """
    if job_id not in analysis_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis job '{job_id}' not found"
        )
    
    del analysis_jobs[job_id]
