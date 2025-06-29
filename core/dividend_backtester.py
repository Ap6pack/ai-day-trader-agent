#!/usr/bin/env python3
"""
Sophisticated backtesting framework for dividend capture strategies.

This module provides realistic simulation of dividend capture trades, accounting for:
- Ex-dividend price adjustments
- Transaction costs and slippage
- Market impact of trades
- Time-based entry/exit constraints
- Multiple position management

The backtester is designed to give you confidence in your strategy before
risking real capital, while also helping identify optimal parameters.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from core.dividend_strategy import DividendCaptureStrategy, DividendEvent
from core.dividend_database import DividendDatabase

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration parameters for backtesting."""
    initial_capital: float = 100000.0
    commission_per_share: float = 0.005  # $0.005 per share
    slippage_bps: int = 5  # 5 basis points slippage
    min_price: float = 5.0  # Minimum stock price to trade
    max_position_pct: float = 0.20  # Max 20% of portfolio in one position
    use_fractional_shares: bool = False
    reinvest_dividends: bool = True
    tax_rate: float = 0.15  # Qualified dividend tax rate
    
    # Dividend capture specific parameters
    min_dividend_yield: float = 0.5  # Minimum 0.5% quarterly yield
    max_days_before_ex: int = 7  # Start looking 7 days before ex-dividend
    min_days_after_ex: int = 1  # Hold at least 1 day after ex-dividend
    max_days_after_ex: int = 5  # Exit within 5 days after ex-dividend
    
    # Risk management
    stop_loss_pct: float = 0.03  # 3% stop loss
    position_sizing_method: str = 'fixed'  # 'fixed', 'volatility', 'kelly'


@dataclass
class Trade:
    """Represents a single trade in the backtest."""
    trade_id: int
    symbol: str
    entry_date: datetime
    entry_price: float
    entry_quantity: int
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    dividend_amount: float = 0.0
    commission_paid: float = 0.0
    slippage_cost: float = 0.0
    profit_loss: Optional[float] = None
    profit_pct: Optional[float] = None
    trade_type: str = 'dividend_capture'
    dividend_event_id: Optional[int] = None
    
    @property
    def is_open(self) -> bool:
        return self.exit_date is None
    
    @property
    def holding_period(self) -> int:
        if self.exit_date:
            return (self.exit_date - self.entry_date).days
        return 0


class DividendBacktester:
    """
    Advanced backtesting engine for dividend capture strategies.
    
    This backtester simulates realistic trading conditions including:
    - Accurate ex-dividend price adjustments
    - Transaction costs and market impact
    - Position sizing constraints
    - Tax implications
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.trades: List[Trade] = []
        self.open_positions: Dict[str, Trade] = {}
        self.portfolio_history = []
        self.cash_history = []
        self.trade_counter = 0
        
    def backtest_strategy(self, 
                         strategy: DividendCaptureStrategy,
                         price_data: pd.DataFrame,
                         dividend_events: List[DividendEvent],
                         start_date: datetime,
                         end_date: datetime) -> Dict:
        """
        Run a complete backtest of the dividend capture strategy.
        
        This simulates trading over the specified period, executing trades
        based on the strategy's signals while accounting for all costs.
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Initialize portfolio
        self.cash = self.config.initial_capital
        self.portfolio_value = self.config.initial_capital
        
        # Prepare dividend calendar
        dividend_calendar = self._prepare_dividend_calendar(dividend_events)
        
        # Filter price data to backtest period
        backtest_prices = price_data[
            (price_data.index >= start_date) & 
            (price_data.index <= end_date)
        ].copy()
        
        # Main backtest loop - process each trading day
        for current_date, day_prices in backtest_prices.iterrows():
            # Check for ex-dividend adjustments
            self._process_ex_dividend_adjustments(current_date, dividend_calendar)
            
            # Check for dividend capture opportunities
            self._check_dividend_opportunities(
                strategy, current_date, day_prices, 
                dividend_calendar, backtest_prices
            )
            
            # Manage existing positions
            self._manage_open_positions(current_date, day_prices, dividend_calendar)
            
            # Update portfolio value
            self._update_portfolio_value(day_prices)
            
            # Record daily snapshot
            self._record_daily_snapshot(current_date)
        
        # Close any remaining positions at end of backtest
        self._close_all_positions(backtest_prices.iloc[-1])
        
        # Calculate performance metrics
        return self._calculate_performance_metrics()
    
    def _prepare_dividend_calendar(self, 
                                 dividend_events: List[DividendEvent]) -> Dict:
        """Organize dividend events by ex-dividend date for quick lookup."""
        calendar = defaultdict(list)
        for event in dividend_events:
            calendar[event.ex_dividend_date.date()].append(event)
        return calendar
    
    def _process_ex_dividend_adjustments(self, current_date: datetime,
                                       dividend_calendar: Dict) -> None:
        """
        Process ex-dividend date adjustments for open positions.
        
        This is crucial for accurate simulation - on ex-dividend date,
        we record the dividend but don't receive cash until payment date.
        """
        events_today = dividend_calendar.get(current_date.date(), [])
        
        for event in events_today:
            if event.symbol in self.open_positions:
                trade = self.open_positions[event.symbol]
                # Record dividend entitlement
                trade.dividend_amount = event.dividend_amount * trade.entry_quantity
                logger.info(f"Recorded ${trade.dividend_amount:.2f} dividend "
                          f"for {trade.entry_quantity} shares of {event.symbol}")
    
    def _check_dividend_opportunities(self, 
                                    strategy: DividendCaptureStrategy,
                                    current_date: datetime,
                                    current_prices: pd.Series,
                                    dividend_calendar: Dict,
                                    price_history: pd.DataFrame) -> None:
        """
        Check for dividend capture entry opportunities.
        
        This evaluates upcoming dividends and determines if conditions
        are favorable for initiating a dividend capture trade.
        """
        # Look ahead for upcoming dividends within our window
        for days_ahead in range(self.config.max_days_before_ex):
            check_date = current_date + timedelta(days=days_ahead)
            events = dividend_calendar.get(check_date.date(), [])
            
            for event in events:
                # Skip if we already have a position
                if event.symbol in self.open_positions:
                    continue
                
                # Skip if dividend yield is too low
                if event.dividend_yield < self.config.min_dividend_yield * 4:  # Annualized
                    continue
                
                # Get strategy signal
                signal = strategy.generate_signals(
                    current_prices['close'],
                    price_history[price_history.index <= current_date].tail(50),
                    None  # Market data not used in basic backtest
                )
                
                if signal['signal'] == 'BUY':
                    self._execute_entry_trade(
                        event.symbol,
                        current_date,
                        current_prices,
                        signal,
                        event
                    )
    
    def _execute_entry_trade(self, symbol: str, entry_date: datetime,
                           prices: pd.Series, signal: Dict,
                           dividend_event: DividendEvent) -> None:
        """Execute entry into a dividend capture position."""
        entry_price = prices['close']
        
        # Calculate position size
        position_size = self._calculate_position_size(
            entry_price,
            signal.get('confidence', 0.5)
        )
        
        if position_size == 0:
            return
        
        # Calculate costs
        commission = position_size * self.config.commission_per_share
        slippage = entry_price * self.config.slippage_bps / 10000
        total_cost = (entry_price + slippage) * position_size + commission
        
        # Check if we have enough cash
        if total_cost > self.cash:
            # Reduce position size to fit available cash
            position_size = int((self.cash - commission) / (entry_price + slippage))
            if position_size < 1:
                return
            total_cost = (entry_price + slippage) * position_size + commission
        
        # Execute trade
        self.trade_counter += 1
        trade = Trade(
            trade_id=self.trade_counter,
            symbol=symbol,
            entry_date=entry_date,
            entry_price=entry_price + slippage,
            entry_quantity=position_size,
            commission_paid=commission,
            slippage_cost=slippage * position_size,
            dividend_event_id=dividend_event.ex_dividend_date.toordinal()
        )
        
        self.open_positions[symbol] = trade
        self.cash -= total_cost
        
        logger.info(f"Entered {symbol}: {position_size} shares @ ${entry_price:.2f} "
                   f"(+${slippage:.3f} slippage), confidence: {signal.get('confidence', 0):.1%}")
    
    def _manage_open_positions(self, current_date: datetime,
                             current_prices: pd.Series,
                             dividend_calendar: Dict) -> None:
        """
        Manage existing open positions, checking for exit conditions.
        
        Exit conditions include:
        - Reached optimal holding period post-dividend
        - Stop loss triggered
        - Maximum holding period exceeded
        """
        positions_to_close = []
        
        for symbol, trade in self.open_positions.items():
            current_price = current_prices.get('close', trade.entry_price)
            
            # Check stop loss
            if self._check_stop_loss(trade, current_price):
                positions_to_close.append((symbol, 'stop_loss'))
                continue
            
            # Check dividend capture exit conditions
            days_held = (current_date - trade.entry_date).days
            
            # Find the dividend event for this trade
            dividend_date = None
            for date, events in dividend_calendar.items():
                for event in events:
                    if event.symbol == symbol:
                        # Check if this is the dividend we're capturing
                        event_date = event.ex_dividend_date.date()
                        if (trade.entry_date.date() < event_date and 
                            event_date <= current_date.date()):
                            dividend_date = event_date
                            break
            
            if dividend_date:
                # We've passed ex-dividend date
                days_since_ex = (current_date.date() - dividend_date).days
                
                if days_since_ex >= self.config.min_days_after_ex:
                    # Check if price has recovered or max holding period reached
                    price_recovery = (current_price - trade.entry_price) / trade.entry_price
                    
                    if (price_recovery > -0.01 or  # Price nearly recovered
                        days_since_ex >= self.config.max_days_after_ex):
                        positions_to_close.append((symbol, 'dividend_capture_exit'))
        
        # Execute closes
        for symbol, reason in positions_to_close:
            self._execute_exit_trade(symbol, current_date, current_prices, reason)
    
    def _execute_exit_trade(self, symbol: str, exit_date: datetime,
                          prices: pd.Series, reason: str) -> None:
        """Execute exit from a position."""
        trade = self.open_positions.get(symbol)
        if not trade:
            return
        
        exit_price = prices.get('close', trade.entry_price)
        
        # Calculate costs
        commission = trade.entry_quantity * self.config.commission_per_share
        slippage = exit_price * self.config.slippage_bps / 10000
        
        # Update trade record
        trade.exit_date = exit_date
        trade.exit_price = exit_price - slippage
        trade.commission_paid += commission
        trade.slippage_cost += slippage * trade.entry_quantity
        
        # Calculate profit/loss including dividend
        total_revenue = (exit_price - slippage) * trade.entry_quantity + trade.dividend_amount
        total_cost = trade.entry_price * trade.entry_quantity
        trade.profit_loss = total_revenue - total_cost - trade.commission_paid
        trade.profit_pct = (trade.profit_loss / total_cost) * 100
        
        # Update cash (including dividend if reinvesting)
        cash_received = (exit_price - slippage) * trade.entry_quantity - commission
        if self.config.reinvest_dividends:
            cash_received += trade.dividend_amount * (1 - self.config.tax_rate)
        
        self.cash += cash_received
        
        # Move to closed trades
        self.trades.append(trade)
        del self.open_positions[symbol]
        
        logger.info(f"Exited {symbol}: {trade.entry_quantity} shares @ ${exit_price:.2f} "
                   f"P&L: ${trade.profit_loss:.2f} ({trade.profit_pct:.2f}%), reason: {reason}")
    
    def _calculate_position_size(self, price: float, confidence: float) -> int:
        """
        Calculate appropriate position size based on configuration and risk.
        
        This implements various position sizing methods including:
        - Fixed percentage of portfolio
        - Volatility-based sizing
        - Kelly Criterion (if configured)
        """
        if self.config.position_sizing_method == 'fixed':
            # Fixed percentage of portfolio
            max_position_value = self.portfolio_value * self.config.max_position_pct
            position_size = int(max_position_value / price)
            
        elif self.config.position_sizing_method == 'volatility':
            # Size inversely proportional to volatility
            # (Implementation would require volatility calculation)
            position_size = int(self.portfolio_value * 0.1 / price)
            
        else:  # Kelly or other advanced methods
            # Simplified Kelly: f = p - q/b
            # Where p = win probability, q = loss probability, b = win/loss ratio
            # Using confidence as proxy for win probability
            kelly_fraction = confidence * 0.25  # Conservative Kelly
            position_value = self.portfolio_value * kelly_fraction
            position_size = int(position_value / price)
        
        # Apply confidence scaling
        position_size = int(position_size * confidence)
        
        # Ensure we don't exceed available cash
        max_affordable = int((self.cash * 0.98) / price)  # Leave some cash for costs
        position_size = min(position_size, max_affordable)
        
        # Round to 10 shares if large enough
        if position_size >= 50:
            position_size = (position_size // 10) * 10
            
        return max(position_size, 0)
    
    def _check_stop_loss(self, trade: Trade, current_price: float) -> bool:
        """Check if stop loss has been triggered."""
        price_change = (current_price - trade.entry_price) / trade.entry_price
        return price_change <= -self.config.stop_loss_pct
    
    def _update_portfolio_value(self, current_prices: pd.Series) -> None:
        """Update total portfolio value including cash and positions."""
        positions_value = 0
        for symbol, trade in self.open_positions.items():
            current_price = current_prices.get('close', trade.entry_price)
            positions_value += current_price * trade.entry_quantity
        
        self.portfolio_value = self.cash + positions_value
    
    def _record_daily_snapshot(self, date: datetime) -> None:
        """Record daily portfolio state for performance analysis."""
        self.portfolio_history.append({
            'date': date,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'positions_value': self.portfolio_value - self.cash,
            'num_positions': len(self.open_positions)
        })
    
    def _close_all_positions(self, final_prices: pd.Series) -> None:
        """Close any remaining open positions at end of backtest."""
        symbols_to_close = list(self.open_positions.keys())
        for symbol in symbols_to_close:
            self._execute_exit_trade(
                symbol, 
                final_prices.name,  # Date from index
                final_prices,
                'backtest_end'
            )
    
    def _calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics for the backtest.
        
        This includes standard metrics plus dividend-specific analytics.
        """
        if not self.portfolio_history:
            return {}
        
        # Convert history to DataFrame for easier analysis
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        portfolio_df['cumulative_returns'] = (1 + portfolio_df['returns']).cumprod() - 1
        
        # Basic metrics
        total_return = (self.portfolio_value - self.config.initial_capital) / self.config.initial_capital
        
        # Calculate annualized metrics
        days = len(portfolio_df)
        years = days / 252
        annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Risk metrics
        daily_returns = portfolio_df['returns'].dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0  # 2% risk-free rate
        
        # Drawdown analysis
        running_max = portfolio_df['portfolio_value'].expanding().max()
        drawdown = (portfolio_df['portfolio_value'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade analysis
        completed_trades = [t for t in self.trades if t.profit_loss is not None]
        winning_trades = [t for t in completed_trades if t.profit_loss > 0]
        losing_trades = [t for t in completed_trades if t.profit_loss < 0]
        
        # Win rate and profit factor
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        
        avg_win = np.mean([t.profit_pct for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t.profit_pct) for t in losing_trades]) if losing_trades else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Dividend-specific metrics
        total_dividends = sum(t.dividend_amount for t in completed_trades)
        dividend_capture_success = sum(1 for t in completed_trades 
                                     if t.profit_loss > 0 and t.dividend_amount > 0)
        
        return {
            # Overall performance
            'total_return': total_return * 100,
            'annual_return': annual_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            
            # Trade statistics
            'total_trades': len(completed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            
            # Dividend metrics
            'total_dividends_captured': total_dividends,
            'dividend_capture_success_rate': (dividend_capture_success / len(completed_trades) * 100) 
                                            if completed_trades else 0,
            'avg_holding_period': np.mean([t.holding_period for t in completed_trades]) 
                                if completed_trades else 0,
            
            # Final values
            'ending_capital': self.portfolio_value,
            'total_commission_paid': sum(t.commission_paid for t in completed_trades),
            'total_slippage_cost': sum(t.slippage_cost for t in completed_trades),
            
            # Additional data
            'portfolio_history': portfolio_df,
            'trades': completed_trades
        }
    
    def plot_results(self, results: Dict) -> None:
        """
        Generate visualization of backtest results.
        
        Creates multiple plots showing:
        - Portfolio value over time
        - Drawdown chart
        - Monthly returns heatmap
        - Trade distribution
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        ax1 = axes[0, 0]
        portfolio_df = results['portfolio_history']
        ax1.plot(portfolio_df.index, portfolio_df['portfolio_value'], 'b-', linewidth=2)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        
        # Drawdown chart
        ax2 = axes[0, 1]
        running_max = portfolio_df['portfolio_value'].expanding().max()
        drawdown = (portfolio_df['portfolio_value'] - running_max) / running_max * 100
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown, 'r-', linewidth=1)
        ax2.set_title('Portfolio Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # Trade profit distribution
        ax3 = axes[1, 0]
        trade_profits = [t.profit_pct for t in results['trades']]
        ax3.hist(trade_profits, bins=30, color='green', alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.set_title('Trade Profit Distribution')
        ax3.set_xlabel('Profit/Loss (%)')
        ax3.set_ylabel('Number of Trades')
        ax3.grid(True, alpha=0.3)
        
        # Win rate by month
        ax4 = axes[1, 1]
        trades_df = pd.DataFrame([{
            'date': t.exit_date,
            'profit': t.profit_loss > 0
        } for t in results['trades']])
        
        if not trades_df.empty:
            monthly_wins = trades_df.set_index('date').resample('M')['profit'].agg(['sum', 'count'])
            monthly_wins['win_rate'] = monthly_wins['sum'] / monthly_wins['count'] * 100
            
            ax4.bar(monthly_wins.index, monthly_wins['win_rate'], color='blue', alpha=0.7)
            ax4.axhline(y=50, color='red', linestyle='--', linewidth=1)
            ax4.set_title('Monthly Win Rate')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Win Rate (%)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive text report of backtest results."""
        report = []
        report.append("=" * 60)
        report.append("DIVIDEND CAPTURE STRATEGY BACKTEST REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Performance Summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 30)
        report.append(f"Total Return: {results['total_return']:.2f}%")
        report.append(f"Annual Return: {results['annual_return']:.2f}%")
        report.append(f"Volatility: {results['volatility']:.2f}%")
        report.append(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        report.append(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        report.append("")
        
        # Trade Statistics
        report.append("TRADE STATISTICS")
        report.append("-" * 30)
        report.append(f"Total Trades: {results['total_trades']}")
        report.append(f"Winning Trades: {results['winning_trades']}")
        report.append(f"Losing Trades: {results['losing_trades']}")
        report.append(f"Win Rate: {results['win_rate']:.2f}%")
        report.append(f"Profit Factor: {results['profit_factor']:.2f}")
        report.append(f"Average Win: {results['avg_win_pct']:.2f}%")
        report.append(f"Average Loss: {results['avg_loss_pct']:.2f}%")
        report.append(f"Average Holding Period: {results['avg_holding_period']:.1f} days")
        report.append("")
        
        # Dividend Metrics
        report.append("DIVIDEND CAPTURE METRICS")
        report.append("-" * 30)
        report.append(f"Total Dividends Captured: ${results['total_dividends_captured']:.2f}")
        report.append(f"Dividend Capture Success Rate: {results['dividend_capture_success_rate']:.2f}%")
        report.append("")
        
        # Cost Analysis
        report.append("COST ANALYSIS")
        report.append("-" * 30)
        report.append(f"Total Commission: ${results['total_commission_paid']:.2f}")
        report.append(f"Total Slippage: ${results['total_slippage_cost']:.2f}")
        total_costs = results['total_commission_paid'] + results['total_slippage_cost']
        cost_impact = (total_costs / self.config.initial_capital) * 100
        report.append(f"Total Trading Costs: ${total_costs:.2f} ({cost_impact:.2f}% of initial capital)")
        report.append("")
        
        # Final Summary
        report.append("FINAL SUMMARY")
        report.append("-" * 30)
        report.append(f"Starting Capital: ${self.config.initial_capital:,.2f}")
        report.append(f"Ending Capital: ${results['ending_capital']:,.2f}")
        report.append(f"Net Profit: ${results['ending_capital'] - self.config.initial_capital:,.2f}")
        
        return "\n".join(report)


# Integration function for running backtests
def run_dividend_backtest(symbol: str, start_date: datetime, end_date: datetime,
                        config: BacktestConfig = None) -> Dict:
    """
    Run a complete dividend capture backtest for a symbol.
    
    This function handles all the setup and execution, making it easy
    to test different configurations and time periods.
    """
    from core.dividend_strategy import DividendCaptureStrategy, DividendDataFetcher
    from config.env_loader import load_env_variables
    
    # Load configuration
    api_keys = load_env_variables()
    config = config or BacktestConfig()
    
    # Initialize components
    strategy = DividendCaptureStrategy(symbol)
    fetcher = DividendDataFetcher(api_keys)
    backtester = DividendBacktester(config)
    
    # Fetch historical data
    logger.info(f"Fetching historical data for {symbol}")
    
    # Get dividend history
    dividend_events = fetcher.fetch_dividend_history(symbol, years=5)
    
    # Get price data (you'll need to implement this based on your data source)
    # For now, using a placeholder
    price_data = pd.DataFrame()  # Replace with actual price data fetching
    
    # Run backtest
    results = backtester.backtest_strategy(
        strategy,
        price_data,
        dividend_events,
        start_date,
        end_date
    )
    
    # Generate report
    report = backtester.generate_report(results)
    print(report)
    
    # Plot results
    backtester.plot_results(results)
    
    return results


if __name__ == "__main__":
    # Example backtest
    config = BacktestConfig(
        initial_capital=100000,
        commission_per_share=0.005,
        max_position_pct=0.25,
        min_dividend_yield=1.5,  # 6% annual
        position_sizing_method='fixed'
    )
    
    # Run backtest for APAM
    results = run_dividend_backtest(
        'APAM',
        datetime(2022, 1, 1),
        datetime(2024, 12, 31),
        config
    )