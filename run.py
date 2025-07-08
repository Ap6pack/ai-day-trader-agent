#!/usr/bin/env python3
"""
Enhanced command-line interface for the AI Day Trader Agent.
Now includes portfolio management, dividend capture capabilities, and multi-strategy analysis.
"""

import sys
import argparse
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from config.env_loader import load_env_variables
from core.pipeline import run_enhanced_analysis, EnhancedTradingPipeline
from core.portfolio_manager import PortfolioManager
from core.dividend_database import DividendDatabase
from utils.logger import get_logger
from utils.formatter import format_analysis_result

# Set up logging
logger = get_logger('ai_day_trader')


class PortfolioCLI:
    """Handles portfolio management CLI commands."""
    
    def __init__(self):
        self.portfolio_manager = PortfolioManager()
    
    def setup_portfolio(self, args):
        """Interactive portfolio setup."""
        print("\n🔧 Portfolio Setup Wizard")
        print("=" * 50)
        
        # Get portfolio name
        name = input("Portfolio name (default: 'default'): ").strip() or "default"
        
        # Check if exists
        existing = self.portfolio_manager.get_portfolio(name)
        if existing:
            print(f"\n⚠️  Portfolio '{name}' already exists!")
            overwrite = input("Overwrite? (y/N): ").lower() == 'y'
            if not overwrite:
                print("Setup cancelled.")
                return
        
        # Get trading capital
        while True:
            try:
                capital_str = input("Trading capital (default: $5000): $").strip()
                capital = float(capital_str) if capital_str else 5000.0
                if capital <= 0:
                    print("Capital must be positive!")
                    continue
                break
            except ValueError:
                print("Invalid amount! Please enter a number.")
        
        # Create portfolio
        try:
            portfolio = self.portfolio_manager.create_portfolio(name, capital)
            print(f"\n✅ Portfolio '{name}' created with ${capital:,.2f} capital")
            
            # Ask about holdings
            add_holdings = input("\nAdd current holdings? (y/N): ").lower() == 'y'
            if add_holdings:
                self._add_holdings_interactive(name)
                
        except Exception as e:
            print(f"\n❌ Error creating portfolio: {e}")
    
    def _add_holdings_interactive(self, portfolio_name: str):
        """Interactively add holdings to portfolio."""
        print("\n📊 Add Holdings (enter 'done' when finished)")
        print("-" * 40)
        
        while True:
            symbol = input("\nStock symbol (or 'done'): ").strip().upper()
            if symbol == 'DONE':
                break
            
            if not symbol:
                continue
            
            try:
                quantity = int(input(f"Quantity of {symbol}: "))
                if quantity <= 0:
                    print("Quantity must be positive!")
                    continue
                
                avg_cost_str = input(f"Average cost per share (optional): $").strip()
                avg_cost = float(avg_cost_str) if avg_cost_str else None
                
                self.portfolio_manager.update_holding(portfolio_name, symbol, quantity, avg_cost)
                print(f"✅ Added {quantity} shares of {symbol}")
                
            except ValueError:
                print("Invalid input! Please try again.")
            except Exception as e:
                print(f"Error adding holding: {e}")
    
    def show_portfolio(self, args):
        """Display portfolio details."""
        name = args.portfolio or "default"
        
        portfolio = self.portfolio_manager.get_portfolio(name)
        if not portfolio:
            print(f"\n❌ Portfolio '{name}' not found")
            return
        
        # Get portfolio value and holdings
        value_info = self.portfolio_manager.get_portfolio_value(name)
        holdings = self.portfolio_manager.get_holdings(name)
        metrics = self.portfolio_manager.get_performance_metrics(name)
        
        # Display portfolio summary
        print(f"\n📊 Portfolio: {name}")
        print("=" * 60)
        print(f"💰 Total Value: ${value_info['total_value']:,.2f}")
        print(f"💵 Cash Available: ${value_info['cash_available']:,.2f}")
        print(f"📈 Holdings Value: ${value_info['holdings_value']:,.2f}")
        
        if metrics:
            print(f"\n📊 Performance Metrics:")
            print(f"   Total Return: ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)")
            print(f"   Realized P&L: ${metrics['realized_pnl']:,.2f}")
            print(f"   Unrealized P&L: ${metrics['unrealized_pnl']:,.2f}")
            print(f"   Win Rate: {metrics['win_rate']:.1f}%")
            print(f"   Total Trades: {metrics['total_trades']}")
        
        if holdings:
            print("\n📈 Current Holdings:")
            print("-" * 60)
            print(f"{'Symbol':<10} {'Quantity':>10} {'Avg Cost':>12} {'Current':>12} {'P&L':>12}")
            print("-" * 60)
            
            for holding in value_info['holdings_details']:
                symbol = holding['symbol']
                quantity = holding['quantity']
                avg_cost = holding['avg_cost']
                current = holding['current_price']
                pnl = holding['unrealized_pnl']
                pnl_pct = holding['unrealized_pnl_pct']
                
                pnl_str = f"${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
                print(f"{symbol:<10} {quantity:>10} ${avg_cost:>11,.2f} ${current:>11,.2f} {pnl_str:>12} ({pnl_pct:+.1f}%)")
        else:
            print("\n📊 No holdings in portfolio")
    
    def update_capital(self, args):
        """Update portfolio trading capital."""
        name = args.portfolio or "default"
        
        if self.portfolio_manager.update_trading_capital(name, args.amount):
            print(f"\n✅ Updated trading capital for '{name}' to ${args.amount:,.2f}")
        else:
            print(f"\n❌ Portfolio '{name}' not found")
    
    def add_holding(self, args):
        """Add or update a holding."""
        name = args.portfolio or "default"
        
        try:
            self.portfolio_manager.update_holding(
                name, args.symbol, args.quantity, args.cost
            )
            print(f"\n✅ Updated holding: {args.quantity} shares of {args.symbol}")
            
            # Show updated portfolio value
            value_info = self.portfolio_manager.get_portfolio_value(name)
            print(f"💰 New portfolio value: ${value_info['total_value']:,.2f}")
            
        except Exception as e:
            print(f"\n❌ Error updating holding: {e}")
    
    def remove_holding(self, args):
        """Remove a holding from portfolio."""
        name = args.portfolio or "default"
        
        try:
            self.portfolio_manager.update_holding(name, args.symbol, 0)
            print(f"\n✅ Removed {args.symbol} from portfolio '{name}'")
        except Exception as e:
            print(f"\n❌ Error removing holding: {e}")
    
    def list_portfolios(self, args):
        """List all portfolios."""
        portfolios = self.portfolio_manager.list_portfolios()
        
        if not portfolios:
            print("\n📊 No portfolios found. Create one with --setup-portfolio")
            return
        
        print("\n📊 Available Portfolios:")
        print("=" * 60)
        
        for p in portfolios:
            value_info = self.portfolio_manager.get_portfolio_value(p['name'])
            print(f"\n📁 {p['name']}")
            print(f"   Capital: ${p['trading_capital']:,.2f}")
            print(f"   Total Value: ${value_info['total_value']:,.2f}")
            print(f"   Created: {p['created_at']}")
    
    def record_trade(self, args):
        """Record a manual trade."""
        name = args.portfolio or "default"
        
        try:
            trade_id = self.portfolio_manager.record_trade(
                name=name,
                symbol=args.symbol,
                action=args.action,
                quantity=args.quantity,
                price=args.price,
                strategy=args.strategy,
                confidence=args.confidence,
                notes=args.notes
            )
            
            print(f"\n✅ Trade recorded (ID: {trade_id})")
            print(f"   {args.action} {args.quantity} shares of {args.symbol} @ ${args.price}")
            
            # Show updated portfolio
            value_info = self.portfolio_manager.get_portfolio_value(name)
            print(f"   New cash available: ${value_info['cash_available']:,.2f}")
            
        except Exception as e:
            print(f"\n❌ Error recording trade: {e}")
    
    def show_trades(self, args):
        """Show trade history."""
        name = args.portfolio or "default"
        days = args.days or 30
        
        trades = self.portfolio_manager.get_trade_history(name, days)
        
        if not trades:
            print(f"\n📊 No trades found in the last {days} days")
            return
        
        print(f"\n📊 Trade History - Last {days} days")
        print("=" * 80)
        print(f"{'Date':<20} {'Symbol':<8} {'Action':<6} {'Qty':>6} {'Price':>10} {'Total':>12} {'Strategy':<15}")
        print("-" * 80)
        
        for trade in trades:
            date = trade['timestamp'][:19]  # Trim to seconds
            symbol = trade['symbol']
            action = trade['action']
            qty = trade['quantity']
            price = trade['price']
            total = trade['total_value']
            strategy = trade['strategy'] or 'manual'
            
            print(f"{date:<20} {symbol:<8} {action:<6} {qty:>6} ${price:>9.2f} ${total:>11.2f} {strategy:<15}")
    
    def backup_database(self, args):
        """Create database backup."""
        try:
            backup_path = self.portfolio_manager.backup_database(args.output)
            print(f"\n✅ Database backed up to: {backup_path}")
        except Exception as e:
            print(f"\n❌ Backup failed: {e}")
    
    def restore_database(self, args):
        """Restore database from backup."""
        try:
            if self.portfolio_manager.restore_database(args.backup_file):
                print(f"\n✅ Database restored from: {args.backup_file}")
            else:
                print("\n❌ Restore failed")
        except Exception as e:
            print(f"\n❌ Restore failed: {e}")
    
    def analyze_portfolio(self, args):
        """Analyze all holdings in a portfolio."""
        name = args.portfolio or "default"
        
        portfolio = self.portfolio_manager.get_portfolio(name)
        if not portfolio:
            print(f"\n❌ Portfolio '{name}' not found")
            return
        
        holdings = self.portfolio_manager.get_holdings(name)
        if not holdings:
            print(f"\n📊 No holdings in portfolio '{name}' to analyze")
            return
        
        print(f"\n🤖 AI Day Trader Agent - Analyzing Portfolio: {name}")
        print("=" * 60)
        print(f"📊 Analyzing {len(holdings)} holdings...")
        
        # Load API keys once
        api_keys = load_env_variables()
        
        # Track recommendations
        buy_recommendations = []
        sell_recommendations = []
        hold_positions = []
        
        for i, holding in enumerate(holdings, 1):
            symbol = holding['symbol']
            quantity = holding['quantity']
            avg_cost = holding['avg_cost']
            
            print(f"\n[{i}/{len(holdings)}] Analyzing {symbol} ({quantity} shares @ ${avg_cost:.2f})...")
            
            try:
                # Initialize pipeline with portfolio context
                pipeline = EnhancedTradingPipeline(symbol, name)
                
                # Run analysis
                result = pipeline.run_analysis(api_keys)
                
                # Check if result is an error
                if result.get('error'):
                    error_type = result.get('error_type', 'unknown')
                    error_msg = result.get('message', 'Unknown error')
                    logger.error(f"Analysis error for {symbol}: {error_type} - {error_msg}")
                    print(f"   ❌ {error_msg}")
                    continue
                
                # Check if result has required fields
                if 'signal' not in result:
                    logger.error(f"Invalid result format for {symbol}: missing 'signal' key")
                    print(f"   ❌ Invalid analysis result format")
                    continue
                
                # Categorize recommendation
                signal = result.get('signal', 'HOLD')
                quantity_rec = result.get('quantity', 0)
                confidence = result.get('confidence', '0%')
                reason = result.get('primary_reason', 'No reason provided')
                
                if signal == 'BUY' and quantity_rec > 0:
                    buy_recommendations.append({
                        'symbol': symbol,
                        'current_holding': quantity,
                        'recommendation': result
                    })
                elif signal == 'SELL' and quantity_rec > 0:
                    sell_recommendations.append({
                        'symbol': symbol,
                        'current_holding': quantity,
                        'recommendation': result
                    })
                else:
                    hold_positions.append({
                        'symbol': symbol,
                        'current_holding': quantity,
                        'recommendation': result
                    })
                
                # Show brief summary
                # Handle confidence as either string or float
                if isinstance(confidence, str):
                    conf_display = confidence
                else:
                    conf_display = f"{confidence:.1%}"
                    
                print(f"   → {signal} {quantity_rec} shares (confidence: {conf_display})")
                print(f"   → Reason: {reason}")
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                print(f"   ❌ Error: {str(e)}")
        
        # Display summary
        print("\n" + "=" * 60)
        print("📊 PORTFOLIO ANALYSIS SUMMARY")
        print("=" * 60)
        
        if buy_recommendations:
            print(f"\n🟢 BUY RECOMMENDATIONS ({len(buy_recommendations)}):")
            for rec in buy_recommendations:
                r = rec['recommendation']
                conf = r.get('confidence', '0%')
                conf_display = conf if isinstance(conf, str) else f"{conf:.1%}"
                print(f"   {rec['symbol']}: BUY {r['quantity']} shares (confidence: {conf_display})")
                print(f"      Reason: {r['primary_reason']}")
        
        if sell_recommendations:
            print(f"\n🔴 SELL RECOMMENDATIONS ({len(sell_recommendations)}):")
            for rec in sell_recommendations:
                r = rec['recommendation']
                conf = r.get('confidence', '0%')
                conf_display = conf if isinstance(conf, str) else f"{conf:.1%}"
                print(f"   {rec['symbol']}: SELL {r['quantity']} shares (confidence: {conf_display})")
                print(f"      Reason: {r['primary_reason']}")
        
        if hold_positions:
            print(f"\n🟡 HOLD POSITIONS ({len(hold_positions)}):")
            for rec in hold_positions:
                print(f"   {rec['symbol']}: HOLD {rec['current_holding']} shares")
        
        # Portfolio value summary
        value_info = self.portfolio_manager.get_portfolio_value(name)
        print(f"\n💰 Portfolio Value: ${value_info['total_value']:,.2f}")
        print(f"💵 Cash Available: ${value_info['cash_available']:,.2f}")
        
        print("\n✅ Portfolio analysis complete!")


def run_analysis(ticker: str, portfolio_name: str = "default", 
                override_capital: Optional[float] = None,
                override_holdings: Optional[int] = None):
    """Run trading analysis with portfolio context."""
    try:
        # Load API keys
        api_keys = load_env_variables()
        
        # Initialize portfolio manager
        pm = PortfolioManager()
        
        # Get portfolio context
        portfolio = pm.get_portfolio(portfolio_name)
        if not portfolio and portfolio_name != "default":
            print(f"\n⚠️  Portfolio '{portfolio_name}' not found, using defaults")
        
        # Initialize pipeline with portfolio context
        pipeline = EnhancedTradingPipeline(ticker)
        
        # Override capital/holdings if specified
        if override_capital:
            pipeline.config.TRADING_CAPITAL = override_capital
        elif portfolio:
            value_info = pm.get_portfolio_value(portfolio_name)
            pipeline.config.TRADING_CAPITAL = value_info['cash_available']
        
        # Run analysis
        print(f"\n🤖 AI Day Trader Agent - Analysis for {ticker}")
        print(f"📊 Portfolio: {portfolio_name}")
        print("=" * 60)
        
        result = pipeline.run_analysis(api_keys)
        
        # Display formatted results
        print("\n📊 Analysis Results:")
        print("-" * 40)
        formatted_output = format_analysis_result(result)
        print(formatted_output)
        
        # If it's a BUY/SELL recommendation, ask about execution
        if result['signal'] in ['BUY', 'SELL'] and result['quantity'] > 0:
            print(f"\n💡 Recommendation: {result['signal']} {result['quantity']} shares")
            
            if portfolio:
                execute = input("\nExecute this trade? (y/N): ").lower() == 'y'
                if execute:
                    # Get current price (use last close as approximation)
                    current_price = result.get('all_signals', {}).get('technical', {}).get('current_price', 0)
                    if current_price > 0:
                        trade_id = pm.record_trade(
                            name=portfolio_name,
                            symbol=ticker,
                            action=result['signal'],
                            quantity=result['quantity'],
                            price=current_price,
                            strategy=result['primary_strategy'],
                            confidence=float(result['confidence'].rstrip('%')) / 100,
                            notes=f"Auto-executed: {result['primary_reason']}"
                        )
                        print(f"\n✅ Trade executed (ID: {trade_id})")
                    else:
                        print("\n❌ Could not execute: price unavailable")
        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


def main():
    """Main entry point with enhanced CLI."""
    parser = argparse.ArgumentParser(
        description="AI Day Trader Agent - Portfolio-based trading analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a stock
  python run.py AAPL
  
  # Analyze with specific portfolio
  python run.py AAPL --portfolio my_portfolio
  
  # Setup a new portfolio
  python run.py --setup-portfolio
  
  # Show portfolio details
  python run.py --show-portfolio
  
  # Add a holding
  python run.py --add-holding AAPL 100 --cost 150.00
  
  # Record a trade
  python run.py --record-trade AAPL BUY 50 155.00
  
  # Analyze entire portfolio
  python run.py --analyze-portfolio
        """
    )
    
    # Analysis mode (default)
    parser.add_argument('ticker', nargs='?', help='Stock ticker to analyze')
    parser.add_argument('--portfolio', '-p', help='Portfolio name (default: "default")')
    parser.add_argument('--capital', type=float, help='Override trading capital')
    parser.add_argument('--holdings', type=int, help='Override current holdings')
    
    # Portfolio management commands
    parser.add_argument('--setup-portfolio', action='store_true', help='Interactive portfolio setup')
    parser.add_argument('--show-portfolio', action='store_true', help='Show portfolio details')
    parser.add_argument('--list-portfolios', action='store_true', help='List all portfolios')
    parser.add_argument('--update-capital', type=float, metavar='AMOUNT', help='Update trading capital')
    parser.add_argument('--analyze-portfolio', action='store_true', help='Analyze all holdings in portfolio')
    
    # Holdings management
    parser.add_argument('--add-holding', nargs=2, metavar=('SYMBOL', 'QTY'), help='Add/update holding')
    parser.add_argument('--cost', type=float, help='Average cost for --add-holding')
    parser.add_argument('--remove-holding', metavar='SYMBOL', help='Remove holding')
    
    # Trade management
    parser.add_argument('--record-trade', nargs=4, metavar=('SYMBOL', 'ACTION', 'QTY', 'PRICE'),
                       help='Record a trade (ACTION: BUY/SELL)')
    parser.add_argument('--strategy', help='Strategy for --record-trade')
    parser.add_argument('--confidence', type=float, help='Confidence (0-1) for --record-trade')
    parser.add_argument('--notes', help='Notes for --record-trade')
    parser.add_argument('--show-trades', action='store_true', help='Show trade history')
    parser.add_argument('--days', type=int, help='Days of history for --show-trades')
    
    # Database operations
    parser.add_argument('--backup', action='store_true', help='Backup database')
    parser.add_argument('--restore', metavar='FILE', help='Restore database from backup')
    parser.add_argument('--output', '-o', help='Output file for backup')
    
    args = parser.parse_args()
    
    # Initialize CLI handler
    cli = PortfolioCLI()
    
    # Handle portfolio management commands
    if args.setup_portfolio:
        cli.setup_portfolio(args)
    elif args.show_portfolio:
        cli.show_portfolio(args)
    elif args.list_portfolios:
        cli.list_portfolios(args)
    elif args.update_capital:
        args.amount = args.update_capital
        cli.update_capital(args)
    elif args.add_holding:
        args.symbol, qty = args.add_holding
        args.quantity = int(qty)
        cli.add_holding(args)
    elif args.remove_holding:
        args.symbol = args.remove_holding
        cli.remove_holding(args)
    elif args.record_trade:
        symbol, action, qty, price = args.record_trade
        args.symbol = symbol
        args.action = action.upper()
        args.quantity = int(qty)
        args.price = float(price)
        cli.record_trade(args)
    elif args.show_trades:
        cli.show_trades(args)
    elif args.backup:
        cli.backup_database(args)
    elif args.restore:
        args.backup_file = args.restore
        cli.restore_database(args)
    elif args.analyze_portfolio:
        cli.analyze_portfolio(args)
    elif args.ticker:
        # Run analysis mode
        run_analysis(
            args.ticker.upper(),
            args.portfolio or "default",
            args.capital,
            args.holdings
        )
    else:
        # No command specified
        parser.print_help()


if __name__ == "__main__":
    main()
