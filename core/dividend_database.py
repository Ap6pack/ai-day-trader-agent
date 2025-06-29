#!/usr/bin/env python3
"""
Database management for dividend capture strategy.

This module provides a persistent storage solution for dividend events,
trading history, and performance metrics. We use SQLite for simplicity,
but the design allows easy migration to PostgreSQL for production use.
"""

import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATABASE_PATH = os.getenv('DATABASE_PATH', str("data/dividend_trading.db"))

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, asdict

def _configure_sqlite_adapters():
    """Configure SQLite datetime adapters for Python 3.12+ compatibility."""
    sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
    sqlite3.register_converter("TIMESTAMP", lambda b: datetime.fromisoformat(b.decode()))

logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """
    Exception raised when database connection fails.
    
    Provides detailed error information and potential recovery suggestions.
    """
    
    def __init__(self, message: str, original_error: Exception = None, db_path: str = None):
        self.message = message
        self.original_error = original_error
        self.db_path = db_path
        
        # Create detailed error message
        error_details = [message]
        
        if db_path:
            error_details.append(f"Database path: {db_path}")
            
        if original_error:
            error_details.append(f"Original error: {type(original_error).__name__}: {str(original_error)}")
            
        # Add recovery suggestions
        error_details.append("Recovery suggestions:")
        error_details.append("1. Check if the database directory exists and is writable")
        error_details.append("2. Verify database file permissions")
        error_details.append("3. Ensure sufficient disk space")
        error_details.append("4. Check if another process is using the database")
        
        super().__init__("\n".join(error_details))
    
    def get_recovery_actions(self) -> List[str]:
        """Get list of potential recovery actions."""
        actions = []
        
        if self.db_path:
            db_dir = Path(self.db_path).parent
            actions.append(f"Create directory: mkdir -p {db_dir}")
            actions.append(f"Check permissions: ls -la {db_dir}")
            
        actions.append("Check disk space: df -h")
        actions.append("Check for locks: lsof | grep .db")
        
        return actions


class DividendDatabase:
    """
    Manages all dividend-related data storage and retrieval.
    
    The database design follows these principles:
    1. Normalize data to prevent redundancy
    2. Index frequently queried fields for performance
    3. Store both planned and actual values for learning
    4. Track all trades for performance analysis
    """
    def __init__(self, db_path: str = "data/dividend_trading.db"):
        _configure_sqlite_adapters()  # Add this line
        self.db_path = db_path
        self._init_database()
        
    def ensure_database_directory(self):
        """Ensure database directory exists with proper permissions."""
        db_dir = Path(self.db_path).parent
        if not db_dir.exists():
            db_dir.mkdir(parents=True, mode=0o755)
            logger.info(f"Created database directory: {db_dir}")
   
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        try:
            self.ensure_database_directory()
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Database error: {e}")
                raise
            finally:
                conn.close()
        except sqlite3.OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            raise DatabaseConnectionError(f"Unable to connect to database: {e}")
    
    def _init_database(self):
        """Initialize database schema if not exists."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Dividend events table - tracks all dividend announcements
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dividend_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    declaration_date DATE,
                    ex_dividend_date DATE NOT NULL,
                    record_date DATE NOT NULL,
                    payment_date DATE NOT NULL,
                    dividend_amount REAL NOT NULL,
                    dividend_yield REAL NOT NULL,
                    frequency TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, ex_dividend_date)
                )
            """)
            
            # Dividend capture trades - tracks specific dividend capture attempts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dividend_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dividend_event_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    entry_date TIMESTAMP,
                    entry_price REAL,
                    entry_quantity INTEGER,
                    exit_date TIMESTAMP,
                    exit_price REAL,
                    exit_quantity INTEGER,
                    dividend_received REAL,
                    profit_loss REAL,
                    profit_percentage REAL,
                    strategy_version TEXT,
                    market_conditions TEXT,  -- JSON field for storing conditions
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dividend_event_id) REFERENCES dividend_events(id)
                )
            """)
            
            # Price patterns around dividends - for machine learning
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dividend_price_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dividend_event_id INTEGER NOT NULL,
                    days_before_ex INTEGER NOT NULL,
                    price_open REAL,
                    price_high REAL,
                    price_low REAL,
                    price_close REAL,
                    volume INTEGER,
                    price_change_pct REAL,
                    volatility REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dividend_event_id) REFERENCES dividend_events(id),
                    UNIQUE(dividend_event_id, days_before_ex)
                )
            """)
            
            # Strategy performance metrics - track how well strategies perform
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    period_start DATE NOT NULL,
                    period_end DATE NOT NULL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    total_profit_loss REAL,
                    average_profit_pct REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    best_trade_id INTEGER,
                    worst_trade_id INTEGER,
                    metadata TEXT,  -- JSON field for additional metrics
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(strategy_name, symbol, period_start, period_end)
                )
            """)
            
            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_dividend_events_symbol_date 
                ON dividend_events(symbol, ex_dividend_date DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_dividend_trades_symbol_date 
                ON dividend_trades(symbol, entry_date DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_price_patterns_event_days 
                ON dividend_price_patterns(dividend_event_id, days_before_ex)
            """)
            
            logger.info("Database schema initialized successfully")
    
    def save_dividend_event(self, event: 'DividendEvent') -> int:
        """
        Save or update a dividend event in the database.
        
        Returns the event ID for linking with trades and patterns.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if event already exists
            cursor.execute("""
                SELECT id FROM dividend_events 
                WHERE symbol = ? AND ex_dividend_date = ?
            """, (event.symbol, event.ex_dividend_date))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing event
                cursor.execute("""
                    UPDATE dividend_events SET
                        declaration_date = ?,
                        record_date = ?,
                        payment_date = ?,
                        dividend_amount = ?,
                        dividend_yield = ?,
                        frequency = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    event.declaration_date,
                    event.record_date,
                    event.payment_date,
                    event.dividend_amount,
                    event.dividend_yield,
                    event.frequency,
                    existing['id']
                ))
                return existing['id']
            else:
                # Insert new event
                cursor.execute("""
                    INSERT INTO dividend_events (
                        symbol, declaration_date, ex_dividend_date,
                        record_date, payment_date, dividend_amount,
                        dividend_yield, frequency
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.symbol,
                    event.declaration_date,
                    event.ex_dividend_date,
                    event.record_date,
                    event.payment_date,
                    event.dividend_amount,
                    event.dividend_yield,
                    event.frequency
                ))
                return cursor.lastrowid
    
    def record_trade_entry(self, symbol: str, dividend_event_id: int,
                          entry_date: datetime, entry_price: float,
                          entry_quantity: int, market_conditions: Dict) -> int:
        """Record entry into a dividend capture trade."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO dividend_trades (
                    dividend_event_id, symbol, entry_date,
                    entry_price, entry_quantity, market_conditions
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                dividend_event_id,
                symbol,
                entry_date,
                entry_price,
                entry_quantity,
                json.dumps(market_conditions)
            ))
            
            return cursor.lastrowid
    
    def record_trade_exit(self, trade_id: int, exit_date: datetime,
                         exit_price: float, exit_quantity: int,
                         dividend_received: float = 0.0) -> None:
        """Record exit from a dividend capture trade and calculate profit/loss."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get entry information
            cursor.execute("""
                SELECT entry_price, entry_quantity 
                FROM dividend_trades WHERE id = ?
            """, (trade_id,))
            
            trade = cursor.fetchone()
            if not trade:
                raise ValueError(f"Trade {trade_id} not found")
            
            # Calculate profit/loss
            entry_cost = trade['entry_price'] * trade['entry_quantity']
            exit_revenue = exit_price * exit_quantity
            profit_loss = exit_revenue - entry_cost + dividend_received
            profit_percentage = (profit_loss / entry_cost) * 100 if entry_cost > 0 else 0
            
            # Update trade record
            cursor.execute("""
                UPDATE dividend_trades SET
                    exit_date = ?,
                    exit_price = ?,
                    exit_quantity = ?,
                    dividend_received = ?,
                    profit_loss = ?,
                    profit_percentage = ?
                WHERE id = ?
            """, (
                exit_date,
                exit_price,
                exit_quantity,
                dividend_received,
                profit_loss,
                profit_percentage,
                trade_id
            ))
            
            logger.info(f"Trade {trade_id} closed with {profit_percentage:.2f}% profit")
    
    def save_price_patterns(self, dividend_event_id: int,
                           price_data: pd.DataFrame,
                           ex_dividend_date: datetime) -> None:
        """
        Save price patterns around a dividend event for pattern analysis.
        
        This data is crucial for training the ML model to recognize
        profitable dividend capture opportunities.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Save data for 15 days before and 10 days after ex-dividend
            start_date = ex_dividend_date - timedelta(days=15)
            end_date = ex_dividend_date + timedelta(days=10)
            
            relevant_data = price_data[
                (price_data.index >= start_date) & 
                (price_data.index <= end_date)
            ].copy()
            
            for date, row in relevant_data.iterrows():
                days_before_ex = (ex_dividend_date - date).days
                
                # Calculate additional metrics
                price_change_pct = row.get('close_pct_change', 0) * 100
                volatility = row.get('volatility', 0)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO dividend_price_patterns (
                        dividend_event_id, days_before_ex, price_open,
                        price_high, price_low, price_close, volume,
                        price_change_pct, volatility
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    dividend_event_id,
                    days_before_ex,
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row.get('volume', 0),
                    price_change_pct,
                    volatility
                ))
    
    def get_historical_patterns(self, symbol: str, 
                              lookback_days: int = 365) -> pd.DataFrame:
        """
        Retrieve historical price patterns around dividend events.
        
        This data is used to train the ML model and identify optimal
        entry/exit points for dividend capture.
        """
        with self.get_connection() as conn:
            query = """
                SELECT 
                    de.symbol,
                    de.ex_dividend_date,
                    de.dividend_amount,
                    de.dividend_yield,
                    dpp.days_before_ex,
                    dpp.price_close,
                    dpp.price_change_pct,
                    dpp.volume,
                    dpp.volatility
                FROM dividend_price_patterns dpp
                JOIN dividend_events de ON dpp.dividend_event_id = de.id
                WHERE de.symbol = ?
                  AND de.ex_dividend_date >= date('now', '-' || ? || ' days')
                ORDER BY de.ex_dividend_date DESC, dpp.days_before_ex DESC
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, lookback_days))
            
            # Convert date strings to datetime
            df['ex_dividend_date'] = pd.to_datetime(df['ex_dividend_date'])
            
            return df
    
    def get_trade_performance(self, symbol: str, 
                            start_date: Optional[datetime] = None) -> Dict:
        """
        Calculate performance metrics for dividend capture trades.
        
        This helps you understand how well the strategy is working and
        identify areas for improvement.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Build query based on parameters
            query = """
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(profit_loss) as total_profit_loss,
                    AVG(profit_percentage) as avg_profit_pct,
                    MAX(profit_percentage) as best_trade_pct,
                    MIN(profit_percentage) as worst_trade_pct,
                    AVG(CASE WHEN profit_loss > 0 THEN profit_percentage END) as avg_win_pct,
                    AVG(CASE WHEN profit_loss < 0 THEN profit_percentage END) as avg_loss_pct
                FROM dividend_trades
                WHERE symbol = ?
                  AND exit_date IS NOT NULL
            """
            
            params = [symbol]
            
            if start_date:
                query += " AND entry_date >= ?"
                params.append(start_date)
            
            cursor.execute(query, params)
            result = cursor.fetchone()
            
            # Calculate additional metrics
            total_trades = result['total_trades'] or 0
            winning_trades = result['winning_trades'] or 0
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            avg_win = result['avg_win_pct'] or 0
            avg_loss = abs(result['avg_loss_pct'] or 0)
            
            # Calculate profit factor
            profit_factor = (avg_win / avg_loss) if avg_loss > 0 else float('inf')
            
            return {
                'symbol': symbol,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': result['losing_trades'] or 0,
                'win_rate': win_rate,
                'total_profit_loss': result['total_profit_loss'] or 0,
                'average_profit_pct': result['avg_profit_pct'] or 0,
                'best_trade_pct': result['best_trade_pct'] or 0,
                'worst_trade_pct': result['worst_trade_pct'] or 0,
                'average_win_pct': avg_win,
                'average_loss_pct': avg_loss,
                'profit_factor': profit_factor
            }
    
    def save_strategy_performance(self, strategy_name: str, symbol: str,
                                period_start: datetime, period_end: datetime,
                                metrics: Dict) -> None:
        """Save aggregated strategy performance metrics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO strategy_performance (
                    strategy_name, symbol, period_start, period_end,
                    total_trades, winning_trades, losing_trades,
                    total_profit_loss, average_profit_pct, sharpe_ratio,
                    max_drawdown, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_name,
                symbol,
                period_start,
                period_end,
                metrics.get('total_trades', 0),
                metrics.get('winning_trades', 0),
                metrics.get('losing_trades', 0),
                metrics.get('total_profit_loss', 0),
                metrics.get('average_profit_pct', 0),
                metrics.get('sharpe_ratio', 0),
                metrics.get('max_drawdown', 0),
                json.dumps(metrics.get('additional_metrics', {}))
            ))
    
    def get_upcoming_dividends(self, symbol: str, days_ahead: int = 30) -> List[Dict]:
        """Get upcoming dividend events for planning."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM dividend_events
                WHERE symbol = ?
                  AND ex_dividend_date >= date('now')
                  AND ex_dividend_date <= date('now', '+' || ? || ' days')
                ORDER BY ex_dividend_date ASC
            """, (symbol, days_ahead))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_dividend_capture_opportunities(self, min_yield: float = 5.0,
                                         days_ahead: int = 14) -> pd.DataFrame:
        """
        Find upcoming dividend capture opportunities across all tracked stocks.
        
        This helps identify the best opportunities when capital is limited.
        """
        with self.get_connection() as conn:
            query = """
                SELECT 
                    de.*,
                    sp.average_profit_pct as historical_avg_profit,
                    sp.winning_trades,
                    sp.total_trades,
                    CAST(sp.winning_trades AS FLOAT) / sp.total_trades * 100 as win_rate
                FROM dividend_events de
                LEFT JOIN strategy_performance sp 
                    ON de.symbol = sp.symbol 
                    AND sp.strategy_name = 'dividend_capture'
                WHERE de.dividend_yield >= ?
                  AND de.ex_dividend_date >= date('now')
                  AND de.ex_dividend_date <= date('now', '+' || ? || ' days')
                ORDER BY de.dividend_yield DESC, de.ex_dividend_date ASC
            """
            
            df = pd.read_sql_query(query, conn, params=(min_yield, days_ahead))
            
            # Convert date strings to datetime
            date_columns = ['declaration_date', 'ex_dividend_date', 'record_date', 'payment_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            return df


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize database
    db = DividendDatabase()
    
    # Example: Save a dividend event
    from dividend_strategy import DividendEvent
    
    event = DividendEvent(
        symbol='APAM',
        declaration_date=datetime(2024, 4, 15),
        ex_dividend_date=datetime(2024, 5, 16),
        record_date=datetime(2024, 5, 17),
        payment_date=datetime(2024, 5, 30),
        dividend_amount=0.88,
        dividend_yield=7.5,
        frequency='quarterly'
    )
    
    event_id = db.save_dividend_event(event)
    print(f"Saved dividend event with ID: {event_id}")
    
    # Example: Record a trade entry
    trade_id = db.record_trade_entry(
        symbol='APAM',
        dividend_event_id=event_id,
        entry_date=datetime(2024, 5, 10),
        entry_price=45.50,
        entry_quantity=50,
        market_conditions={'vix': 15.2, 'market_trend': 'bullish'}
    )
    print(f"Recorded trade entry with ID: {trade_id}")
    
    # Example: Get performance metrics
    performance = db.get_trade_performance('APAM')
    print("\nPerformance Metrics:")
    for key, value in performance.items():
        print(f"  {key}: {value}")
    
    # Example: Find upcoming opportunities
    opportunities = db.get_dividend_capture_opportunities(min_yield=5.0)
    print(f"\nFound {len(opportunities)} dividend capture opportunities")
    print(opportunities[['symbol', 'ex_dividend_date', 'dividend_yield', 'historical_avg_profit']])
