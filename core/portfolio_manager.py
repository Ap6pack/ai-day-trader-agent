#!/usr/bin/env python3
"""
Portfolio Manager - Database-backed portfolio management system.
Handles all portfolio operations including holdings, trades, and performance tracking.
"""

import sqlite3
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import os

logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    Manages portfolios, holdings, and trades using SQLite database.
    Provides thread-safe operations with proper transaction handling.
    """
    
    def __init__(self, db_path: str = "data/portfolios.db"):
        """Initialize the portfolio manager with database connection."""
        self.db_path = db_path
        self._ensure_data_directory()
        self.init_database()
        logger.info(f"Portfolio Manager initialized with database at {db_path}")
    
    def _ensure_data_directory(self):
        """Ensure the data directory exists."""
        data_dir = os.path.dirname(self.db_path)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, mode=0o755)
            logger.info(f"Created data directory: {data_dir}")
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def init_database(self):
        """Initialize database schema from the roadmap specification."""
        schema_sql = """
        -- Users for authentication
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            is_admin BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            CONSTRAINT username_length CHECK (length(username) >= 3 AND length(username) <= 50),
            CONSTRAINT email_format CHECK (email LIKE '%_@__%.__%')
        );

        -- User portfolios
        CREATE TABLE IF NOT EXISTS portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT UNIQUE NOT NULL,
            trading_capital REAL NOT NULL DEFAULT 5000.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        -- Current holdings
        CREATE TABLE IF NOT EXISTS holdings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 0,
            avg_cost REAL NOT NULL DEFAULT 0.0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id),
            UNIQUE(portfolio_id, symbol)
        );

        -- Trade history
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL, -- 'BUY', 'SELL'
            quantity INTEGER NOT NULL,
            price REAL NOT NULL,
            total_value REAL NOT NULL,
            fees REAL DEFAULT 0.0,
            strategy TEXT, -- 'dividend_capture', 'technical', 'sentiment'
            confidence REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
        );

        -- Portfolio performance snapshots
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            total_value REAL NOT NULL,
            cash_available REAL NOT NULL,
            daily_change REAL,
            daily_change_pct REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
        );

        -- Price alerts and notifications
        CREATE TABLE IF NOT EXISTS price_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            alert_type TEXT NOT NULL, -- 'stop_loss', 'take_profit', 'price_target', 'dividend_reminder'
            target_price REAL,
            current_price REAL,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            triggered_at TIMESTAMP,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
        );

        -- Pending orders (for future automation)
        CREATE TABLE IF NOT EXISTS pending_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL, -- 'BUY', 'SELL'
            quantity INTEGER NOT NULL,
            order_type TEXT NOT NULL, -- 'MARKET', 'LIMIT', 'STOP'
            price REAL,
            status TEXT DEFAULT 'PENDING', -- 'PENDING', 'EXECUTED', 'CANCELLED', 'EXPIRED'
            created_by_chat BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            executed_at TIMESTAMP,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
        );

        -- Chat history for AI chatbot
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            intent_type TEXT, -- 'dividend_timing', 'portfolio_status', 'trade_execution', etc.
            extracted_data JSON, -- Parsed entities (symbol, quantity, etc.)
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
        );

        -- Watchlists
        CREATE TABLE IF NOT EXISTS watchlists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id),
            UNIQUE(portfolio_id, symbol)
        );
        
        -- JWT token blacklist for logout functionality
        CREATE TABLE IF NOT EXISTS token_blacklist (
            jti TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            blacklisted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Portfolio analysis jobs
        CREATE TABLE IF NOT EXISTS analysis_jobs (
            job_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            portfolio_name TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending', 'running', 'completed', 'failed')),
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            result_json TEXT,
            error TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        CREATE INDEX IF NOT EXISTS idx_token_blacklist_expires ON token_blacklist(expires_at);
        CREATE INDEX IF NOT EXISTS idx_analysis_jobs_user ON analysis_jobs(user_id);
        CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status ON analysis_jobs(status);
        CREATE INDEX IF NOT EXISTS idx_holdings_portfolio ON holdings(portfolio_id);
        CREATE INDEX IF NOT EXISTS idx_trades_portfolio ON trades(portfolio_id);
        CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
        CREATE INDEX IF NOT EXISTS idx_alerts_portfolio_active ON price_alerts(portfolio_id, is_active);
        CREATE INDEX IF NOT EXISTS idx_snapshots_portfolio_timestamp ON portfolio_snapshots(portfolio_id, timestamp);
        
        -- Trigger to update portfolio updated_at timestamp
        CREATE TRIGGER IF NOT EXISTS update_portfolio_timestamp 
        AFTER UPDATE ON portfolios
        BEGIN
            UPDATE portfolios SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;
        """
        
        with self._get_connection() as conn:
            conn.executescript(schema_sql)
            self._migrate_portfolio_ownership_schema(conn)
            logger.info("Database schema initialized successfully")

    def _migrate_portfolio_ownership_schema(self, conn: sqlite3.Connection) -> None:
        """Migrate portfolios to user-owned rows with per-user names."""
        columns = {
            row["name"]: row
            for row in conn.execute("PRAGMA table_info(portfolios)").fetchall()
        }
        needs_rebuild = "user_id" not in columns

        if not needs_rebuild:
            indexes = conn.execute("PRAGMA index_list(portfolios)").fetchall()
            for index in indexes:
                if not index["unique"]:
                    continue
                index_columns = [
                    row["name"]
                    for row in conn.execute(f"PRAGMA index_info({index['name']})").fetchall()
                ]
                if index_columns == ["name"]:
                    needs_rebuild = True
                    break

        if not needs_rebuild:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_portfolios_user ON portfolios(user_id)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_portfolios_user_name "
                "ON portfolios(user_id, name)"
            )
            return

        logger.info("Migrating portfolios table to user-owned schema")
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            """
            CREATE TABLE portfolios_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT NOT NULL,
                trading_capital REAL NOT NULL DEFAULT 5000.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, name)
            )
            """
        )

        select_user_id = "user_id" if "user_id" in columns else "NULL AS user_id"
        conn.execute(
            f"""
            INSERT INTO portfolios_new (id, user_id, name, trading_capital, created_at, updated_at)
            SELECT id, {select_user_id}, name, trading_capital, created_at, updated_at
            FROM portfolios
            """
        )
        conn.execute("DROP TABLE portfolios")
        conn.execute("ALTER TABLE portfolios_new RENAME TO portfolios")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_portfolios_user ON portfolios(user_id)")
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_portfolios_user_name "
            "ON portfolios(user_id, name)"
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS update_portfolio_timestamp
            AFTER UPDATE ON portfolios
            BEGIN
                UPDATE portfolios SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
            """
        )
        conn.execute("PRAGMA foreign_keys = ON")

    def claim_unowned_portfolios(self, user_id: int) -> int:
        """Assign legacy unowned portfolios to a user."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE portfolios SET user_id = ? WHERE user_id IS NULL",
                (user_id,),
            )
            return cursor.rowcount

    def create_portfolio(self, name: str, trading_capital: float = 5000.0,
                         user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Create a new portfolio.
        
        Args:
            name: Unique portfolio name
            trading_capital: Initial trading capital (default: 5000.0)
            
        Returns:
            Dictionary with portfolio details
            
        Raises:
            ValueError: If portfolio name already exists
        """
        if not name:
            raise ValueError("Portfolio name must be a non-empty string")
        
        if trading_capital <= 0:
            raise ValueError("Trading capital must be positive")
        
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "INSERT INTO portfolios (user_id, name, trading_capital) VALUES (?, ?, ?)",
                    (user_id, name.strip(), trading_capital)
                )
                portfolio_id = cursor.lastrowid
                
                # Return the created portfolio
                return {
                    'id': portfolio_id,
                    'user_id': user_id,
                    'name': name.strip(),
                    'trading_capital': trading_capital,
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }
        except sqlite3.IntegrityError:
            raise ValueError(f"Portfolio '{name}' already exists")
    
    def get_portfolio(self, name: str = "default",
                      user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get portfolio details by name.
        
        Args:
            name: Portfolio name (default: "default")
            
        Returns:
            Dictionary with portfolio details or None if not found
        """
        with self._get_connection() as conn:
            if user_id is None:
                row = conn.execute(
                    "SELECT * FROM portfolios WHERE name = ?",
                    (name,)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM portfolios WHERE name = ? AND user_id = ?",
                    (name, user_id)
                ).fetchone()
            
            if row:
                return dict(row)
            return None
    
    def get_portfolio_by_id(self, portfolio_id: int) -> Optional[Dict[str, Any]]:
        """Get portfolio details by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM portfolios WHERE id = ?",
                (portfolio_id,)
            ).fetchone()
            
            if row:
                return dict(row)
            return None
    
    def list_portfolios(self, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """List all portfolios."""
        with self._get_connection() as conn:
            if user_id is None:
                rows = conn.execute(
                    "SELECT * FROM portfolios ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM portfolios WHERE user_id = ? ORDER BY created_at DESC",
                    (user_id,),
                ).fetchall()
            
            return [dict(row) for row in rows]
    
    def update_trading_capital(self, name: str, capital: float,
                               user_id: Optional[int] = None) -> bool:
        """
        Update available trading capital for a portfolio.
        
        Args:
            name: Portfolio name
            capital: New trading capital amount
            
        Returns:
            True if updated successfully, False if portfolio not found
        """
        if capital < 0:
            raise ValueError("Trading capital cannot be negative")
        
        with self._get_connection() as conn:
            if user_id is None:
                cursor = conn.execute(
                    "UPDATE portfolios SET trading_capital = ? WHERE name = ?",
                    (capital, name)
                )
            else:
                cursor = conn.execute(
                    "UPDATE portfolios SET trading_capital = ? WHERE name = ? AND user_id = ?",
                    (capital, name, user_id)
                )
            return cursor.rowcount > 0

    def delete_portfolio(self, name: str, user_id: Optional[int] = None) -> bool:
        """
        Delete a portfolio and all dependent local records.

        This is intentionally explicit instead of relying on SQLite cascades,
        because the existing schema does not define ON DELETE CASCADE.
        """
        portfolio = self.get_portfolio(name, user_id=user_id)
        if not portfolio:
            return False

        dependent_tables = [
            "holdings",
            "trades",
            "portfolio_snapshots",
            "price_alerts",
            "pending_orders",
            "chat_sessions",
            "watchlists",
        ]

        with self._get_connection() as conn:
            for table in dependent_tables:
                conn.execute(f"DELETE FROM {table} WHERE portfolio_id = ?", (portfolio["id"],))

            cursor = conn.execute(
                "DELETE FROM portfolios WHERE id = ?",
                (portfolio["id"],),
            )
            return cursor.rowcount > 0
    
    def get_holdings(self, name: str = "default",
                     user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get current holdings for a portfolio.
        
        Args:
            name: Portfolio name
            
        Returns:
            List of holdings with symbol, quantity, avg_cost
        """
        portfolio = self.get_portfolio(name, user_id=user_id)
        if not portfolio:
            return []
        
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT symbol, quantity, avg_cost, last_updated
                FROM holdings
                WHERE portfolio_id = ? AND quantity > 0
                ORDER BY symbol
                """,
                (portfolio['id'],)
            ).fetchall()
            
            return [dict(row) for row in rows]
    
    def update_holding(self, name: str, symbol: str, quantity: int,
                       avg_cost: Optional[float] = None,
                       user_id: Optional[int] = None) -> bool:
        """
        Update or add a holding in a portfolio.
        
        Args:
            name: Portfolio name
            symbol: Stock symbol
            quantity: Number of shares (0 to remove)
            avg_cost: Average cost per share (optional)
            
        Returns:
            True if successful
        """
        portfolio = self.get_portfolio(name, user_id=user_id)
        if not portfolio:
            raise ValueError(f"Portfolio '{name}' not found")
        
        if quantity < 0:
            raise ValueError("Quantity cannot be negative")
        
        symbol = symbol.upper().strip()
        
        with self._get_connection() as conn:
            if quantity == 0:
                # Remove holding
                conn.execute(
                    "DELETE FROM holdings WHERE portfolio_id = ? AND symbol = ?",
                    (portfolio['id'], symbol)
                )
            else:
                # Update or insert holding
                if avg_cost is not None and avg_cost < 0:
                    raise ValueError("Average cost cannot be negative")
                
                conn.execute(
                    """
                    INSERT INTO holdings (portfolio_id, symbol, quantity, avg_cost)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(portfolio_id, symbol) 
                    DO UPDATE SET 
                        quantity = excluded.quantity,
                        avg_cost = CASE 
                            WHEN excluded.avg_cost IS NOT NULL THEN excluded.avg_cost 
                            ELSE holdings.avg_cost 
                        END,
                        last_updated = CURRENT_TIMESTAMP
                    """,
                    (portfolio['id'], symbol, quantity, avg_cost)
                )
            
            return True
    
    def record_trade(self, name: str, symbol: str, action: str, quantity: int, 
                    price: float, strategy: Optional[str] = None, 
                    confidence: Optional[float] = None, notes: Optional[str] = None,
                    user_id: Optional[int] = None) -> int:
        """
        Record a trade execution.
        
        Args:
            name: Portfolio name
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            quantity: Number of shares
            price: Price per share
            strategy: Trading strategy used (optional)
            confidence: Confidence level 0-1 (optional)
            notes: Additional notes (optional)
            
        Returns:
            Trade ID
        """
        portfolio = self.get_portfolio(name, user_id=user_id)
        if not portfolio:
            raise ValueError(f"Portfolio '{name}' not found")
        
        if action not in ['BUY', 'SELL']:
            raise ValueError("Action must be 'BUY' or 'SELL'")
        
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if price <= 0:
            raise ValueError("Price must be positive")
        
        if confidence is not None and not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        symbol = symbol.upper().strip()
        total_value = quantity * price
        
        # Calculate fees (simplified - could be enhanced)
        fees = max(1.0, total_value * 0.001)  # 0.1% or $1 minimum
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO trades (
                    portfolio_id, symbol, action, quantity, price, 
                    total_value, fees, strategy, confidence, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (portfolio['id'], symbol, action, quantity, price, 
                 total_value, fees, strategy, confidence, notes)
            )
            
            trade_id = cursor.lastrowid
            if trade_id is None:
                raise RuntimeError("Failed to insert trade: no row ID returned")
            
            # Update holdings based on the trade
            self._update_holdings_from_trade(conn, portfolio['id'], symbol, action, quantity, price)
            
            return trade_id
    
    def _update_holdings_from_trade(self, conn: sqlite3.Connection, portfolio_id: int, 
                                   symbol: str, action: str, quantity: int, price: float):
        """Update holdings based on a trade (internal method)."""
        # Get current holding
        current = conn.execute(
            "SELECT quantity, avg_cost FROM holdings WHERE portfolio_id = ? AND symbol = ?",
            (portfolio_id, symbol)
        ).fetchone()
        
        if action == 'BUY':
            if current:
                # Update existing holding with new average cost
                new_quantity = current['quantity'] + quantity
                new_avg_cost = ((current['quantity'] * current['avg_cost']) + (quantity * price)) / new_quantity
                
                conn.execute(
                    """
                    UPDATE holdings 
                    SET quantity = ?, avg_cost = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE portfolio_id = ? AND symbol = ?
                    """,
                    (new_quantity, new_avg_cost, portfolio_id, symbol)
                )
            else:
                # Create new holding
                conn.execute(
                    "INSERT INTO holdings (portfolio_id, symbol, quantity, avg_cost) VALUES (?, ?, ?, ?)",
                    (portfolio_id, symbol, quantity, price)
                )
        
        elif action == 'SELL':
            if current and current['quantity'] >= quantity:
                new_quantity = current['quantity'] - quantity
                
                if new_quantity == 0:
                    # Remove holding
                    conn.execute(
                        "DELETE FROM holdings WHERE portfolio_id = ? AND symbol = ?",
                        (portfolio_id, symbol)
                    )
                else:
                    # Update quantity (avg_cost remains the same)
                    conn.execute(
                        """
                        UPDATE holdings 
                        SET quantity = ?, last_updated = CURRENT_TIMESTAMP
                        WHERE portfolio_id = ? AND symbol = ?
                        """,
                        (new_quantity, portfolio_id, symbol)
                    )
            else:
                raise ValueError(f"Insufficient shares to sell. Current: {current['quantity'] if current else 0}, Requested: {quantity}")
    
    def get_portfolio_value(self, name: str = "default",
                            current_prices: Optional[Dict[str, float]] = None,
                            user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate current portfolio value.
        
        Args:
            name: Portfolio name
            current_prices: Dictionary of symbol -> current price (optional)
            
        Returns:
            Dictionary with total_value, holdings_value, cash_available, holdings_details
        """
        portfolio = self.get_portfolio(name, user_id=user_id)
        if not portfolio:
            return {
                'total_value': 0,
                'holdings_value': 0,
                'cash_available': 0,
                'holdings_details': []
            }
        
        holdings = self.get_holdings(name, user_id=user_id)
        holdings_value = 0
        holdings_details: List[Dict[str, Any]] = []
        
        for holding in holdings:
            symbol = holding['symbol']
            quantity = holding['quantity']
            avg_cost = holding['avg_cost']
            
            # Use provided price or avg_cost as fallback
            current_price = current_prices.get(symbol, avg_cost) if current_prices else avg_cost
            market_value = quantity * current_price
            holdings_value += market_value
            
            holdings_details.append({
                'symbol': symbol,
                'quantity': quantity,
                'avg_cost': avg_cost,
                'current_price': current_price,
                'market_value': market_value,
                'unrealized_pnl': market_value - (quantity * avg_cost),
                'unrealized_pnl_pct': ((current_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0
            })
        
        # Calculate cash from trades
        with self._get_connection() as conn:
            # Sum of all sells minus sum of all buys
            cash_flow = conn.execute(
                """
                SELECT 
                    SUM(CASE WHEN action = 'SELL' THEN total_value - fees ELSE 0 END) -
                    SUM(CASE WHEN action = 'BUY' THEN total_value + fees ELSE 0 END) as net_cash_flow
                FROM trades
                WHERE portfolio_id = ?
                """,
                (portfolio['id'],)
            ).fetchone()
            
            net_cash_flow = cash_flow['net_cash_flow'] or 0
            cash_available = portfolio['trading_capital'] + net_cash_flow
        
        total_value = holdings_value + cash_available
        
        return {
            'total_value': round(total_value, 2),
            'holdings_value': round(holdings_value, 2),
            'cash_available': round(cash_available, 2),
            'holdings_details': holdings_details
        }
    
    def get_trade_history(self, name: str = "default", days: int = 30,
                          user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent trade history for a portfolio.
        
        Args:
            name: Portfolio name
            days: Number of days to look back (default: 30)
            
        Returns:
            List of trades ordered by timestamp descending
        """
        portfolio = self.get_portfolio(name, user_id=user_id)
        if not portfolio:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM trades
                WHERE portfolio_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                """,
                (portfolio['id'], cutoff_date.isoformat())
            ).fetchall()
            
            return [dict(row) for row in rows]
    
    def get_performance_metrics(self, name: str = "default",
                                user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            name: Portfolio name
            
        Returns:
            Dictionary with performance metrics
        """
        portfolio = self.get_portfolio(name, user_id=user_id)
        if not portfolio:
            return {}
        
        with self._get_connection() as conn:
            # Get trade statistics
            trade_stats = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN action = 'BUY' THEN 1 ELSE 0 END) as buy_trades,
                    SUM(CASE WHEN action = 'SELL' THEN 1 ELSE 0 END) as sell_trades,
                    SUM(fees) as total_fees,
                    MIN(timestamp) as first_trade,
                    MAX(timestamp) as last_trade
                FROM trades
                WHERE portfolio_id = ?
                """,
                (portfolio['id'],)
            ).fetchone()
            
            # Get realized P&L from closed positions
            realized_pnl = self._calculate_realized_pnl(conn, portfolio['id'])
            
            # Get current portfolio value
            portfolio_value = self.get_portfolio_value(name, user_id=user_id)
            
            # Calculate returns
            initial_capital = portfolio['trading_capital']
            current_value = portfolio_value['total_value']
            total_return = current_value - initial_capital
            total_return_pct = (total_return / initial_capital * 100) if initial_capital > 0 else 0
            
            return {
                'total_trades': trade_stats['total_trades'] or 0,
                'buy_trades': trade_stats['buy_trades'] or 0,
                'sell_trades': trade_stats['sell_trades'] or 0,
                'total_fees': round(trade_stats['total_fees'] or 0, 2),
                'realized_pnl': round(realized_pnl, 2),
                'unrealized_pnl': round(sum(h['unrealized_pnl'] for h in portfolio_value['holdings_details']), 2),
                'total_return': round(total_return, 2),
                'total_return_pct': round(total_return_pct, 2),
                'first_trade': trade_stats['first_trade'],
                'last_trade': trade_stats['last_trade']
            }
    
    def _calculate_realized_pnl(self, conn: sqlite3.Connection, portfolio_id: int) -> float:
        """Calculate realized P&L from closed positions."""
        # This is a simplified calculation
        # In production, you'd want FIFO/LIFO accounting
        result = conn.execute(
            """
            SELECT SUM(
                CASE 
                    WHEN action = 'SELL' THEN (quantity * price) - fees
                    WHEN action = 'BUY' THEN -(quantity * price) - fees
                END
            ) as realized_pnl
            FROM trades
            WHERE portfolio_id = ?
            """,
            (portfolio_id,)
        ).fetchone()
        
        return result['realized_pnl'] or 0.0
    
    def create_price_alert(self, name: str, symbol: str, alert_type: str, 
                          target_price: Optional[float] = None) -> int:
        """Create a price alert for a portfolio."""
        portfolio = self.get_portfolio(name)
        if not portfolio:
            raise ValueError(f"Portfolio '{name}' not found")
        
        valid_types = ['stop_loss', 'take_profit', 'price_target', 'dividend_reminder']
        if alert_type not in valid_types:
            raise ValueError(f"Alert type must be one of: {valid_types}")
        
        if alert_type != 'dividend_reminder' and (target_price is None or target_price <= 0):
            raise ValueError("Target price required for this alert type")
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO price_alerts (portfolio_id, symbol, alert_type, target_price)
                VALUES (?, ?, ?, ?)
                """,
                (portfolio['id'], symbol.upper(), alert_type, target_price)
            )
            alert_id = cursor.lastrowid
            if alert_id is None:
                raise RuntimeError("Failed to insert price alert: no row ID returned")
            return alert_id
    
    def get_active_alerts(self, name: str = "default") -> List[Dict[str, Any]]:
        """Get active price alerts for a portfolio."""
        portfolio = self.get_portfolio(name)
        if not portfolio:
            return []
        
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM price_alerts
                WHERE portfolio_id = ? AND is_active = 1
                ORDER BY created_at DESC
                """,
                (portfolio['id'],)
            ).fetchall()
            
            return [dict(row) for row in rows]
    
    def add_to_watchlist(self, name: str, symbol: str, notes: Optional[str] = None) -> bool:
        """Add a symbol to portfolio watchlist."""
        portfolio = self.get_portfolio(name)
        if not portfolio:
            raise ValueError(f"Portfolio '{name}' not found")
        
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT INTO watchlists (portfolio_id, symbol, notes) VALUES (?, ?, ?)",
                    (portfolio['id'], symbol.upper(), notes)
                )
                return True
        except sqlite3.IntegrityError:
            return False  # Already in watchlist
    
    def get_watchlist(self, name: str = "default") -> List[Dict[str, Any]]:
        """Get watchlist for a portfolio."""
        portfolio = self.get_portfolio(name)
        if not portfolio:
            return []
        
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT symbol, notes, added_at FROM watchlists
                WHERE portfolio_id = ?
                ORDER BY added_at DESC
                """,
                (portfolio['id'],)
            ).fetchall()
            
            return [dict(row) for row in rows]
    
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Custom backup path (optional)
            
        Returns:
            Path to the backup file
        """
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"data/backups/portfolios_backup_{timestamp}.db"
        
        # Ensure backup directory exists
        backup_dir = os.path.dirname(backup_path)
        if backup_dir and not os.path.exists(backup_dir):
            os.makedirs(backup_dir, mode=0o755)
        
        # Use SQLite backup API
        with self._get_connection() as source_conn:
            with sqlite3.connect(backup_path) as backup_conn:
                source_conn.backup(backup_conn)
        
        logger.info(f"Database backed up to: {backup_path}")
        return backup_path
    
    def restore_database(self, backup_path: str) -> bool:
        """
        Restore database from a backup.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            True if successful
        """
        if not os.path.exists(backup_path):
            raise ValueError(f"Backup file not found: {backup_path}")
        
        # Close any existing connections
        # In production, you'd want to handle this more gracefully
        
        # Backup current database before restore
        pre_restore_backup = self.backup_database()
        logger.info(f"Created pre-restore backup: {pre_restore_backup}")
        
        try:
            # Restore from backup
            with sqlite3.connect(backup_path) as backup_conn:
                with sqlite3.connect(self.db_path) as target_conn:
                    backup_conn.backup(target_conn)
            
            logger.info(f"Database restored from: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            raise

    # ==========================================
    # User Management Methods
    # ==========================================

    def create_user(self, username: str, email: str, hashed_password: str,
                    is_admin: bool = False) -> Dict[str, Any]:
        """
        Create a new user.

        Args:
            username: Unique username (3-50 characters)
            email: Valid email address
            hashed_password: Bcrypt hashed password
            is_admin: Whether user has admin privileges

        Returns:
            Dictionary with user details

        Raises:
            ValueError: If username or email already exists
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO users (username, email, hashed_password, is_admin)
                    VALUES (?, ?, ?, ?)
                    """,
                    (username, email, hashed_password, int(is_admin))
                )
                user_id = cursor.lastrowid

                # Fetch and return the created user
                row = conn.execute(
                    "SELECT * FROM users WHERE id = ?", (user_id,)
                ).fetchone()

                logger.info(f"Created user: {username} (ID: {user_id})")
                return dict(row)

        except sqlite3.IntegrityError as e:
            if "username" in str(e).lower():
                raise ValueError(f"Username '{username}' already exists")
            elif "email" in str(e).lower():
                raise ValueError(f"Email '{email}' already registered")
            else:
                raise ValueError(f"User creation failed: {e}")

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()

            return dict(row) if row else None

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()

            return dict(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()

            return dict(row) if row else None

    def update_last_login(self, username: str) -> bool:
        """Update user's last login timestamp."""
        try:
            with self._get_connection() as conn:
                from datetime import datetime, timezone
                conn.execute(
                    "UPDATE users SET last_login = ? WHERE username = ?",
                    (datetime.now(timezone.utc).isoformat(), username)
                )
                return True
        except Exception as e:
            logger.error(f"Failed to update last login for {username}: {e}")
            return False

    def deactivate_user(self, username: str) -> bool:
        """Deactivate a user account."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE users SET is_active = 0 WHERE username = ?",
                    (username,)
                )
                logger.info(f"Deactivated user: {username}")
                return True
        except Exception as e:
            logger.error(f"Failed to deactivate user {username}: {e}")
            return False

    def list_users(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all users."""
        with self._get_connection() as conn:
            if active_only:
                rows = conn.execute(
                    "SELECT id, username, email, is_admin, created_at, last_login FROM users WHERE is_active = 1"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, username, email, is_admin, is_active, created_at, last_login FROM users"
                ).fetchall()

            return [dict(row) for row in rows]

    # ==========================================
    # Analysis Job Methods
    # ==========================================

    def create_analysis_job(
        self,
        job_id: str,
        user_id: int,
        portfolio_name: str,
        status: str = "pending",
    ) -> Dict[str, Any]:
        """Create a portfolio analysis job."""
        if status not in {"pending", "running", "completed", "failed"}:
            raise ValueError(f"Invalid analysis job status: {status}")

        created_at = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO analysis_jobs (job_id, user_id, portfolio_name, status, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, user_id, portfolio_name, status, created_at),
            )

        return {
            "job_id": job_id,
            "user_id": user_id,
            "portfolio_name": portfolio_name,
            "status": status,
            "created_at": created_at,
            "completed_at": None,
            "result": None,
            "error": None,
        }

    def get_analysis_job(self, job_id: str, user_id: int) -> Optional[Dict[str, Any]]:
        """Get a user's analysis job by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM analysis_jobs
                WHERE job_id = ? AND user_id = ?
                """,
                (job_id, user_id),
            ).fetchone()

        return self._analysis_job_from_row(row) if row else None

    def update_analysis_job(
        self,
        job_id: str,
        user_id: int,
        *,
        status: Optional[str] = None,
        result: Optional[Any] = None,
        error: Optional[str] = None,
        completed_at: Optional[datetime] = None,
    ) -> bool:
        """Update a user's analysis job."""
        if status is not None and status not in {"pending", "running", "completed", "failed"}:
            raise ValueError(f"Invalid analysis job status: {status}")

        updates = []
        values: List[Any] = []
        if status is not None:
            updates.append("status = ?")
            values.append(status)
        if result is not None:
            updates.append("result_json = ?")
            values.append(json.dumps(result, default=str))
        if error is not None:
            updates.append("error = ?")
            values.append(error)
        if completed_at is not None:
            updates.append("completed_at = ?")
            values.append(completed_at.isoformat())

        if not updates:
            return False

        values.extend([job_id, user_id])
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"""
                UPDATE analysis_jobs
                SET {", ".join(updates)}
                WHERE job_id = ? AND user_id = ?
                """,
                values,
            )
            return cursor.rowcount > 0

    def delete_analysis_job(self, job_id: str, user_id: int) -> bool:
        """Delete a user's analysis job."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM analysis_jobs WHERE job_id = ? AND user_id = ?",
                (job_id, user_id),
            )
            return cursor.rowcount > 0

    def _analysis_job_from_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert an analysis_jobs row into API-friendly data."""
        result_json = row["result_json"]
        return {
            "job_id": row["job_id"],
            "user_id": row["user_id"],
            "portfolio_name": row["portfolio_name"],
            "status": row["status"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
            "result": json.loads(result_json) if result_json else None,
            "error": row["error"],
        }

    # ==========================================
    # JWT Token Blacklist Methods
    # ==========================================

    def blacklist_token(self, jti: str, username: str, expires_at: datetime) -> bool:
        """Add a JWT token to the blacklist."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT INTO token_blacklist (jti, username, expires_at) VALUES (?, ?, ?)",
                    (jti, username, expires_at.isoformat())
                )
                logger.info(f"Blacklisted token {jti} for user {username}")
                return True
        except Exception as e:
            logger.error(f"Failed to blacklist token: {e}")
            return False

    def is_token_blacklisted(self, jti: str) -> bool:
        """Check if a token is blacklisted."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM token_blacklist WHERE jti = ?", (jti,)
            ).fetchone()
            return row is not None

    def cleanup_expired_tokens(self) -> int:
        """Remove expired tokens from blacklist."""
        try:
            from datetime import datetime, timezone
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM token_blacklist WHERE expires_at < ?",
                    (datetime.now(timezone.utc).isoformat(),)
                )
                count = cursor.rowcount
                logger.info(f"Cleaned up {count} expired tokens")
                return count
        except Exception as e:
            logger.error(f"Failed to cleanup expired tokens: {e}")
            return 0


# Example usage and testing
if __name__ == "__main__":
    # Initialize portfolio manager
    pm = PortfolioManager()
    
    # Create a test portfolio
    try:
        portfolio = pm.create_portfolio("test_portfolio", 10000.0)
        print(f"Created portfolio: {portfolio}")
    except ValueError as e:
        print(f"Portfolio creation failed: {e}")
    
    # Add some holdings
    pm.update_holding("test_portfolio", "AAPL", 100, 150.0)
    pm.update_holding("test_portfolio", "MSFT", 50, 300.0)
    
    # Record a trade
    trade_id = pm.record_trade(
        "test_portfolio", "AAPL", "BUY", 10, 155.0,
        strategy="technical", confidence=0.75
    )
    print(f"Recorded trade ID: {trade_id}")
    
    # Get portfolio value
    value = pm.get_portfolio_value("test_portfolio", {"AAPL": 160.0, "MSFT": 310.0})
    print(f"Portfolio value: {value}")
    
    # Get performance metrics
    metrics = pm.get_performance_metrics("test_portfolio")
    print(f"Performance metrics: {metrics}")
