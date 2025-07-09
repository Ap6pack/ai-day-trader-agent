# Changelog

All notable changes to this project will be documented in this file.

---

## [3.2.0] - 2025-07-09

### 🎯 **Intelligent API Rate Limiting - Professional Rate Limit Handling**

#### Added
- **Professional Rate Limiting System** (`core/candle_fetcher.py`):
  - **RateLimitConfig class**: Centralized configuration from environment variables
  - **RateLimiter class**: Thread-safe rate limiter with sliding window tracking
  - Automatic detection of API rate limit errors
  - Exponential backoff with configurable jitter
  - Request tracking to prevent hitting limits proactively
  - Automatic retry logic with configurable attempts

- **Enhanced Data Fetching**:
  - Both Twelve Data and Alpha Vantage now handle rate limits gracefully
  - Automatic waiting and retry when rate limited
  - Clear logging of rate limit events and wait times
  - Seamless failover to alternative data sources when rate limited
  - No more crashes from "API credits exceeded" errors

- **Configuration Options** (via `.env`):
  - `TWELVE_DATA_RATE_LIMIT_WAIT`: Seconds to wait when rate limited (default: 60)
  - `TWELVE_DATA_MAX_RETRIES`: Maximum retry attempts (default: 3)
  - `TWELVE_DATA_CALLS_PER_MINUTE`: Your API plan's limit (default: 8)
  - `ALPHA_VANTAGE_RATE_LIMIT_WAIT`: Seconds to wait when rate limited (default: 60)
  - `ALPHA_VANTAGE_MAX_RETRIES`: Maximum retry attempts (default: 3)
  - `ALPHA_VANTAGE_CALLS_PER_MINUTE`: Your API plan's limit (default: 5)
  - `API_BACKOFF_FACTOR`: Exponential backoff multiplier (default: 2.0)
  - `API_MAX_BACKOFF_SECONDS`: Maximum wait time (default: 300)
  - `API_JITTER_ENABLED`: Add randomness to prevent thundering herd (default: true)

- **Test Script** (`test_rate_limiting.py`):
  - Simple script to verify rate limiting functionality
  - Shows rate limiting in action with clear output
  - Useful for testing configuration changes

#### Enhanced
- **Error Handling**: Rate limit errors now trigger automatic retry instead of failure
- **User Experience**: No more manual intervention needed for rate limits
- **Logging**: Clear, informative messages about rate limiting and retry attempts
- **Documentation**: Updated README with rate limiting configuration options

#### Technical Details
- **Sliding Window Algorithm**: Tracks requests within the last minute
- **Thread-Safe Operations**: Uses locks to prevent race conditions
- **Smart Backoff**: Exponential backoff prevents aggressive retrying
- **Jitter Implementation**: Random delays prevent synchronized retries
- **Graceful Degradation**: Falls back to other data sources when one is rate limited

---

## [3.1.0] - 2025-07-08

### 🎯 **Phase 3: API Layer Development - Professional REST API & WebSockets**

#### Added
- **Professional FastAPI REST API** (`config/api/`):
  - JWT-based authentication with access and refresh tokens
  - Secure user registration and login system
  - Complete portfolio management endpoints (CRUD operations)
  - Holdings management with real-time market data
  - Trade recording and history endpoints
  - Performance metrics API
  - Single and batch symbol analysis endpoints
  - Asynchronous portfolio analysis with job tracking
  - Comprehensive error handling and validation

- **WebSocket Support** (`config/api/websockets.py`):
  - Real-time portfolio updates
  - Trade execution notifications
  - Analysis progress updates
  - Authenticated WebSocket connections
  - Portfolio-specific subscriptions
  - Connection management with automatic cleanup

- **API Security Features**:
  - OAuth2 password flow implementation
  - Bcrypt password hashing
  - JWT token authentication with expiration
  - Role-based access control (admin/user)
  - CORS configuration for web clients
  - Input validation with Pydantic models
  - SQL injection protection

- **API Documentation** (`API_DOCUMENTATION.md`):
  - Complete endpoint documentation
  - Authentication flow examples
  - WebSocket usage guide
  - curl, Python, and JavaScript examples
  - Error response documentation
  - Security best practices

- **API Server** (`api_server.py`):
  - Production-ready server startup script
  - Environment-based configuration
  - Multi-worker support for production
  - Auto-reload for development
  - Security warnings for default secrets

#### Enhanced
- **Logging System** (`utils/logger.py`):
  - Fixed log file location to only use `logs/` directory
  - No more duplicate logs in project root
  - Automatic log directory creation

- **Requirements** (`requirements.txt`):
  - Added FastAPI and dependencies
  - Added uvicorn for ASGI server
  - Added python-jose for JWT
  - Added passlib for password hashing
  - Added python-multipart for form data

#### API Endpoints Summary
- **Authentication**: Login, register, refresh tokens, user profile
- **Portfolios**: List, create, read, update portfolios
- **Holdings**: Get, add, update, remove holdings
- **Trades**: Record trades, view history with filtering
- **Performance**: Get portfolio performance metrics
- **Analysis**: Analyze symbols, batch analysis, portfolio analysis
- **WebSocket**: Real-time updates at `/ws`

---

## [3.0.0] - 2025-07-08

### 🎯 **Phase 1: Database Foundation - Portfolio Management System**

#### Added
- **Portfolio Analysis Feature** (`--analyze-portfolio`):
  - Analyze all holdings in a portfolio with a single command
  - Categorizes recommendations into BUY, SELL, and HOLD
  - Shows confidence levels and reasons for each recommendation
  - Displays portfolio summary with total value and cash available
  - Supports analyzing specific portfolios with `--portfolio` flag

- **Complete SQLite Database System** (`core/portfolio_manager.py`):
  - Implemented full database schema from roadmap with 8 tables
  - Thread-safe database operations with proper transaction handling
  - Comprehensive portfolio CRUD operations
  - Holdings management with automatic average cost calculations
  - Trade recording with strategy and confidence tracking
  - Performance metrics calculation (P&L, win rate, returns)
  - Price alerts and watchlist functionality
  - Database backup and restore capabilities

- **Enhanced CLI Portfolio Management** (`run.py`):
  - Interactive portfolio setup wizard
  - Portfolio switching and management commands
  - Holdings import/export functionality
  - Trade history tracking and display
  - Performance reporting commands
  - Manual trade recording with strategy tracking
  - Database backup/restore commands

- **Portfolio-Aware Trading Pipeline** (`core/pipeline.py`):
  - Automatic portfolio context loading
  - Cash availability-based position sizing
  - Current holdings awareness for SELL signals
  - Portfolio-specific trading capital management
  - Integration with existing multi-strategy analysis

- **Migration Tools** (`scripts/migrate_config.py`):
  - Environment variable to database migration
  - JSON import/export for portfolio data
  - Migration logging and validation
  - Support for historical trade imports

#### Changed
- **Trading Capital Management**: Now uses actual portfolio cash instead of static config
- **Position Sizing**: Considers current holdings when calculating SELL quantities
- **Analysis Pipeline**: Accepts portfolio context for personalized recommendations

#### Security Enhancements
- Parameterized queries throughout to prevent SQL injection
- Proper file permissions (0o755) for database directories
- Input validation and sanitization for all user inputs
- Secure database backup/restore with pre-restore backups

---

## [2.1.0] - 2025-06-29

### 🎯 **Position Sizing Revolution & Dividend Data Fix**

#### Fixed
- **Critical Position Sizing Issue**: Completely redesigned position calculation system
  - **Removed hardcoded "core position" assumption** that required owning 100 shares of every stock
  - **Eliminated "0 shares" recommendations** for SELL signals when not holding excess positions
  - **Implemented flexible trading capital approach** that works with any current holdings (0, 5, 100, or 4,933 shares)
- **Critical Dividend Data Issue**: Resolved "No upcoming dividends found" problem
  - **Root Cause**: Alpha Vantage API rate limiting (25 requests/day exceeded)
  - **Solution**: Implemented triple-API fallback system for dividend data
  - **Primary Source**: Twelve Data dividend API (premium users)
  - **Secondary**: Alpha Vantage (when not rate limited)
  - **Tertiary**: Yahoo Finance (free, unlimited fallback)
- **Critical Ticker Validation Issue**: Fixed inconsistent behavior for invalid/rate-limited tickers
  - **Root Cause**: Invalid tickers showed fake analysis, valid tickers could hang
  - **Solution**: Comprehensive ticker validation and consistent error handling
  - **Validation**: Format checking + existence verification via Yahoo Finance
  - **Error Handling**: Clear, user-friendly error messages for all failure scenarios

#### Added
- **Portfolio-Based Position Sizing** (`config/settings.py`):
  - `TRADING_CAPITAL`: Configurable trading capital (default: $5,000)
  - `MIN_POSITION_PERCENTAGE`: Minimum position size (default: 2% of capital)
  - `MAX_POSITION_PERCENTAGE`: Maximum position size (default: 10% of capital)
- **Dual-API Dividend System** (`core/dividend_strategy.py`):
  - **Twelve Data Integration**: Primary dividend data source with comprehensive dividend information
  - **Enhanced Error Handling**: Detects rate limiting and API failures
  - **Intelligent Caching**: 7-day cache to minimize API calls
  - **Robust Fallback Logic**: Twelve Data → Alpha Vantage → Cached Data
- **Enhanced Position Calculations** (`core/pipeline.py`):
  - Dynamic position sizing based on signal confidence and market volatility
  - Realistic share quantities calculated from dollar amounts and stock prices
  - Intelligent lot sizing (rounds to 5s, 10s for larger positions)
  - Minimum position logic (ensures at least 1 share for viable trades)
- **Enhanced Output Formatting** (`utils/formatter.py`):
  - Added `format_dividend_info()` function for dividend capture details
  - Enhanced quantity display: "**Quantity:** 47 shares ($2,350.00)"
  - Integrated dividend information formatting in analysis results
  - Added comprehensive error case formatting with clear user guidance
- **Ticker Validation System** (`core/pipeline.py`):
  - Format validation (1-5 letters only)
  - Existence verification via Yahoo Finance API
  - Market data validation to detect dummy fallback data
  - Consistent error responses for all failure scenarios
- **Smart API Tier Detection** (`core/dividend_strategy.py`):
  - Auto-detects Twelve Data API tier (free vs premium)
  - 24-hour caching to avoid repeated detection calls
  - Skips premium endpoints for free users to eliminate error messages
  - Manual override option via `TWELVE_DATA_PREMIUM` environment variable

#### Changed
- **Position Tracker Simplification**: Removed complex "core + trading" position logic
- **Realistic Trading Scenarios**: System now works for actual dividend stock trading
- **Flexible Recommendations**: Adapts to user's actual capital and holdings
- **Dividend Data Reliability**: More reliable dividend detection for quarterly dividend stocks

#### Enhanced
- **Trading Capital Management**: 
  - High confidence signals (70%+): Up to 10% of capital
  - Medium confidence signals (50-70%): 2-6% of capital
  - Low confidence signals (<50%): 2% minimum allocation
- **Risk-Adjusted Sizing**: Volatility factor reduces position sizes for volatile stocks
- **User-Friendly Output**: Clear dollar amounts alongside share quantities
- **Dividend Data Accuracy**: Better ex-dividend dates, payment dates, and dividend amounts from Twelve Data

---

## [2.0.0] - 2025-06-29

### 🚀 **Major Release: Enhanced Multi-Strategy Trading Intelligence**

#### Added
- **Enhanced Multi-Strategy Pipeline**: Complete rewrite with intelligent signal fusion
  - Advanced dividend capture strategy (`core/dividend_strategy.py`)
  - Sophisticated dividend backtesting framework (`core/dividend_backtester.py`)
  - Comprehensive dividend database management (`core/dividend_database.py`)
  - Multi-strategy signal fusion with priority-based decision making
- **Advanced Risk Management**: 
  - Stop-loss and take-profit calculations based on Average True Range (ATR)
  - Position sizing based on volatility and confidence levels
  - Comprehensive risk parameters for every trade recommendation
- **Enhanced Analysis Output**:
  - Current price tracking with moving average comparisons
  - Detailed technical indicator status (Oversold/Overbought, Bullish/Bearish)
  - Signal strength metrics for all strategies
  - Complete risk management information display

#### Fixed
- **Critical Import Path Issues**: Removed all hacky `sys.path.append()` workarounds
  - Fixed `core/sentiment_analyzer.py` import structure
  - Fixed `core/news_fetcher.py` import structure  
  - Fixed `core/trade_recommender.py` import structure
  - Fixed `core/pipeline.py` import structure
- **Enhanced Config Module**: Proper module exports in `config/__init__.py`
- **Function Name Conflicts**: Resolved duplicate `format_analysis_result()` functions
- **Data Processing Issues**: 
  - Robust string-to-float conversion for API responses
  - Enhanced data validation and error handling
  - Increased data volume from 100 to 500 candlesticks for better analysis

#### Enhanced
- **Technical Analysis Intelligence**:
  - Improved RSI interpretation with oversold/overbought status
  - Enhanced MACD trend analysis with bullish/bearish indicators
  - Better moving average comparison with current price positioning
  - More sensitive signal thresholds for improved detection
- **Market Data Processing**:
  - **500+ data points** for robust technical indicator calculations
  - Enhanced dual-API system with improved fallback logic
  - Better handling of both string and numeric API responses
- **Professional Code Architecture**:
  - Proper Python import patterns throughout codebase
  - Clean, maintainable module organization
  - Comprehensive error handling without breaking functionality
  - Production-ready configuration management

#### Changed
- **Enhanced Analysis Output Format**: Complete redesign with detailed insights
- **Improved Signal Generation**: Multi-strategy approach with intelligent fusion
- **Better Risk Assessment**: Comprehensive risk parameters for all trades
- **Enhanced Discord Bot**: Updated with new analysis capabilities

---

## [1.0.0] - Previous Release

### Added
- Initial modular pipeline for AI Day Trader Agent, including:
  - Discord bot interface (`core/discord_bot.py`)
  - CLI entry point (`run.py`)
  - Candlestick data fetcher (`core/candle_fetcher.py`)
  - News fetcher (`core/news_fetcher.py`)
  - Sentiment analyzer (`core/sentiment_analyzer.py`)
  - Technical indicator engine (`core/indicator_engine.py`)
  - Trade recommender (`core/trade_recommender.py`)
  - Pipeline orchestrator (`core/pipeline.py`)
  - Logging and formatting utilities (`utils/logger.py`, `utils/formatter.py`)
  - Environment variable loader (`config/settings.py`)
- Security best practices: all secrets in `.env`, no sensitive data in logs
- Added comprehensive `.gitignore` to exclude sensitive and unnecessary files

### Changed
- Updated OpenAI API usage to latest syntax and upgraded to v1.93.0
- Improved error handling and user-facing error messages
- Added ticker symbol validation and robust API response checks
- Truncate candlestick data sent to OpenAI to avoid context window errors
- Removed debug logging of OpenAI prompt and raw response from `trade_recommender.py`
- **Integrated Alpha Vantage API in parallel with Twelve Data for candlestick data**
- Updated LICENSE to MIT for personal project

### Documentation
- Added `README.md` with setup, usage, and security notes
- Updated documentation to reflect new .gitignore, Alpha Vantage integration, and recent improvements

---

## [Planned]
- Machine learning model training for pattern recognition
- Advanced portfolio management features
- Real-time trade execution integration
- Automated tests and CI integration
- Performance optimization and async/await refactor

---
