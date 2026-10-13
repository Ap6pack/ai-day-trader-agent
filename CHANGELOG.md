# Changelog

All notable changes to this project will be documented in this file.

---

## [Unreleased]

### Added
- Continuous integration: `.github/workflows/ci.yml` runs the test suite on Python
  3.11, 3.12 and 3.13 on every push and pull request. CI holds no credentials and no
  `.env`, so every credential-gated path takes its "not configured" branch.
- `tests/regressions/`: deterministic, offline regression tests for every defect found
  in two independent reviews of `d6e0768`. Each asserts the intended safe behaviour and
  names its finding id; the ones not yet fixed carry `xfail(strict=True)`, so fixing a
  defect turns the test XPASS and fails the build until the marker is removed. The
  package's `conftest.py` makes any real HTTP call an error and strips every trader
  environment variable, so no test can reach a broker or read a credential.
  `tests/regressions/FINDINGS.md` maps each finding id to its test.
- `docs/evidence/`: the sanitized reproduction probes behind those findings, with
  instructions. Deliberately excluded from CI — some bind sockets, spawn processes or
  configure a live account mode to exercise real-money code paths.
- The agents run on their own: `python -m trader.desk` starts the news bot as a
  supervised process (`trader.supervisor`: restarted with backoff if it stops; setup
  errors reported, not retried; output in `data/newsbot.log`) and starts the autopilot
  from `DESK_AUTOPILOT_MODE` / `DESK_AUTOPILOT_SYMBOLS`. The NEWS BOT tab shows the
  process and has START / STOP BOT. `python -m trader.desk service` prints a systemd
  user service that keeps the desk and its agents running at boot and after crashes.
- Autopilot `AUTO` symbols: today's in-play stocks (watchlist + Alpaca most-active and
  movers, the news bot's list), refreshed each cycle, capped by `AUTOPILOT_MAX_SYMBOLS`,
  plus the positions it holds.
- The after-close review may run `trader.status`; status shows the Alpaca account mode.
- Opt-in real money on Alpaca. `ALPACA_TRADING_BASE_URL=https://api.alpaca.markets`
  plus `ALPACA_LIVE_TRADING=true` (both required; anything else refuses every order).
  Every live buy is checked in `trader.alpaca` against `ALPACA_LIVE_MAX_ORDER_USD`,
  `ALPACA_LIVE_MAX_ORDERS_PER_DAY` and `ALPACA_LIVE_MAX_DAILY_LOSS_USD`; sells always
  pass. `NEWSBOT_EXECUTION=live` lets the news bot trade it (the mode must match the
  account); on live it flattens only the shares it bought (`flatten_owned`), and
  `close_all_positions` is paper only. The desk shows the live account with a LIVE
  MONEY badge and requires typing LIVE for each ticket order. The desk autopilot has a
  `live` mode (typing LIVE to start): it sizes on live equity, caps buys at the live
  per-order limit and only sells shares it bought. In paper and live modes it now
  closes its own positions `AUTOPILOT_FLATTEN_MINUTES` before the close.
- Features from the original desk (PR #5), rebuilt on the new app:
  - `trader.analysis`: the multi-strategy analysis. `trader.technicals` (RSI, MACD,
    SMA/EMA, ATR) gives the technical signal with the original rules, Jev's headline
    judgments give sentiment, and `trader.dividends` gives dividend capture from Alpaca
    corporate actions. They are fused into BUY / SELL / HOLD with a confidence, then
    sized (quantity, ATR stop and target, risk/reward, position value, risk). The desk's
    **ANALYZE** button (`SYM AN`) shows it with the original signal panel.
  - `trader.portfolios`: any number of local paper portfolios (cash, holdings, average
    price, realized and unrealized P&L) in the journal database. The desk can create
    them, show one in POSITIONS and route the ticket to it.
  - `trader.autotrader`: the desk autopilot, started and stopped from the desk's
    AUTOPILOT tab (`AP ON` / `AP OFF`): chosen symbols, every 1 min to 1 hr, in signals,
    record-to-portfolio or Alpaca paper order mode, with `AUTOPILOT_*` limits (minimum
    confidence, daily cap, cooldown, minimum price, entry cutoff). Every step streams to
    the activity feed; decisions and orders are journaled.
  - News bot pause: the desk's NEWS BOT tab can pause and resume the bot's trading
    (`BOT PAUSE` / `BOT RESUME`); while paused it keeps judging and journaling headlines.
- `trader.desk`: the live trading desk (from PR #5) rebuilt on the new app and served by
  `python -m trader.desk`. Its front end (chart, watchlist, tape, news, positions,
  blotter, command line) is kept; it now streams the journal (news bot signals and orders
  with Jev's numbers and latency, flattens, Claude's decisions, guard reviews), shows the
  news bot's status, and JUDGE runs Jev with the bot's rules on a symbol's recent
  headlines. The manual ticket sends Alpaca paper market orders, journaled as desk orders.
  Local-only by default; `DESK_TOKEN` protects a non-local bind. `trader.market_feed`
  keeps its Alpaca data with an optional Yahoo fallback (only if `yfinance` is installed).
- `NEWSBOT_SYMBOLS=auto`: the news bot trades today's in-play stocks, refreshed every
  `NEWSBOT_UNIVERSE_REFRESH_MINUTES` from Alpaca's most-active and movers screeners plus
  `TRADER_WATCHLIST`; it judges nothing until a list is loaded.
- `trader.autopilot`: Claude's unattended after-close runs. `score` records outcomes,
  `review` reports the news bot's results by event type and probability bucket with
  proposed `NEWSBOT_*` changes (never applied automatically), optional `run` analyzes the
  scan shortlist, `check` and `cron` help set it up. Its tool allowlist never includes
  order placement.
- `trader.newsbot`: autonomous news trading on the Alpaca paper account (PLAN Phase 3).
  Streams Alpaca news, judges each headline and symbol with Jev, applies the
  `NEWSBOT_*` thresholds and places bracket orders with a size per trade, a daily cap,
  a per-symbol cooldown, a minimum price and an entry cutoff before the close. Long
  only; `NEWSBOT_EXECUTION=off` by default. `replay` tunes thresholds on recent
  headlines; `flatten` closes all paper positions.
- `trader.alpaca`: market clock, latest price and paper-only order entry. Order and
  position functions refuse any base URL other than `https://paper-api.alpaca.markets`.
- `NEWSBOT_ENTRY_CUTOFF_MINUTES` and `NEWSBOT_FLATTEN_MINUTES` (not in the original
  plan): `run` flattens from Alpaca's clock before the close, including early-close
  days a fixed cron time would miss, and takes no entries after that. Bracket legs are
  day orders, so a position left at the close would be held overnight with no stop.
- Stream errors after authentication are logged and the stream continues; only auth
  failures stop `run`.
- `websockets` dependency.

### Security
- `trader.alpaca.market_clock()` now refuses any host but the Alpaca paper host, or the
  live host with `ALPACA_LIVE_TRADING=true`, like every other function that reaches the
  trading API. It attaches the API key and secret, so before this a mistyped or pasted
  `ALPACA_TRADING_BASE_URL` sent live trading credentials to whatever that pointed at,
  in cleartext if the URL said `http`. It is the most-called function in the system, so
  the window was seconds from start-up. Every caller already fails closed on the raise.
- Enabling `TRADER_ALLOW_OPTIONS` or `TRADER_ALLOW_CRYPTO` no longer disables the rest
  of the order guard. Those orders skipped the dollar cap, the symbol allowlist and the
  sizeability requirement, leaving only the daily count — a $900,000 crypto buy passed
  with the shipped $500 cap. They now take the same path as equity orders. Option
  notional uses the 100-share contract multiplier, so 1,000 contracts at $50 is priced
  as $5,000,000 of exposure rather than $50,000.
- Non-finite numeric limits are refused instead of silently not being enforced.
  `float()` accepts `nan` and `inf`, and every comparison against NaN is False, so
  `TRADER_MAX_ORDER_USD=nan` let a $10,000 buy through the guard and
  `ALPACA_LIVE_MAX_DAILY_LOSS_USD=nan` disabled the real-money daily loss stop outright.
  One validator (`config.finite` / `config.env_float`) now covers every money limit,
  replacing four near-identical `os.getenv` float helpers that had drifted apart.

### Fixed
- The test suite was red on any day more than 24 hours after 2026-09-28: a desk test
  pinned its headlines to a fixed date while `desk.judge_symbol` read the wall clock,
  so the fixtures aged past the 24-hour judge window. `judge_symbol` now takes an
  optional `now`, matching the seam `newsbot.decide` and `dividends.dividend_signal`
  already offer, and the fixtures are built relative to the current time.
- `guard.order_notional` treated a zero price as absent (`limit_price or stop_price`),
  so a `limit_price` of 0 silently fell through to the stop price. A non-positive price
  is now not a price: the order is reported unsizeable and denied.

### Removed
- `trader.scan.market_clock`, which nothing called; `trader.alpaca.market_clock` replaces it.

### Known issues
Two independent reviews of `d6e0768` found defects that are **not yet fixed**, including
agents that can place orders after being switched off, flattening that can double-sell or
strand a working entry, a live daily-order cap that concurrent agents can exceed, and a
tokenless desk that a page on another origin can read and drive. Each is reproduced by an
`xfail(strict=True)` test in `tests/regressions/`; `tests/regressions/FINDINGS.md` lists
them with severity. **Keep live automation disabled** (`ALPACA_LIVE_TRADING=false`,
`NEWSBOT_EXECUTION=off`, `DESK_AUTOPILOT_MODE=signals`) until they are closed. Set those
values explicitly rather than commenting them out: `config.load_env` only overrides keys
present in `.env`, so a commented-out key leaves a stale shell export in force.

---

## [4.0.0] - 2026-09-28

### Redesign around Claude Code and Robinhood

Claude now does the analysis and operates Robinhood through its MCP tools. The
repository keeps only what those tools do not provide. The previous version is
preserved under the `v1-legacy` tag.

#### Added
- `trader.guard` and `.claude/hooks/order_guard.sh`: a PreToolUse hook on Robinhood
  order tools. Review-only mode by default; in live mode a dollar cap on buys, a daily
  order cap, a required recent review, options and crypto off. Fails closed.
- `trader.journal`: SQLite journal of reviews, guard decisions and trading decisions,
  with outcome scoring and a summary.
- `trader.sizing`: quantity and stop-loss from account figures.
- `trader.news`: Jev headline judgments (`--json` for Claude, `--sample` to test Jev).
- `trader.status`: mode, limits and credential status without printing secrets.
- `.claude/skills/day-trader`: the analysis, sizing, review and journaling playbook.
- Project permission rules that stop Claude's file tools from reading `.env` or editing
  the guard configuration.

#### Removed
- Market data fetching and indicators (replaced by Robinhood historicals and indicators).
- Portfolio database, FastAPI server, authentication, WebSockets and web dashboard.
- OpenAI trade recommender and sentiment analyzer, Discord bot and CLI.
- Alpaca order execution and the dividend capture engine.
- Dependencies reduced to `requests` and `typesafe-sdk` (plus `pytest`).

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
