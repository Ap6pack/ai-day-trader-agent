# Changelog

All notable changes to this project will be documented in this file.

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
