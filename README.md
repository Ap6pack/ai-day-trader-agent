# AI Day Trader Agent

A sophisticated, multi-strategy AI-powered trading agent that combines technical analysis, sentiment analysis, and dividend capture strategies to generate intelligent trade recommendations. Features enhanced signal fusion, comprehensive risk management, and professional-grade architecture.

---

## Features

### 🚀 **Enhanced Multi-Strategy Analysis**
- **Technical Analysis**: RSI, MACD, SMA/EMA with intelligent signal fusion
- **Sentiment Analysis**: Real-time news sentiment using OpenAI GPT-4.1
- **Dividend Capture Strategy**: Advanced dividend timing and capture optimization
- **Signal Fusion**: Intelligent combination of all strategies with priority-based decision making

### 📊 **Advanced Market Data**
- Multi-timeframe candlestick data (1m, 15m, 1h) via **both Twelve Data and Alpha Vantage APIs**
- **500+ data points** for robust technical indicator calculations
- Dual API fallback system for maximum reliability
- Real-time price tracking with moving average comparisons

### 🤖 **AI-Powered Intelligence**
- Discord bot interface for real-time trade analysis
- Enhanced analysis output with detailed technical indicators
- Comprehensive risk management with stop-loss and take-profit calculations
- Position sizing based on volatility and confidence levels

### 🏗️ **Professional Architecture**
- Modular, testable, and maintainable codebase
- Proper Python import structure and module organization
- Comprehensive error handling and logging
- Production-ready configuration management

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Ap6pack/ai-day-trader-agent.git
cd ai-day-trader-agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the project root with the following keys:

```
# Required API Keys
DISCORD_BOT_TOKEN=your_discord_bot_token
TWELVE_DATA_API_KEY=your_12data_api_key
ALPHA_VANTAGE_API_KEY=your_alphavantage_api_key
NEWS_API_KEY=your_newsapi_key
OPENAI_API_KEY=your_openai_key

# Optional Trading Configuration
TRADING_CAPITAL=5000.0                    # Your trading capital in dollars
MIN_POSITION_PERCENTAGE=0.02              # Minimum 2% of capital per trade
MAX_POSITION_PERCENTAGE=0.10              # Maximum 10% of capital per trade

# Optional API Tier Configuration
TWELVE_DATA_PREMIUM=auto                  # auto (detect), true (premium), false (free)
```

**Never commit your `.env` file to version control.**  
The `.gitignore` is configured to exclude `.env`, logs, and other sensitive or unnecessary files.

### 4. Trading Capital Configuration

The system uses **portfolio-based position sizing** that adapts to your actual trading capital:

- **Default**: $5,000 trading capital
- **Flexible**: Works whether you own 0, 5, 100, or 4,933 shares of any stock
- **Risk-Managed**: Position sizes scale with signal confidence and market volatility
- **Configurable**: Adjust via environment variables or config files

---

## Usage

### Discord Bot

Start the bot:

```bash
python core/discord_bot.py
```

In your Discord server, use:

```
!trade <TICKER>
```

Example:

```
!trade AAPL
```

### Command-Line Interface

Run a trade analysis from the terminal:

```bash
python run.py <TICKER>
```

Example:

```bash
python run.py TSLA
```

---

## Security & Compliance

- All API keys and secrets are loaded from environment variables.
- No sensitive data is logged or exposed in error messages.
- License information is included in the LICENSE file (MIT License).
- Follows best practices for modularity, error handling, and input validation.
- `.gitignore` ensures secrets and logs are not tracked by git.

---

## Project Structure

- `core/`: Main pipeline modules
- `utils/`: Logging and formatting utilities
- `config/`: Environment variable loader
- `run.py`: CLI entry point
- `requirements.txt`: Python dependencies
- `.gitignore`: Excludes secrets, logs, and unnecessary files

---

## Enhanced Analysis Output

The system now provides comprehensive trading analysis with detailed insights:

```
🤖 AI Day Trader Agent - Enhanced Analysis for APAM

 Analysis Results:
----------------------------------------
**Primary Strategy:** SENTIMENT
**Recommendation:** HOLD
**Confidence:** 50.0%
**Quantity:** 0 shares
**Reason:** Sentiment score: 0.40

**Technical Indicators:**
  Current Price: $40.60
  RSI: 50.99 (Neutral)
  MACD: -0.0529 / Signal: -0.0547 (Bullish)
  SMA(20): $40.61 Below
  EMA(20): $40.54 Above

**All Strategy Signals:**
  Technical: HOLD (strength: 0.20)
  Sentiment: HOLD (score: 0.40)
  Dividend: HOLD (reason: Outside capture window. Next dividend in 46 days)

**Analysis Time:** 2025-06-29 19:58:52
```

## Recent Major Updates

### 🎯 **v2.1.0 - Position Sizing Revolution & Triple-API Dividend System**
- **Fixed Critical "0 shares" Issue**: Completely redesigned position calculation system
- **Portfolio-Based Sizing**: No longer assumes you own 100 shares of every stock
- **Flexible Trading Capital**: Works with any current holdings (0, 5, 100, or 4,933 shares)
- **Enhanced Output**: Shows both share quantity and dollar amount: "47 shares ($2,350.00)"
- **Configurable Capital**: Set your trading capital via environment variables
- **Risk-Adjusted Positions**: Scales with signal confidence and market volatility
- **Triple-API Dividend System**: Solved "No upcoming dividends found" issue
  - **Primary**: Twelve Data (premium users get enhanced data)
  - **Secondary**: Alpha Vantage (when not rate limited)
  - **Fallback**: Yahoo Finance (free, unlimited access)

### 🔧 **v2.0.0 - Code Quality & Architecture**
- **Fixed all import path issues**: Removed hacky `sys.path.append()` workarounds
- **Enhanced config module**: Proper module exports and centralized configuration
- **Resolved function name conflicts**: Clean, maintainable code structure
- **Professional Python patterns**: Proper relative imports throughout

### 📈 **Enhanced Trading Intelligence**
- **Multi-strategy signal fusion**: Intelligent combination of technical, sentiment, and dividend signals
- **Advanced technical analysis**: Improved RSI, MACD, and moving average interpretation
- **Comprehensive risk management**: Stop-loss, take-profit, and position sizing calculations
- **Enhanced data processing**: 500+ candlesticks for robust indicator calculations

### 🎯 **Improved Analysis Output**
- **Detailed technical indicators**: Current price vs moving averages with trend analysis
- **Signal strength metrics**: Quantified confidence levels for each strategy
- **Risk parameters**: Complete risk management information for every trade
- **Enhanced formatting**: Clear, actionable trading intelligence

### 🛠 **Technical Improvements**
- **Robust data conversion**: Handles both string and numeric API responses
- **Better error handling**: Graceful fallbacks without breaking functionality
- **Increased data volume**: 500 data points vs previous 100 for better analysis
- **API reliability**: Enhanced dual-API system with improved fallback logic

---

## License

MIT License. See LICENSE file for details.
