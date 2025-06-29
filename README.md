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
git clone https://github.com/Ap6pack/ai-day-trader-agent.get
cd ai-day-trader-agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the project root with the following keys:

```
DISCORD_BOT_TOKEN=your_discord_bot_token
TWELVE_DATA_API_KEY=your_12data_api_key
ALPHA_VANTAGE_API_KEY=your_alphavantage_api_key
NEWS_API_KEY=your_newsapi_key
OPENAI_API_KEY=your_openai_key
```

**Never commit your `.env` file to version control.**  
The `.gitignore` is configured to exclude `.env`, logs, and other sensitive or unnecessary files.

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
🤖 AI Day Trader Agent - Enhanced Analysis for FIS

 Analysis Results:
----------------------------------------
**Primary Strategy:** TECHNICAL
**Recommendation:** BUY
**Confidence:** 65.2%
**Quantity:** 30 shares
**Reason:** Technical: 2 bullish indicators

**Technical Indicators:**
  Current Price: $80.80
  RSI: 35.37 (Oversold)
  MACD: -0.4306 / Signal: -0.3197 (Bearish)
  SMA(20): $73.49 Below
  EMA(20): $73.21 Below

**All Strategy Signals:**
  Technical: BUY (strength: 0.60)
  Sentiment: HOLD (score: 0.00)
  Dividend: HOLD (reason: Outside capture window. Next dividend in 88 days)

**Risk Management:**
  Stop Loss: $76.45
  Take Profit: $85.15
  Position Value: $2,424.00
  Risk: 2.15%

**Analysis Time:** 2025-06-29 02:03:46
```

## Recent Major Updates (v2.0)

### 🔧 **Code Quality & Architecture**
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
