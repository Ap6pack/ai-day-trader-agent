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

### 🔄 **Intelligent Rate Limiting**
- Automatic detection and handling of API rate limits
- Exponential backoff with jitter for optimal retry timing
- Request tracking to prevent hitting limits proactively
- Configurable retry attempts and wait times
- Seamless failover between data sources when rate limited

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
# Primary API Keys
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_TRADING_BASE_URL=https://paper-api.alpaca.markets/v2
OPENAI_API_KEY=your_openai_key

# Optional notifications
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_GUILD_ID=your_discord_guild_id
DISCORD_CHANNEL_ID=your_discord_channel_id

# Optional market/news fallbacks
MARKET_DATA_PROVIDERS=alpaca,yahoo_finance
ALPACA_DATA_FEED=iex
ALPACA_DATA_BASE_URL=https://data.alpaca.markets
TWELVE_DATA_API_KEY=your_12data_api_key
ALPHA_VANTAGE_API_KEY=your_alphavantage_api_key
NEWS_API_KEY=your_newsapi_key

# Optional Trading Configuration
TRADING_CAPITAL=5000.0                    # Your trading capital in dollars
MIN_POSITION_PERCENTAGE=0.02              # Minimum 2% of capital per trade
MAX_POSITION_PERCENTAGE=0.10              # Maximum 10% of capital per trade

# Optional API Tier Configuration
TWELVE_DATA_PREMIUM=auto                  # auto (detect), true (premium), false (free)

# API Rate Limiting Configuration
ALPACA_RATE_LIMIT_WAIT=60                # Seconds to wait when rate limited
ALPACA_MAX_RETRIES=3                     # Max retry attempts
ALPACA_CALLS_PER_MINUTE=180              # Conservative default below Alpaca Basic historical limit

TWELVE_DATA_RATE_LIMIT_WAIT=60           # Seconds to wait when rate limited
TWELVE_DATA_MAX_RETRIES=3                # Max retry attempts
TWELVE_DATA_CALLS_PER_MINUTE=8           # Your plan's limit

ALPHA_VANTAGE_RATE_LIMIT_WAIT=60         # Seconds to wait when rate limited
ALPHA_VANTAGE_MAX_RETRIES=3              # Max retry attempts
ALPHA_VANTAGE_CALLS_PER_MINUTE=5         # Your plan's limit

# Advanced Rate Limiting
API_BACKOFF_FACTOR=2.0                   # Exponential backoff multiplier
API_MAX_BACKOFF_SECONDS=300              # Maximum wait time (5 minutes)
API_JITTER_ENABLED=true                  # Add randomness to prevent thundering herd
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

### Portfolio Management

#### Setup a Portfolio
```bash
python run.py --setup-portfolio
```
Interactive wizard will guide you through creating a portfolio with trading capital and initial holdings.

#### Portfolio Commands
```bash
# Show portfolio details
python run.py --show-portfolio

# List all portfolios
python run.py --list-portfolios

# Update trading capital
python run.py --update-capital 10000

# Add/update holdings
python run.py --add-holding AAPL 100 --cost 150.00

# Remove holdings
python run.py --remove-holding AAPL

# Record manual trades
python run.py --record-trade AAPL BUY 50 155.00 --strategy technical --confidence 0.75

# Show trade history
python run.py --show-trades --days 30

# Backup database
python run.py --backup --output backups/portfolio_backup.db

# Restore from backup
python run.py --restore backups/portfolio_backup.db

# Analyze entire portfolio
python run.py --analyze-portfolio

# Analyze specific portfolio
python run.py --analyze-portfolio --portfolio my_portfolio
```

### REST API Server

The AI Day Trader Agent now includes a professional REST API with WebSocket support for real-time updates.

#### API Features
- **JWT Authentication**: Secure token-based authentication
- **Portfolio Management**: Full CRUD operations via REST endpoints
- **Real-time Updates**: WebSocket support for live portfolio updates
- **Trading Analysis**: Run analysis on symbols and portfolios
- **Interactive Documentation**: Available at `http://localhost:8000/docs`

**📚 For complete API documentation, examples, and WebSocket usage, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md)**

### Trade Analysis

#### With Portfolio Context
```bash
# Analyze using default portfolio
python run.py AAPL

# Analyze with specific portfolio
python run.py AAPL --portfolio my_portfolio

# Submit an actionable recommendation to Alpaca paper trading
python run.py AAPL --portfolio my_portfolio --paper-trade

# Record an actionable recommendation locally without submitting an Alpaca order
python run.py AAPL --portfolio my_portfolio --record-paper-trade
```

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

---

## License

MIT License. See [LICENSE](LICENSE) file for details.
