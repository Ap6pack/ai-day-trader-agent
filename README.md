# AI Day Trader Agent

A modular, production-ready AI-powered trading agent that analyzes market data and news sentiment to generate actionable trade recommendations. Supports Discord bot and CLI usage.

---

## Features

- Discord bot interface for real-time trade analysis
- Multi-timeframe candlestick data (1m, 15m, 1h) via **both Twelve Data and Alpha Vantage APIs** (used in parallel, with fallback)
- News sentiment analysis using OpenAI GPT-4
- Technical indicators (RSI, MACD, SMA, EMA) via `ta` and `pandas`
- Modular, testable, and secure architecture
- Handles OpenAI context window limits by truncating data sent to the model
- Comprehensive `.gitignore` to protect secrets and exclude unnecessary files

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
- Copyright headers and license information included.
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

## Recent Improvements

- Integrated Alpha Vantage API in parallel with Twelve Data for candlestick data (pipeline uses both, with fallback).
- Upgraded OpenAI API usage to v1.93.0
- Truncate candlestick data sent to OpenAI to avoid context window errors
- Removed debug logging of OpenAI prompt and raw response
- Added comprehensive `.gitignore`

---

## License

MIT License. See LICENSE file for details.
