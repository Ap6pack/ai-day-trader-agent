# Changelog

All notable changes to this project will be documented in this file.

---

## [Unreleased]

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
- Security best practices: all secrets in `.env`, no sensitive data in logs.
- Added comprehensive `.gitignore` to exclude sensitive and unnecessary files.

### Changed
- Updated OpenAI API usage to latest syntax and upgraded to v1.93.0.
- Improved error handling and user-facing error messages.
- Added ticker symbol validation and robust API response checks.
- Truncate candlestick data sent to OpenAI to avoid context window errors.
- Removed debug logging of OpenAI prompt and raw response from `trade_recommender.py`.
- **Integrated Alpha Vantage API in parallel with Twelve Data for candlestick data.** The pipeline now fetches from both APIs and uses Twelve Data as primary, falling back to Alpha Vantage if needed.
- Updated LICENSE to MIT for personal project.

### Documentation
- Added `README.md` with setup, usage, and security notes.
- Updated documentation to reflect new .gitignore, Alpha Vantage integration, and recent improvements.

---

## [Planned]
- Async/await refactor for performance.
- Automated tests and CI integration.
- Advanced logging and monitoring.
- More robust data validation and edge case handling.

---
