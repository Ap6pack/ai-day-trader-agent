#!/usr/bin/env python3
from core.candle_fetcher import get_candlestick_data
from core.news_fetcher import get_news_articles
from core.sentiment_analyzer import analyze_sentiment
from core.indicator_engine import compute_indicators
from core.trade_recommender import recommend_trade
from utils.formatter import format_trade_recommendation
from utils.logger import get_logger

logger = get_logger("pipeline")

import re

def is_valid_ticker(ticker):
    # US stock tickers: 1-5 upper/lowercase letters, no spaces or special chars
    return bool(re.fullmatch(r"[A-Za-z]{1,5}", ticker))

def run_trading_pipeline(ticker):
    """
    Main pipeline orchestrator that coordinates all trading analysis steps.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'appl')
        
    Returns:
        str: Formatted trade recommendation for Discord output
    """
    try:
        logger.info(f"Starting trading analysis for {ticker}")

        # Validate ticker symbol
        if not is_valid_ticker(ticker):
            return f"❌ Invalid ticker symbol: `{ticker}`. Please enter a valid US stock ticker (1-5 uppercase letters)."

        # Step 1: Fetch candlestick data for multiple timeframes from both APIs
        logger.info("Fetching candlestick data from Twelve Data and Alpha Vantage...")
        all_candles = get_candlestick_data(ticker)
        td_candles = all_candles.get("twelvedata", {})
        av_candles = all_candles.get("alphavantage", {})

        # Use Twelve Data as primary for now, fallback to Alpha Vantage if empty
        candles = td_candles if any(td_candles.values()) else av_candles
        if not any(candles.values()):
            return f"❌ Unable to fetch market data for `{ticker}` from either API. The symbol may be invalid or the market may be closed."

        # Step 2: Fetch recent news articles
        logger.info("Fetching news articles...")
        articles = get_news_articles(ticker)
        if articles is None:
            articles = []

        # Step 3: Analyze news sentiment
        logger.info("Analyzing news sentiment...")
        sentiment = analyze_sentiment(articles) if articles else {
            "category": "neutral", 
            "score": 0, 
            "rationale": "No recent news available"
        }

        # Step 4: Compute technical indicators
        logger.info("Computing technical indicators...")
        indicators = compute_indicators(candles)

        # Step 5: Generate trade recommendation
        logger.info("Generating trade recommendation...")
        recommendation = recommend_trade(ticker, candles, indicators, sentiment)

        # Step 6: Format output for Discord
        formatted_output = format_trade_recommendation(recommendation)

        logger.info(f"Analysis complete for {ticker}: {recommendation.get('recommendation', 'N/A')}")
        return formatted_output

    except Exception as e:
        logger.error(f"Pipeline failed for {ticker}: {str(e)}")
        return f"❌ Analysis failed for `{ticker}`. Please try again later."
