#!/usr/bin/env python3
"""
Sentiment analysis module for news articles using OpenAI GPT models.

This module analyzes the sentiment of financial news articles to provide
trading signals based on market sentiment.
"""

import openai
import json
from config import settings
from utils.logger import get_logger

def analyze_sentiment(articles, symbol=None):
    """
    Analyze sentiment of news articles.
    Uses TypeSafe Jev per-article judgments when TYPESAFE_API_KEY and a symbol
    are available, falling back to OpenAI otherwise.
    Returns a dict: {category, score, rationale}
    """
    if symbol:
        from core import jev_news_analyzer

        if jev_news_analyzer.is_configured():
            try:
                return jev_news_analyzer.analyze_news(symbol, articles)
            except Exception as e:
                get_logger("sentiment_analyzer").warning(
                    f"Jev news analysis failed, falling back to OpenAI: {e}"
                )

    openai.api_key = settings.OPENAI_API_KEY
    prompt = (
        "You are a financial sentiment analysis model. "
        "Analyze the following news articles (in JSON) and return a JSON object with:\n"
        "- category: one of 'positive', 'neutral', or 'negative'\n"
        "- score: a number from -1 (extremely negative) to 1 (extremely positive)\n"
        "- rationale: a brief explanation for your sentiment\n"
        "Articles:\n"
        f"{json.dumps(articles)}"
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.2,
        )
        content = response.choices[0].message.content
        # Try to parse the response as JSON
        try:
            result = json.loads(content)
        except Exception:
            # Fallback: try to extract JSON from text
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                result = json.loads(content[start:end+1])
            else:
                result = {"category": "neutral", "score": 0, "rationale": "No valid response."}
        return result
    except Exception:
        return {"category": "neutral", "score": 0, "rationale": "Sentiment analysis failed."}
