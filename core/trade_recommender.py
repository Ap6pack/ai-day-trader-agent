#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
import json
from config import settings

def recommend_trade(symbol, candles, indicators, sentiment):
    """
    Use OpenAI to generate a trade recommendation based on candles, indicators, and sentiment.
    Returns a dict: {recommendation, entry_price, stop_loss, exit_price}
    """
    openai.api_key = settings.OPENAI_API_KEY
    prompt = (
        "You are an expert day trader AI. "
        "Given the following data, output ONLY a JSON object with the following keys:\n"
        "- recommendation: one of 'buy', 'sell', or 'hold'\n"
        "- entry_price: the suggested entry price (or null if not applicable)\n"
        "- stop_loss: the suggested stop-loss price (or null if not applicable)\n"
        "- exit_price: the suggested exit/target price (or null if not applicable)\n"
        "Do not include any rationale or extra commentary.\n\n"
        f"Symbol: {symbol}\n"
        f"Candles: {json.dumps(candles)}\n"
        f"Indicators: {json.dumps(indicators)}\n"
        f"Sentiment: {json.dumps(sentiment)}"
    )
    try:
        from utils.logger import get_logger
        logger = get_logger("trade_recommender")
        response = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": prompt}
            ],
            max_tokens=500,
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
                logger.error("OpenAI response could not be parsed as JSON. Returning HOLD fallback.")
                result = {
                    "recommendation": "hold",
                    "entry_price": None,
                    "stop_loss": None,
                    "exit_price": None
                }
        return result
    except Exception as e:
        from utils.logger import get_logger
        logger = get_logger("trade_recommender")
        logger.error(f"OpenAI call failed: {e}")
        return {
            "recommendation": "hold",
            "entry_price": None,
            "stop_loss": None,
            "exit_price": None
        }
