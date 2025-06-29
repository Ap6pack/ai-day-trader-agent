#!/usr/bin/env python3
"""
Enhanced command-line interface for the AI Day Trader Agent.
Now includes dividend capture capabilities and multi-strategy analysis.
"""

import sys
import logging
from datetime import datetime
from config.env_loader import load_env_variables
from core.pipeline import run_enhanced_analysis
from core.dividend_database import DividendDatabase
from utils.logger import get_logger
from utils.formatter import format_analysis_result

# Set up logging
logger = get_logger('ai_day_trader')


def main():
    """Main entry point for the enhanced trading analysis."""
    if len(sys.argv) != 2:
        print("Usage: python run.py <TICKER>")
        print("Example: python run.py APAM")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    logger.info(f"Starting enhanced analysis for {ticker}")
    
    try:
        # Load API keys
        api_keys = load_env_variables()
        
        # Initialize database for tracking
        db = DividendDatabase()
        
        # Run enhanced multi-strategy analysis
        print(f"\n🤖 AI Day Trader Agent - Enhanced Analysis for {ticker}")
        print("=" * 60)
        
        result = run_enhanced_analysis(ticker, api_keys)
        
        # Display formatted results
        print("\n📊 Analysis Results:")
        print("-" * 40)
        formatted_output = format_analysis_result(result)
        print(formatted_output)
        
        # Log to database if it's a dividend capture trade
        if result.get('primary_strategy') == 'dividend' and result['signal'] != 'HOLD':
            logger.info(f"Recording dividend capture signal in database")
            # Database logging would go here
        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
