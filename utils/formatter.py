#!/usr/bin/env python3
def format_basic_trade_recommendation(rec):
    """
    Format a basic trade recommendation dict for Discord output.
    
    Args:
        rec: Dictionary containing basic trade recommendation data with keys:
             recommendation, entry_price, stop_loss, exit_price
             
    Returns:
        str: Formatted string for Discord display
    """
    lines = []
    lines.append(f"**Trade Recommendation:** `{rec.get('recommendation', 'N/A').upper()}`")
    entry = rec.get("entry_price")
    stop = rec.get("stop_loss")
    exit_ = rec.get("exit_price")
    lines.append(f"**Entry Price:** {entry if entry is not None else 'N/A'}")
    lines.append(f"**Stop Loss:** {stop if stop is not None else 'N/A'}")
    lines.append(f"**Exit/Target Price:** {exit_ if exit_ is not None else 'N/A'}")
    return "\n".join(lines)

def format_analysis_result(result):
    """
    Format the enhanced analysis result for CLI output.
    Handles both old and new result formats for backward compatibility.
    """
    lines = []
    
    # Handle enhanced pipeline results
    if isinstance(result, dict) and 'primary_strategy' in result:
        # New enhanced format
        lines.append(f"**Primary Strategy:** {result.get('primary_strategy', 'N/A').upper()}")
        lines.append(f"**Recommendation:** {result.get('recommendation', 'N/A').upper()}")
        lines.append(f"**Confidence:** {result.get('confidence', 'N/A')}")
        lines.append(f"**Quantity:** {result.get('quantity', 'N/A')} shares")
        lines.append(f"**Reason:** {result.get('reason', 'N/A')}")
        
        # Add technical indicators if available
        if 'technical_indicators' in result:
            indicators = result['technical_indicators']
            lines.append("\n**Technical Indicators:**")
            
            # Get current price from all_signals if available
            current_price = None
            if 'all_signals' in result and 'technical' in result['all_signals']:
                # Try to extract current price from technical analysis
                tech_data = result['all_signals']['technical']
                if 'current_price' in tech_data:
                    current_price = tech_data['current_price']
            
            if current_price:
                lines.append(f"  Current Price: ${current_price:.2f}")
            
            if indicators.get('rsi') is not None:
                rsi_status = "Oversold" if indicators['rsi'] < 30 else "Overbought" if indicators['rsi'] > 70 else "Neutral"
                lines.append(f"  RSI: {indicators['rsi']:.2f} ({rsi_status})")
            if indicators.get('macd') is not None and indicators.get('macd_signal') is not None:
                macd_trend = "Bullish" if indicators['macd'] > indicators['macd_signal'] else "Bearish"
                lines.append(f"  MACD: {indicators['macd']:.4f} / Signal: {indicators['macd_signal']:.4f} ({macd_trend})")
            if indicators.get('sma_20') is not None:
                sma_trend = "Above" if current_price and current_price > indicators['sma_20'] else "Below" if current_price else ""
                lines.append(f"  SMA(20): ${indicators['sma_20']:.2f} {sma_trend}")
            if indicators.get('ema_20') is not None:
                ema_trend = "Above" if current_price and current_price > indicators['ema_20'] else "Below" if current_price else ""
                lines.append(f"  EMA(20): ${indicators['ema_20']:.2f} {ema_trend}")
        
        # Add all signals summary
        if 'all_signals' in result:
            signals = result['all_signals']
            lines.append("\n**All Strategy Signals:**")
            
            # Technical signal
            tech = signals.get('technical', {})
            lines.append(f"  Technical: {tech.get('signal', 'N/A')} (strength: {tech.get('strength', 0):.2f})")
            
            # Sentiment signal  
            sent = signals.get('sentiment', {})
            lines.append(f"  Sentiment: {sent.get('signal', 'N/A')} (score: {sent.get('sentiment_score', 0):.2f})")
            
            # Dividend signal
            div = signals.get('dividend', {})
            lines.append(f"  Dividend: {div.get('signal', 'N/A')} (reason: {div.get('reason', 'N/A')})")
        
        # Add risk parameters if available
        if 'risk_parameters' in result and result['risk_parameters']:
            risk = result['risk_parameters']
            lines.append("\n**Risk Management:**")
            if 'stop_loss' in risk:
                lines.append(f"  Stop Loss: ${risk['stop_loss']}")
            if 'take_profit' in risk:
                lines.append(f"  Take Profit: ${risk['take_profit']}")
            if 'risk_percentage' in risk:
                lines.append(f"  Risk: {risk['risk_percentage']}%")
            if 'position_value' in risk:
                lines.append(f"  Position Value: ${risk['position_value']}")
        
        # Add dividend info if available
        if 'dividend_info' in result:
            div_info = result['dividend_info']
            lines.append("\n**Dividend Information:**")
            lines.append(f"  Days to Ex-Dividend: {div_info.get('days_to_ex_dividend', 'N/A')}")
            lines.append(f"  Expected Dividend: ${div_info.get('expected_dividend', 'N/A')}")
        
        lines.append(f"\n**Analysis Time:** {result.get('timestamp', 'N/A')}")
        
    else:
        # Fallback for old format or simple results
        if isinstance(result, dict):
            lines.append(f"**Recommendation:** {result.get('recommendation', 'N/A').upper()}")
            if 'entry_price' in result:
                lines.append(f"**Entry Price:** {result.get('entry_price', 'N/A')}")
            if 'stop_loss' in result:
                lines.append(f"**Stop Loss:** {result.get('stop_loss', 'N/A')}")
            if 'exit_price' in result:
                lines.append(f"**Exit Price:** {result.get('exit_price', 'N/A')}")
        else:
            lines.append(f"**Result:** {str(result)}")
    
    return "\n".join(lines)
