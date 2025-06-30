#!/usr/bin/env python3
def format_dividend_info(dividend_data: dict) -> str:
    """Format dividend capture information."""
    if not dividend_data:
        return ""
    
    lines = ["\n**Dividend Capture Details:**"]
    
    if 'days_to_ex_dividend' in dividend_data:
        lines.append(f"Days to Ex-Dividend: {dividend_data['days_to_ex_dividend']}")
    
    if 'expected_dividend' in dividend_data:
        lines.append(f"Expected Dividend: ${dividend_data['expected_dividend']:.2f}")
    
    if 'capture_confidence' in dividend_data:
        lines.append(f"Capture Confidence: {dividend_data['capture_confidence']:.1%}")
    
    return '\n'.join(lines)

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
    
    # Handle error cases
    if isinstance(result, dict) and result.get('error'):
        lines.append(f"❌ **Error:** {result.get('message', 'Unknown error')}")
        lines.append(f"**Symbol:** {result.get('symbol', 'N/A')}")
        lines.append(f"**Error Type:** {result.get('error_type', 'unknown')}")
        lines.append(f"**Time:** {result.get('timestamp', 'N/A')}")
        return "\n".join(lines)
    
    # Handle enhanced pipeline results
    if isinstance(result, dict) and 'primary_strategy' in result:
        # New enhanced format
        lines.append(f"**Primary Strategy:** {result.get('primary_strategy', 'N/A').upper()}")
        lines.append(f"**Recommendation:** {result.get('recommendation', 'N/A').upper()}")
        lines.append(f"**Confidence:** {result.get('confidence', 'N/A')}")
        # Enhanced quantity display with position value
        quantity = result.get('quantity', 0)
        if quantity > 0 and 'risk_parameters' in result and 'position_value' in result['risk_parameters']:
            position_value = result['risk_parameters']['position_value']
            lines.append(f"**Quantity:** {quantity} shares (${position_value:,.2f})")
        else:
            lines.append(f"**Quantity:** {quantity} shares")
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
            
            # Show actual position values if quantity > 0
            if 'position_value' in risk and risk['position_value'] > 0:
                lines.append(f"  Position Value: ${risk['position_value']}")
                if 'risk_percentage' in risk:
                    lines.append(f"  Risk: {risk['risk_percentage']}%")
            else:
                # Show theoretical values for HOLD signals
                lines.append(f"  Position Value: $0.00 (No position)")
                if 'theoretical_position_value' in risk:
                    lines.append(f"  Theoretical Position (1 share): ${risk['theoretical_position_value']}")
                if 'theoretical_risk' in risk:
                    lines.append(f"  Theoretical Risk (1 share): ${risk['theoretical_risk']}")
        
        # Add dividend info if available using the dedicated formatter
        if 'dividend_info' in result:
            dividend_formatted = format_dividend_info(result['dividend_info'])
            if dividend_formatted:
                lines.append(dividend_formatted)
        
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
