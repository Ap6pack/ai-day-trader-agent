def format_trade_recommendation(rec):
    """
    Format the trade recommendation dict for Discord output.
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
