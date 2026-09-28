#!/usr/bin/env python3
"""
Hard pre-trade risk limits.

Pure policy code with no network calls: the caller supplies the account
figures (from Robinhood, Alpaca or the local portfolio) and gets back whether
the order fits the configured limits, what it violates, the largest quantity
that would fit, and a suggested stop-loss. Nothing here places orders.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from config.settings import trading_config


@dataclass
class RiskDecision:
    approved: bool
    side: str
    symbol: str
    quantity: int
    price: float
    order_value: float
    max_quantity: int
    suggested_stop_loss: Optional[float]
    violations: List[str] = field(default_factory=list)
    limits: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def check_trade(
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    account_equity: float,
    current_position_value: float = 0.0,
    shares_held: Optional[int] = None,
    buying_power: Optional[float] = None,
    trades_today: int = 0,
    config=trading_config,
) -> RiskDecision:
    side = side.upper()
    symbol = symbol.upper()
    violations: List[str] = []

    if side not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if price <= 0:
        raise ValueError("price must be positive")
    if account_equity <= 0:
        raise ValueError("account_equity must be positive")

    order_value = quantity * price
    max_order_value = config.MAX_POSITION_PERCENTAGE * account_equity
    max_symbol_value = config.MAX_PORTFOLIO_ALLOCATION * account_equity
    limits = {
        "max_order_value": round(max_order_value, 2),
        "max_symbol_value": round(max_symbol_value, 2),
        "max_daily_trades": config.MAX_DAILY_TRADES,
        "stop_loss_pct": config.STOP_LOSS_PERCENTAGE,
    }

    if trades_today >= config.MAX_DAILY_TRADES:
        violations.append(
            f"Daily trade limit reached ({trades_today}/{config.MAX_DAILY_TRADES})."
        )

    if side == "BUY":
        room = min(max_order_value, max_symbol_value - current_position_value)
        if buying_power is not None:
            room = min(room, buying_power)
        max_quantity = max(int(math.floor(room / price)), 0)

        if order_value > max_order_value:
            violations.append(
                f"Order value ${order_value:,.2f} exceeds the per-order limit "
                f"${max_order_value:,.2f} ({config.MAX_POSITION_PERCENTAGE:.0%} of equity)."
            )
        if current_position_value + order_value > max_symbol_value:
            violations.append(
                f"Position in {symbol} would be ${current_position_value + order_value:,.2f}, "
                f"over the ${max_symbol_value:,.2f} limit "
                f"({config.MAX_PORTFOLIO_ALLOCATION:.0%} of equity)."
            )
        if buying_power is not None and order_value > buying_power:
            violations.append(
                f"Order value ${order_value:,.2f} exceeds buying power ${buying_power:,.2f}."
            )
        suggested_stop_loss = round(price * (1 - config.STOP_LOSS_PERCENTAGE / 100), 2)
    else:
        max_quantity = shares_held if shares_held is not None else quantity
        if shares_held is not None and quantity > shares_held:
            violations.append(
                f"Selling {quantity} shares but only {shares_held} held; short selling is not allowed."
            )
        suggested_stop_loss = None

    if trades_today >= config.MAX_DAILY_TRADES:
        max_quantity = 0

    return RiskDecision(
        approved=not violations,
        side=side,
        symbol=symbol,
        quantity=quantity,
        price=price,
        order_value=round(order_value, 2),
        max_quantity=max_quantity,
        suggested_stop_loss=suggested_stop_loss,
        violations=violations,
        limits=limits,
    )
