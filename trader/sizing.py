"""
Position sizing from account figures.

Pure policy code with no network calls: Claude reads equity, buying power and
the current position from Robinhood, and this reports the largest quantity
that fits the percentage limits plus a suggested stop-loss. It advises; the
hard dollar limits are enforced separately by the order guard.

CLI:
    python -m trader.sizing --symbol AAPL --side buy --price 201.5 --equity 12000 \
        [--position-value 0] [--buying-power 3000] [--quantity 5]
    python -m trader.sizing --symbol AAPL --side sell --price 201.5 --equity 12000 \
        --shares-held 10 --quantity 10
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from trader import config


@dataclass
class SizingDecision:
    approved: bool
    side: str
    symbol: str
    price: float
    quantity: Optional[int]
    max_quantity: int
    suggested_stop_loss: Optional[float]
    violations: List[str] = field(default_factory=list)
    limits: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def size_trade(
    symbol: str,
    side: str,
    price: float,
    account_equity: float,
    quantity: Optional[int] = None,
    current_position_value: float = 0.0,
    shares_held: Optional[int] = None,
    buying_power: Optional[float] = None,
    rules: Optional[config.Sizing] = None,
) -> SizingDecision:
    rules = rules or config.load_sizing()
    side = side.upper()
    symbol = symbol.upper()
    if side not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    if quantity is not None and quantity <= 0:
        raise ValueError("quantity must be positive")
    if price <= 0:
        raise ValueError("price must be positive")
    if account_equity <= 0:
        raise ValueError("account_equity must be positive")

    max_order_value = rules.max_position_pct * account_equity
    max_symbol_value = rules.max_symbol_pct * account_equity
    limits = {
        "max_order_value": round(max_order_value, 2),
        "max_symbol_value": round(max_symbol_value, 2),
        "stop_loss_pct": rules.stop_loss_pct,
    }
    violations: List[str] = []

    if side == "BUY":
        room = min(max_order_value, max_symbol_value - current_position_value)
        if buying_power is not None:
            room = min(room, buying_power)
        max_quantity = max(int(math.floor(room / price)), 0)
        suggested_stop_loss = round(price * (1 - rules.stop_loss_pct / 100), 2)
        if quantity is not None:
            order_value = quantity * price
            if order_value > max_order_value:
                violations.append(
                    f"Order value ${order_value:,.2f} exceeds ${max_order_value:,.2f} "
                    f"({rules.max_position_pct:.0%} of equity)."
                )
            if current_position_value + order_value > max_symbol_value:
                violations.append(
                    f"Position in {symbol} would be ${current_position_value + order_value:,.2f}, "
                    f"over ${max_symbol_value:,.2f} ({rules.max_symbol_pct:.0%} of equity)."
                )
            if buying_power is not None and order_value > buying_power:
                violations.append(
                    f"Order value ${order_value:,.2f} exceeds buying power ${buying_power:,.2f}."
                )
    else:
        max_quantity = shares_held if shares_held is not None else (quantity or 0)
        suggested_stop_loss = None
        if quantity is not None and shares_held is not None and quantity > shares_held:
            violations.append(
                f"Selling {quantity} shares but only {shares_held} held; short selling is not allowed."
            )

    return SizingDecision(
        approved=not violations,
        side=side,
        symbol=symbol,
        price=price,
        quantity=quantity,
        max_quantity=max_quantity,
        suggested_stop_loss=suggested_stop_loss,
        violations=violations,
        limits=limits,
    )


def main(argv: Optional[List[str]] = None) -> int:
    config.load_env()
    parser = argparse.ArgumentParser(prog="python -m trader.sizing")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--side", required=True, choices=("buy", "sell"))
    parser.add_argument("--price", required=True, type=float)
    parser.add_argument("--equity", required=True, type=float, help="Account equity from Robinhood")
    parser.add_argument("--quantity", type=int, help="Proposed quantity to check")
    parser.add_argument("--position-value", type=float, default=0.0)
    parser.add_argument("--shares-held", type=int)
    parser.add_argument("--buying-power", type=float)
    args = parser.parse_args(argv)
    decision = size_trade(
        args.symbol, args.side, args.price, args.equity,
        quantity=args.quantity,
        current_position_value=args.position_value,
        shares_held=args.shares_held,
        buying_power=args.buying_power,
    )
    print(json.dumps(decision.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
