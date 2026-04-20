#!/usr/bin/env python3
"""
Alpaca Executor - Sends real orders to Alpaca paper trading account.
Sits between pipeline.py (signals) and portfolio_manager.py (recording).

Usage:
    Set these in your .env file:
        ALPACA_API_KEY=your_key_here
        ALPACA_SECRET_KEY=your_secret_here
        ALPACA_TRADING_BASE_URL=https://paper-api.alpaca.markets/v2
"""

import os
import logging
from typing import Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class AlpacaExecutor:
    """
    Sends orders to Alpaca and returns results.
    Uses Alpaca paper trading by default.
    """

    def __init__(self, base_url: Optional[str] = None):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = self._normalize_base_url(
            base_url
            or os.getenv("ALPACA_TRADING_BASE_URL")
            or os.getenv("ALPACA_BASE_URL")
            or "https://paper-api.alpaca.markets/v2"
        )

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in your .env file"
            )

        if "paper-api.alpaca.markets" not in self.base_url:
            raise ValueError(
                "Paper trading requires ALPACA_TRADING_BASE_URL=https://paper-api.alpaca.markets/v2"
            )

        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }
        logger.info("AlpacaExecutor ready in PAPER mode")

    def _normalize_base_url(self, base_url: str) -> str:
        """Accept either the Alpaca root URL or the versioned v2 URL."""
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v2"):
            normalized = f"{normalized}/v2"
        return normalized

    # ------------------------------------------------------------------
    # Account helpers
    # ------------------------------------------------------------------

    def get_account(self) -> Dict:
        """Return account details (buying power, equity, etc.)."""
        resp = requests.get(
            f"{self.base_url}/account", headers=self.headers, timeout=10
        )
        resp.raise_for_status()
        return resp.json()

    def get_positions(self) -> list:
        """Return all open positions."""
        resp = requests.get(
            f"{self.base_url}/positions", headers=self.headers, timeout=10
        )
        resp.raise_for_status()
        return resp.json()

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Return a single open position, or None if not held."""
        try:
            resp = requests.get(
                f"{self.base_url}/positions/{symbol}",
                headers=self.headers,
                timeout=10,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError:
            return None

    def is_market_open(self) -> bool:
        """Return True if the US market is currently open."""
        resp = requests.get(
            f"{self.base_url}/clock", headers=self.headers, timeout=10
        )
        resp.raise_for_status()
        return resp.json().get("is_open", False)

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def execute_signal(self, signal: Dict) -> Optional[Dict]:
        """
        Main entry point.  Pass the dict that pipeline.py returns and
        this will place the order when conditions are right.

        Returns the Alpaca order dict on success, None if skipped.
        """
        action = str(signal.get("recommendation") or signal.get("signal") or "HOLD").upper()
        symbol = str(signal.get("symbol") or "").upper()
        quantity = int(signal.get("quantity") or 0)

        if action == "HOLD" or quantity <= 0 or not symbol:
            logger.info(f"Skipping execution: {action} {quantity} {symbol}")
            return None

        # Safety check — make sure market is open (paper trading still
        # queues orders, but this prevents runaway scripts overnight)
        if not self.is_market_open():
            logger.warning(
                "Market is closed. Order queued but won't fill until open."
            )

        if action == "BUY":
            return self._place_order(symbol, quantity, "buy")
        elif action == "SELL":
            return self._verify_and_sell(symbol, quantity)

        return None

    def _place_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
    ) -> Dict:
        """Place a market order and return the Alpaca response."""
        payload = {
            "symbol":        symbol,
            "qty":           str(quantity),
            "side":          side,
            "type":          order_type,
            "time_in_force": time_in_force,
        }

        logger.info(f"Placing order: {side.upper()} {quantity} {symbol}")
        resp = requests.post(
            f"{self.base_url}/orders",
            headers=self.headers,
            json=payload,
            timeout=10,
        )

        if not resp.ok:
            logger.error(f"Order failed: {resp.status_code} {resp.text}")
            resp.raise_for_status()

        order = resp.json()
        logger.info(
            f"Order placed — ID: {order['id']} | "
            f"{order['side'].upper()} {order['qty']} {order['symbol']} "
            f"@ {order.get('filled_avg_price', 'pending')}"
        )
        return order

    def _verify_and_sell(self, symbol: str, quantity: int) -> Optional[Dict]:
        """Only sell what we actually own — avoids shorting by accident."""
        position = self.get_position(symbol)

        if not position:
            logger.warning(f"SELL skipped: no position in {symbol}")
            return None

        owned = int(float(position.get("qty", 0)))
        sell_qty = min(quantity, owned)

        if sell_qty <= 0:
            logger.warning(f"SELL skipped: qty {sell_qty} for {symbol}")
            return None

        return self._place_order(symbol, sell_qty, "sell")

    # ------------------------------------------------------------------
    # Trailing stop helper (from video concept)
    # ------------------------------------------------------------------

    def place_trailing_stop(
        self, symbol: str, quantity: int, trail_percent: float = 5.0
    ) -> Dict:
        """
        Buy shares and immediately attach a trailing stop order.
        trail_percent: how far below peak to set the floor (default 5 %).
        """
        buy_order = self._place_order(symbol, quantity, "buy")

        stop_payload = {
            "symbol":           symbol,
            "qty":              str(quantity),
            "side":             "sell",
            "type":             "trailing_stop",
            "trail_percent":    str(trail_percent),
            "time_in_force":    "gtc",
        }

        resp = requests.post(
            f"{self.base_url}/orders",
            headers=self.headers,
            json=stop_payload,
            timeout=10,
        )
        resp.raise_for_status()
        stop_order = resp.json()
        logger.info(
            f"Trailing stop set: {trail_percent}% below peak for {symbol}"
        )
        return {"buy_order": buy_order, "stop_order": stop_order}

    # ------------------------------------------------------------------
    # Cancel helpers
    # ------------------------------------------------------------------

    def cancel_all_orders(self) -> list:
        """Cancel every open order — useful for end-of-day cleanup."""
        resp = requests.delete(
            f"{self.base_url}/orders", headers=self.headers, timeout=10
        )
        resp.raise_for_status()
        logger.info("All open orders cancelled")
        return resp.json() if resp.text else []

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order by ID."""
        resp = requests.delete(
            f"{self.base_url}/orders/{order_id}",
            headers=self.headers,
            timeout=10,
        )
        return resp.status_code == 204


# ------------------------------------------------------------------
# Quick connection test — run this file directly to verify keys work
# python core/alpaca_executor.py
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    executor = AlpacaExecutor()

    print("\n--- Account ---")
    account = executor.get_account()
    print(f"Status:       {account['status']}")
    print(f"Equity:       ${float(account['equity']):,.2f}")
    print(f"Buying Power: ${float(account['buying_power']):,.2f}")
    print(f"Market open:  {executor.is_market_open()}")

    print("\n--- Open Positions ---")
    positions = executor.get_positions()
    if positions:
        for p in positions:
            print(
                f"  {p['symbol']:6} {p['qty']:>6} shares  "
                f"avg ${float(p['avg_entry_price']):.2f}  "
                f"P&L ${float(p['unrealized_pl']):.2f}"
            )
    else:
        print("  No open positions")

    print("\nConnection OK ✓")
