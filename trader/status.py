"""
Show the trading mode, limits and which credentials are configured.

Never prints secret values, so Claude can check setup without reading .env.

    python -m trader.status
"""

from __future__ import annotations

import json
from dataclasses import asdict

from trader import alpaca, config, newsbot
from trader.journal import Journal

CREDENTIALS = ("TYPESAFE_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "NEWS_API_KEY")


def status() -> dict:
    limits = config.load_limits()
    journal = Journal()
    bot = newsbot.load_settings()
    return {
        "mode": limits.mode,
        "limits": {**asdict(limits), "allowed_symbols": sorted(limits.allowed_symbols)},
        "sizing": asdict(config.load_sizing()),
        "orders_passed_guard_today": journal.orders_allowed_today(),
        "newsbot": {
            **{k: v for k, v in asdict(bot).items() if k != "symbols"},
            "symbols": sorted(bot.symbols),
            "trading_base_url": alpaca.trading_base_url(),
            "paper": alpaca.is_paper(),
            "orders_today": journal.count_events_today(newsbot.ORDER_EVENT),
        },
        "journal": str(journal.path),
        "credentials": {name: "set" if config.secret(name) else "missing" for name in CREDENTIALS},
    }


def main() -> int:
    config.load_env()
    print(json.dumps(status(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
