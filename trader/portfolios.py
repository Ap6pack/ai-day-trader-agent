"""
Local paper portfolios: simulated accounts with cash, holdings and a fill history.

No broker is involved: fills are recorded at the price given (the desk and its
autopilot use the live quote). Use several to compare approaches, e.g. an
"aggressive" and a "cautious" portfolio. Stored in the journal database.

    python -m trader.portfolios list
    python -m trader.portfolios create aggressive --cash 10000
    python -m trader.portfolios show aggressive
    python -m trader.portfolios delete aggressive
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from trader import config

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,39}$")
DEFAULT_NAME = "default"

SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolios (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    starting_cash REAL NOT NULL,
    cash REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS holdings (
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    qty REAL NOT NULL,
    avg_price REAL NOT NULL,
    PRIMARY KEY (portfolio_id, symbol)
);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL NOT NULL,
    realized_pl REAL,
    source TEXT,
    note TEXT
);
"""


class PortfolioError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_capital() -> float:
    import os

    try:
        return float(os.getenv("TRADING_CAPITAL") or 5000.0)
    except ValueError:
        return 5000.0


class Portfolios:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else config.journal_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _row(self, conn: sqlite3.Connection, name: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM portfolios WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise PortfolioError(f"No portfolio named {name!r}")
        return row

    # --- portfolios -------------------------------------------------------------

    def create(self, name: str, cash: float) -> Dict[str, Any]:
        name = name.strip()
        if not NAME_RE.match(name):
            raise PortfolioError("Name: 1-40 letters, digits, spaces, '.', '_' or '-'")
        if cash <= 0:
            raise PortfolioError("Starting cash must be positive")
        try:
            with closing(self._connect()) as conn, conn:
                conn.execute("INSERT INTO portfolios (name, starting_cash, cash, created_at) VALUES (?, ?, ?, ?)",
                             (name, float(cash), float(cash), _now()))
        except sqlite3.IntegrityError:
            raise PortfolioError(f"Portfolio {name!r} already exists")
        return self.get(name)

    def ensure_default(self) -> None:
        with closing(self._connect()) as conn:
            exists = conn.execute("SELECT 1 FROM portfolios LIMIT 1").fetchone()
        if not exists:
            self.create(DEFAULT_NAME, default_capital())

    def names(self) -> List[str]:
        with closing(self._connect()) as conn:
            return [r["name"] for r in conn.execute("SELECT name FROM portfolios ORDER BY id")]

    def delete(self, name: str) -> None:
        with closing(self._connect()) as conn, conn:
            row = self._row(conn, name)
            conn.execute("DELETE FROM portfolios WHERE id = ?", (row["id"],))

    def get(self, name: str) -> Dict[str, Any]:
        with closing(self._connect()) as conn:
            row = self._row(conn, name)
            holdings = conn.execute("SELECT symbol, qty, avg_price FROM holdings WHERE portfolio_id = ? AND qty > 0 "
                                    "ORDER BY symbol", (row["id"],)).fetchall()
            realized = conn.execute("SELECT COALESCE(SUM(realized_pl), 0) FROM fills WHERE portfolio_id = ?",
                                    (row["id"],)).fetchone()[0]
        return {
            "name": row["name"], "starting_cash": row["starting_cash"], "cash": row["cash"],
            "created_at": row["created_at"], "realized_pl": round(realized, 2),
            "holdings": [dict(h) for h in holdings],
        }

    def held(self, name: str, symbol: str) -> float:
        with closing(self._connect()) as conn:
            row = self._row(conn, name)
            h = conn.execute("SELECT qty FROM holdings WHERE portfolio_id = ? AND symbol = ?",
                             (row["id"], symbol.upper())).fetchone()
        return h["qty"] if h else 0.0

    def fills(self, name: str, limit: int = 50) -> List[Dict[str, Any]]:
        with closing(self._connect()) as conn:
            row = self._row(conn, name)
            return [dict(r) for r in conn.execute(
                "SELECT ts, symbol, side, qty, price, realized_pl, source, note FROM fills "
                "WHERE portfolio_id = ? ORDER BY id DESC LIMIT ?", (row["id"], limit))]

    # --- trading ------------------------------------------------------------------

    def record_fill(self, name: str, symbol: str, side: str, qty: float, price: float,
                    source: str = "manual", note: str = "") -> Dict[str, Any]:
        symbol, side = symbol.upper(), side.lower()
        if side not in ("buy", "sell"):
            raise PortfolioError("side must be buy or sell")
        if qty <= 0 or price <= 0:
            raise PortfolioError("quantity and price must be positive")
        with closing(self._connect()) as conn, conn:
            row = self._row(conn, name)
            pid, cash = row["id"], row["cash"]
            h = conn.execute("SELECT qty, avg_price FROM holdings WHERE portfolio_id = ? AND symbol = ?",
                             (pid, symbol)).fetchone()
            held, avg = (h["qty"], h["avg_price"]) if h else (0.0, 0.0)
            realized = None
            if side == "buy":
                cost = qty * price
                if cost > cash + 1e-9:
                    raise PortfolioError(f"Not enough cash in {name!r}: need ${cost:,.2f}, have ${cash:,.2f}")
                new_qty = held + qty
                new_avg = (held * avg + cost) / new_qty
                cash -= cost
            else:
                if qty > held + 1e-9:
                    raise PortfolioError(f"{name!r} holds {held:g} {symbol}; cannot sell {qty:g} (long only)")
                new_qty, new_avg = held - qty, avg
                realized = round((price - avg) * qty, 2)
                cash += qty * price
            conn.execute("INSERT INTO holdings (portfolio_id, symbol, qty, avg_price) VALUES (?, ?, ?, ?) "
                         "ON CONFLICT(portfolio_id, symbol) DO UPDATE SET qty = excluded.qty, avg_price = excluded.avg_price",
                         (pid, symbol, new_qty, new_avg if new_qty > 0 else 0.0))
            conn.execute("UPDATE portfolios SET cash = ? WHERE id = ?", (cash, pid))
            conn.execute("INSERT INTO fills (portfolio_id, ts, symbol, side, qty, price, realized_pl, source, note) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (pid, _now(), symbol, side, qty, price, realized, source, note))
        return {"portfolio": name, "symbol": symbol, "side": side, "qty": qty, "price": price,
                "realized_pl": realized, "cash": round(cash, 2)}

    # --- valuation --------------------------------------------------------------------

    def valuation(self, name: str, quotes: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
        """Portfolio value at the given quotes ({symbol: {"last": price, ...}})."""
        p = self.get(name)
        positions, market_value, unrealized = [], 0.0, 0.0
        for h in p["holdings"]:
            last = (quotes.get(h["symbol"]) or {}).get("last") or h["avg_price"]
            value = h["qty"] * last
            upl = (last - h["avg_price"]) * h["qty"]
            market_value += value
            unrealized += upl
            positions.append({
                "symbol": h["symbol"], "qty": h["qty"], "avg_entry_price": round(h["avg_price"], 4),
                "current_price": last, "market_value": round(value, 2), "unrealized_pl": round(upl, 2),
                "unrealized_plpc": round((last / h["avg_price"] - 1) * 100, 2) if h["avg_price"] else None,
            })
        equity = p["cash"] + market_value
        return {
            **{k: p[k] for k in ("name", "starting_cash", "created_at", "realized_pl")},
            "cash": round(p["cash"], 2),
            "market_value": round(market_value, 2),
            "equity": round(equity, 2),
            "unrealized_pl": round(unrealized, 2),
            "total_return_pct": round((equity / p["starting_cash"] - 1) * 100, 2),
            "positions": positions,
        }


def main(argv: Optional[List[str]] = None) -> int:
    config.load_env()
    parser = argparse.ArgumentParser(prog="python -m trader.portfolios")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    create = sub.add_parser("create")
    create.add_argument("name")
    create.add_argument("--cash", type=float, default=None, help="starting cash (default TRADING_CAPITAL)")
    for cmd in ("show", "delete"):
        sub.add_parser(cmd).add_argument("name")
    args = parser.parse_args(argv)
    store = Portfolios()
    try:
        if args.command == "list":
            store.ensure_default()
            print(json.dumps([store.get(n) for n in store.names()], indent=2))
        elif args.command == "create":
            print(json.dumps(store.create(args.name, args.cash or default_capital()), indent=2))
        elif args.command == "show":
            from trader import market_feed

            p = store.get(args.name)
            quotes = market_feed.get_quotes([h["symbol"] for h in p["holdings"]]) if p["holdings"] else {}
            print(json.dumps({**store.valuation(args.name, quotes), "recent_fills": store.fills(args.name, 20)},
                             indent=2))
        else:
            store.delete(args.name)
            print(json.dumps({"deleted": args.name}))
    except PortfolioError as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
