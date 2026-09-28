"""
Trade journal: every order review, guard decision and trading decision.

Standard library only (SQLite) so the order guard can use it. The journal is
the evidence for whether the strategy works: decisions are logged with their
reasoning and entry price, then scored later against what the price did.

CLI (run from the project root):
    python -m trader.journal decide --symbol AAPL --action buy --price 201.5 \
        --confidence 0.6 --thesis "Earnings beat, holding above VWAP"
    python -m trader.journal list [--limit 20]
    python -m trader.journal pending          # decisions from earlier days, not yet scored
    python -m trader.journal outcome --id 7 --price 204.1   # close of the decision's day
    python -m trader.journal summary
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from trader import config

MARKET_TZ = ZoneInfo("America/New_York")
ACTIONS = ("buy", "sell", "pass")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    tool TEXT,
    symbol TEXT,
    side TEXT,
    quantity REAL,
    price REAL,
    notional REAL,
    detail TEXT,
    payload TEXT
);
CREATE INDEX IF NOT EXISTS events_kind_ts ON events (kind, ts);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    price REAL NOT NULL,
    confidence REAL,
    thesis TEXT,
    mode TEXT NOT NULL,
    outcome_price REAL,
    outcome_return REAL,
    outcome_ts TEXT
);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


class Journal:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else config.journal_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    # --- events -----------------------------------------------------------

    def log_event(
        self,
        kind: str,
        *,
        tool: Optional[str] = None,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        notional: Optional[float] = None,
        detail: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        now: Optional[datetime] = None,
    ) -> int:
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "INSERT INTO events (ts, kind, tool, symbol, side, quantity, price, notional, detail, payload)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _iso(now or _now()), kind, tool,
                    symbol.upper() if symbol else None,
                    side.lower() if side else None,
                    quantity, price, notional, detail,
                    json.dumps(payload, default=str) if payload is not None else None,
                ),
            )
            return int(cur.lastrowid)

    def orders_allowed_today(self, now: Optional[datetime] = None) -> int:
        """Orders the guard let through since midnight US/Eastern."""
        return self.count_events_today("order_allowed", now=now)

    def count_events_today(self, kind: str, now: Optional[datetime] = None) -> int:
        """Events of this kind since midnight US/Eastern."""
        now = now or _now()
        local_midnight = now.astimezone(MARKET_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM events WHERE kind = ? AND ts >= ?", (kind, _iso(local_midnight)),
            ).fetchone()
        return int(row[0])

    def last_event_for_symbol(self, kind: str, symbol: str) -> Optional[sqlite3.Row]:
        with closing(self._connect()) as conn:
            return conn.execute(
                "SELECT * FROM events WHERE kind = ? AND symbol = ? ORDER BY ts DESC LIMIT 1",
                (kind, symbol.upper()),
            ).fetchone()

    def latest_review(
        self, symbol: str, side: str, within_minutes: int, now: Optional[datetime] = None
    ) -> Optional[sqlite3.Row]:
        since = (now or _now()) - timedelta(minutes=within_minutes)
        with closing(self._connect()) as conn:
            return conn.execute(
                "SELECT * FROM events WHERE kind = 'review' AND symbol = ? AND side = ? AND ts >= ?"
                " ORDER BY ts DESC LIMIT 1",
                (symbol.upper(), side.lower(), _iso(since)),
            ).fetchone()

    def events(self, limit: int = 20) -> List[sqlite3.Row]:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    # --- decisions --------------------------------------------------------

    def add_decision(
        self,
        symbol: str,
        action: str,
        price: float,
        *,
        confidence: Optional[float] = None,
        thesis: Optional[str] = None,
        mode: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> int:
        action = action.lower()
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}")
        if price <= 0:
            raise ValueError("price must be positive")
        if confidence is not None and not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        mode = mode or config.load_limits().mode
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "INSERT INTO decisions (ts, symbol, action, price, confidence, thesis, mode)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (_iso(now or _now()), symbol.upper(), action, price, confidence, thesis, mode),
            )
            return int(cur.lastrowid)

    def decisions(self, limit: int = 20) -> List[sqlite3.Row]:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    def decision(self, decision_id: int) -> Optional[sqlite3.Row]:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,)).fetchone()

    def unscored_decisions(self, older_than: datetime) -> List[sqlite3.Row]:
        with closing(self._connect()) as conn:
            return conn.execute(
                "SELECT * FROM decisions WHERE outcome_ts IS NULL AND ts < ? ORDER BY ts",
                (_iso(older_than),),
            ).fetchall()

    def set_outcome(self, decision_id: int, outcome_price: float, entry_price: float,
                    now: Optional[datetime] = None) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE decisions SET outcome_price = ?, outcome_return = ?, outcome_ts = ? WHERE id = ?",
                (outcome_price, outcome_price / entry_price - 1, _iso(now or _now()), decision_id),
            )

    def summary(self) -> Dict[str, Any]:
        """Hit rate and average signed return of scored buy/sell decisions."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT action, outcome_return FROM decisions WHERE outcome_return IS NOT NULL AND action != 'pass'"
            ).fetchall()
            counts = dict(conn.execute("SELECT action, COUNT(*) FROM decisions GROUP BY action").fetchall())
        signed = [r["outcome_return"] * (1 if r["action"] == "buy" else -1) for r in rows]
        return {
            "decisions": counts,
            "scored": len(signed),
            "hit_rate": round(sum(1 for s in signed if s > 0) / len(signed), 3) if signed else None,
            "avg_signed_return": round(sum(signed) / len(signed), 5) if signed else None,
        }


def _print_rows(rows: List[sqlite3.Row]) -> None:
    for row in rows:
        print(json.dumps(dict(row)))


def main(argv: Optional[List[str]] = None) -> int:
    config.load_env()
    parser = argparse.ArgumentParser(prog="python -m trader.journal")
    sub = parser.add_subparsers(dest="command", required=True)

    decide = sub.add_parser("decide", help="Record a buy/sell/pass decision")
    decide.add_argument("--symbol", required=True)
    decide.add_argument("--action", required=True, choices=ACTIONS)
    decide.add_argument("--price", required=True, type=float, help="Price when the decision was made")
    decide.add_argument("--confidence", type=float)
    decide.add_argument("--thesis", help="One or two sentences on why")

    ls = sub.add_parser("list", help="Show recent decisions")
    ls.add_argument("--limit", type=int, default=20)
    ev = sub.add_parser("events", help="Show recent reviews and guard decisions")
    ev.add_argument("--limit", type=int, default=20)
    sub.add_parser("pending", help="Unscored decisions made before today (US/Eastern)")
    outcome = sub.add_parser("outcome", help="Record the closing price for a decision's day")
    outcome.add_argument("--id", required=True, type=int)
    outcome.add_argument("--price", required=True, type=float)
    sub.add_parser("summary", help="Hit rate and returns of scored decisions")

    args = parser.parse_args(argv)
    journal = Journal()
    if args.command == "decide":
        decision_id = journal.add_decision(
            args.symbol, args.action, args.price, confidence=args.confidence, thesis=args.thesis
        )
        print(json.dumps({"recorded_decision_id": decision_id}))
    elif args.command == "list":
        _print_rows(journal.decisions(args.limit))
    elif args.command == "events":
        _print_rows(journal.events(args.limit))
    elif args.command == "pending":
        today = _now().astimezone(MARKET_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
        _print_rows(journal.unscored_decisions(older_than=today))
    elif args.command == "outcome":
        row = journal.decision(args.id)
        if row is None:
            print(json.dumps({"error": f"no decision {args.id}"}))
            return 1
        journal.set_outcome(args.id, args.price, row["price"])
        print(json.dumps(dict(journal.decision(args.id))))
    else:
        print(json.dumps(journal.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
