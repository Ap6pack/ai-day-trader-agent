"""
Settings for the trader package.

Standard library only, so the order guard hook runs under any python3 even if
the project's virtualenv is missing. Values in the project .env win over
variables already exported in the shell: a stale export must never silently
shadow the key or limit in .env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

MODES = ("review", "live")


def parse_env_file(path: Path) -> Dict[str, str]:
    """Parse KEY=VALUE lines, handling `export`, quotes and trailing comments."""
    values: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value[:1] in ("'", '"'):
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end != -1 else value[1:]
        else:
            hash_at = value.find(" #")
            if hash_at != -1:
                value = value[:hash_at].rstrip()
        if key:
            values[key] = value
    return values


def load_env(path: Path = ENV_PATH) -> None:
    """Load .env into os.environ, overriding existing values."""
    os.environ.update(parse_env_file(path))


def _get(env: Dict[str, str], name: str, default: str) -> str:
    value = env.get(name, "").strip()
    return value if value else default


def _float(env: Dict[str, str], name: str, default: float) -> float:
    return float(_get(env, name, str(default)))


def _int(env: Dict[str, str], name: str, default: int) -> int:
    return int(_get(env, name, str(default)))


def _bool(env: Dict[str, str], name: str) -> bool:
    return _get(env, name, "false").lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Limits:
    """Hard limits enforced by the order guard. Dollar amounts, not percentages:
    the guard sees only the order, not the account balance."""

    mode: str = "review"
    max_order_usd: float = 500.0
    max_orders_per_day: int = 3
    review_window_minutes: int = 15
    allowed_symbols: FrozenSet[str] = field(default_factory=frozenset)
    allow_options: bool = False
    allow_crypto: bool = False


@dataclass(frozen=True)
class Sizing:
    """Soft sizing rules used when proposing trades (percent of account equity)."""

    max_position_pct: float = 0.10
    max_symbol_pct: float = 0.25
    stop_loss_pct: float = 3.0


def load_limits(env: Optional[Dict[str, str]] = None) -> Limits:
    env = dict(os.environ) if env is None else env
    mode = _get(env, "TRADER_MODE", "review").lower()
    if mode not in MODES:
        raise ValueError(f"TRADER_MODE must be one of {MODES}, got {mode!r}")
    symbols = _get(env, "TRADER_ALLOWED_SYMBOLS", "")
    return Limits(
        mode=mode,
        max_order_usd=_float(env, "TRADER_MAX_ORDER_USD", 500.0),
        max_orders_per_day=_int(env, "TRADER_MAX_ORDERS_PER_DAY", 3),
        review_window_minutes=_int(env, "TRADER_REVIEW_WINDOW_MINUTES", 15),
        allowed_symbols=frozenset(s.strip().upper() for s in symbols.split(",") if s.strip()),
        allow_options=_bool(env, "TRADER_ALLOW_OPTIONS"),
        allow_crypto=_bool(env, "TRADER_ALLOW_CRYPTO"),
    )


def load_sizing(env: Optional[Dict[str, str]] = None) -> Sizing:
    env = dict(os.environ) if env is None else env
    return Sizing(
        max_position_pct=_float(env, "MAX_POSITION_PERCENTAGE", 0.10),
        max_symbol_pct=_float(env, "MAX_PORTFOLIO_ALLOCATION", 0.25),
        stop_loss_pct=_float(env, "STOP_LOSS_PCT", 3.0),
    )


def journal_path(env: Optional[Dict[str, str]] = None) -> Path:
    env = dict(os.environ) if env is None else env
    path = Path(_get(env, "TRADER_JOURNAL_PATH", str(PROJECT_ROOT / "data" / "journal.db")))
    return path if path.is_absolute() else PROJECT_ROOT / path


def secret(name: str) -> Optional[str]:
    value = os.environ.get(name, "").strip()
    return value or None
