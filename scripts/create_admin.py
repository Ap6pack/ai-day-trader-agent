#!/usr/bin/env python3
"""Create an administrator account for the API."""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

from passlib.context import CryptContext

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.portfolio_manager import PortfolioManager  # noqa: E402


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _prompt(name: str, env_name: str, *, secret: bool = False) -> str:
    value = os.getenv(env_name)
    if value:
        return value
    prompt = f"{name}: "
    return getpass.getpass(prompt) if secret else input(prompt).strip()


def main() -> int:
    username = _prompt("Username", "ADMIN_USERNAME")
    email = _prompt("Email", "ADMIN_EMAIL")
    password = _prompt("Password", "ADMIN_PASSWORD", secret=True)

    if not username or not email or not password:
        print("Username, email, and password are required.", file=sys.stderr)
        return 1

    db = PortfolioManager(os.getenv("PORTFOLIO_DB_PATH", "data/portfolios.db"))
    hashed_password = pwd_context.hash(password)

    try:
        user = db.create_user(
            username=username,
            email=email,
            hashed_password=hashed_password,
            is_admin=True,
        )
    except ValueError as exc:
        print(f"Could not create admin: {exc}", file=sys.stderr)
        return 1

    print(f"Created admin user: {user['username']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
