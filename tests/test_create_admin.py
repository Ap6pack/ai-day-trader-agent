from __future__ import annotations

from pathlib import Path

from core.portfolio_manager import PortfolioManager
from scripts import create_admin


def test_create_admin_from_environment(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "admin.db"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(db_path))
    monkeypatch.setenv("ADMIN_USERNAME", "rootuser")
    monkeypatch.setenv("ADMIN_EMAIL", "rootuser@example.com")
    monkeypatch.setenv("ADMIN_PASSWORD", "change-me-12345")

    assert create_admin.main() == 0

    db = PortfolioManager(str(db_path))
    user = db.get_user_by_username("rootuser")
    assert user is not None
    assert user["email"] == "rootuser@example.com"
    assert user["is_admin"] == 1


def test_create_admin_rejects_duplicates(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "admin.db"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(db_path))
    monkeypatch.setenv("ADMIN_USERNAME", "rootuser")
    monkeypatch.setenv("ADMIN_EMAIL", "rootuser@example.com")
    monkeypatch.setenv("ADMIN_PASSWORD", "change-me-12345")

    assert create_admin.main() == 0
    assert create_admin.main() == 1
