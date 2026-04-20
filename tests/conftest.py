from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.portfolio_manager import PortfolioManager


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    return tmp_path / "portfolios.db"


@pytest.fixture
def portfolio_manager(test_db_path: Path) -> PortfolioManager:
    manager = PortfolioManager(str(test_db_path))
    manager.create_user(
        username="api-user",
        email="api-user@example.com",
        hashed_password="test-password-hash",
    )
    manager.create_user(
        username="other-user",
        email="other-user@example.com",
        hashed_password="test-password-hash",
    )
    return manager
