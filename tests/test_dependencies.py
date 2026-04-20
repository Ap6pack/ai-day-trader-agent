from __future__ import annotations

from config.api.dependencies import (
    clear_portfolio_manager_cache,
    get_portfolio_manager,
)
from core.pipeline import EnhancedTradingPipeline


def test_portfolio_manager_dependency_is_cached_per_database_path(monkeypatch, tmp_path):
    clear_portfolio_manager_cache()
    db_path = tmp_path / "cached.db"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(db_path))

    first = get_portfolio_manager()
    second = get_portfolio_manager()

    assert first is second
    assert first.db_path == str(db_path)

    clear_portfolio_manager_cache()


def test_portfolio_manager_dependency_uses_new_cache_entry_for_new_path(monkeypatch, tmp_path):
    clear_portfolio_manager_cache()
    first_db = tmp_path / "first.db"
    second_db = tmp_path / "second.db"

    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(first_db))
    first = get_portfolio_manager()

    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(second_db))
    second = get_portfolio_manager()

    assert first is not second
    assert first.db_path == str(first_db)
    assert second.db_path == str(second_db)

    clear_portfolio_manager_cache()


def test_pipeline_reuses_cached_portfolio_manager(monkeypatch, tmp_path):
    clear_portfolio_manager_cache()
    db_path = tmp_path / "pipeline.db"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(db_path))

    first = EnhancedTradingPipeline("FIS", "missing")
    second = EnhancedTradingPipeline("AAPL", "missing")

    assert first.portfolio_manager is second.portfolio_manager
    assert first.portfolio_manager is get_portfolio_manager()

    clear_portfolio_manager_cache()
