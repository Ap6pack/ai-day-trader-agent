from __future__ import annotations

import pytest
from fastapi import HTTPException

from config.api.auth import User
from config.api.portfolios import (
    HoldingCreate,
    PortfolioCreate,
    TradeCreate,
    add_or_update_holding,
    create_portfolio,
    delete_portfolio,
    get_holdings,
    get_portfolio,
    get_performance,
    list_portfolios,
    record_trade,
    remove_holding,
    update_portfolio,
    PortfolioUpdate,
)
from core.portfolio_manager import PortfolioManager


@pytest.fixture
def current_user() -> User:
    return User(
        id=1,
        username="api-user",
        email="api-user@example.com",
        is_active=True,
        is_admin=False,
        created_at="2026-01-01T00:00:00",
    )


@pytest.fixture
def other_user() -> User:
    return User(
        id=2,
        username="other-user",
        email="other-user@example.com",
        is_active=True,
        is_admin=False,
        created_at="2026-01-01T00:00:00",
    )


@pytest.mark.asyncio
async def test_portfolio_routes_use_injected_database(
    portfolio_manager: PortfolioManager,
    current_user: User,
) -> None:
    created = await create_portfolio(
        portfolio_data=PortfolioCreate(name="main", trading_capital=10_000),
        current_user=current_user,
        db=portfolio_manager,
    )

    assert created["name"] == "main"
    assert created["cash_available"] == 10_000

    fetched = await get_portfolio(
        portfolio_name="main",
        current_user=current_user,
        db=portfolio_manager,
    )
    assert fetched["total_value"] == 10_000

    holding = await add_or_update_holding(
        portfolio_name="main",
        holding_data=HoldingCreate(symbol="aapl", quantity=3, avg_cost=150),
        current_user=current_user,
        db=portfolio_manager,
    )
    assert holding["symbol"] == "AAPL"
    assert holding["market_value"] == 450

    holdings = await get_holdings(
        portfolio_name="main",
        current_user=current_user,
        db=portfolio_manager,
    )
    assert [item["symbol"] for item in holdings] == ["AAPL"]


@pytest.mark.asyncio
async def test_record_trade_updates_holding(
    portfolio_manager: PortfolioManager,
    current_user: User,
) -> None:
    await create_portfolio(
        portfolio_data=PortfolioCreate(name="trades", trading_capital=5_000),
        current_user=current_user,
        db=portfolio_manager,
    )

    trade = await record_trade(
        portfolio_name="trades",
        trade_data=TradeCreate(
            symbol="msft",
            action="buy",
            quantity=2,
            price=300,
            strategy="technical",
            confidence=0.8,
        ),
        current_user=current_user,
        db=portfolio_manager,
    )

    assert trade["id"] > 0
    assert trade["symbol"] == "MSFT"
    assert trade["action"] == "BUY"

    holdings = portfolio_manager.get_holdings("trades")
    assert holdings == [
        {
            "symbol": "MSFT",
            "quantity": 2,
            "avg_cost": 300.0,
            "last_updated": holdings[0]["last_updated"],
        }
    ]


@pytest.mark.asyncio
async def test_portfolio_update_delete_and_holding_remove(
    portfolio_manager: PortfolioManager,
    current_user: User,
) -> None:
    await create_portfolio(
        portfolio_data=PortfolioCreate(name="crud", trading_capital=1_000),
        current_user=current_user,
        db=portfolio_manager,
    )

    updated = await update_portfolio(
        portfolio_name="crud",
        portfolio_update=PortfolioUpdate(trading_capital=2_500),
        current_user=current_user,
        db=portfolio_manager,
    )
    assert updated["trading_capital"] == 2_500

    await add_or_update_holding(
        portfolio_name="crud",
        holding_data=HoldingCreate(symbol="aapl", quantity=5, avg_cost=100),
        current_user=current_user,
        db=portfolio_manager,
    )
    await remove_holding(
        portfolio_name="crud",
        symbol="AAPL",
        current_user=current_user,
        db=portfolio_manager,
    )
    assert portfolio_manager.get_holdings("crud") == []

    await record_trade(
        portfolio_name="crud",
        trade_data=TradeCreate(symbol="msft", action="buy", quantity=1, price=300),
        current_user=current_user,
        db=portfolio_manager,
    )
    assert portfolio_manager.get_trade_history("crud", 1)

    await delete_portfolio(
        portfolio_name="crud",
        current_user=current_user,
        db=portfolio_manager,
    )
    assert portfolio_manager.get_portfolio("crud") is None


@pytest.mark.asyncio
async def test_performance_response_includes_required_fields_without_trades(
    portfolio_manager: PortfolioManager,
    current_user: User,
) -> None:
    await create_portfolio(
        portfolio_data=PortfolioCreate(name="perf", trading_capital=1_000),
        current_user=current_user,
        db=portfolio_manager,
    )
    await add_or_update_holding(
        portfolio_name="perf",
        holding_data=HoldingCreate(symbol="AAPL", quantity=5, avg_cost=100),
        current_user=current_user,
        db=portfolio_manager,
    )

    metrics = await get_performance(
        portfolio_name="perf",
        current_user=current_user,
        db=portfolio_manager,
    )

    assert metrics["total_trades"] == 0
    assert metrics["buy_trades"] == 0
    assert metrics["sell_trades"] == 0
    assert metrics["total_fees"] == 0


@pytest.mark.asyncio
async def test_portfolios_are_scoped_to_current_user(
    portfolio_manager: PortfolioManager,
    current_user: User,
    other_user: User,
) -> None:
    await create_portfolio(
        portfolio_data=PortfolioCreate(name="private", trading_capital=1_000),
        current_user=current_user,
        db=portfolio_manager,
    )

    owner_portfolios = await list_portfolios(
        current_user=current_user,
        db=portfolio_manager,
    )
    other_portfolios = await list_portfolios(
        current_user=other_user,
        db=portfolio_manager,
    )

    assert [portfolio["name"] for portfolio in owner_portfolios] == ["private"]
    assert other_portfolios == []

    with pytest.raises(HTTPException) as exc_info:
        await get_portfolio(
            portfolio_name="private",
            current_user=other_user,
            db=portfolio_manager,
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_cross_user_portfolio_mutations_are_denied(
    portfolio_manager: PortfolioManager,
    current_user: User,
    other_user: User,
) -> None:
    await create_portfolio(
        portfolio_data=PortfolioCreate(name="protected", trading_capital=1_000),
        current_user=current_user,
        db=portfolio_manager,
    )

    blocked_calls = [
        update_portfolio(
            portfolio_name="protected",
            portfolio_update=PortfolioUpdate(trading_capital=2_000),
            current_user=other_user,
            db=portfolio_manager,
        ),
        add_or_update_holding(
            portfolio_name="protected",
            holding_data=HoldingCreate(symbol="AAPL", quantity=1, avg_cost=100),
            current_user=other_user,
            db=portfolio_manager,
        ),
        record_trade(
            portfolio_name="protected",
            trade_data=TradeCreate(symbol="MSFT", action="BUY", quantity=1, price=300),
            current_user=other_user,
            db=portfolio_manager,
        ),
        delete_portfolio(
            portfolio_name="protected",
            current_user=other_user,
            db=portfolio_manager,
        ),
    ]

    for call in blocked_calls:
        with pytest.raises(HTTPException) as exc_info:
            await call
        assert exc_info.value.status_code == 404

    assert portfolio_manager.get_portfolio("protected", user_id=current_user.id) is not None


@pytest.mark.asyncio
async def test_different_users_can_use_same_portfolio_name(
    portfolio_manager: PortfolioManager,
    current_user: User,
    other_user: User,
) -> None:
    owner_portfolio = await create_portfolio(
        portfolio_data=PortfolioCreate(name="main", trading_capital=1_000),
        current_user=current_user,
        db=portfolio_manager,
    )
    other_portfolio = await create_portfolio(
        portfolio_data=PortfolioCreate(name="main", trading_capital=2_000),
        current_user=other_user,
        db=portfolio_manager,
    )

    assert owner_portfolio["id"] != other_portfolio["id"]
    assert owner_portfolio["trading_capital"] == 1_000
    assert other_portfolio["trading_capital"] == 2_000

    owner_fetched = await get_portfolio(
        portfolio_name="main",
        current_user=current_user,
        db=portfolio_manager,
    )
    other_fetched = await get_portfolio(
        portfolio_name="main",
        current_user=other_user,
        db=portfolio_manager,
    )

    assert owner_fetched["id"] == owner_portfolio["id"]
    assert other_fetched["id"] == other_portfolio["id"]
