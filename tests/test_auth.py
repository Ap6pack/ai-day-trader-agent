from __future__ import annotations

from datetime import datetime, timezone

import jwt
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from config.api import auth
from core.portfolio_manager import PortfolioManager


def _create_user(db: PortfolioManager, username: str = "alice") -> dict:
    return db.create_user(
        username=username,
        email=f"{username}@example.com",
        hashed_password=auth.get_password_hash("correct-horse-battery-staple"),
        is_admin=False,
    )


def test_authenticate_user_uses_database(portfolio_manager: PortfolioManager) -> None:
    _create_user(portfolio_manager)

    assert auth.authenticate_user(
        "alice",
        "correct-horse-battery-staple",
        portfolio_manager,
    )
    assert auth.authenticate_user("alice", "wrong-password", portfolio_manager) is None
    assert auth.authenticate_user("missing", "correct-horse-battery-staple", portfolio_manager) is None


@pytest.mark.asyncio
async def test_get_current_user_rejects_blacklisted_token(
    portfolio_manager: PortfolioManager,
) -> None:
    user = _create_user(portfolio_manager)
    token = auth.create_access_token({"sub": user["username"], "user_id": user["id"]})

    current_user = await auth.get_current_user(token=token, db=portfolio_manager)
    assert current_user.username == "alice"

    payload = jwt.decode(token, auth.JWT_SECRET_KEY, algorithms=[auth.JWT_ALGORITHM])
    portfolio_manager.blacklist_token(
        payload["jti"],
        user["username"],
        datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
    )

    with pytest.raises(HTTPException) as exc_info:
        await auth.get_current_user(token=token, db=portfolio_manager)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_refresh_token_accepts_json_body_contract(
    portfolio_manager: PortfolioManager,
) -> None:
    user = _create_user(portfolio_manager)
    refresh = auth.create_refresh_token({"sub": user["username"], "user_id": user["id"]})
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/auth/refresh",
            "headers": [],
            "client": ("127.0.0.1", 12345),
        }
    )

    response = await auth.refresh_token(
        request=request,
        token_request=auth.TokenRefreshRequest(refresh_token=refresh),
        db=portfolio_manager,
    )

    assert response["refresh_token"] == refresh
    assert response["token_type"] == "bearer"
    payload = jwt.decode(response["access_token"], auth.JWT_SECRET_KEY, algorithms=[auth.JWT_ALGORITHM])
    assert payload["sub"] == "alice"
