#!/usr/bin/env python3
"""
JWT Authentication for AI Day Trader Agent API.
Implements secure OAuth2 password flow with JWT tokens following industry best practices.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import os
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from passlib.context import CryptContext
import jwt
from jwt.exceptions import PyJWTError
from slowapi import Limiter
from slowapi.util import get_remote_address

# Logging
logger = logging.getLogger(__name__)

# Import database manager
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.portfolio_manager import PortfolioManager
from config.api.dependencies import get_portfolio_manager

# Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    # For development/testing only - use a consistent test key
    # In production, this will raise an error
    if os.getenv("ENVIRONMENT", "development") == "development":
        JWT_SECRET_KEY = "dev_secret_key_for_testing_only_change_in_production"
        logger.warning("⚠️  Using development JWT secret. Set JWT_SECRET_KEY env var for production!")
    else:
        raise RuntimeError(
            "JWT_SECRET_KEY environment variable must be set in production. "
            "Generate one with: openssl rand -hex 32"
        )

JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Router
router = APIRouter()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Pydantic models for request/response validation
class Token(BaseModel):
    """OAuth2 compatible token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Token expiration time in seconds")


class TokenRefreshRequest(BaseModel):
    """Refresh token request body."""
    refresh_token: str = Field(..., min_length=1)


class TokenData(BaseModel):
    """Token payload data"""
    username: Optional[str] = None
    user_id: Optional[int] = None
    scopes: list[str] = []


class User(BaseModel):
    """User model"""
    id: int
    username: str
    email: str
    is_active: bool = True
    is_admin: bool = False
    created_at: datetime


class UserInDB(User):
    """User model with hashed password for database storage"""
    hashed_password: str


class UserCreate(BaseModel):
    """User registration model"""
    username: str = Field(..., min_length=3, max_length=50, pattern="^[a-zA-Z0-9_-]+$")
    email: str = Field(..., pattern="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$")
    password: str = Field(..., min_length=8, max_length=100)


class UserLogin(BaseModel):
    """User login model"""
    username: str
    password: str


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)


def get_user(username: str, db: Optional[PortfolioManager] = None) -> Optional[UserInDB]:
    """Get user from database"""
    try:
        manager = db or get_portfolio_manager()
        user_dict = manager.get_user_by_username(username)
        if user_dict:
            return UserInDB(**user_dict)
        return None
    except Exception as e:
        logger.error(f"Error fetching user {username}: {e}")
        return None


def authenticate_user(
    username: str,
    password: str,
    db: Optional[PortfolioManager] = None,
) -> Optional[UserInDB]:
    """Authenticate user with username and password"""
    user = get_user(username, db)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    import uuid
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    # Add unique token ID for blacklist functionality
    jti = str(uuid.uuid4())
    to_encode.update({"exp": expire, "type": "access", "jti": jti})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: PortfolioManager = Depends(get_portfolio_manager),
) -> User:
    """Get current authenticated user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        token_type: str = payload.get("type")
        jti: str = payload.get("jti")

        if username is None or token_type != "access":
            raise credentials_exception

        # Check if token is blacklisted
        if jti:
            is_blacklisted = await run_in_threadpool(db.is_token_blacklisted, jti)
            if is_blacklisted:
                raise credentials_exception

        token_data = TokenData(username=username)
    except PyJWTError:
        raise credentials_exception

    user = await run_in_threadpool(get_user, token_data.username, db)
    if user is None:
        raise credentials_exception

    return User(
        id=user.id,
        username=user.username,
        email=user.email,
        is_active=user.is_active,
        is_admin=user.is_admin,
        created_at=user.created_at
    )


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Ensure current user is active"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_admin_user(current_user: User = Depends(get_current_active_user)) -> User:
    """Ensure current user is admin"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


# API Endpoints
@router.post("/login", response_model=Token)
@limiter.limit("5/minute")  # 5 login attempts per minute per IP
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: PortfolioManager = Depends(get_portfolio_manager),
):
    """
    OAuth2 compatible token login, get an access token for future requests.
    
    Submit OAuth2 username and password form fields to receive access tokens.
    """
    user = await run_in_threadpool(authenticate_user, form_data.username, form_data.password, db)
    if not user:
        logger.warning(f"Failed login attempt for username: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update last login timestamp
    await run_in_threadpool(db.update_last_login, user.username)

    # Create tokens
    access_token_expires = timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    token_data = {"sub": user.username, "user_id": user.id}
    access_token = create_access_token(data=token_data, expires_delta=access_token_expires)
    refresh_token = create_refresh_token(data=token_data)

    logger.info(f"Successful login for user: {user.username}")

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60  # Convert to seconds
    }


@router.post("/refresh", response_model=Token)
@limiter.limit("10/minute")  # 10 refresh attempts per minute per IP
async def refresh_token(
    request: Request,
    token_request: TokenRefreshRequest,
    db: PortfolioManager = Depends(get_portfolio_manager),
):
    """
    Refresh access token using refresh token.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token_request.refresh_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if username is None or token_type != "refresh":
            raise credentials_exception
            
    except PyJWTError:
        raise credentials_exception
    
    user = await run_in_threadpool(get_user, username, db)
    if user is None:
        raise credentials_exception
    
    # Create new access token
    access_token_expires = timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    token_data = {"sub": user.username, "user_id": user.id}
    new_access_token = create_access_token(data=token_data, expires_delta=access_token_expires)
    
    return {
        "access_token": new_access_token,
        "refresh_token": token_request.refresh_token,  # Return same refresh token
        "token_type": "bearer",
        "expires_in": JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }


@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Get current user information.
    """
    return current_user


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
@limiter.limit("3/hour")  # 3 registrations per hour per IP
async def register(
    request: Request,
    user_data: UserCreate,
    db: PortfolioManager = Depends(get_portfolio_manager),
):
    """
    Register a new user.
    
    Requirements:
    - Username: 3-50 characters, alphanumeric with _ and -
    - Email: Valid email format
    - Password: Minimum 8 characters
    """
    try:
        # Hash the password
        hashed_password = await run_in_threadpool(get_password_hash, user_data.password)

        # Create user in database
        user_dict = await run_in_threadpool(
            db.create_user,
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password,
            is_admin=False
        )

        logger.info(f"New user registered: {user_data.username}")

        return User(
            id=user_dict['id'],
            username=user_dict['username'],
            email=user_dict['email'],
            is_active=bool(user_dict['is_active']),
            is_admin=bool(user_dict['is_admin']),
            created_at=datetime.fromisoformat(user_dict['created_at'])
        )

    except ValueError as e:
        # Handle duplicate username/email
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/logout")
async def logout(
    token: str = Depends(oauth2_scheme),
    current_user: User = Depends(get_current_active_user),
    db: PortfolioManager = Depends(get_portfolio_manager),
):
    """
    Logout current user by blacklisting the JWT token.

    This properly invalidates the token so it cannot be used again.
    """
    try:
        # Decode token to get JTI and expiration
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        jti = payload.get("jti")
        exp = payload.get("exp")

        if jti and exp:
            expires_at = datetime.fromtimestamp(exp, tz=timezone.utc)

            # Add token to blacklist
            await run_in_threadpool(db.blacklist_token, jti, current_user.username, expires_at)

            logger.info(f"User logged out and token blacklisted: {current_user.username}")
            return {"message": "Successfully logged out"}
        else:
            logger.warning(f"Token missing JTI for logout: {current_user.username}")
            return {"message": "Logged out (token format not supported for blacklist)"}

    except Exception as e:
        logger.error(f"Logout error: {e}")
        # Still return success to user, just log the error
        return {"message": "Successfully logged out"}
