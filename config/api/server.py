#!/usr/bin/env python3
"""
Main FastAPI server for AI Day Trader Agent API Layer.
Implements secure, standards-compliant REST endpoints for portfolio management and trading.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request
import logging

from config.api.auth import router as auth_router
from config.api.portfolios import router as portfolios_router
from config.api.analysis import router as analysis_router
from config.api.websockets import websocket_endpoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_day_trader_api")

app = FastAPI(
    title="AI Day Trader Agent API",
    description="Secure REST API for portfolio management, trading, and analysis.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to trusted domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(portfolios_router, prefix="/api/portfolios", tags=["portfolios"])
app.include_router(analysis_router, prefix="/api/analysis", tags=["analysis"])

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please contact support."}
    )

@app.get("/api/health", tags=["system"])
async def health_check():
    return {"status": "ok", "message": "API is running."}

# WebSocket endpoint
app.websocket("/ws")(websocket_endpoint)
