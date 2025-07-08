#!/usr/bin/env python3
"""
WebSocket implementation for real-time updates in AI Day Trader Agent.
Provides real-time portfolio updates, trade notifications, and analysis results.
"""

from typing import Dict, Set, Optional
from datetime import datetime
import json
import asyncio
import logging

from fastapi import WebSocket, WebSocketDisconnect, Depends, status
from fastapi.websockets import WebSocketState
import jwt
from jwt.exceptions import PyJWTError

from config.api.auth import JWT_SECRET_KEY, JWT_ALGORITHM, get_user

# Logging
logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        # Store active connections by user ID
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Store portfolio subscriptions
        self.portfolio_subscriptions: Dict[str, Set[str]] = {}  # portfolio_name -> set of user_ids
        
    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        
        # Add to active connections
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
        
        logger.info(f"WebSocket connected for user: {user_id}")
        
        # Send welcome message
        await self.send_personal_message(
            {
                "type": "connection",
                "status": "connected",
                "message": "Welcome to AI Day Trader real-time updates",
                "timestamp": datetime.utcnow().isoformat()
            },
            websocket
        )
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        """Remove a WebSocket connection."""
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                
        # Remove from portfolio subscriptions
        for portfolio_name, subscribers in list(self.portfolio_subscriptions.items()):
            if user_id in subscribers:
                subscribers.discard(user_id)
                if not subscribers:
                    del self.portfolio_subscriptions[portfolio_name]
                    
        logger.info(f"WebSocket disconnected for user: {user_id}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message: {e}")
    
    async def send_user_message(self, message: dict, user_id: str):
        """Send a message to all connections for a specific user."""
        if user_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[user_id]:
                try:
                    await self.send_personal_message(message, connection)
                except:
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                self.active_connections[user_id].discard(conn)
    
    async def broadcast_portfolio_update(self, portfolio_name: str, update: dict):
        """Broadcast portfolio updates to all subscribers."""
        if portfolio_name in self.portfolio_subscriptions:
            message = {
                "type": "portfolio_update",
                "portfolio_name": portfolio_name,
                "data": update,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send to all subscribed users
            for user_id in self.portfolio_subscriptions[portfolio_name]:
                await self.send_user_message(message, user_id)
    
    async def broadcast_trade_notification(self, portfolio_name: str, trade: dict):
        """Broadcast trade notifications to portfolio subscribers."""
        if portfolio_name in self.portfolio_subscriptions:
            message = {
                "type": "trade_notification",
                "portfolio_name": portfolio_name,
                "trade": trade,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            for user_id in self.portfolio_subscriptions[portfolio_name]:
                await self.send_user_message(message, user_id)
    
    async def broadcast_analysis_update(self, user_id: str, analysis: dict):
        """Send analysis updates to a specific user."""
        message = {
            "type": "analysis_update",
            "data": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.send_user_message(message, user_id)
    
    def subscribe_to_portfolio(self, user_id: str, portfolio_name: str):
        """Subscribe a user to portfolio updates."""
        if portfolio_name not in self.portfolio_subscriptions:
            self.portfolio_subscriptions[portfolio_name] = set()
        self.portfolio_subscriptions[portfolio_name].add(user_id)
        logger.info(f"User {user_id} subscribed to portfolio: {portfolio_name}")
    
    def unsubscribe_from_portfolio(self, user_id: str, portfolio_name: str):
        """Unsubscribe a user from portfolio updates."""
        if portfolio_name in self.portfolio_subscriptions:
            self.portfolio_subscriptions[portfolio_name].discard(user_id)
            if not self.portfolio_subscriptions[portfolio_name]:
                del self.portfolio_subscriptions[portfolio_name]
        logger.info(f"User {user_id} unsubscribed from portfolio: {portfolio_name}")


# Global connection manager instance
manager = ConnectionManager()


async def get_current_user_ws(websocket: WebSocket, token: Optional[str] = None) -> Optional[str]:
    """Authenticate WebSocket connection using JWT token."""
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return None
    
    try:
        # Decode JWT token
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        
        if username is None:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return None
            
        # Verify user exists
        user = get_user(username)
        if not user:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return None
            
        return username
        
    except PyJWTError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return None


async def websocket_endpoint(websocket: WebSocket, token: str):
    """
    WebSocket endpoint for real-time updates.
    
    Clients should connect with their JWT token as a query parameter:
    ws://localhost:8000/ws?token=<jwt_token>
    """
    # Authenticate user
    user_id = await get_current_user_ws(websocket, token)
    if not user_id:
        return
    
    # Connect
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            
            # Handle different message types
            message_type = data.get("type")
            
            if message_type == "subscribe_portfolio":
                portfolio_name = data.get("portfolio_name")
                if portfolio_name:
                    manager.subscribe_to_portfolio(user_id, portfolio_name)
                    await manager.send_personal_message(
                        {
                            "type": "subscription_confirmed",
                            "portfolio_name": portfolio_name,
                            "timestamp": datetime.utcnow().isoformat()
                        },
                        websocket
                    )
            
            elif message_type == "unsubscribe_portfolio":
                portfolio_name = data.get("portfolio_name")
                if portfolio_name:
                    manager.unsubscribe_from_portfolio(user_id, portfolio_name)
                    await manager.send_personal_message(
                        {
                            "type": "unsubscription_confirmed",
                            "portfolio_name": portfolio_name,
                            "timestamp": datetime.utcnow().isoformat()
                        },
                        websocket
                    )
            
            elif message_type == "ping":
                # Respond to ping with pong
                await manager.send_personal_message(
                    {
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    websocket
                )
            
            else:
                # Unknown message type
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": f"Unknown message type: {message_type}",
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        manager.disconnect(websocket, user_id)
        await websocket.close()


# Example usage functions for sending updates from other parts of the application
async def notify_portfolio_update(portfolio_name: str, update_data: dict):
    """Send portfolio update notification to all subscribers."""
    await manager.broadcast_portfolio_update(portfolio_name, update_data)


async def notify_trade_execution(portfolio_name: str, trade_data: dict):
    """Send trade execution notification to all subscribers."""
    await manager.broadcast_trade_notification(portfolio_name, trade_data)


async def notify_analysis_complete(user_id: str, analysis_data: dict):
    """Send analysis completion notification to user."""
    await manager.broadcast_analysis_update(user_id, analysis_data)
