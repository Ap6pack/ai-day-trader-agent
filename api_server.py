#!/usr/bin/env python3
"""
Startup script for AI Day Trader Agent API Server.
Run this to start the FastAPI server with proper configuration.
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the FastAPI app
from config.api.server import app

def main():
    """Start the API server with production-ready settings."""
    
    # Get configuration from environment or use defaults
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    workers = int(os.getenv("API_WORKERS", "1"))
    log_level = os.getenv("API_LOG_LEVEL", "info").lower()
    
    # Security warning for JWT secret
    jwt_secret = os.getenv("JWT_SECRET_KEY")
    if not jwt_secret or jwt_secret == "CHANGE_ME_IN_PRODUCTION_USE_STRONG_SECRET":
        print("\n⚠️  WARNING: Using default JWT secret key!")
        print("   Set JWT_SECRET_KEY environment variable for production use.")
        print("   Generate a secure key with: openssl rand -hex 32\n")
    
    print(f"""
🚀 Starting AI Day Trader Agent API Server
==========================================
Host: {host}
Port: {port}
Workers: {workers}
Reload: {reload}
Log Level: {log_level}

📚 API Documentation: http://localhost:{port}/docs
📊 ReDoc: http://localhost:{port}/redoc

Default credentials:
- Username: admin
- Password: secret
""")
    
    # Run the server
    if workers > 1 and not reload:
        # Production mode with multiple workers
        uvicorn.run(
            "config.api.server:app",
            host=host,
            port=port,
            workers=workers,
            log_level=log_level,
            access_log=True
        )
    else:
        # Development mode or single worker
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=True
        )

if __name__ == "__main__":
    main()
