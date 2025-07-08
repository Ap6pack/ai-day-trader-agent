# AI Day Trader Agent API Documentation

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Quick Start](#quick-start)
- [Authentication](#authentication)
  - [Login](#login)
  - [Register New User](#register-new-user)
- [API Endpoints](#api-endpoints)
  - [Authentication Endpoints](#authentication-endpoints)
  - [Portfolio Endpoints](#portfolio-endpoints)
  - [Holdings Endpoints](#holdings-endpoints)
  - [Trade Endpoints](#trade-endpoints)
  - [Performance Endpoints](#performance-endpoints)
  - [Analysis Endpoints](#analysis-endpoints)
  - [System Endpoints](#system-endpoints)
- [WebSocket Endpoint](#websocket-endpoint)
  - [WebSocket Messages](#websocket-messages)
- [Example Usage](#example-usage)
  - [1. Create a Portfolio](#1-create-a-portfolio)
  - [2. Add Holdings](#2-add-holdings)
  - [3. Run Analysis](#3-run-analysis)
  - [4. Record a Trade](#4-record-a-trade)
  - [5. Get Portfolio Performance](#5-get-portfolio-performance)
- [Error Responses](#error-responses)
- [Rate Limiting](#rate-limiting)
- [Security Considerations](#security-considerations)
- [Development](#development)
  - [Running the API Server](#running-the-api-server)
  - [Environment Variables](#environment-variables)
- [Testing the API](#testing-the-api)
  - [Using curl](#using-curl)
  - [Using Python](#using-python)
  - [Using JavaScript](#using-javascript)
- [WebSocket Client Example](#websocket-client-example)
- [Support](#support)

## Overview

The AI Day Trader Agent API provides secure REST endpoints for portfolio management, trading analysis, and real-time updates. The API uses JWT authentication and follows RESTful design principles.

## Base URL

```
http://localhost:8000
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Add to your `.env` file:
```bash
# Generate a secure JWT secret
JWT_SECRET_KEY=$(openssl rand -hex 32)

# Optional API configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_LOG_LEVEL=info
```

### 3. Start the API Server
```bash
# Production mode
python api_server.py

# Development mode with auto-reload
API_RELOAD=true python api_server.py

# Custom configuration
API_HOST=0.0.0.0 API_PORT=8080 API_WORKERS=4 python api_server.py
```

### 4. Access the API
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/health

### 5. Quick Authentication
```bash
# Get access token
TOKEN=$(curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=secret" \
  | jq -r '.access_token')

# Use token in requests
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/portfolios
```

### Default Credentials
- **Username**: admin
- **Password**: secret

⚠️ **Security Warning**: Change default credentials and JWT secret in production!

## Authentication

The API uses JWT (JSON Web Token) authentication. To access protected endpoints:

1. Login to get an access token
2. Include the token in the Authorization header: `Bearer <token>`

### Login

```http
POST /api/auth/login
Content-Type: application/x-www-form-urlencoded

username=admin&password=secret
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Register New User

```http
POST /api/auth/register
Content-Type: application/json

{
  "username": "newuser",
  "email": "user@example.com",
  "password": "securepassword123"
}
```

## API Endpoints

### Authentication Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/login` | Login and get JWT tokens |
| POST | `/api/auth/refresh` | Refresh access token |
| POST | `/api/auth/register` | Register new user |
| GET | `/api/auth/me` | Get current user info |
| POST | `/api/auth/logout` | Logout (client-side) |

### Portfolio Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/portfolios` | List all portfolios |
| POST | `/api/portfolios` | Create new portfolio |
| GET | `/api/portfolios/{name}` | Get portfolio details |
| PUT | `/api/portfolios/{name}` | Update portfolio |
| DELETE | `/api/portfolios/{name}` | Delete portfolio (not implemented) |

### Holdings Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/portfolios/{name}/holdings` | Get all holdings |
| POST | `/api/portfolios/{name}/holdings` | Add/update holding |
| DELETE | `/api/portfolios/{name}/holdings/{symbol}` | Remove holding |

### Trade Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/portfolios/{name}/trades` | Get trade history |
| POST | `/api/portfolios/{name}/trades` | Record new trade |

### Performance Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/portfolios/{name}/performance` | Get performance metrics |

### Analysis Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analysis/symbol` | Analyze single symbol |
| POST | `/api/analysis/batch` | Analyze multiple symbols |
| POST | `/api/analysis/portfolio/{name}` | Start portfolio analysis job |
| GET | `/api/analysis/portfolio/status/{job_id}` | Check analysis job status |
| GET | `/api/analysis/portfolio/result/{job_id}` | Get analysis results |
| DELETE | `/api/analysis/portfolio/job/{job_id}` | Delete completed job |

### System Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/docs` | Interactive API documentation |
| GET | `/redoc` | Alternative API documentation |

## WebSocket Endpoint

Connect to real-time updates:

```
ws://localhost:8000/ws?token=<jwt_token>
```

### WebSocket Messages

#### Client to Server

Subscribe to portfolio updates:
```json
{
  "type": "subscribe_portfolio",
  "portfolio_name": "default"
}
```

Unsubscribe from portfolio:
```json
{
  "type": "unsubscribe_portfolio",
  "portfolio_name": "default"
}
```

Ping (keep-alive):
```json
{
  "type": "ping"
}
```

#### Server to Client

Portfolio update:
```json
{
  "type": "portfolio_update",
  "portfolio_name": "default",
  "data": { ... },
  "timestamp": "2025-01-08T20:00:00Z"
}
```

Trade notification:
```json
{
  "type": "trade_notification",
  "portfolio_name": "default",
  "trade": { ... },
  "timestamp": "2025-01-08T20:00:00Z"
}
```

Analysis update:
```json
{
  "type": "analysis_update",
  "data": { ... },
  "timestamp": "2025-01-08T20:00:00Z"
}
```

## Example Usage

### 1. Create a Portfolio

```bash
# Login first
TOKEN=$(curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=secret" \
  | jq -r '.access_token')

# Create portfolio
curl -X POST http://localhost:8000/api/portfolios \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_portfolio",
    "trading_capital": 10000,
    "description": "My trading portfolio"
  }'
```

### 2. Add Holdings

```bash
curl -X POST http://localhost:8000/api/portfolios/my_portfolio/holdings \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "quantity": 100,
    "avg_cost": 150.50
  }'
```

### 3. Run Analysis

```bash
curl -X POST http://localhost:8000/api/analysis/symbol \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "portfolio_name": "my_portfolio"
  }'
```

### 4. Record a Trade

```bash
curl -X POST http://localhost:8000/api/portfolios/my_portfolio/trades \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "action": "BUY",
    "quantity": 50,
    "price": 155.25,
    "strategy": "technical",
    "confidence": 0.85,
    "notes": "Strong technical signals"
  }'
```

### 5. Get Portfolio Performance

```bash
curl -X GET http://localhost:8000/api/portfolios/my_portfolio/performance \
  -H "Authorization: Bearer $TOKEN"
```

## Error Responses

The API returns standard HTTP status codes:

- `200 OK` - Success
- `201 Created` - Resource created
- `204 No Content` - Success with no response body
- `400 Bad Request` - Invalid request
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Access denied
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

Error response format:
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Rate Limiting

Currently, there are no rate limits implemented. In production, consider adding rate limiting to prevent abuse.

## Security Considerations

1. **JWT Secret**: Change the default JWT secret key in production
2. **CORS**: Configure CORS to only allow trusted domains
3. **HTTPS**: Use HTTPS in production
4. **Input Validation**: All inputs are validated using Pydantic models
5. **SQL Injection**: Protected through parameterized queries
6. **Password Security**: Passwords are hashed using bcrypt

## Development

### Running the API Server

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python api_server.py

# With auto-reload for development
API_RELOAD=true python api_server.py
```

### Environment Variables

```bash
# API Server
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_LOG_LEVEL=info

# Security
JWT_SECRET_KEY=your-secret-key-here

# Generate a secure key
openssl rand -hex 32
```

## Testing the API

### Using curl

```bash
# Health check
curl http://localhost:8000/api/health

# Login
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=secret"
```

### Using Python

```python
import requests

# Login
response = requests.post(
    "http://localhost:8000/api/auth/login",
    data={"username": "admin", "password": "secret"}
)
token = response.json()["access_token"]

# Use token for authenticated requests
headers = {"Authorization": f"Bearer {token}"}
portfolios = requests.get(
    "http://localhost:8000/api/portfolios",
    headers=headers
).json()
```

### Using JavaScript

```javascript
// Login
const loginResponse = await fetch('http://localhost:8000/api/auth/login', {
  method: 'POST',
  headers: {'Content-Type': 'application/x-www-form-urlencoded'},
  body: 'username=admin&password=secret'
});
const { access_token } = await loginResponse.json();

// Use token
const portfolios = await fetch('http://localhost:8000/api/portfolios', {
  headers: {'Authorization': `Bearer ${access_token}`}
}).then(r => r.json());
```

## WebSocket Client Example

```javascript
// Connect to WebSocket
const ws = new WebSocket(`ws://localhost:8000/ws?token=${access_token}`);

ws.onopen = () => {
  console.log('Connected to WebSocket');
  
  // Subscribe to portfolio updates
  ws.send(JSON.stringify({
    type: 'subscribe_portfolio',
    portfolio_name: 'default'
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Received:', message);
  
  switch(message.type) {
    case 'portfolio_update':
      // Handle portfolio update
      break;
    case 'trade_notification':
      // Handle trade notification
      break;
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
```

## Support

For issues or questions:
- Check the interactive documentation at `/docs`
- Review error messages in API responses
- Check server logs for detailed error information
