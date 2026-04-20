/**
 * AI Day Trader - API Client
 * Handles all REST calls, JWT token management, and auto-refresh.
 */
const API = (() => {
  const BASE = '/api';

  /* ── Token helpers ── */
  function getAccessToken()  { return localStorage.getItem('adt_access_token'); }
  function getRefreshToken() { return localStorage.getItem('adt_refresh_token'); }

  function setTokens(access, refresh) {
    localStorage.setItem('adt_access_token', access);
    if (refresh) localStorage.setItem('adt_refresh_token', refresh);
  }

  function clearTokens() {
    localStorage.removeItem('adt_access_token');
    localStorage.removeItem('adt_refresh_token');
  }

  function isLoggedIn() { return !!getAccessToken(); }

  /* ── Error class ── */
  class APIError extends Error {
    constructor(status, message) {
      super(message);
      this.status = status;
      this.name = 'APIError';
    }
  }

  /* ── Token refresh ── */
  let refreshPromise = null;

  async function refreshAccessToken() {
    if (refreshPromise) return refreshPromise;

    const rt = getRefreshToken();
    if (!rt) return false;

    refreshPromise = (async () => {
      try {
        const resp = await fetch(`${BASE}/auth/refresh`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: rt }),
        });
        if (!resp.ok) { clearTokens(); return false; }
        const data = await resp.json();
        setTokens(data.access_token, data.refresh_token);
        return true;
      } catch {
        clearTokens();
        return false;
      } finally {
        refreshPromise = null;
      }
    })();

    return refreshPromise;
  }

  /* ── Generic request ── */
  async function request(method, path, body, opts = {}) {
    const headers = {};
    const token = getAccessToken();
    if (token && !opts.noAuth) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const fetchOpts = { method, headers };

    if (body !== undefined && body !== null) {
      if (opts.formEncoded) {
        headers['Content-Type'] = 'application/x-www-form-urlencoded';
        fetchOpts.body = body;
      } else {
        headers['Content-Type'] = 'application/json';
        fetchOpts.body = JSON.stringify(body);
      }
    }

    let resp = await fetch(`${BASE}${path}`, fetchOpts);

    // Auto-refresh on 401
    if (resp.status === 401 && !opts.noAuth && !opts._isRetry) {
      const ok = await refreshAccessToken();
      if (ok) {
        headers['Authorization'] = `Bearer ${getAccessToken()}`;
        resp = await fetch(`${BASE}${path}`, { method, headers, body: fetchOpts.body });
      }
    }

    if (!resp.ok) {
      let detail = resp.statusText;
      try { detail = (await resp.json()).detail || detail; } catch {}
      throw new APIError(resp.status, detail);
    }

    if (resp.status === 204) return null;
    return resp.json();
  }

  /* ── Public API ── */
  return {
    APIError,
    isLoggedIn,
    clearTokens,

    /* Auth */
    async login(username, password) {
      const body = new URLSearchParams({ username, password });
      const resp = await fetch(`${BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body,
      });
      if (!resp.ok) {
        let detail = 'Login failed';
        try { detail = (await resp.json()).detail || detail; } catch {}
        throw new APIError(resp.status, detail);
      }
      const data = await resp.json();
      setTokens(data.access_token, data.refresh_token);
      return data;
    },

    register(username, email, password) {
      return request('POST', '/auth/register', { username, email, password }, { noAuth: true });
    },

    getMe() { return request('GET', '/auth/me'); },

    async logout() {
      try { await request('POST', '/auth/logout'); } catch {}
      clearTokens();
    },

    async healthCheck() {
      try { return (await fetch(`${BASE}/health`)).ok; }
      catch { return false; }
    },

    /* Portfolios */
    getPortfolios()    { return request('GET', '/portfolios/'); },
    getPortfolio(name) { return request('GET', `/portfolios/${encodeURIComponent(name)}`); },

    createPortfolio(name, capital, desc) {
      return request('POST', '/portfolios/', {
        name,
        trading_capital: capital,
        description: desc || null,
      });
    },

    updatePortfolio(name, capital, desc) {
      return request('PUT', `/portfolios/${encodeURIComponent(name)}`, {
        trading_capital: capital,
        description: desc || null,
      });
    },

    deletePortfolio(name) {
      return request('DELETE', `/portfolios/${encodeURIComponent(name)}`);
    },

    getHoldings(name) {
      return request('GET', `/portfolios/${encodeURIComponent(name)}/holdings`);
    },

    saveHolding(portfolioName, symbol, quantity, avgCost) {
      return request('POST', `/portfolios/${encodeURIComponent(portfolioName)}/holdings`, {
        symbol,
        quantity,
        avg_cost: avgCost,
      });
    },

    removeHolding(portfolioName, symbol) {
      return request('DELETE', `/portfolios/${encodeURIComponent(portfolioName)}/holdings/${encodeURIComponent(symbol)}`);
    },

    getTradeHistory(name, days = 30) {
      return request('GET', `/portfolios/${encodeURIComponent(name)}/trades?days=${days}`);
    },

    recordTrade(portfolioName, trade) {
      return request('POST', `/portfolios/${encodeURIComponent(portfolioName)}/trades`, trade);
    },

    getPortfolioSummary(name) {
      return request('GET', `/portfolios/${encodeURIComponent(name)}/performance`);
    },

    /* Trading */
    getProviderStatus() { return request('GET', '/trading/provider-status'); },
    getAlpacaAccount()  { return request('GET', '/trading/alpaca/account'); },

    submitPaperOrder(symbol, action, qty) {
      return request('POST', '/trading/paper-order', { symbol, action, quantity: qty });
    },

    analyzeAndPaperTrade(symbol, portfolioName, submitPaperOrder = true, recordLocalTrade = true) {
      return request('POST', '/trading/analyze-and-paper-trade', {
        symbol,
        portfolio_name: portfolioName || 'default',
        submit_paper_order: submitPaperOrder,
        record_local_trade: recordLocalTrade,
      });
    },

    /* Analysis */
    analyzeSymbol(symbol, portfolioName) {
      return request('POST', '/analysis/symbol', {
        symbol,
        portfolio_name: portfolioName || 'default',
      });
    },
  };
})();
