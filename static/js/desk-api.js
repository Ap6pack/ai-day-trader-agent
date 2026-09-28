/**
 * ADT Desk API client for trader.desk (python -m trader.desk).
 * The desk runs locally; a token is only needed when DESK_TOKEN is set on the server.
 */
const API = (() => {
  'use strict';
  const TOKEN_KEY = 'adt_desk_token';
  let authRequired = null;

  function getAccessToken() {
    try { return localStorage.getItem(TOKEN_KEY) || ''; } catch { return ''; }
  }
  function setToken(token) {
    try { token ? localStorage.setItem(TOKEN_KEY, token) : localStorage.removeItem(TOKEN_KEY); } catch {}
  }

  async function request(method, path, body) {
    const headers = { Accept: 'application/json' };
    const token = getAccessToken();
    if (token) headers.Authorization = `Bearer ${token}`;
    if (body !== undefined) headers['Content-Type'] = 'application/json';
    const res = await fetch(`/api${path}`, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
    let data = null;
    try { data = await res.json(); } catch { /* empty body */ }
    if (!res.ok) {
      const detail = data && (data.detail || data.message);
      const text = Array.isArray(detail) ? detail.map((d) => `${(d.loc || []).slice(-1)[0] || ''} ${d.msg}`.trim()).join('; ') : detail;
      const err = new Error(text || `HTTP ${res.status}`);
      err.status = res.status;
      throw err;
    }
    return data;
  }

  return {
    getAccessToken,
    async isAuthRequired() {
      if (authRequired === null) authRequired = !!(await request('GET', '/desk/auth')).auth_required;
      return authRequired;
    },
    async isLoggedIn() { return !(await this.isAuthRequired()) || !!getAccessToken(); },
    async login(_user, token) {
      setToken(token);
      try { await request('GET', '/desk/config'); } catch (err) { setToken(''); throw err; }
    },
    async logout() { setToken(''); },
    async refreshAccessToken() { return !(await this.isAuthRequired()); },
    async getMe() { return { username: (await this.isAuthRequired()) ? 'TOKEN' : 'LOCAL' }; },

    getDeskConfig()   { return request('GET', '/desk/config'); },
    getDeskAccount()  { return request('GET', '/desk/account'); },
    getDeskNews(sym)  { return request('GET', `/desk/news/${encodeURIComponent(sym)}`); },
    getDeskBars(sym, timeframe, limit = 300) {
      return request('GET', `/desk/bars/${encodeURIComponent(sym)}?timeframe=${encodeURIComponent(timeframe)}&limit=${limit}`);
    },
    getNewsbot()      { return request('GET', '/desk/newsbot'); },
    judgeSymbol(sym)  { return request('POST', `/desk/judge/${encodeURIComponent(sym)}`); },
    analyzeSymbol(sym, portfolio = 'default') {
      return request('POST', `/desk/analyze/${encodeURIComponent(sym)}?portfolio=${encodeURIComponent(portfolio)}`);
    },
    getAutopilot()    { return request('GET', '/desk/autopilot'); },
    startAutopilot(cfg) { return request('POST', '/desk/autopilot', cfg); },
    stopAutopilot()   { return request('DELETE', '/desk/autopilot'); },
    pauseNewsbot(paused) { return request('POST', '/desk/newsbot/pause', { paused }); },
    getPortfolios()   { return request('GET', '/desk/portfolios'); },
    getPortfolio(name) { return request('GET', `/desk/portfolios/${encodeURIComponent(name)}`); },
    createPortfolio(name, cash) { return request('POST', '/desk/portfolios', { name, cash }); },
    deletePortfolio(name) { return request('DELETE', `/desk/portfolios/${encodeURIComponent(name)}`); },
    submitPaperOrder(symbol, side, qty, destination = 'alpaca', confirmLive = false) {
      return request('POST', '/desk/orders', { symbol, side, qty, destination, confirm_live: confirmLive });
    },
    cancelOrder(id)   { return request('POST', `/desk/orders/${encodeURIComponent(id)}/cancel`); },
  };
})();
