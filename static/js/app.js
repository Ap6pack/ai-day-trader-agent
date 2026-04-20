/**
 * AI Day Trader - Dashboard Application
 * State management, rendering, analysis workflow, paper trade submission.
 */
const App = (() => {
  /* ── State ── */
  const state = {
    user: null,
    portfolios: [],
    selectedPortfolio: null,
    holdings: [],
    trades: [],
    portfolioSummary: null,
    alpacaAccount: null,
    providerStatus: null,
    analysisResult: null,
    recentOrder: null,
    apiHealthy: false,
  };

  let pollTimer = null;

  /* ── DOM helpers ── */
  const $ = (id) => document.getElementById(id);
  const qs = (sel) => document.querySelector(sel);

  function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  /* ── Formatting ── */
  function fmtCurrency(v) {
    if (v == null || isNaN(v)) return '\u2014';
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(v);
  }

  function fmtPercent(v) {
    if (v == null || isNaN(v)) return '\u2014';
    return (v >= 0 ? '+' : '') + v.toFixed(2) + '%';
  }

  function fmtNum(v, dec = 2) {
    if (v == null || isNaN(v)) return '\u2014';
    return Number(v).toFixed(dec);
  }

  function fmtDate(s) {
    if (!s) return '\u2014';
    return new Date(s).toLocaleDateString('en-US', {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    });
  }

  function signalClass(sig) {
    switch ((sig || '').toUpperCase()) {
      case 'BUY':  return 'signal-buy';
      case 'SELL': return 'signal-sell';
      case 'HOLD': return 'signal-hold';
      default:     return '';
    }
  }

  function signalColor(sig) {
    switch ((sig || '').toUpperCase()) {
      case 'BUY':  return 'var(--green)';
      case 'SELL': return 'var(--red)';
      case 'HOLD': return 'var(--yellow)';
      default:     return 'var(--text-secondary)';
    }
  }

  /* ── Toast ── */
  function toast(msg, type = 'info') {
    let c = $('toast-container');
    if (!c) {
      c = document.createElement('div');
      c.id = 'toast-container';
      c.className = 'toast-container';
      document.body.appendChild(c);
    }
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = msg;
    c.appendChild(el);
    setTimeout(() => {
      el.style.opacity = '0';
      el.style.transition = 'opacity 0.3s';
      setTimeout(() => el.remove(), 300);
    }, 4000);
  }

  /* ── View switching ── */
  function showLogin() {
    $('login-view').style.display = 'flex';
    $('dashboard-view').style.display = 'none';
    stopPolling();
  }

  function showDashboard() {
    $('login-view').style.display = 'none';
    $('dashboard-view').style.display = 'flex';
    renderUserInfo();
    loadAllData();
    startPolling();
  }

  /* ── Event binding ── */
  function bindEvents() {
    $('login-form').addEventListener('submit', handleLogin);
    $('register-form').addEventListener('submit', handleRegister);

    $('show-register').addEventListener('click', (e) => {
      e.preventDefault();
      $('login-form').style.display = 'none';
      $('login-toggle').style.display = 'none';
      $('register-form').style.display = 'block';
      $('register-toggle').style.display = 'block';
    });

    $('show-login').addEventListener('click', (e) => {
      e.preventDefault();
      $('register-form').style.display = 'none';
      $('register-toggle').style.display = 'none';
      $('login-form').style.display = 'block';
      $('login-toggle').style.display = 'block';
    });

    $('logout-btn').addEventListener('click', handleLogout);
    $('analyze-btn').addEventListener('click', runAnalysis);
    $('ticker-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') runAnalysis();
    });

    $('create-portfolio-btn').addEventListener('click', () => {
      const f = $('create-portfolio-form');
      f.style.display = f.style.display === 'none' ? 'block' : 'none';
    });
    $('create-portfolio-form').addEventListener('submit', handleCreatePortfolio);
    $('cancel-create-portfolio').addEventListener('click', () => {
      $('create-portfolio-form').style.display = 'none';
    });

    $('edit-portfolio-form').addEventListener('submit', handleUpdatePortfolio);
    $('delete-portfolio-btn').addEventListener('click', handleDeletePortfolio);

    $('add-holding-btn').addEventListener('click', () => showHoldingForm());
    $('holding-form').addEventListener('submit', handleSaveHolding);
    $('cancel-holding').addEventListener('click', hideHoldingForm);

    $('record-trade-btn').addEventListener('click', () => {
      const f = $('trade-form');
      f.style.display = f.style.display === 'none' ? 'block' : 'none';
    });
    $('trade-form').addEventListener('submit', handleRecordTrade);
    $('cancel-trade').addEventListener('click', () => {
      $('trade-form').style.display = 'none';
    });
  }

  /* ── Auth handlers ── */
  async function handleLogin(e) {
    e.preventDefault();
    const user = $('login-username').value.trim();
    const pass = $('login-password').value;
    const err = $('login-error');
    err.style.display = 'none';

    const btn = e.target.querySelector('button[type="submit"]');
    btn.disabled = true;
    btn.textContent = 'Signing in\u2026';

    try {
      await API.login(user, pass);
      state.user = await API.getMe();
      showDashboard();
    } catch (ex) {
      err.textContent = ex.message || 'Login failed';
      err.style.display = 'block';
    } finally {
      btn.disabled = false;
      btn.textContent = 'Sign In';
    }
  }

  async function handleRegister(e) {
    e.preventDefault();
    const user = $('reg-username').value.trim();
    const email = $('reg-email').value.trim();
    const pass = $('reg-password').value;
    const err = $('register-error');
    err.style.display = 'none';

    const btn = e.target.querySelector('button[type="submit"]');
    btn.disabled = true;
    btn.textContent = 'Creating account\u2026';

    try {
      await API.register(user, email, pass);
      toast('Account created! Please sign in.', 'success');
      $('show-login').click();
      $('login-username').value = user;
    } catch (ex) {
      err.textContent = ex.message || 'Registration failed';
      err.style.display = 'block';
    } finally {
      btn.disabled = false;
      btn.textContent = 'Create Account';
    }
  }

  async function handleLogout() {
    await API.logout();
    state.user = null;
    state.selectedPortfolio = null;
    showLogin();
  }

  /* ── Data loading ── */
  async function loadAllData() {
    await Promise.allSettled([
      loadPortfolios(),
      loadStatusIndicators(),
      loadProviderStatus(),
      loadAlpacaAccount(),
    ]);
  }

  async function loadPortfolios() {
    try {
      state.portfolios = await API.getPortfolios();
      renderPortfolios();
      if (!state.selectedPortfolio && state.portfolios.length > 0) {
        selectPortfolio(state.portfolios[0].name);
      } else if (state.selectedPortfolio && !state.portfolios.some((p) => p.name === state.selectedPortfolio)) {
        state.selectedPortfolio = null;
        state.holdings = [];
        state.trades = [];
        state.portfolioSummary = null;
        renderPortfolioActions();
        renderHoldings();
        renderTradeHistory();
        renderPortfolioSummary();
      } else if (state.selectedPortfolio) {
        selectPortfolio(state.selectedPortfolio);
      } else {
        renderPortfolioActions();
      }
    } catch (ex) {
      if (ex.status === 401) { showLogin(); return; }
      console.error('Failed to load portfolios:', ex);
    }
  }

  async function selectPortfolio(name) {
    state.selectedPortfolio = name;
    renderPortfolios();

    try {
      const [holdings, trades, perf] = await Promise.all([
        API.getHoldings(name).catch(() => []),
        API.getTradeHistory(name, 30).catch(() => []),
        API.getPortfolioSummary(name).catch(() => null),
      ]);
      state.holdings = holdings;
      state.trades = trades;
      state.portfolioSummary = perf;
      renderPortfolioActions();
      renderHoldings();
      renderTradeHistory();
      renderPortfolioSummary();
    } catch (ex) {
      console.error('Failed to load portfolio data:', ex);
    }
  }

  async function loadStatusIndicators() {
    try { state.apiHealthy = await API.healthCheck(); }
    catch { state.apiHealthy = false; }
    renderTopBarStatus();
  }

  async function loadAlpacaAccount() {
    try {
      state.alpacaAccount = await API.getAlpacaAccount();
    } catch (ex) {
      state.alpacaAccount = { connected: false, message: ex.message };
    }
    renderAlpacaAccount();
    renderTopBarStatus();
  }

  async function loadProviderStatus() {
    try {
      state.providerStatus = await API.getProviderStatus();
    } catch {
      state.providerStatus = null;
    }
    renderProviderStatus();
  }

  /* ── Polling ── */
  function startPolling() {
    stopPolling();
    pollTimer = setInterval(() => {
      Promise.allSettled([loadStatusIndicators(), loadAlpacaAccount()]);
    }, 30000);
  }

  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  /* ── Render: top bar ── */
  function renderUserInfo() {
    $('user-display').textContent = state.user?.username || '';
  }

  function renderTopBarStatus() {
    // API
    const apiDot = qs('#api-status .status-dot');
    const apiLbl = qs('#api-status .status-label');
    apiDot.className = `status-dot ${state.apiHealthy ? 'connected' : 'disconnected'}`;
    apiLbl.textContent = state.apiHealthy ? 'API Connected' : 'API Down';

    // Alpaca
    const alpDot = qs('#alpaca-status .status-dot');
    const alpLbl = qs('#alpaca-status .status-label');
    const alp = state.alpacaAccount;
    alpDot.className = `status-dot ${alp?.connected ? 'connected' : 'disconnected'}`;
    alpLbl.textContent = alp?.connected ? 'Alpaca Paper' : 'Alpaca Off';

    // Market
    const mktDot = qs('#market-status .status-dot');
    const mktLbl = qs('#market-status .status-label');
    if (alp?.market_open === true) {
      mktDot.className = 'status-dot connected';
      mktLbl.textContent = 'Market Open';
    } else if (alp?.market_open === false) {
      mktDot.className = 'status-dot warning';
      mktLbl.textContent = 'Market Closed';
    } else {
      mktDot.className = 'status-dot';
      mktLbl.textContent = 'Market Unknown';
    }
  }

  /* ── Render: portfolios ── */
  function renderPortfolios() {
    const c = $('portfolio-list');
    if (!state.portfolios.length) {
      c.innerHTML = '<div class="empty-state">No portfolios yet</div>';
      return;
    }
    c.innerHTML = state.portfolios.map((p) => `
      <div class="portfolio-item ${p.name === state.selectedPortfolio ? 'selected' : ''}"
           data-name="${escapeHtml(p.name)}">
        <span class="portfolio-name">${escapeHtml(p.name)}</span>
        <span class="portfolio-capital">${fmtCurrency(p.total_value || p.trading_capital)}</span>
      </div>
    `).join('');

    c.querySelectorAll('.portfolio-item').forEach((el) => {
      el.addEventListener('click', () => selectPortfolio(el.dataset.name));
    });
  }

  function renderPortfolioActions() {
    const selected = state.portfolios.find((p) => p.name === state.selectedPortfolio);
    $('edit-portfolio-capital').value = selected ? selected.trading_capital : '';
    $('edit-portfolio-form').style.display = selected ? 'block' : 'none';
  }

  /* ── Render: holdings ── */
  function renderHoldings() {
    const c = $('holdings-container');
    if (!state.holdings?.length) {
      c.innerHTML = '<div class="empty-state">No holdings</div>';
      return;
    }
    const rows = state.holdings.map((h) => {
      const cls = h.unrealized_pnl >= 0 ? 'text-green' : 'text-red';
      return `<tr>
        <td>${escapeHtml(h.symbol)}</td>
        <td class="text-right text-mono">${h.quantity}</td>
        <td class="text-right text-mono">${fmtCurrency(h.avg_cost)}</td>
        <td class="text-right text-mono">${fmtCurrency(h.market_value)}</td>
        <td class="text-right text-mono ${cls}">${fmtPercent(h.unrealized_pnl_pct)}</td>
        <td class="text-right">
          <button class="btn btn-sm btn-outline holding-edit"
                  data-symbol="${escapeHtml(h.symbol)}"
                  data-quantity="${h.quantity}"
                  data-avg-cost="${h.avg_cost}">Edit</button>
          <button class="btn btn-sm btn-danger holding-remove"
                  data-symbol="${escapeHtml(h.symbol)}">Remove</button>
        </td>
      </tr>`;
    }).join('');

    c.innerHTML = `<table class="data-table">
      <thead><tr>
        <th>Symbol</th><th class="text-right">Qty</th>
        <th class="text-right">Avg Cost</th><th class="text-right">Value</th>
        <th class="text-right">P&amp;L</th><th class="text-right">Actions</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;

    c.querySelectorAll('.holding-edit').forEach((btn) => {
      btn.addEventListener('click', () => showHoldingForm({
        symbol: btn.dataset.symbol,
        quantity: Number(btn.dataset.quantity),
        avg_cost: Number(btn.dataset.avgCost),
      }));
    });
    c.querySelectorAll('.holding-remove').forEach((btn) => {
      btn.addEventListener('click', () => handleRemoveHolding(btn.dataset.symbol));
    });
  }

  /* ── Render: trade history ── */
  function renderTradeHistory() {
    const c = $('trades-container');
    if (!state.trades?.length) {
      c.innerHTML = '<div class="empty-state">No recent trades</div>';
      return;
    }
    const rows = state.trades.slice(0, 20).map((t) => {
      const cls = t.action === 'BUY' ? 'text-green' : 'text-red';
      return `<tr>
        <td>${fmtDate(t.timestamp)}</td>
        <td>${escapeHtml(t.symbol)}</td>
        <td class="${cls}">${t.action}</td>
        <td class="text-right text-mono">${t.quantity}</td>
        <td class="text-right text-mono">${fmtCurrency(t.price)}</td>
      </tr>`;
    }).join('');

    c.innerHTML = `<table class="data-table">
      <thead><tr>
        <th>Date</th><th>Symbol</th><th>Action</th>
        <th class="text-right">Qty</th><th class="text-right">Price</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  }

  /* ── Render: portfolio summary ── */
  function renderPortfolioSummary() {
    const c = $('portfolio-summary-container');
    if (!c) return;
    const p = state.portfolioSummary;
    if (!p) { c.innerHTML = ''; return; }

    const retCls = (p.total_return || 0) >= 0 ? 'text-green' : 'text-red';
    const realizedCls = (p.realized_pnl || 0) >= 0 ? 'text-green' : 'text-red';
    const unrealizedCls = (p.unrealized_pnl || 0) >= 0 ? 'text-green' : 'text-red';
    c.innerHTML = `<div class="perf-grid">
      <div class="perf-metric">
        <div class="value ${retCls}">${fmtPercent(p.total_return_pct)}</div>
        <div class="label">Total Return</div>
      </div>
      <div class="perf-metric">
        <div class="value ${realizedCls}">${fmtCurrency(p.realized_pnl)}</div>
        <div class="label">Realized P&amp;L</div>
      </div>
      <div class="perf-metric">
        <div class="value ${unrealizedCls}">${fmtCurrency(p.unrealized_pnl)}</div>
        <div class="label">Unrealized P&amp;L</div>
      </div>
      <div class="perf-metric">
        <div class="value">${p.total_trades || 0}</div>
        <div class="label">Trades</div>
      </div>
      <div class="perf-metric">
        <div class="value">${p.buy_trades || 0}/${p.sell_trades || 0}</div>
        <div class="label">Buy / Sell</div>
      </div>
      <div class="perf-metric">
        <div class="value">${fmtCurrency(p.total_fees)}</div>
        <div class="label">Fees</div>
      </div>
    </div>`;
  }

  /* ── Render: Alpaca account ── */
  function renderAlpacaAccount() {
    const c = $('alpaca-account-container');
    const a = state.alpacaAccount;
    if (!a) { c.innerHTML = '<div class="empty-state">Loading\u2026</div>'; return; }

    if (!a.connected) {
      c.innerHTML = `
        <div class="connection-badge disconnected">Disconnected</div>
        <div class="text-secondary" style="font-size:12px">${escapeHtml(a.message || 'Not connected')}</div>`;
      return;
    }

    c.innerHTML = `
      <div class="connection-badge connected">Connected</div>
      <div class="info-card">
        <div class="info-row"><span class="label">Mode</span>
          <span class="value">${a.paper_trading ? 'Paper' : 'Live'}</span></div>
        <div class="info-row"><span class="label">Status</span>
          <span class="value">${escapeHtml(a.account_status || '\u2014')}</span></div>
        <div class="info-row"><span class="label">Buying Power</span>
          <span class="value">${fmtCurrency(a.buying_power)}</span></div>
        <div class="info-row"><span class="label">Cash</span>
          <span class="value">${fmtCurrency(a.cash)}</span></div>
        <div class="info-row"><span class="label">Equity</span>
          <span class="value">${fmtCurrency(a.equity)}</span></div>
      </div>`;
  }

  /* ── Render: provider status ── */
  function renderProviderStatus() {
    const c = $('provider-status-container');
    const ps = state.providerStatus;
    if (!ps) { c.innerHTML = '<div class="empty-state">Loading\u2026</div>'; return; }

    const items = Object.entries(ps.configured_providers).map(([name, ok]) => `
      <div class="provider-item">
        <span class="provider-dot ${ok ? 'active' : 'inactive'}"></span>
        <span>${escapeHtml(name)}</span>
      </div>`).join('');

    c.innerHTML = `
      <div style="margin-bottom:8px">${items}</div>
      <div class="info-card">
        <div class="info-row"><span class="label">Data Feed</span>
          <span class="value">${escapeHtml(ps.alpaca_data_feed)}</span></div>
        <div class="info-row"><span class="label">Paper Trading</span>
          <span class="value">${ps.paper_trading_endpoint ? 'Yes' : 'No'}</span></div>
      </div>`;
  }

  /* ── Render: recent order ── */
  function renderRecentOrder() {
    const c = $('recent-order-container');
    const order = state.recentOrder;
    if (!order) { c.innerHTML = '<div class="empty-state">No recent orders</div>'; return; }

    if (order.submitted && order.order) {
      const o = order.order;
      const cls = (o.side || '') === 'buy' ? 'text-green' : 'text-red';
      const idShort = String(o.id || '').substring(0, 12);
      c.innerHTML = `<div class="order-result success">
        <div class="info-row"><span class="label">Order ID</span>
          <span class="value" style="font-size:10px">${escapeHtml(idShort)}\u2026</span></div>
        <div class="info-row"><span class="label">Symbol</span>
          <span class="value">${escapeHtml(o.symbol || '\u2014')}</span></div>
        <div class="info-row"><span class="label">Side</span>
          <span class="value ${cls}">${(o.side || '\u2014').toUpperCase()}</span></div>
        <div class="info-row"><span class="label">Quantity</span>
          <span class="value">${o.qty || '\u2014'}</span></div>
        <div class="info-row"><span class="label">Status</span>
          <span class="value">${escapeHtml(o.status || '\u2014')}</span></div>
        <div class="info-row"><span class="label">Local Trade</span>
          <span class="value">${order.recorded_trade_id || '\u2014'}</span></div>
        <div class="info-row"><span class="label">Filled Price</span>
          <span class="value">${o.filled_avg_price ? fmtCurrency(o.filled_avg_price) : 'Pending'}</span></div>
      </div>`;
    } else {
      c.innerHTML = `<div class="order-result failed">
        <div class="text-red" style="font-size:13px">Order not submitted</div>
        <div class="text-secondary" style="font-size:12px;margin-top:4px">
          ${escapeHtml(order.skipped_reason || 'Unknown reason')}</div>
      </div>`;
    }
  }

  /* ── Analysis ── */
  async function runAnalysis() {
    const ticker = $('ticker-input').value.trim().toUpperCase();
    if (!ticker) { toast('Enter a ticker symbol', 'error'); return; }

    const btn = $('analyze-btn');
    btn.disabled = true;
    btn.textContent = 'Analyzing\u2026';
    $('analysis-loading').style.display = 'flex';
    $('analysis-result').style.display = 'none';
    $('analysis-result').innerHTML = '';

    try {
      state.analysisResult = await API.analyzeSymbol(ticker, state.selectedPortfolio || 'default');
      renderAnalysisResult();
    } catch (ex) {
      toast(`Analysis failed: ${ex.message}`, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Analyze';
      $('analysis-loading').style.display = 'none';
    }
  }

  function renderAnalysisResult() {
    const c = $('analysis-result');
    const r = state.analysisResult;
    if (!r) { c.style.display = 'none'; return; }
    c.style.display = 'block';

    const signal     = r.recommendation || 'HOLD';
    const confidence = (r.confidence || 0) * 100;
    const ti         = r.technical_indicators || {};
    const risk       = r.risk_parameters || null;
    const signals    = r.all_signals || {};

    // Technical indicators
    const techRows = [];
    if (ti.current_price != null) techRows.push(['Price', fmtCurrency(ti.current_price)]);
    if (ti.rsi != null)           techRows.push(['RSI', fmtNum(ti.rsi, 1)]);
    if (ti.macd != null)          techRows.push(['MACD', fmtNum(ti.macd, 4)]);
    if (ti.sma_20 != null)        techRows.push(['SMA 20', fmtCurrency(ti.sma_20)]);
    if (ti.ema_20 != null)        techRows.push(['EMA 20', fmtCurrency(ti.ema_20)]);
    if (ti.volume != null)        techRows.push(['Volume', Number(ti.volume).toLocaleString()]);
    if (ti.price_change_pct != null) {
      const pcCls = ti.price_change_pct >= 0 ? 'text-green' : 'text-red';
      techRows.push(['Change', `<span class="${pcCls}">${fmtPercent(ti.price_change_pct)}</span>`]);
    }
    const techHtml = techRows.length
      ? techRows.map(([l, v]) => `<div class="detail-row"><span class="label">${l}</span><span class="value">${v}</span></div>`).join('')
      : '<div class="text-muted">No data</div>';

    // Risk parameters
    const riskRows = [];
    if (risk) {
      if (risk.stop_loss != null)        riskRows.push(['Stop Loss', `<span class="text-red">${fmtCurrency(risk.stop_loss)}</span>`]);
      if (risk.take_profit != null)      riskRows.push(['Take Profit', `<span class="text-green">${fmtCurrency(risk.take_profit)}</span>`]);
      if (risk.position_value != null)   riskRows.push(['Position Value', fmtCurrency(risk.position_value)]);
      if (risk.risk_reward_ratio != null) riskRows.push(['Risk/Reward', fmtNum(risk.risk_reward_ratio, 2)]);
    }
    const riskHtml = riskRows.length
      ? riskRows.map(([l, v]) => `<div class="detail-row"><span class="label">${l}</span><span class="value">${v}</span></div>`).join('')
      : '<div class="text-muted">No risk data</div>';

    // Strategy signals
    const sigEntries = Object.entries(signals);
    const sigHtml = sigEntries.map(([name, s]) => `
      <div class="strategy-signal">
        <span class="strategy-name">${escapeHtml(name)}</span>
        <span class="strategy-signal-badge ${signalClass(s.signal)}">${s.signal}</span>
        <span class="strategy-strength">${fmtNum((s.strength || 0) * 100, 0)}%</span>
      </div>`).join('');

    // Submit trade button
    let tradeHtml = '';
    if (signal !== 'HOLD' && r.quantity > 0) {
      const btnCls = signal === 'BUY' ? 'btn-success' : 'btn-danger';
      tradeHtml = `
        <div class="trade-submit-section">
          <div class="trade-info">
            ${signal} ${r.quantity} shares of ${escapeHtml(r.symbol)} at ~${fmtCurrency(ti.current_price)}
          </div>
          <button class="btn ${btnCls}" id="submit-trade-btn">Submit Paper Trade</button>
        </div>`;
    }

    c.innerHTML = `<div class="analysis-result">
      <div class="signal-banner ${signalClass(signal)}">
        <div class="signal-info">
          <div class="signal-badge">${signal}</div>
          <div class="signal-reason">${escapeHtml(r.primary_strategy || '')} \u2014 ${escapeHtml(r.primary_reason || '')}</div>
        </div>
        <div class="confidence-meter">
          <div class="confidence-bar-bg">
            <div class="confidence-bar-fill" style="width:${confidence}%;background:${signalColor(signal)}"></div>
          </div>
          <span class="confidence-label">${fmtNum(confidence, 0)}%</span>
        </div>
      </div>

      <div class="analysis-details">
        <div class="detail-grid">
          <div class="detail-section">
            <h4>Technical Indicators</h4>
            ${techHtml}
          </div>
          <div class="detail-section">
            <h4>Risk Parameters</h4>
            ${riskHtml}
          </div>
        </div>

        ${sigEntries.length ? `
        <div class="detail-section">
          <h4>Strategy Signals (${r.confirming_strategies || 0} confirming, ${r.conflicting_strategies || 0} conflicting)</h4>
          <div class="signal-list">${sigHtml}</div>
        </div>` : ''}
      </div>

      ${tradeHtml}
    </div>`;

    // Bind submit button
    const submitBtn = $('submit-trade-btn');
    if (submitBtn) submitBtn.addEventListener('click', submitPaperTrade);
  }

  /* ── Paper trade submission ── */
  async function submitPaperTrade() {
    const r = state.analysisResult;
    if (!r || r.recommendation === 'HOLD') return;

    const btn = $('submit-trade-btn');
    btn.disabled = true;
    btn.textContent = 'Submitting\u2026';

    try {
      const workflowResult = await API.analyzeAndPaperTrade(
        r.symbol,
        state.selectedPortfolio || 'default',
        true,
        true,
      );
      state.analysisResult = workflowResult.analysis || state.analysisResult;
      const result = {
        submitted: !!workflowResult.alpaca_order,
        order: workflowResult.alpaca_order,
        skipped_reason: workflowResult.skipped_reason,
        recorded_trade_id: workflowResult.recorded_trade_id,
      };
      state.recentOrder = result;
      renderRecentOrder();

      if (result.submitted) {
        toast(`Paper order submitted and recorded: ${r.recommendation} ${r.quantity} ${r.symbol}`, 'success');
        if (state.selectedPortfolio) {
          await selectPortfolio(state.selectedPortfolio);
        }
      } else {
        toast(`Order skipped: ${result.skipped_reason || 'Unknown'}`, 'error');
      }
    } catch (ex) {
      toast(`Order failed: ${ex.message}`, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Submit Paper Trade';
    }
  }

  /* ── Create portfolio ── */
  async function handleCreatePortfolio(e) {
    e.preventDefault();
    const name = $('new-portfolio-name').value.trim();
    const capital = parseFloat($('new-portfolio-capital').value);
    const err = $('create-portfolio-error');
    err.style.display = 'none';

    if (!name || isNaN(capital) || capital <= 0) {
      err.textContent = 'Enter a valid name and capital amount';
      err.style.display = 'block';
      return;
    }

    try {
      await API.createPortfolio(name, capital);
      $('create-portfolio-form').style.display = 'none';
      $('new-portfolio-name').value = '';
      $('new-portfolio-capital').value = '';
      toast(`Portfolio "${name}" created`, 'success');
      await loadPortfolios();
      selectPortfolio(name);
    } catch (ex) {
      err.textContent = ex.message || 'Failed to create portfolio';
      err.style.display = 'block';
    }
  }

  async function handleUpdatePortfolio(e) {
    e.preventDefault();
    if (!state.selectedPortfolio) { toast('Select a portfolio first', 'error'); return; }

    const capital = parseFloat($('edit-portfolio-capital').value);
    const err = $('edit-portfolio-error');
    err.style.display = 'none';

    if (isNaN(capital) || capital <= 0) {
      err.textContent = 'Enter a valid trading capital amount';
      err.style.display = 'block';
      return;
    }

    try {
      await API.updatePortfolio(state.selectedPortfolio, capital);
      toast(`Portfolio "${state.selectedPortfolio}" updated`, 'success');
      await loadPortfolios();
      await selectPortfolio(state.selectedPortfolio);
    } catch (ex) {
      err.textContent = ex.message || 'Failed to update portfolio';
      err.style.display = 'block';
    }
  }

  async function handleDeletePortfolio() {
    if (!state.selectedPortfolio) { toast('Select a portfolio first', 'error'); return; }
    const name = state.selectedPortfolio;
    const confirmed = window.confirm(`Delete portfolio "${name}" and all local holdings/trades? This cannot be undone.`);
    if (!confirmed) return;

    try {
      await API.deletePortfolio(name);
      toast(`Portfolio "${name}" deleted`, 'success');
      state.selectedPortfolio = null;
      state.holdings = [];
      state.trades = [];
      state.portfolioSummary = null;
      await loadPortfolios();
      renderHoldings();
      renderTradeHistory();
      renderPortfolioSummary();
      renderPortfolioActions();
    } catch (ex) {
      toast(`Delete failed: ${ex.message}`, 'error');
    }
  }

  function showHoldingForm(holding = null) {
    if (!state.selectedPortfolio) { toast('Select a portfolio first', 'error'); return; }
    $('holding-symbol').value = holding?.symbol || '';
    $('holding-symbol').disabled = !!holding;
    $('holding-quantity').value = holding?.quantity || '';
    $('holding-avg-cost').value = holding?.avg_cost || '';
    $('holding-error').style.display = 'none';
    $('holding-form').style.display = 'block';
  }

  function hideHoldingForm() {
    $('holding-form').style.display = 'none';
    $('holding-symbol').disabled = false;
    $('holding-symbol').value = '';
    $('holding-quantity').value = '';
    $('holding-avg-cost').value = '';
  }

  async function handleSaveHolding(e) {
    e.preventDefault();
    if (!state.selectedPortfolio) { toast('Select a portfolio first', 'error'); return; }

    const symbol = $('holding-symbol').value.trim().toUpperCase();
    const quantity = parseInt($('holding-quantity').value, 10);
    const avgCost = parseFloat($('holding-avg-cost').value);
    const err = $('holding-error');
    err.style.display = 'none';

    if (!symbol || isNaN(quantity) || quantity <= 0 || isNaN(avgCost) || avgCost <= 0) {
      err.textContent = 'Enter a valid symbol, quantity, and average cost';
      err.style.display = 'block';
      return;
    }

    try {
      await API.saveHolding(state.selectedPortfolio, symbol, quantity, avgCost);
      toast(`Holding saved: ${symbol}`, 'success');
      hideHoldingForm();
      await selectPortfolio(state.selectedPortfolio);
    } catch (ex) {
      err.textContent = ex.message || 'Failed to save holding';
      err.style.display = 'block';
    }
  }

  async function handleRemoveHolding(symbol) {
    if (!state.selectedPortfolio || !symbol) return;
    const confirmed = window.confirm(`Remove ${symbol} from "${state.selectedPortfolio}"?`);
    if (!confirmed) return;

    try {
      await API.removeHolding(state.selectedPortfolio, symbol);
      toast(`Holding removed: ${symbol}`, 'success');
      await selectPortfolio(state.selectedPortfolio);
    } catch (ex) {
      toast(`Remove failed: ${ex.message}`, 'error');
    }
  }

  async function handleRecordTrade(e) {
    e.preventDefault();
    if (!state.selectedPortfolio) { toast('Select a portfolio first', 'error'); return; }

    const symbol = $('trade-symbol').value.trim().toUpperCase();
    const action = $('trade-action').value;
    const quantity = parseInt($('trade-quantity').value, 10);
    const price = parseFloat($('trade-price').value);
    const err = $('trade-error');
    err.style.display = 'none';

    if (!symbol || !['BUY', 'SELL'].includes(action) || isNaN(quantity) || quantity <= 0 || isNaN(price) || price <= 0) {
      err.textContent = 'Enter a valid symbol, action, quantity, and price';
      err.style.display = 'block';
      return;
    }

    try {
      await API.recordTrade(state.selectedPortfolio, {
        symbol,
        action,
        quantity,
        price,
        strategy: 'manual',
        notes: 'Recorded from dashboard',
      });
      toast(`Trade recorded: ${action} ${quantity} ${symbol}`, 'success');
      $('trade-form').style.display = 'none';
      $('trade-symbol').value = '';
      $('trade-quantity').value = '';
      $('trade-price').value = '';
      await selectPortfolio(state.selectedPortfolio);
    } catch (ex) {
      err.textContent = ex.message || 'Failed to record trade';
      err.style.display = 'block';
    }
  }

  /* ── Init ── */
  async function init() {
    bindEvents();

    if (API.isLoggedIn()) {
      try {
        state.user = await API.getMe();
        showDashboard();
      } catch {
        API.clearTokens();
        showLogin();
      }
    } else {
      showLogin();
    }
  }

  return { init };
})();

document.addEventListener('DOMContentLoaded', App.init);
