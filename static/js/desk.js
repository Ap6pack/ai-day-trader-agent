/**
 * ADT Desk — live window onto the agents (Jev news bot, Claude, order guard),
 * with a manual Alpaca paper ticket. Streams quotes, journal activity, the news
 * bot's status and the Alpaca paper account over /ws/desk (python -m trader.desk).
 */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
  };

  const TF_LABEL = { '1Min': '1M', '5Min': '5M', '15Min': '15M', '1Hour': '1H', '1Day': '1D' };
  const TF_SECONDS = { '1Min': 60, '5Min': 300, '15Min': 900, '1Hour': 3600, '1Day': 86400 };
  const MAX_EVENTS = 400;
  const EVENT_GROUP = {
    news_signal: 'signal', news_skip: 'signal', judgment: 'signal', decision: 'signal',
    order_submitted: 'order', order_failed: 'order', order_cancelled: 'order', flatten: 'order', guard: 'order',
  };
  const EVENT_TAG = {
    news_signal: 'JEV', news_skip: 'PASS', judgment: 'JUDGE', decision: 'CALL',
    order_submitted: 'ORDER', order_failed: 'REJ', order_cancelled: 'CXL', flatten: 'FLAT', guard: 'GUARD',
  };

  const S = {
    config: null,
    watchlist: [],
    quotes: {},
    symbol: null,
    tf: store.get('adt_desk_tf', '5Min'),
    bars: [],
    barsSource: null,
    decisions: {},
    events: [],
    feedFilter: 'all',
    account: null,
    side: 'BUY',
    indicators: store.get('adt_desk_ind', { sma: true, ema: true, vwap: false, levels: true }),
    newsbot: null,
    ws: null,
    wsRetry: 0,
    analyzing: new Set(),
    selectedRow: -1,
  };

  /* ═══ Formatting ═══ */
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const num = (v) => (v === null || v === undefined || v === '' || Number.isNaN(Number(v)) ? null : Number(v));
  function px(v) {
    v = num(v); if (v === null) return '—';
    const d = Math.abs(v) < 1 ? 4 : 2;
    return v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  function money(v, sign = false) {
    v = num(v); if (v === null) return '—';
    const s = Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    return `${v < 0 ? '-' : sign && v > 0 ? '+' : ''}$${s}`;
  }
  function signed(v, d = 2, suffix = '') {
    v = num(v); if (v === null) return '—';
    return `${v > 0 ? '+' : ''}${v.toFixed(d)}${suffix}`;
  }
  function vol(v) {
    v = num(v); if (v === null) return '—';
    const a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(2) + 'B';
    if (a >= 1e6) return (v / 1e6).toFixed(2) + 'M';
    if (a >= 1e3) return (v / 1e3).toFixed(1) + 'K';
    return String(Math.round(v));
  }
  const dir = (v) => (num(v) === null || num(v) === 0 ? 'flat' : num(v) > 0 ? 'up' : 'down');
  const etTime = (d) => new Date(d).toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour12: false });
  function ago(iso) {
    const s = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 1000));
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m`;
    if (s < 86400) return `${Math.floor(s / 3600)}h`;
    return `${Math.floor(s / 86400)}d`;
  }

  // lightweight-charts renders timestamps as UTC; shift them so the axis reads New York time.
  const etOffsetCache = new Map();
  function etOffset(ts) {
    const hourKey = Math.floor(ts / 3600);
    if (etOffsetCache.has(hourKey)) return etOffsetCache.get(hourKey);
    const parts = new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York', hourCycle: 'h23',
      year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit',
    }).formatToParts(new Date(ts * 1000)).reduce((o, p) => (o[p.type] = p.value, o), {});
    const asUtc = Date.UTC(+parts.year, +parts.month - 1, +parts.day, +parts.hour, +parts.minute, +parts.second) / 1000;
    const off = Math.round((asUtc - ts) / 60) * 60;
    etOffsetCache.set(hourKey, off);
    return off;
  }
  const chartTime = (ts) => (S.tf === '1Day' ? ts : ts + etOffset(ts));

  function toast(msg, ms = 2600) {
    const el = document.createElement('div');
    el.className = 'toast'; el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), ms);
  }

  /* ═══ Auth ═══ */
  function showLogin() {
    $('desk').hidden = true; $('login').hidden = false;
    if (S.ws) { S.ws.onclose = null; S.ws.close(); S.ws = null; }
    $('login-user').focus();
  }

  $('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    $('login-error').textContent = '';
    try {
      await API.login($('login-user').value.trim(), $('login-pass').value);
      start();
    } catch (err) {
      $('login-error').textContent = err.message || 'Login failed';
    }
  });

  $('logout').addEventListener('click', async () => { await API.logout(); showLogin(); });

  async function boot() {
    try {
      if (await API.isLoggedIn()) start(); else showLogin();
    } catch (err) {
      toast(`Desk server unavailable: ${err.message}`, 6000);
    }
  }

  /* ═══ Boot ═══ */
  async function start() {
    try {
      const [me, config] = await Promise.all([API.getMe(), API.getDeskConfig()]);
      S.config = config;
      $('user').textContent = me.username.toUpperCase();
    } catch (err) {
      if (err.status === 401) return showLogin();
      toast(`Desk unavailable: ${err.message}`);
      return;
    }
    $('login').hidden = true; $('desk').hidden = false;
    $('logout').hidden = !S.config.auth_required;
    $('demo-badge').hidden = !S.config.demo_mode;
    if (!S.config.timeframes.includes(S.tf)) S.tf = S.config.default_timeframe;

    S.watchlist = store.get('adt_desk_watchlist', null) || S.config.watchlist.slice();
    buildTimeframes();
    syncIndicatorButtons();
    initChart();
    renderWatchlist();
    connect();
    loadSymbol(store.get('adt_desk_symbol', null) || S.watchlist[0] || 'SPY');
    refreshAccount();
    if (!S.config.alpaca_configured) {
      $('tk-msg').innerHTML = '<span class="muted">Set ALPACA_API_KEY / ALPACA_SECRET_KEY to route paper orders.</span>';
    }
  }

  /* ═══ WebSocket ═══ */
  function setConn(state) {
    const el = $('conn');
    el.className = `conn ${state}`;
    el.querySelector('b').textContent = { live: 'LIVE', connecting: 'CONNECTING', '': 'OFFLINE' }[state] ?? 'OFFLINE';
  }

  function connect() {
    const token = API.getAccessToken();
    if (S.config?.auth_required && !token) return showLogin();
    setConn('connecting');
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws/desk${token ? `?token=${encodeURIComponent(token)}` : ''}`);
    S.ws = ws;
    let opened = false;

    ws.onopen = () => { opened = true; S.wsRetry = 0; setConn('live'); sendWatch(); };
    ws.onmessage = (m) => { try { handleMessage(JSON.parse(m.data)); } catch (e) { console.error(e); } };
    ws.onclose = async () => {
      if (S.ws !== ws) return;
      setConn('');
      if (!opened) {
        // Rejected at handshake: most likely a wrong or missing DESK_TOKEN.
        const ok = await API.refreshAccessToken();
        if (!ok) return showLogin();
      }
      const delay = Math.min(15000, 1000 * 2 ** S.wsRetry++);
      setTimeout(() => { if (S.ws === ws) connect(); }, delay);
    };
  }

  setInterval(() => { if (S.ws && S.ws.readyState === 1) S.ws.send(JSON.stringify({ type: 'ping' })); }, 25000);

  function watchedSymbols() {
    const set = new Set(S.watchlist);
    if (S.symbol) set.add(S.symbol);
    (S.account?.positions || []).forEach((p) => set.add(p.symbol));
    return [...set].slice(0, 40);
  }

  function sendWatch() {
    if (S.ws && S.ws.readyState === 1) S.ws.send(JSON.stringify({ type: 'watch', symbols: watchedSymbols() }));
  }

  function handleMessage(msg) {
    switch (msg.type) {
      case 'hello':
        S.events = msg.data.events || [];
        S.events.slice().reverse().forEach((e) => { if (e.data?.signal && e.symbol) S.decisions[e.symbol] = e.data.signal; });
        renderFeed();
        setNewsbot(msg.data.newsbot);
        if (msg.data.account) setAccount(msg.data.account);
        if (S.decisions[S.symbol]) renderSignal(S.decisions[S.symbol]);
        break;
      case 'quotes': onQuotes(msg.data); break;
      case 'event': onEvent(msg.data); break;
      case 'account': setAccount(msg.data); break;
      case 'newsbot': setNewsbot(msg.data); break;
    }
  }

  /* ═══ Quotes ═══ */
  function onQuotes(quotes) {
    for (const [sym, q] of Object.entries(quotes)) {
      const prev = S.quotes[sym];
      S.quotes[sym] = q;
      updateWatchRow(sym, q, prev);
      if (sym === S.symbol) { renderQuoteHeader(q, prev); liveCandle(q); updateTicketEst(); }
    }
    renderTape();
  }

  function flash(el, prevVal, newVal) {
    if (!el || prevVal === undefined || prevVal === null || newVal === prevVal) return;
    el.classList.remove('flash-up', 'flash-down');
    void el.offsetWidth;
    el.classList.add(newVal > prevVal ? 'flash-up' : 'flash-down');
  }

  /* ═══ Watchlist ═══ */
  function renderWatchlist() {
    const body = $('watch-body');
    body.innerHTML = S.watchlist.map((sym) => `
      <tr data-sym="${esc(sym)}" class="${sym === S.symbol ? 'sel' : ''}">
        <td>${esc(sym)}</td><td class="r cell-last">—</td><td class="r cell-chg">—</td>
        <td class="r"><span class="pill cell-pct">—</span></td><td class="r cell-vol muted">—</td>
        <td class="r"><span class="x" title="Remove">✕</span></td>
      </tr>`).join('');
    S.watchlist.forEach((sym) => S.quotes[sym] && updateWatchRow(sym, S.quotes[sym]));
    $('watch-count').textContent = `${S.watchlist.length} SYM`;
  }

  function updateWatchRow(sym, q, prev) {
    const row = $('watch-body').querySelector(`tr[data-sym="${CSS.escape(sym)}"]`);
    if (!row) return;
    const d = dir(q.change);
    const last = row.querySelector('.cell-last');
    last.textContent = px(q.last); last.className = `r cell-last ${d}`;
    flash(last, prev?.last, q.last);
    const chg = row.querySelector('.cell-chg');
    chg.textContent = signed(q.change); chg.className = `r cell-chg ${d}`;
    const pct = row.querySelector('.cell-pct');
    pct.textContent = signed(q.change_pct, 2, '%'); pct.className = `pill cell-pct ${d}`;
    row.querySelector('.cell-vol').textContent = vol(q.volume);
  }

  $('watch-body').addEventListener('click', (e) => {
    const row = e.target.closest('tr[data-sym]');
    if (!row) return;
    if (e.target.classList.contains('x')) return removeWatch(row.dataset.sym);
    loadSymbol(row.dataset.sym);
  });

  $('watch-add').addEventListener('submit', (e) => {
    e.preventDefault();
    addWatch($('watch-input').value);
    $('watch-input').value = '';
  });

  function addWatch(sym) {
    sym = (sym || '').trim().toUpperCase();
    if (!/^[A-Z][A-Z.]{0,9}$/.test(sym) || S.watchlist.includes(sym)) return;
    S.watchlist.push(sym);
    store.set('adt_desk_watchlist', S.watchlist);
    renderWatchlist(); sendWatch();
  }

  function removeWatch(sym) {
    S.watchlist = S.watchlist.filter((s) => s !== sym);
    store.set('adt_desk_watchlist', S.watchlist);
    renderWatchlist(); sendWatch(); renderTape();
  }

  function renderTape() {
    const html = S.watchlist.filter((s) => S.quotes[s]).map((s) => {
      const q = S.quotes[s];
      const d = dir(q.change);
      return `<span class="tape-item" data-sym="${esc(s)}"><b>${esc(s)}</b>${px(q.last)} <span class="${d}">${d === 'up' ? '▲' : d === 'down' ? '▼' : '■'} ${signed(q.change_pct, 2, '%')}</span></span>`;
    }).join('');
    const track = $('tape-track');
    if (track.dataset.html !== html) { track.innerHTML = html; track.dataset.html = html; }
  }
  $('tape-track').addEventListener('click', (e) => {
    const it = e.target.closest('.tape-item'); if (it) loadSymbol(it.dataset.sym);
  });

  /* ═══ Symbol ═══ */
  function loadSymbol(sym) {
    sym = (sym || '').trim().toUpperCase();
    if (!/^[A-Z][A-Z.]{0,9}$/.test(sym)) return toast(`Bad symbol: ${sym}`);
    S.symbol = sym;
    store.set('adt_desk_symbol', sym);
    $('q-sym').textContent = sym;
    $('tk-sym').value = sym;
    $('news-sym').textContent = sym;
    document.title = `${sym} · ADT Desk`;
    $('watch-body').querySelectorAll('tr').forEach((r) => r.classList.toggle('sel', r.dataset.sym === sym));
    S.selectedRow = S.watchlist.indexOf(sym);
    renderQuoteHeader(S.quotes[sym]);
    if (S.decisions[sym]) renderSignal(S.decisions[sym]); else renderSignalEmpty();
    if (S.analyzing.has(sym)) renderAnalyzing();
    sendWatch();
    loadBars();
    loadNews();
    updateTicketEst();
  }

  function renderQuoteHeader(q, prev) {
    const last = $('q-last');
    if (!q) {
      last.textContent = '—'; $('q-chg').textContent = '';
      ['q-bid', 'q-ask', 'q-open', 'q-high', 'q-low', 'q-vol'].forEach((id) => { $(id).textContent = '—'; });
      return;
    }
    const d = dir(q.change);
    last.textContent = px(q.last); last.className = `q-last ${d}`;
    flash(last, prev?.last, q.last);
    $('q-chg').innerHTML = `<span class="${d}">${signed(q.change)} (${signed(q.change_pct, 2, '%')})</span>`;
    $('q-bid').textContent = q.bid ? `${px(q.bid)}${q.bid_size ? ' ×' + vol(q.bid_size) : ''}` : '—';
    $('q-ask').textContent = q.ask ? `${px(q.ask)}${q.ask_size ? ' ×' + vol(q.ask_size) : ''}` : '—';
    $('q-open').textContent = px(q.open);
    $('q-high').textContent = px(q.high);
    $('q-low').textContent = px(q.low);
    $('q-vol').textContent = vol(q.volume);
  }

  /* ═══ Chart ═══ */
  let chart, candles, volumeSeries, smaSeries, emaSeries, vwapSeries;
  let levelLines = [];

  function initChart() {
    const LC = window.LightweightCharts;
    if (!LC) { chartMessage('Chart library failed to load.'); return; }
    chart = LC.createChart($('chart'), {
      autoSize: true,
      localization: { locale: 'en-US' },
      layout: { background: { color: '#0b0f14' }, textColor: '#6c7a89', fontFamily: getComputedStyle(document.body).fontFamily, fontSize: 11 },
      grid: { vertLines: { color: '#121922' }, horzLines: { color: '#121922' } },
      crosshair: { mode: LC.CrosshairMode.Normal, vertLine: { color: '#ff9f1a55', labelBackgroundColor: '#7a4b0a' }, horzLine: { color: '#ff9f1a55', labelBackgroundColor: '#7a4b0a' } },
      rightPriceScale: { borderColor: '#1b232d' },
      timeScale: { borderColor: '#1b232d', timeVisible: true, secondsVisible: false, rightOffset: 6 },
    });
    candles = chart.addCandlestickSeries({
      upColor: '#1fd67f', downColor: '#ff4d5e', borderUpColor: '#1fd67f', borderDownColor: '#ff4d5e',
      wickUpColor: '#1fd67f', wickDownColor: '#ff4d5e',
    });
    volumeSeries = chart.addHistogramSeries({ priceScaleId: 'vol', priceFormat: { type: 'volume' }, lastValueVisible: false, priceLineVisible: false });
    chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    const line = (color) => chart.addLineSeries({ color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
    smaSeries = line('#ff9f1a');
    emaSeries = line('#4aa8ff');
    vwapSeries = line('#b48cff');
    chart.subscribeCrosshairMove(renderLegend);
  }

  function chartMessage(msg) {
    const el = $('chart-msg');
    el.hidden = !msg; el.textContent = msg || '';
  }

  function buildTimeframes() {
    $('tf-seg').innerHTML = S.config.timeframes.map((tf) =>
      `<button data-tf="${tf}" class="${tf === S.tf ? 'on' : ''}">${TF_LABEL[tf] || tf}</button>`).join('');
  }
  $('tf-seg').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-tf]'); if (b) setTimeframe(b.dataset.tf);
  });
  function setTimeframe(tf) {
    if (!S.config.timeframes.includes(tf)) return;
    S.tf = tf; store.set('adt_desk_tf', tf);
    $('tf-seg').querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.tf === tf));
    loadBars();
  }

  function syncIndicatorButtons() {
    $('ind-seg').querySelectorAll('button').forEach((b) => b.classList.toggle('on', !!S.indicators[b.dataset.ind]));
  }
  $('ind-seg').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-ind]'); if (!b) return;
    S.indicators[b.dataset.ind] = !S.indicators[b.dataset.ind];
    store.set('adt_desk_ind', S.indicators);
    syncIndicatorButtons(); drawStudies(); drawLevels();
  });

  let barsReq = 0;
  async function loadBars() {
    if (!chart) return;
    const req = ++barsReq;
    const sym = S.symbol, tf = S.tf;
    chartMessage(`Loading ${sym} ${TF_LABEL[tf]}…`);
    try {
      const res = await API.getDeskBars(sym, tf, 400);
      if (req !== barsReq) return;
      S.bars = res.bars || [];
      S.barsSource = res.source;
      if (!S.bars.length) {
        candles.setData([]); volumeSeries.setData([]); drawStudies();
        chartMessage(`No bars for ${sym}. ${(res.errors || []).join(' · ')}`);
        return;
      }
      chartMessage('');
      drawBars(true);
    } catch (err) {
      if (req === barsReq) chartMessage(`Chart data error: ${err.message}`);
    }
  }
  setInterval(() => { if (S.symbol && !document.hidden) loadBars(); }, 60000);

  function drawBars(fit) {
    candles.setData(S.bars.map((b) => ({ time: chartTime(b.time), open: b.open, high: b.high, low: b.low, close: b.close })));
    volumeSeries.setData(S.bars.map((b) => ({ time: chartTime(b.time), value: b.volume, color: b.close >= b.open ? '#1fd67f40' : '#ff4d5e40' })));
    drawStudies();
    drawLevels();
    renderLegend();
    if (fit) {
      const n = S.bars.length;
      chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, n - 150), to: n + 5 });
    }
  }

  function sma(bars, n) {
    const out = []; let sum = 0;
    bars.forEach((b, i) => {
      sum += b.close; if (i >= n) sum -= bars[i - n].close;
      if (i >= n - 1) out.push({ time: chartTime(b.time), value: sum / n });
    });
    return out;
  }
  function ema(bars, n) {
    const k = 2 / (n + 1); let prev = null;
    return bars.map((b) => { prev = prev === null ? b.close : b.close * k + prev * (1 - k); return { time: chartTime(b.time), value: prev }; });
  }
  function vwap(bars) {
    let pv = 0, v = 0, day = null;
    return bars.map((b) => {
      const d = new Date((b.time + etOffset(b.time)) * 1000).getUTCDate();
      if (d !== day) { pv = 0; v = 0; day = d; }
      const tp = (b.high + b.low + b.close) / 3;
      pv += tp * (b.volume || 1); v += b.volume || 1;
      return { time: chartTime(b.time), value: pv / v };
    });
  }
  function drawStudies() {
    if (!chart) return;
    smaSeries.setData(S.indicators.sma ? sma(S.bars, 20) : []);
    emaSeries.setData(S.indicators.ema ? ema(S.bars, 9) : []);
    vwapSeries.setData(S.indicators.vwap && S.tf !== '1Day' ? vwap(S.bars) : []);
  }

  function drawLevels() {
    if (!candles) return;
    levelLines.forEach((l) => candles.removePriceLine(l));
    levelLines = [];
    const d = S.decisions[S.symbol];
    if (!S.indicators.levels || !d) return;
    const n = normalizeDecision(d);
    const add = (price, color, title, style = 2) => {
      if (num(price) && num(price) > 0) levelLines.push(candles.createPriceLine({ price: +price, color, lineWidth: 1, lineStyle: style, axisLabelVisible: true, title }));
    };
    if (n.rec !== 'HOLD') {
      add(n.risk.stop_loss, '#ff4d5e', 'STOP');
      add(n.risk.take_profit, '#1fd67f', 'TARGET');
      add(n.price, '#ff9f1a', `${n.rec} ENTRY`, 0);
    }
  }

  function renderLegend(param) {
    const el = $('chart-legend');
    let bar = null;
    if (param && param.time !== undefined && param.seriesData) {
      const c = param.seriesData.get(candles);
      const v = param.seriesData.get(volumeSeries);
      if (c) bar = { ...c, volume: v?.value };
    }
    if (!bar && S.bars.length) bar = S.bars[S.bars.length - 1];
    if (!bar) { el.innerHTML = ''; return; }
    const d = bar.close >= bar.open ? 'up' : 'down';
    const src = S.barsSource ? ` · ${S.barsSource === 'demo' ? 'SIMULATED' : S.barsSource.toUpperCase()}` : '';
    el.innerHTML = `<span>${esc(S.symbol)} · ${TF_LABEL[S.tf]}${esc(src)}</span>
      <span>O <b>${px(bar.open)}</b></span><span>H <b>${px(bar.high)}</b></span>
      <span>L <b>${px(bar.low)}</b></span><span>C <b class="${d}">${px(bar.close)}</b></span>
      <span>V <b>${vol(bar.volume)}</b></span>
      ${S.indicators.sma ? '<span style="color:#ff9f1a">SMA20</span>' : ''}
      ${S.indicators.ema ? '<span style="color:#4aa8ff">EMA9</span>' : ''}
      ${S.indicators.vwap && S.tf !== '1Day' ? '<span style="color:#b48cff">VWAP</span>' : ''}`;
  }

  // Fold each streaming quote into the forming bar so the chart ticks live.
  function liveCandle(q) {
    if (!chart || !S.bars.length || !num(q.last)) return;
    const period = TF_SECONDS[S.tf];
    const last = S.bars[S.bars.length - 1];
    const ts = q.timestamp || Math.floor(Date.now() / 1000);
    if (ts < last.time) return;
    // Only open new bars during the session; after hours, polled quotes just refine the last bar.
    const open = S.config.demo_mode || (S.clock ? S.clock.is_open : localMarketOpen(new Date()));
    const start = S.tf === '1Day' || !open ? last.time : Math.floor(ts / period) * period;
    const price = +q.last;
    let bar;
    if (start > last.time) {
      bar = { time: start, open: last.close, high: Math.max(last.close, price), low: Math.min(last.close, price), close: price, volume: 0 };
      S.bars.push(bar);
      if (S.bars.length > 1000) S.bars.shift();
    } else {
      bar = last;
      bar.close = price; bar.high = Math.max(bar.high, price); bar.low = Math.min(bar.low, price);
    }
    candles.update({ time: chartTime(bar.time), open: bar.open, high: bar.high, low: bar.low, close: bar.close });
    volumeSeries.update({ time: chartTime(bar.time), value: bar.volume, color: bar.close >= bar.open ? '#1fd67f40' : '#ff4d5e40' });
    renderLegend();
  }

  /* ═══ Agent signal: the news bot's (or JUDGE's) latest call ═══ */
  function normalizeDecision(d) {
    return {
      rec: d.call === 'BUY' ? 'BUY' : d.call === 'BEARISH' ? 'SELL' : 'HOLD',
      call: d.call || 'NONE',
      conf: num(d.probability) ?? 0,
      reason: d.reason || '',
      price: num(d.price) ?? num(S.quotes[d.symbol]?.last),
      risk: { stop_loss: d.stop, take_profit: d.target },
      ts: d.timestamp,
    };
  }

  function renderSignalEmpty() {
    $('sig-time').textContent = '';
    $('signal-body').innerHTML = '<div class="empty">No call from the news bot on this symbol yet. Press JUDGE (or type <kbd>SYM JG</kbd>) to have Jev judge its recent headlines with the bot\'s rules.</div>';
  }

  function renderAnalyzing() {
    $('signal-body').innerHTML = `<div class="analyzing"><div class="spin"></div>Jev judging recent ${esc(S.symbol)} headlines…</div>`;
  }

  const pct = (v) => (num(v) === null ? '—' : `${Math.round(num(v) * 100)}%`);

  function renderSignal(d) {
    const n = normalizeDecision(d);
    const who = d.source === 'judge' ? 'JUDGE' : 'NEWS BOT';
    $('sig-time').textContent = `${who}${n.ts ? ` @ ${etTime(n.ts)} ET` : ''}`;
    const rows = (d.headlines || []).map((h) => `
      <div class="strat ${h.action === 'buy' ? 'BUY' : h.action === 'bearish' ? 'SELL' : 'HOLD'}">
        <div class="strat-h"><b>${esc((h.event_type || 'news').toUpperCase())}</b>
          <span class="${h.action === 'buy' ? 'up' : h.action === 'bearish' ? 'down' : 'muted'}">${esc((h.action || 'none').toUpperCase())}</span>
          <div class="meter"><i style="width:${Math.round((num(h.p_bullish) ?? 0) * 100)}%"></i></div>
          <span class="muted">↑${pct(h.p_bullish)} ↓${pct(h.p_bearish)}</span></div>
        <div class="strat-r" title="${esc(h.reason)}">${esc(h.headline || '—')}</div>
        <div class="strat-r muted">rel ${pct(h.relevance)} · mat ${pct(h.materiality)}${h.published_at ? ` · ${ago(h.published_at)} ago` : ''} · ${esc(h.blocked || h.reason || '')}</div>
      </div>`).join('');
    $('signal-body').innerHTML = `
      <div class="sig">
        <div class="sig-call">
          <div class="sig-rec ${esc(n.rec)}">${esc(n.call)}</div>
          <div class="sig-meta">P <b>${pct(n.conf)}</b>${d.jev_ms !== undefined && d.jev_ms !== null ? ` · JEV <b>${d.jev_ms}ms</b>` : ''}${d.execution ? ` · BOT <b>${esc(d.execution.toUpperCase())}</b>` : ''}</div>
          <div class="meter"><i style="width:${Math.round(n.conf * 100)}%"></i></div>
          <div class="sig-reason">${esc(n.reason)}</div>
          ${d.note ? `<div class="sig-meta muted">${esc(d.note)}</div>` : ''}
        </div>
        <div class="strats">${rows || '<div class="empty">No headlines judged.</div>'}</div>
        <div class="kv">
          <span>ENTRY</span><b>${px(n.price)}</b>
          <span>STOP</span><b class="down">${px(n.risk.stop_loss)}</b>
          <span>TARGET</span><b class="up">${px(n.risk.take_profit)}</b>
          <span>ORDER</span><b>${d.order_id ? esc(String(d.order_id).slice(0, 8)) : '—'}</b>
        </div>
      </div>`;
  }

  async function analyze(sym = S.symbol) {
    if (!sym || S.analyzing.has(sym)) return;
    S.analyzing.add(sym);
    if (sym === S.symbol) { renderAnalyzing(); $('analyze-btn').disabled = true; }
    try {
      S.decisions[sym] = await API.judgeSymbol(sym);
    } catch (err) {
      toast(`${sym}: ${err.message}`, 4000);
    } finally {
      S.analyzing.delete(sym);
      if (sym === S.symbol) {
        $('analyze-btn').disabled = false;
        if (S.decisions[sym]) renderSignal(S.decisions[sym]); else renderSignalEmpty();
        drawLevels();
      }
    }
  }
  $('analyze-btn').addEventListener('click', () => analyze());

  /* ═══ Activity feed ═══ */
  function onEvent(ev) {
    S.events.unshift(ev);
    if (S.events.length > MAX_EVENTS) S.events.length = MAX_EVENTS;
    if (ev.data?.signal && ev.symbol && !S.analyzing.has(ev.symbol)) {
      S.decisions[ev.symbol] = ev.data.signal;
      if (ev.symbol === S.symbol) { renderSignal(ev.data.signal); drawLevels(); }
    }
    if (EVENT_GROUP[ev.type] === 'order') refreshAccountSoon();
    if (matchesFilter(ev)) {
      $('feed').insertAdjacentHTML('afterbegin', eventHtml(ev, true));
      const rows = $('feed').children;
      while (rows.length > MAX_EVENTS) rows[rows.length - 1].remove();
    }
  }

  const matchesFilter = (ev) => S.feedFilter === 'all' || (EVENT_GROUP[ev.type] || 'system') === S.feedFilter;

  function eventHtml(ev, fresh) {
    return `<div class="ev lv-${esc(ev.level)} ${fresh ? 'new' : ''}">
      <span class="ev-t">${etTime(ev.timestamp)}</span>
      <span class="ev-s" data-sym="${esc(ev.symbol || '')}">${esc(ev.symbol || '')}</span>
      <span class="ev-m"><span class="ev-tag">${esc(EVENT_TAG[ev.type] || ev.type.toUpperCase())}</span>${esc(ev.message)}</span>
    </div>`;
  }

  function renderFeed() {
    const list = S.events.filter(matchesFilter);
    $('feed').innerHTML = list.length ? list.map((e) => eventHtml(e, false)).join('')
      : '<div class="empty">Waiting for agent activity… the news bot\'s signals and orders, Claude\'s decisions and guard reviews appear here as they are journaled.</div>';
  }

  $('feed-filter').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-f]'); if (!b) return;
    S.feedFilter = b.dataset.f;
    $('feed-filter').querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
    renderFeed();
  });
  $('feed').addEventListener('click', (e) => {
    const s = e.target.closest('.ev-s'); if (s && s.dataset.sym) loadSymbol(s.dataset.sym);
  });

  /* ═══ News ═══ */
  let newsReq = 0;
  async function loadNews() {
    const req = ++newsReq, sym = S.symbol;
    $('news').innerHTML = '<div class="empty">Loading…</div>';
    try {
      const items = await API.getDeskNews(sym);
      if (req !== newsReq) return;
      $('news').innerHTML = items.length ? items.map((n) => `
        <div class="news-item">
          <a href="${esc(/^https?:\/\//i.test(n.url || '') ? n.url : '#')}" target="_blank" rel="noopener noreferrer">${esc(n.title)}</a>
          <div class="news-meta"><b>${esc(n.source || '')}</b> · ${n.published_at ? ago(n.published_at) + ' ago' : ''}${n.symbols?.length ? ' · ' + esc(n.symbols.slice(0, 5).join(' ')) : ''}</div>
        </div>`).join('') : `<div class="empty">No news for ${esc(sym)} in the last 24h.</div>`;
    } catch (err) {
      if (req === newsReq) $('news').innerHTML = `<div class="empty">News unavailable: ${esc(err.message)}</div>`;
    }
  }
  setInterval(() => { if (S.symbol && !document.hidden) loadNews(); }, 120000);

  /* ═══ Account / positions / orders ═══ */
  async function refreshAccount() {
    if (!S.config?.alpaca_configured || S.config.demo_mode) { setAccount({ connected: false, message: S.config?.demo_mode ? 'Demo mode' : 'Alpaca not configured' }); return; }
    try { setAccount(await API.getDeskAccount()); } catch { /* stream will retry */ }
  }
  let accountTimer = null;
  function refreshAccountSoon() { clearTimeout(accountTimer); accountTimer = setTimeout(refreshAccount, 1200); }

  function setAccount(snap) {
    const hadPositions = new Set((S.account?.positions || []).map((p) => p.symbol));
    S.account = snap;
    if (!snap?.connected) {
      $('t-equity').textContent = '—'; $('t-daypl').textContent = '—'; $('t-bp').textContent = '—';
      $('pos-body').innerHTML = `<tr><td colspan="6" class="empty">${esc(snap?.message || '—')}</td></tr>`;
      $('orders-body').innerHTML = `<tr><td colspan="8" class="empty">${esc(snap?.message || '—')}</td></tr>`;
      return;
    }
    const a = snap.account;
    $('t-equity').textContent = money(a.equity);
    $('t-daypl').innerHTML = `<span class="${dir(a.day_pl)}">${money(a.day_pl, true)} ${a.day_pl_pct !== null ? `(${signed(a.day_pl_pct, 2, '%')})` : ''}</span>`;
    $('t-bp').textContent = money(a.buying_power);

    const pos = snap.positions || [];
    let upl = 0;
    $('pos-body').innerHTML = pos.length ? pos.map((p) => {
      upl += p.unrealized_pl || 0;
      return `<tr data-sym="${esc(p.symbol)}" style="cursor:pointer">
        <td style="color:var(--amber);font-weight:700">${esc(p.symbol)}</td><td class="r">${p.qty}</td>
        <td class="r">${px(p.avg_entry_price)}</td><td class="r">${px(p.current_price)}</td>
        <td class="r ${dir(p.unrealized_pl)}">${money(p.unrealized_pl, true)}</td>
        <td class="r ${dir(p.unrealized_plpc)}">${signed(p.unrealized_plpc, 2, '%')}</td></tr>`;
    }).join('') : '<tr><td colspan="6" class="empty">Flat — no open positions</td></tr>';
    $('pos-upl').innerHTML = pos.length ? `U P&amp;L <span class="${dir(upl)}">${money(upl, true)}</span>` : '';

    const orders = snap.orders || [];
    $('orders-body').innerHTML = orders.length ? orders.map((o) => {
      const open = ['new', 'accepted', 'pending_new', 'partially_filled', 'held'].includes(o.status);
      return `<tr>
        <td class="muted">${o.submitted_at ? etTime(o.submitted_at) : '—'}</td>
        <td style="color:var(--amber);font-weight:700">${esc(o.symbol)}</td>
        <td class="side-${esc(o.side)}">${esc((o.side || '').toUpperCase())}</td>
        <td class="r">${o.qty ?? '—'}</td><td class="r">${o.filled_qty ?? 0}</td>
        <td class="r">${px(o.filled_avg_price ?? o.limit_price)}</td>
        <td><span class="st st-${esc(o.status)}">${esc((o.status || '').toUpperCase())}</span></td>
        <td class="r">${open ? `<span class="cancel" data-id="${esc(o.id)}" title="Cancel">✕</span>` : ''}</td></tr>`;
    }).join('') : '<tr><td colspan="8" class="empty">No orders</td></tr>';

    if (snap.clock) S.clock = snap.clock;
    const now = new Set(pos.map((p) => p.symbol));
    if ([...now].some((s) => !hadPositions.has(s))) sendWatch();
  }

  $('pos-body').addEventListener('click', (e) => { const r = e.target.closest('tr[data-sym]'); if (r) loadSymbol(r.dataset.sym); });
  $('orders-body').addEventListener('click', async (e) => {
    const c = e.target.closest('.cancel'); if (!c) return;
    if (!confirm('Cancel this paper order?')) return;
    try { await API.cancelOrder(c.dataset.id); refreshAccountSoon(); } catch (err) { toast(err.message); }
  });

  /* ═══ Clock / market status ═══ */
  function localMarketOpen(now) {
    const et = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
    const day = et.getDay(), mins = et.getHours() * 60 + et.getMinutes();
    return day > 0 && day < 6 && mins >= 570 && mins < 960;
  }
  function countdown(iso) {
    const s = Math.max(0, Math.round((new Date(iso) - Date.now()) / 1000));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return h > 47 ? `${Math.floor(h / 24)}d` : h ? `${h}h${String(m).padStart(2, '0')}m` : `${m}m${String(s % 60).padStart(2, '0')}s`;
  }
  function tick() {
    const now = new Date();
    $('t-clock').textContent = etTime(now);
    const c = S.clock;
    const open = c ? c.is_open : localMarketOpen(now);
    let text = open ? 'OPEN' : 'CLOSED';
    if (c) text += open ? ` · ${countdown(c.next_close)}` : ` · ${countdown(c.next_open)}`;
    $('t-mkt').innerHTML = `<span class="${open ? 'up' : 'down'}">${text}</span>`;
    $('t-mkt-label').textContent = c ? (open ? 'MARKET · CLOSES' : 'MARKET · OPENS') : 'MARKET (EST.)';
    renderNewsbot();
  }
  setInterval(tick, 1000);

  /* ═══ Order ticket ═══ */
  function setSide(side) {
    S.side = side;
    document.querySelectorAll('.side-toggle button').forEach((b) => b.classList.toggle('on', b.dataset.side === side));
    const btn = $('tk-submit');
    btn.textContent = `SEND ${side}`;
    btn.className = `btn ${side === 'BUY' ? 'btn-buy' : 'btn-sell'}`;
  }
  document.querySelector('.side-toggle').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-side]'); if (b) setSide(b.dataset.side);
  });
  function updateTicketEst() {
    const q = S.quotes[$('tk-sym').value.trim().toUpperCase()];
    const qty = +$('tk-qty').value || 0;
    $('tk-est').textContent = q?.last ? `${money(q.last * qty)} @ ${px(q.last)}` : '—';
  }
  $('tk-qty').addEventListener('input', updateTicketEst);
  $('tk-sym').addEventListener('input', updateTicketEst);

  $('ticket').addEventListener('submit', async (e) => {
    e.preventDefault();
    const sym = $('tk-sym').value.trim().toUpperCase();
    const qty = parseInt($('tk-qty').value, 10);
    const msg = $('tk-msg');
    if (!S.config.alpaca_configured) { msg.innerHTML = '<span class="down">Alpaca keys not configured.</span>'; return; }
    if (!/^[A-Z]+$/.test(sym) || !(qty > 0)) { msg.innerHTML = '<span class="down">Enter a symbol and quantity.</span>'; return; }
    if (!confirm(`Send PAPER ${S.side} ${qty} ${sym} at market?`)) return;
    $('tk-submit').disabled = true;
    msg.textContent = 'Routing…';
    try {
      const res = await API.submitPaperOrder(sym, S.side, qty);
      msg.innerHTML = res.submitted
        ? `<span class="up">✓ ${esc(res.order?.status || 'submitted').toUpperCase()} · ${esc((res.order?.id || '').slice(0, 8))}</span>`
        : `<span class="down">${esc(res.skipped_reason || 'Not submitted')}</span>`;
      refreshAccountSoon();
    } catch (err) {
      msg.innerHTML = `<span class="down">${esc(err.message)}</span>`;
    } finally {
      $('tk-submit').disabled = false;
    }
  });

  /* ═══ News bot (runs on its own; the desk only shows it) ═══ */
  const BOT_ACTIVE_MS = 10 * 60 * 1000;

  function setNewsbot(status) {
    S.newsbot = status || null;
    renderNewsbot();
  }

  async function refreshNewsbot() {
    try { setNewsbot(await API.getNewsbot()); } catch (err) { toast(`News bot: ${err.message}`); }
  }

  function renderNewsbot() {
    const b = S.newsbot;
    if (!b) return;
    const active = b.last_activity && Date.now() - new Date(b.last_activity).getTime() < BOT_ACTIVE_MS;
    $('ap-badge').hidden = !active;
    const mode = (b.execution || '—').toUpperCase();
    $('ap-state').innerHTML = b.execution === 'paper' ? `<span class="up">${mode}</span>` : mode;
    const st = b.settings || {};
    const universe = Array.isArray(b.universe) ? b.universe.join(',') : (b.universe || '—').toUpperCase();
    $('ap-status').innerHTML = `
      <div class="kv">
        <span>MODE</span><b>${b.execution === 'paper' ? 'ALPACA PAPER ORDERS' : 'SIGNALS ONLY'}</b>
        <span>STOCKS</span><b title="${esc(universe)}">${esc(universe.length > 28 ? universe.slice(0, 28) + '…' : universe)}</b>
        <span>TRADES TODAY</span><b>${b.orders_today ?? 0} / ${b.max_trades_per_day ?? '—'}</b>
        <span>SIGNALS TODAY</span><b>${b.signals_today ?? 0}</b>
        <span>LAST ACTIVITY</span><b>${b.last_activity ? `${ago(b.last_activity)} ago` : 'none yet'}</b>
        <span>JEV LATENCY</span><b>${b.median_jev_ms !== null && b.median_jev_ms !== undefined ? `${b.median_jev_ms}ms med · ${b.max_jev_ms}ms max` : '—'}</b>
        <span>THRESHOLDS</span><b>rel ${st.min_relevance ?? '—'} · mat ${st.min_materiality ?? '—'} · p ${st.min_probability ?? '—'}</b>
        <span>PER TRADE</span><b>${money(st.order_usd)} · TP ${st.take_profit_pct ?? '—'}% · SL ${st.stop_loss_pct ?? '—'}%</b>
        <span>COOLDOWN</span><b>${st.cooldown_minutes ?? '—'} min · cutoff ${st.entry_cutoff_minutes ?? '—'} min</b>
      </div>
      <div class="muted small">${active ? '' : 'No bot activity in the last 10 min. '}The bot runs on its own (python -m trader.newsbot run); change its rules in .env.</div>`;
  }

  /* ═══ Command line ═══ */
  function runCommand(raw) {
    const parts = raw.trim().toUpperCase().replace(/<GO>/g, '').split(/\s+/).filter(Boolean);
    if (!parts.length) return;
    const [a, b] = parts;
    const tfAlias = { '1M': '1Min', '5M': '5Min', '15M': '15Min', '1H': '1Hour', '1D': '1Day' };
    if (a === 'HELP' || a === '?') { $('help').hidden = false; return; }
    if (tfAlias[a]) return setTimeframe(tfAlias[a]);
    if (['AN', 'ANALYZE', 'JG', 'JUDGE'].includes(a)) return analyze(b || S.symbol);
    if (a === 'BOT') return refreshNewsbot();
    if (a === 'ADD' && b) return addWatch(b);
    if ((a === 'DEL' || a === 'RM') && b) return removeWatch(b);
    if ((a === 'BUY' || a === 'SELL') && b) {
      setSide(a);
      $('tk-qty').value = parseInt(b, 10) || 1;
      if (parts[2]) $('tk-sym').value = parts[2];
      updateTicketEst();
      $('tk-submit').focus();
      return;
    }
    loadSymbol(a);
    if (['AN', 'ANALYZE', 'JG', 'JUDGE'].includes(b)) analyze(a);
  }

  $('cmd-form').addEventListener('submit', (e) => {
    e.preventDefault();
    runCommand($('cmd').value);
    $('cmd').value = '';
  });
  $('help-close').addEventListener('click', () => { $('help').hidden = true; });
  $('help').addEventListener('click', (e) => { if (e.target === $('help')) $('help').hidden = true; });

  document.addEventListener('keydown', (e) => {
    const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement?.tagName);
    if (e.key === 'Escape') { $('help').hidden = true; document.activeElement?.blur(); return; }
    if (typing || $('desk').hidden) return;
    if (e.key === '/') { e.preventDefault(); $('cmd').focus(); return; }
    if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') && S.watchlist.length) {
      e.preventDefault();
      const i = S.watchlist.indexOf(S.symbol);
      const next = e.key === 'ArrowDown' ? Math.min(S.watchlist.length - 1, i + 1) : Math.max(0, i - 1);
      loadSymbol(S.watchlist[next]);
      return;
    }
    // Any letter starts a command, Bloomberg-style.
    if (/^[a-zA-Z]$/.test(e.key) && !e.metaKey && !e.ctrlKey && !e.altKey) $('cmd').focus();
  });

  /* ═══ Go ═══ */
  setSide('BUY');
  boot();
})();
