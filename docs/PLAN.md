# Plan

The tool is being rebuilt around Claude Code and Robinhood's MCP tools. Robinhood
covers market data, indicators, account data and order entry, so this repository keeps
only what Robinhood does not provide: enforced limits, cheap always-on judgment (Jev),
a playbook, and a record of results. The old standalone version is tagged `v1-legacy`.

## Phase 1: safe foundation (done)

- Order guard hook: review-only mode by default; in live mode a dollar cap on buys, a
  daily order cap, a review before every order, options and crypto off. Fails closed.
- Jev headline judgments with Alpaca news (NewsAPI fallback).
- Sizing helper from Robinhood account figures.
- Trade journal with decisions, outcomes and a summary.
- Day-trader skill: the analysis, sizing, review and journaling workflow.

## Phase 2: run in review-only mode and measure

- Use the skill on real sessions in review mode; journal every decision, including passes.
- Score each day's decisions at the close. After enough decisions to mean something
  (for example 50 or more across at least 20 trading days), compare the hit rate and
  average return with simple baselines such as holding the same symbols for the day.
- Improve scoring: judge each trade against its own stop and target within the day
  (from intraday bars), not only against the close.

Done so far for automation: `python -m trader.scan` finds candidates on its own
(watchlist plus Alpaca's most-active and movers screeners, ranked with Jev).

## Phase 3: autonomous news trading on Alpaca paper

Goal: trade news without a person in the loop. Jev makes each per-headline decision in
milliseconds; plain code applies the rules and places the order; Claude is the brain
that reviews results and tunes the rules, not a step in the trade path.

```
Alpaca news stream ─► trader.newsbot ─► Jev (one request per headline, ~300 ms)
                          │                    │ relevance, direction probabilities,
                          │                    │ materiality, event type
                          ▼                    ▼
                    rules in code ──► Alpaca paper bracket order (entry + stop + target)
                          │
                          └──► journal (every signal, order, latency)
                                    │
     after the close: Claude (claude -p, day-trader skill) scores outcomes, reads the
     journal, and writes a report proposing threshold changes for the user to apply
```

### Scope and safety rules

- **No approval step.** The bot places orders on its own.
- **Alpaca paper trading only.** Order functions refuse to run unless
  `ALPACA_TRADING_BASE_URL` is `https://paper-api.alpaca.markets`. Live execution is out
  of scope until Phase 2/3 results justify it (see Phase 4). Robinhood cannot be driven
  this way: its orders only go through Claude's MCP tools.
- **Execution is opt-in:** `NEWSBOT_EXECUTION=off` (default: signals are journaled only)
  or `paper`.
- **Hard limits in code**, all from `.env`: dollar size per trade, trades per day,
  per-symbol cooldown, minimum price, bracket stop-loss and take-profit on every order,
  and an end-of-day flatten.
- **Long only.** Bearish signals are journaled (as `sell` decisions, so they are scored)
  but never shorted.
- The bot never edits `.env`; Claude only proposes threshold changes in its report.

### Status

Built: `trader/alpaca.py`, `trader/newsbot.py`, `trader/autopilot.py`, the live desk
(`trader/desk.py`, `static/desk.html`, watching all of it) and their tests.
Differences from the design below:

- `run` flattens itself `NEWSBOT_FLATTEN_MINUTES` (default 10) before the close from
  Alpaca's clock, so early-close days (13:00) are covered; the 15:50 cron `flatten` is a
  backup. `NEWSBOT_ENTRY_CUTOFF_MINUTES` (default 15, must exceed the flatten) blocks new
  entries before that. Bracket legs are day orders: anything open at the close would be
  held overnight with no stop.
- Stream errors after authentication are logged, not fatal; only auth errors stop `run`.
- `client_order_id` is a UUID derived from the news id and symbol, so a re-delivered
  headline cannot open a second order.
- `trader.scan` had no caller for its `market_clock`, so it was removed rather than
  redirected.

### Files to add

**`trader/alpaca.py`**, a small REST client (uses `alpaca_headers()` and
`alpaca_data_url()` from `trader/news_sources.py`):
- `trading_base_url()`: `ALPACA_TRADING_BASE_URL`, default paper host, trailing `/v2`
  stripped. `is_paper()` checks the hostname is `paper-api.alpaca.markets`.
- `market_clock()`: `GET {trading}/v2/clock` returns `{is_open, next_open, next_close}`.
- `latest_price(symbol)`: `GET {data}/v2/stocks/{symbol}/trades/latest?feed={ALPACA_DATA_FEED or iex}`,
  price at `trade.p`.
- `bracket_order_payload(symbol, qty, entry_price, take_profit_pct, stop_loss_pct)`:
  `{symbol, qty, side: "buy", type: "market", time_in_force: "day", order_class: "bracket",
  take_profit: {limit_price}, stop_loss: {stop_price}, client_order_id: "newsbot-<uuid>"}`.
- `submit_order(payload)`: `POST {trading}/v2/orders`; `close_all_positions()`:
  `DELETE {trading}/v2/positions?cancel_orders=true`; `positions()`. All three raise
  unless `is_paper()`.
- `trader/scan.py` should then use `alpaca.market_clock()` instead of its own copy.

**`trader/newsbot.py`**:
- `Settings` from `.env`:

  | Variable | Default | Meaning |
  |---|---|---|
  | `NEWSBOT_EXECUTION` | `off` | `off` or `paper` |
  | `NEWSBOT_SYMBOLS` | empty (all) | `auto` (today's in-play universe from `trader.scan`, refreshed every `NEWSBOT_UNIVERSE_REFRESH_MINUTES`), a fixed list, or empty for all news |
  | `NEWSBOT_MIN_RELEVANCE` | 0.8 | Jev "is this about the company" probability |
  | `NEWSBOT_MIN_MATERIALITY` | 0.6 | Jev materiality score, 0 to 1 |
  | `NEWSBOT_MIN_PROBABILITY` | 0.75 | bullish (or bearish) probability mass needed |
  | `NEWSBOT_MAX_HEADLINE_AGE_SECONDS` | 120 | ignore older headlines |
  | `NEWSBOT_MAX_SYMBOLS_PER_HEADLINE` | 2 | skip multi-stock roundups |
  | `NEWSBOT_ORDER_USD` | 500 | size per trade; qty = floor(usd / price) |
  | `NEWSBOT_TAKE_PROFIT_PCT` | 2.0 | bracket take-profit |
  | `NEWSBOT_STOP_LOSS_PCT` | 1.0 | bracket stop-loss |
  | `NEWSBOT_MAX_TRADES_PER_DAY` | 5 | counted from the journal, US/Eastern day |
  | `NEWSBOT_COOLDOWN_MINUTES` | 30 | per symbol |
  | `NEWSBOT_MIN_PRICE` | 5 | skip penny stocks |

- `decide(symbol, article, judgment, settings, now) -> Signal`: a pure function, fully
  unit-tested. In order: judgment missing means `none`; headline older than max age means
  `none`; relevance or materiality below threshold means `none`; `p_bullish >= min_probability`
  means `buy`; `p_bearish >= min_probability` means `bearish`; otherwise `none`. The
  reason is always recorded.
- `symbols_to_judge(item_symbols, settings)`: empty if the headline is tagged with more
  than `max_symbols_per_headline` symbols, else the tagged symbols filtered by the allowlist.
- `NewsBot.handle(symbol, article, received_at)`: calls `jev_news.judge_headline()`
  (already in `trader/jev_news.py`, returns `p_bullish`/`p_bearish`), then `decide()`.
  For `buy`/`bearish` it gets `latest_price` and records a journal decision
  (`buy`/`sell`, confidence = the probability, thesis `[newsbot] <event>: <headline>`,
  mode `newsbot-<execution>`). For `buy` it checks execution mode, market open (clock
  cached for 60 s), daily cap (`journal.count_events_today("newsbot_order")`), cooldown
  (`journal.last_event_for_symbol("newsbot_order", symbol)`), minimum price and qty >= 1,
  then submits the bracket order. It logs a `newsbot_order` or `newsbot_signal` event with
  the signal, Jev latency, total latency from receipt, order id or the reason it was blocked.
- Stream: `wss://stream.data.alpaca.markets/v1beta1/news`. Send
  `{"action":"auth","key":...,"secret":...}`, wait for
  `{"T":"success","msg":"authenticated"}` (stop on `{"T":"error"}`), then
  `{"action":"subscribe","news":[symbols or "*"]}`. Messages are JSON arrays; news items
  have `T:"n"`, `id`, `headline`, `summary`, `created_at`, `url`, `symbols`, `source`.
  De-duplicate by `id` (items can be re-sent when updated). Run `handle` in a thread pool
  so a slow Jev call never blocks the stream, wrap it so exceptions are logged, and
  reconnect with backoff (`async for ws in websockets.asyncio.client.connect(...)`).
  Confirm the message formats against Alpaca's docs when building.
- CLI: `run` (stream), `replay SYMBOL [--hours 24]` (apply the rules to recent REST
  headlines, never trades or journals, for tuning thresholds), `flatten` (close all paper
  positions).
- Add `websockets` to `requirements.txt`.

**`trader/autopilot.py`**, Claude's part, run by cron:
- `score` (16:20 ET): `claude -p` with the day-trader skill, `--permission-mode dontAsk`,
  and `--allowedTools` limited to read-only Robinhood tools plus
  `Bash(python -m trader.journal *)`, to fill in outcomes for pending decisions.
- `review` (after scoring): Claude reads `journal list`/`events`, breaks results down by
  event type and probability bucket, and writes a report to `data/runs/<ts>/report.md`
  proposing threshold changes. It never edits `.env`.
- `cron`: prints crontab lines with `CRON_TZ=America/New_York`.

### Schedule (cron, US/Eastern, weekdays)

```
25 9  * * 1-5  timeout 7h python -m trader.newsbot run   >> data/newsbot.log 2>&1
50 15 * * 1-5  python -m trader.newsbot flatten           >> data/newsbot.log 2>&1
20 16 * * 1-5  python -m trader.autopilot score           >> data/autopilot.log 2>&1
40 16 * * 1-5  python -m trader.autopilot review          >> data/autopilot.log 2>&1
```

### Tests

- `decide()`: each threshold, stale headlines, bullish vs bearish vs none.
- `symbols_to_judge()`: roundups skipped, allowlist honoured.
- `alpaca`: bracket payload prices; every order function refuses a non-paper base URL.
- `NewsBot.handle()` with a fake Jev and fake Alpaca: execution off journals only;
  market closed, daily cap, cooldown, min price and tiny size each block with a reason;
  a passing signal submits exactly one bracket order and logs latency.
- Stream parsing with canned messages: auth failure stops, duplicates are ignored.

### Measure before trusting it

Latency around 400 ms is fast for a person but slow next to professional news trading,
which often reacts before retail feeds deliver the headline. Run in paper for several
weeks and compare the journal's hit rate and average return (and the bracket outcomes in
the Alpaca paper account) against doing nothing, before considering any live money.

## Phase 4: live trading with small limits

- Only if Phase 3's paper results show an edge: decide whether live news trading runs on
  Alpaca live (the same bot, with a separate explicit switch) and at what size.
- For Robinhood, switch `TRADER_MODE=live` only after Phase 2 shows an edge, with low
  `TRADER_MAX_ORDER_USD` and `TRADER_MAX_ORDERS_PER_DAY` and an allowlist.
- Add a daily loss stop: record account equity from `get_portfolio` in a PostToolUse
  hook and have the guard block new buys after a set drawdown for the day.

## Known gaps

- Hooks run only in Claude Code, so Robinhood tools used from Claude Desktop or claude.ai
  chat are not guarded.
- The guard sees only the order, so its limits are fixed dollar amounts. Percentage
  sizing is advisory (`trader.sizing`).
- News: NewsAPI's free plan is about a day late; Alpaca news needs valid Trading API keys.
- A dividend check (ex-dividend dates near a trade) could be added as a small part of the
  analysis step; the v1 dividend capture engine is not coming back as a strategy.
