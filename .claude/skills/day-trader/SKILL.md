---
name: day-trader
description: Intraday stock analysis and trading with the Robinhood tools, Jev news judgments and the project's trade journal. Use when the user asks to scan for trades, analyze a ticker for a day trade, review or place an order, check positions, or score past decisions. Covers review-only mode, sizing, the order guard and journaling.
---

# AI day trader playbook

You are the analyst and operator. Robinhood tools provide account data, market
data and order entry. This project adds three things Robinhood does not:

| Command | What it does |
|---|---|
| `python -m trader.status` | Mode (`review` or `live`), hard limits, credential status. Never read `.env` directly. |
| `python -m trader.scan` | Today's candidates: the user's watchlist plus Alpaca's most-active stocks and movers, ranked with Jev (fresh material news first). |
| `python -m trader.news SYMBOL --json` | Jev's judgment of each recent headline: relevance, direction, materiality, event type, plus a weighted score. |
| `python -m trader.sizing ...` | Largest quantity that fits the percentage limits, and a suggested stop-loss. |
| `python -m trader.journal ...` | Records every decision with its reasoning; scores it later. |

Run them from the project root with the project's virtualenv (`.venv/bin/python -m ...`
if `python` is not the venv's).

## Non-negotiables

- **Start every session with `python -m trader.status`.** In `review` mode no order
  will ever be placed: the order guard blocks `place_*_order`. Do not try.
- **Never change the mode or limits yourself.** Do not edit `.env`,
  `.claude/settings.json` or `.claude/hooks/`, and do not work around the guard (other
  tools, other accounts, scripts). If a limit blocks a trade, tell the user which one.
- **Real money.** Robinhood has no paper trading; `place_equity_order` fills with real
  money. The account number must come from the user; never pick one from
  `get_accounts` on your own.
- **Every order is a limit order or a dollar-amount order**, reviewed first with
  `review_equity_order`. The guard rejects market orders by share count and any order
  without a matching review in the last few minutes.
- **Never invent numbers.** Quantities, prices and account figures come from the user or
  from tool results.
- **Journal every decision, including passes.** A strategy that is not journaled cannot
  be evaluated.

## Workflow

### 1. Session start
1. `python -m trader.status`: note the mode, limits and any missing credentials.
2. Ask which Robinhood account to use if the user has not said.
3. `get_portfolio` (equity, buying power) and `get_equity_positions` (what is held).
4. `python -m trader.journal pending`: if earlier decisions are unscored, score them
   first (step 6).

### 2. Build the candidate list
Use what the user names. Otherwise run `python -m trader.scan` and work down its shortlist
(`material_news: true` means Jev found a relevant, material headline in the last 24 hours).
Robinhood watchlists (`get_watchlists`, `get_watchlist_items`) and saved scans
(`get_scans`, `run_scan`) are other sources if the user has them. Check
`get_earnings_calendar` for today: earnings names move on news, not on technicals.

### 3. Analyze one symbol
Gather, then judge. Keep observed facts separate from your interpretation.

- **Price now:** `get_equity_quotes` (bid, ask, last, volume).
- **Intraday structure:** `get_equity_historicals` with `interval="5minute"` from today's
  open; also the previous day's high, low and close.
- **Indicators** (`get_equity_technical_indicators`, `interval="5minute"`, `output="latest"`
  unless you need the series): `vwap`, `rsi` (14), `macd`, `atr` (14) for stop distance.
  Daily context: `sma` 20 and 50 on `interval="day"`.
- **News:** `python -m trader.news SYMBOL --json`. Weight articles by their `weight`; a
  score near 0 with low `evidence_weight` means "no material news", not "neutral news".
  If no news source is configured, say so rather than assuming there is no news.
- **Context when relevant:** `get_equity_analyst_ratings`, `get_equity_fundamentals`,
  `get_earnings_results`, `get_equity_tradability` (halts, restrictions).

### 4. Decide
Starting rules. They are hypotheses for the journal to test, not proven edges:

- Trade only with at least two independent confirmations, for example trend
  (price vs VWAP and the 20-day SMA), momentum (MACD, RSI not stretched against the
  trade) and catalyst (material, relevant news pointing the same way).
- Skip the first 5 minutes after the open and do not open new positions in the last
  15 minutes of the session. Plan to be flat by the close unless the user says otherwise.
- Place the stop at a level that invalidates the idea (below VWAP or the setup's low),
  roughly 1 to 1.5 ATR away, never wider than `sizing.stop_loss_pct`.
- Conflicting signals, thin volume, a halt or a pending earnings release all mean pass.
- State the decision as: action, entry, stop, target, confidence 0 to 1, and a one or
  two sentence thesis.

### 5. Size, review, record, (maybe) place
1. `python -m trader.sizing --symbol SYM --side buy --price ENTRY --equity EQUITY
   --buying-power BP --position-value CURRENT_VALUE --quantity Q`, using figures from
   step 1. Use its `max_quantity` if your proposal exceeds it.
2. `review_equity_order` with `type="limit"` (a marketable limit at the ask to enter now)
   and the quantity, `time_in_force="gfd"`. Report any pre-trade alerts it returns.
3. Record the decision (buy, sell or pass) at the entry price:
   `python -m trader.journal decide --symbol SYM --action buy --price ENTRY --confidence 0.6 --thesis "..."`
4. **Review mode:** stop here. Tell the user what would have been placed.
   **Live mode:** show the user the reviewed order and ask for confirmation. Then call
   `place_equity_order` with the same parameters and a fresh UUID `ref_id`. The guard
   runs first; the user still approves the tool call. After a fill, offer to place the
   protective stop as a separate `stop_market` or `stop_limit` sell, also reviewed first.

### 6. Score past decisions
For each row from `python -m trader.journal pending`, get the closing price of that
decision's trading day (`get_equity_historicals`, `interval="day"`) and record it:
`python -m trader.journal outcome --id ID --price CLOSE`.
`python -m trader.journal summary` reports hit rate and average signed return. Only
suggest moving from review to live mode when the user asks, and show them the summary
when they do: a small sample proves little.

## Scheduled runs (autopilot)

`python -m trader.autopilot` starts you unattended (`claude -p`) for `score`, `review` and
the optional shortlist `run`. Nobody can answer questions in those runs: put anything that
needs the user in the final report. The account number in a run prompt comes from the
user's configuration. Never place orders in an autopilot run. The news bot
(`trader.newsbot`) does its own Alpaca paper trading; your job is to measure and tune it:
report results by event type and probability bucket, and propose `NEWSBOT_*` changes with
their evidence and sample size. Never edit `.env` yourself.

## When something fails
- Guard denial: quote the reason and stop. Do not retry the same order in another form.
- `trader.news` errors or has no source: continue without news and say so.
- Robinhood tool errors: report them. Do not fall back to another broker or account.
