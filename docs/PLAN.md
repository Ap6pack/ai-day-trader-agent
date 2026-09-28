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

## Phase 3: real time

- A small always-on watcher: stream Alpaca news for the watchlist, score each headline
  with Jev as it arrives, and only when one is relevant and material start a Claude Code
  session (`claude -p` with the day-trader skill) to analyze it, or notify you.
- Needs working Alpaca keys.

## Phase 4: live trading with small limits

- Switch `TRADER_MODE=live` only after Phase 2 shows an edge, with low
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
