# AI Day Trader

An intraday trading assistant built on Claude Code. Claude does the analysis and
operates your Robinhood account through Robinhood's MCP tools; this repository adds
what those tools do not provide:

- **An order guard** that Claude cannot skip. A Claude Code hook checks every Robinhood
  order against hard limits before it reaches the permission prompt, and blocks all
  orders while the project is in review-only mode.
- **Jev news judgments.** Each headline is scored by TypeSafe's Jev model for relevance,
  direction, materiality and event type, then aggregated with recency weighting.
- **A trade journal** that records every decision (including passes) with its thesis and
  entry price, then scores it against the day's close, so you can see whether the
  strategy works before risking money.
- **A playbook** (`.claude/skills/day-trader`) that tells Claude how to analyze, size,
  review and journal a trade.

The previous standalone version (multi-provider data fetching, FastAPI server, web
dashboard, dividend capture engine) is preserved under the `v1-legacy` tag.

## How it fits together

```
you ──► Claude Code ──► Robinhood MCP tools (quotes, indicators, account, orders)
             │               ▲
             │               └── .claude/hooks/order_guard.sh ─► trader.guard
             │                   (runs before every order; fails closed)
             ├──► python -m trader.news    (Jev headline judgments)
             ├──► python -m trader.sizing  (quantity and stop from account figures)
             └──► python -m trader.journal (decisions, outcomes, hit rate)
```

## Setup

```bash
git clone https://github.com/Ap6pack/ai-day-trader-agent.git
cd ai-day-trader-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then add your keys; leave TRADER_MODE=review
```

Start Claude Code in the project directory (`claude`). When it asks about the
`robinhood-trading` MCP server from `.mcp.json`, enable it for this project and sign in
to Robinhood. Order placement needs a Robinhood account with agentic trading enabled.

Check the setup without exposing secrets:

```bash
python -m trader.status
```

Then ask Claude, for example: *"Scan my watchlist for day-trade setups"* or
*"Analyze NVDA for a day trade"*. The day-trader skill guides the rest.

## Review-only mode first

`TRADER_MODE=review` is the default. In this mode the guard blocks every
`place_*_order` call; Claude reviews the order with `review_equity_order` and records
what it would have done. Score those decisions each day
(`python -m trader.journal pending`, then `outcome`) and check
`python -m trader.journal summary`. Switch `.env` to `TRADER_MODE=live` yourself, and
only once the record justifies it.

## The order guard

Configured in `.claude/settings.json`; logic in `trader/guard.py`. In live mode an
order passes only if:

- it is an equity order (options and crypto are off unless enabled),
- the symbol is on `TRADER_ALLOWED_SYMBOLS` when that list is set,
- it is sizeable from its own fields (a limit or stop price with a quantity, or a
  dollar amount), and a buy is at most `TRADER_MAX_ORDER_USD`,
- fewer than `TRADER_MAX_ORDERS_PER_DAY` orders passed today, and
- the same symbol and side were reviewed within `TRADER_REVIEW_WINDOW_MINUTES`.

Passing the guard does not approve an order: Claude Code still asks you. Any error in
the guard blocks the order. The project settings also stop Claude's file tools from
reading `.env` or editing the guard's configuration.

**Limits of this protection:** hooks run only in Claude Code. Robinhood tools used from
Claude Desktop or claude.ai chat are not guarded by this project. Use Claude Code for
anything that can place an order.

## Commands

| Command | Purpose |
|---|---|
| `python -m trader.status` | Mode, limits, credential status |
| `python -m trader.news SYMBOL [--json] [--sample]` | Jev judgments of recent headlines |
| `python -m trader.sizing --symbol S --side buy --price P --equity E [...]` | Max quantity and stop-loss |
| `python -m trader.journal decide ...` / `list` / `events` / `pending` / `outcome` / `summary` | Journal |

## Tests

```bash
python -m pytest
```

## Plan

See [docs/PLAN.md](docs/PLAN.md) for what comes next.

## Disclaimer

Educational software. Trading involves risk of loss. You are responsible for every
order placed from your account.
