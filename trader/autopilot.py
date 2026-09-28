"""
Run the day trader unattended.

    python -m trader.autopilot run      # scan, then have Claude analyze the shortlist
    python -m trader.autopilot score    # after the close: score pending decisions
    python -m trader.autopilot review   # after scoring: Claude reviews results, proposes tuning
    python -m trader.autopilot check    # show setup and which MCP servers Claude sees
    python -m trader.autopilot cron     # print crontab lines for a trading-day schedule

`run` and `score` start Claude Code non-interactively (`claude -p`) in this
project, so the order guard hook and the day-trader skill load as usual.
Claude gets only read-only Robinhood tools, review_equity_order, and the
project's own trader commands. It is never given the order-placing tools, and
in review mode the guard blocks them anyway. Reports are saved under
data/runs/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from trader import alpaca, config, scan
from trader.journal import MARKET_TZ
from trader.news_sources import alpaca_configured

RUNS_DIR = config.PROJECT_ROOT / "data" / "runs"
DEFAULT_ROBINHOOD_SERVER = "claude_ai_Robinhood"
CLAUDE_TIMEOUT_SECONDS = 20 * 60

ROBINHOOD_READ_TOOLS = (
    "get_portfolio",
    "get_equity_positions",
    "get_equity_quotes",
    "get_equity_historicals",
    "get_equity_technical_indicators",
    "get_equity_fundamentals",
    "get_equity_analyst_ratings",
    "get_equity_tradability",
    "get_earnings_calendar",
    "get_earnings_results",
    "review_equity_order",
)
# Both interpreter spellings are allowed; claude_env() puts .venv/bin first on PATH.
PYTHONS = ("python", ".venv/bin/python")
TRADER_COMMANDS = tuple(
    f"Bash({py} -m {command})"
    for py in PYTHONS
    for command in ("trader.status", "trader.news *", "trader.sizing *", "trader.journal *")
)

# Allowlist rules match one plain command at a time, so a chained or wrapped command
# (`source ... && python -m ...`) is denied in dontAsk mode.
COMMAND_RULES = """Run each project command on its own, exactly as `python -m trader.<module> ...`: no `source`,
no virtualenv activation, no `cd`, and no `&&`, `;` or pipes. The project's virtualenv is already
first on PATH. Any other command form is denied in this unattended run."""


def robinhood_server() -> str:
    return config.secret("TRADER_ROBINHOOD_SERVER") or DEFAULT_ROBINHOOD_SERVER


def allowed_tools(server: str) -> List[str]:
    return [f"mcp__{server}__{tool}" for tool in ROBINHOOD_READ_TOOLS] + list(TRADER_COMMANDS)


def claude_command(prompt: str, tools: List[str]) -> List[str]:
    claude = config.secret("TRADER_CLAUDE_BIN") or shutil.which("claude") or "claude"
    return [
        claude, "-p", prompt,
        "--permission-mode", "dontAsk",
        "--allowedTools", ",".join(tools),
        "--output-format", "json",
    ]


def claude_env() -> Dict[str, str]:
    """Put the project's virtualenv first so `python -m trader...` resolves to it."""
    env = dict(os.environ)
    venv_bin = config.PROJECT_ROOT / ".venv" / "bin"
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def market_clock() -> Optional[Dict[str, Any]]:
    """Alpaca's market clock, or None without keys or on error."""
    if not alpaca_configured():
        return None
    try:
        return alpaca.market_clock()
    except Exception:
        return None


def market_is_open(now: datetime) -> bool:
    clock = market_clock()
    if clock is not None:
        return bool(clock.get("is_open"))
    # Fallback without Alpaca: regular hours on weekdays (ignores holidays).
    local = now.astimezone(MARKET_TZ)
    return local.weekday() < 5 and time(9, 30) <= local.time() < time(16, 0)


def run_prompt(shortlist: List[Dict[str, Any]], mode: str, account: Optional[str], now: datetime) -> str:
    account_line = (
        f"Robinhood account for review_equity_order: {account} (from the user's configuration)."
        if account else
        "No Robinhood account is configured (TRADER_ROBINHOOD_ACCOUNT): skip review_equity_order "
        "and note that in the report."
    )
    return f"""/day-trader

Autopilot run at {now.astimezone(MARKET_TZ):%Y-%m-%d %H:%M} ET. Nobody is watching this session: do not
ask questions. Anything that needs the user goes in the report.
{COMMAND_RULES}
Mode: {mode}. {account_line}
Never call a place_*_order or exercise tool in an autopilot run, whatever the mode.

Shortlist from `python -m trader.scan`, ranked. material_news means Jev found a relevant, material
headline from the last 24 hours; percent_change is today's move.
{json.dumps(shortlist, indent=2)}

For each symbol in order, follow playbook steps 3 to 5 (analyze, decide, size, review, journal).
Journal every decision, passes included, with `python -m trader.journal decide`. Skip a symbol if its data is unavailable and say why.

Finish with a short Markdown report: a table of symbol, action, entry, stop, target, confidence and a
one-line thesis, then anything the user should look at."""


SCORE_PROMPT = f"""/day-trader

Autopilot scoring run after the close. Nobody is watching this session: do not ask questions.
{COMMAND_RULES}
Follow playbook step 6 for every row from `python -m trader.journal pending`, using the closing price of
each decision's trading day from get_equity_historicals. Then run `python -m trader.journal summary` and
finish with a short Markdown report of what was scored and the summary figures."""


REVIEW_PROMPT = f"""/day-trader

Autopilot review run after scoring. Nobody is watching this session: do not ask questions.
{COMMAND_RULES}
You are reviewing the automated news bot (trader.newsbot). Use `python -m trader.journal summary`,
`python -m trader.journal list --limit 200` and `python -m trader.journal events --limit 500`.
Newsbot decisions have a thesis starting with "[newsbot]" and mode "newsbot-off" or "newsbot-paper";
newsbot_signal and newsbot_order events carry each signal's probabilities, relevance, materiality,
event type, latency and why a trade was blocked.

Write a short Markdown report:
1. Today and to date: signals, trades, scored hit rate and average signed return, split by event
   type and by probability bucket (0.75-0.85, 0.85-0.95, 0.95+).
2. Latency: median and worst Jev and total times.
3. Proposed changes to NEWSBOT_* thresholds, each with the evidence and the sample size. Say plainly
   when the sample is too small to justify a change. Do not edit any file; the user applies changes."""

REVIEW_TOOLS = [f"Bash({py} -m trader.journal *)" for py in PYTHONS]


def invoke_claude(prompt: str, run_dir: Path, tools: Optional[List[str]] = None) -> Dict[str, Any]:
    command = claude_command(prompt, tools if tools is not None else allowed_tools(robinhood_server()))
    (run_dir / "prompt.md").write_text(prompt)
    proc = subprocess.run(
        command, cwd=config.PROJECT_ROOT, env=claude_env(), capture_output=True, text=True,
        timeout=CLAUDE_TIMEOUT_SECONDS,
    )
    (run_dir / "claude.stderr.log").write_text(proc.stderr)
    try:
        output = json.loads(proc.stdout)
    except json.JSONDecodeError:
        output = {"result": proc.stdout, "is_error": True}
    (run_dir / "claude.json").write_text(json.dumps(output, indent=2))
    report = output.get("result") or "(no result)"
    (run_dir / "report.md").write_text(report)
    return {
        "exit_code": proc.returncode,
        "is_error": bool(output.get("is_error")) or proc.returncode != 0,
        "cost_usd": output.get("total_cost_usd"),
        "denials": output.get("permission_denials"),
        "report": str(run_dir / "report.md"),
    }


def notify(title: str, body: str) -> None:
    if shutil.which("notify-send"):
        subprocess.run(["notify-send", title, body], check=False)


def new_run_dir(kind: str, now: datetime) -> Path:
    run_dir = RUNS_DIR / f"{now.astimezone(MARKET_TZ):%Y%m%d-%H%M%S}-{kind}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def cmd_run(force: bool, dry_run: bool, now: Optional[datetime] = None) -> int:
    now = now or datetime.now(timezone.utc)
    if not force and not market_is_open(now):
        print(json.dumps({"skipped": "market closed"}))
        return 0

    result = scan.scan(now=now)
    shortlist = result["shortlist"]
    limits = config.load_limits()
    prompt = run_prompt(shortlist, limits.mode, config.secret("TRADER_ROBINHOOD_ACCOUNT"), now)

    if dry_run:
        print(json.dumps({"scan": result, "command": claude_command(prompt, allowed_tools(robinhood_server()))},
                         indent=2))
        return 0

    run_dir = new_run_dir("run", now)
    (run_dir / "scan.json").write_text(json.dumps(result, indent=2))
    if not shortlist:
        (run_dir / "report.md").write_text("Scan found no candidates.")
        print(json.dumps({"run_dir": str(run_dir), "shortlist": []}))
        return 0

    outcome = invoke_claude(prompt, run_dir)
    symbols = ", ".join(c["symbol"] for c in shortlist)
    notify("AI Day Trader", f"Report ready for {symbols}" if not outcome["is_error"] else "Autopilot run failed")
    print(json.dumps({"run_dir": str(run_dir), "shortlist": [c["symbol"] for c in shortlist], **outcome}))
    return 1 if outcome["is_error"] else 0


def cmd_score(now: Optional[datetime] = None) -> int:
    now = now or datetime.now(timezone.utc)
    run_dir = new_run_dir("score", now)
    outcome = invoke_claude(SCORE_PROMPT, run_dir)
    notify("AI Day Trader", "Scoring done" if not outcome["is_error"] else "Scoring run failed")
    print(json.dumps({"run_dir": str(run_dir), **outcome}))
    return 1 if outcome["is_error"] else 0


def cmd_review(now: Optional[datetime] = None) -> int:
    now = now or datetime.now(timezone.utc)
    run_dir = new_run_dir("review", now)
    outcome = invoke_claude(REVIEW_PROMPT, run_dir, tools=REVIEW_TOOLS)
    notify("AI Day Trader", "Daily review ready" if not outcome["is_error"] else "Review run failed")
    print(json.dumps({"run_dir": str(run_dir), **outcome}))
    return 1 if outcome["is_error"] else 0


def cmd_check() -> int:
    claude = config.secret("TRADER_CLAUDE_BIN") or shutil.which("claude")
    info: Dict[str, Any] = {
        "claude": claude or "NOT FOUND (set TRADER_CLAUDE_BIN)",
        "robinhood_server": robinhood_server(),
        "robinhood_account": "set" if config.secret("TRADER_ROBINHOOD_ACCOUNT") else "missing",
        "mode": config.load_limits().mode,
        "market_clock": market_clock(),
    }
    print(json.dumps(info, indent=2))
    if claude:
        print("\nMCP servers Claude Code sees (tool names are mcp__<server>__<tool>):")
        subprocess.run([claude, "-p", "/mcp"], cwd=config.PROJECT_ROOT, env=claude_env(), check=False)
    return 0


def cmd_cron() -> int:
    python = config.PROJECT_ROOT / ".venv" / "bin" / "python"
    root = config.PROJECT_ROOT
    log = root / "data" / "autopilot.log"
    claude = shutil.which("claude")
    path_line = f"PATH={Path(claude).parent}:/usr/local/bin:/usr/bin:/bin" if claude else "# PATH=<dir containing claude>:/usr/bin:/bin"
    newsbot_log = root / "data" / "newsbot.log"

    def line(schedule: str, command: str, logfile: Path) -> str:
        return f"{schedule}  cd {root} && {command} >> {logfile} 2>&1"

    print(f"""# AI Day Trader (add with `crontab -e`). Times are US/Eastern, weekdays.
CRON_TZ=America/New_York
{path_line}
# News bot: stream from just before the open until it is stopped at 16:35.
{line("25 9 * * 1-5", f"timeout 7h10m {python} -m trader.newsbot run", newsbot_log)}
# Close all paper positions before the close.
{line("50 15 * * 1-5", f"{python} -m trader.newsbot flatten", newsbot_log)}
# Claude scores the day's decisions, then reviews the bot's results.
{line("20 16 * * 1-5", f"{python} -m trader.autopilot score", log)}
{line("40 16 * * 1-5", f"{python} -m trader.autopilot review", log)}
# Optional: Claude analyzes the scan shortlist during the day (review mode, no orders).
# {line("45 9,11,13,15 * * 1-5", f"{python} -m trader.autopilot run", log)}""")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    config.load_env()
    parser = argparse.ArgumentParser(prog="python -m trader.autopilot")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Scan and have Claude analyze the shortlist")
    run.add_argument("--force", action="store_true", help="run even if the market is closed")
    run.add_argument("--dry-run", action="store_true", help="scan and print the Claude command only")
    sub.add_parser("score", help="Score pending decisions after the close")
    sub.add_parser("review", help="Claude reviews the news bot's results and proposes tuning")
    sub.add_parser("check", help="Show autopilot setup")
    sub.add_parser("cron", help="Print suggested crontab lines")
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args.force, args.dry_run)
    if args.command == "score":
        return cmd_score()
    if args.command == "review":
        return cmd_review()
    if args.command == "check":
        return cmd_check()
    return cmd_cron()


if __name__ == "__main__":
    sys.exit(main())
