from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from trader import autopilot

NOW = datetime(2026, 9, 28, 15, 0, tzinfo=timezone.utc)


def test_allowed_tools_never_include_order_placement():
    tools = autopilot.allowed_tools("claude_ai_Robinhood")
    assert "mcp__claude_ai_Robinhood__review_equity_order" in tools
    assert not any("place_" in t or "exercise" in t or "cancel" in t for t in tools)
    assert all(t.startswith(("mcp__claude_ai_Robinhood__", "Bash(python -m trader.",
                             "Bash(.venv/bin/python -m trader."))
               for t in tools)


def test_review_only_gets_the_journal():
    assert autopilot.REVIEW_TOOLS == ["Bash(python -m trader.journal *)",
                                      "Bash(.venv/bin/python -m trader.journal *)"]
    assert "Do not edit any file" in autopilot.REVIEW_PROMPT


def test_every_prompt_tells_claude_to_run_plain_commands():
    # A chained `source ... && python -m ...` command is denied by the allowlist in dontAsk mode.
    prompts = [autopilot.SCORE_PROMPT, autopilot.REVIEW_PROMPT,
               autopilot.run_prompt([], "review", None, NOW)]
    for prompt in prompts:
        assert "no `source`" in prompt and "`&&`" in prompt
    assert "Bash(.venv/bin/python -m trader.status)" in autopilot.allowed_tools("x")


def test_claude_command_is_locked_down(monkeypatch):
    monkeypatch.setenv("TRADER_CLAUDE_BIN", "/bin/claude")
    cmd = autopilot.claude_command("hi", ["Bash(python -m trader.journal *)"])
    assert cmd[:3] == ["/bin/claude", "-p", "hi"]
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    assert cmd[cmd.index("--allowedTools") + 1] == "Bash(python -m trader.journal *)"


def test_run_prompt_forbids_orders_and_carries_the_shortlist():
    prompt = autopilot.run_prompt([{"symbol": "ACME"}], "review", None, NOW)
    assert prompt.startswith("/day-trader")
    assert "Never call a place_*_order" in prompt
    assert '"symbol": "ACME"' in prompt
    assert "No Robinhood account is configured" in prompt


def test_run_skips_when_market_closed(monkeypatch, capsys):
    monkeypatch.setattr(autopilot, "market_is_open", lambda now: False)
    monkeypatch.setattr(autopilot.scan, "scan", lambda now: pytest.fail("should not scan"))
    assert autopilot.cmd_run(force=False, dry_run=False, now=NOW) == 0
    assert json.loads(capsys.readouterr().out) == {"skipped": "market closed"}


def test_market_hours_fallback_without_alpaca(monkeypatch):
    monkeypatch.setattr(autopilot, "market_clock", lambda: None)
    assert autopilot.market_is_open(datetime(2026, 9, 28, 15, 0, tzinfo=timezone.utc))      # Mon 11:00 ET
    assert not autopilot.market_is_open(datetime(2026, 9, 28, 21, 0, tzinfo=timezone.utc))  # Mon 17:00 ET
    assert not autopilot.market_is_open(datetime(2026, 9, 27, 15, 0, tzinfo=timezone.utc))  # Sunday


def test_invoke_claude_saves_report(monkeypatch, tmp_path):
    class Proc:
        returncode = 0
        stdout = json.dumps({"result": "## Report", "total_cost_usd": 0.12, "is_error": False})
        stderr = ""

    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        return Proc()

    monkeypatch.setattr(autopilot.subprocess, "run", fake_run)
    outcome = autopilot.invoke_claude("prompt", tmp_path, tools=["Bash(python -m trader.journal *)"])
    assert outcome["is_error"] is False and outcome["cost_usd"] == 0.12
    assert (tmp_path / "report.md").read_text() == "## Report"
    assert (tmp_path / "prompt.md").read_text() == "prompt"
    assert "Bash(python -m trader.journal *)" in seen["command"]


def test_invoke_claude_flags_failures(monkeypatch, tmp_path):
    class Proc:
        returncode = 1
        stdout = "not json"
        stderr = "boom"

    monkeypatch.setattr(autopilot.subprocess, "run", lambda *a, **k: Proc())
    outcome = autopilot.invoke_claude("prompt", tmp_path, tools=[])
    assert outcome["is_error"] is True
    assert (tmp_path / "claude.stderr.log").read_text() == "boom"


def test_cron_schedule_includes_bot_flatten_score_review(capsys):
    autopilot.cmd_cron()
    out = capsys.readouterr().out
    assert "CRON_TZ=America/New_York" in out
    for fragment in ("trader.newsbot run", "trader.newsbot flatten", "trader.autopilot score",
                     "trader.autopilot review"):
        assert fragment in out
