#!/usr/bin/env bash
# Runs trader.guard for Robinhood order/review hooks. Fails closed: if the
# guard cannot run or errors, exit 2 so Claude Code blocks the tool call.
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || { echo "Order guard: project dir not found" >&2; exit 2; }
PY=".venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
[ -n "$PY" ] || { echo "Order guard: no python3 found; blocking" >&2; exit 2; }
"$PY" -m trader.guard
status=$?
if [ "$status" -ne 0 ]; then
  echo "Order guard exited with $status; blocking" >&2
  exit 2
fi
