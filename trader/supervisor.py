"""
Keeps the news bot running as a child process of the desk (DESK_START_NEWSBOT).

The bot runs as its own process (`python -m trader.newsbot run`), exactly as it
would from a terminal, so a crash in one cannot take down the other. If it
exits it is restarted with a growing delay (5 s up to 5 min). Exit code 2
means a setup problem (missing keys, NEWSBOT_EXECUTION not matching the
Alpaca account): that is reported and not retried, since retrying cannot fix it.
Output goes to data/newsbot.log.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from trader import config

logger = logging.getLogger("trader.supervisor")

SETUP_ERROR_EXIT = 2
MIN_BACKOFF, MAX_BACKOFF = 5.0, 300.0
# A run this long counts as healthy: the next crash restarts after the minimum delay.
HEALTHY_SECONDS = 600
LOG_TAIL_LINES = 8


def _now() -> datetime:
    return datetime.now(timezone.utc)


class NewsbotProcess:
    def __init__(
        self,
        log_path: Optional[Path] = None,
        command: Optional[List[str]] = None,
        popen: Callable[..., Any] = subprocess.Popen,
        on_change: Callable[[Dict[str, Any]], None] = lambda status: None,
        now: Callable[[], datetime] = _now,
    ) -> None:
        self.log_path = Path(log_path) if log_path else config.journal_path().with_name("newsbot.log")
        self.command = command or [sys.executable, "-m", "trader.newsbot", "run"]
        self.popen = popen
        self.on_change = on_change
        self.now = now
        self.proc: Any = None
        self.wanted = False
        self.restarts = 0
        self.started_at: Optional[datetime] = None
        self.last_exit: Optional[int] = None
        self.error: Optional[str] = None
        self.backoff = MIN_BACKOFF
        self._retry_at: Optional[datetime] = None

    # --- control -------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> Dict[str, Any]:
        self.wanted, self.error, self._retry_at = True, None, None
        if not self.running:
            self._spawn()
        return self.status()

    def stop(self, timeout: float = 10.0) -> Dict[str, Any]:
        self.wanted, self._retry_at = False, None
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=timeout)
        self._changed()
        return self.status()

    def _spawn(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "ab") as log:
            log.write(f"\n--- {self.now().isoformat(timespec='seconds')} starting: news bot ---\n".encode())
            log.flush()
            self.proc = self.popen(self.command, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=str(config.PROJECT_ROOT))
        self.started_at = self.now()
        logger.info(f"News bot started (pid {self.proc.pid}); output in {self.log_path}")
        self._changed()

    # --- supervision -----------------------------------------------------------------

    def check(self) -> None:
        """Notice an exit and restart (with backoff) unless stopped or misconfigured."""
        if not self.wanted:
            return
        if self.proc is not None:
            code = self.proc.poll()
            if code is None:
                return
            self.proc, self.last_exit = None, code
            ran = (self.now() - self.started_at).total_seconds() if self.started_at else 0
            if code == SETUP_ERROR_EXIT:
                self.wanted = False
                self.error = f"news bot setup error (exit 2): {self.log_tail(3) or 'see ' + str(self.log_path)}"
                logger.error(self.error)
                self._changed()
                return
            self.backoff = MIN_BACKOFF if ran >= HEALTHY_SECONDS else min(MAX_BACKOFF, self.backoff * 2)
            self._retry_at = self.now() + timedelta(seconds=self.backoff)
            logger.warning(f"News bot exited with code {code} after {ran:.0f}s; restarting in {self.backoff:.0f}s")
            self._changed()
            return
        if self._retry_at is not None and self.now() >= self._retry_at:
            self._retry_at = None
            self.restarts += 1
            self._spawn()

    async def watch(self, interval: float = 2.0) -> None:
        while True:
            try:
                self.check()
            except Exception:
                logger.exception("News bot supervision check failed")
            await asyncio.sleep(interval)

    # --- reporting -------------------------------------------------------------------

    def log_tail(self, lines: int = LOG_TAIL_LINES) -> str:
        try:
            text = self.log_path.read_text(errors="replace")
        except OSError:
            return ""
        body = [line.strip() for line in text.strip().splitlines() if line.strip() and not line.startswith("--- ")]
        return " | ".join(body[-lines:])

    def status(self) -> Dict[str, Any]:
        return {
            "managed": self.wanted or self.running or self.error is not None,
            "running": self.running,
            "pid": self.proc.pid if self.running else None,
            "started_at": self.started_at.isoformat(timespec="seconds") if self.started_at and self.running else None,
            "restarts": self.restarts,
            "last_exit": self.last_exit,
            "restart_at": self._retry_at.isoformat(timespec="seconds") if self._retry_at else None,
            "error": self.error,
            "log": str(self.log_path),
        }

    def _changed(self) -> None:
        try:
            self.on_change(self.status())
        except Exception:
            logger.exception("News bot status callback failed")
