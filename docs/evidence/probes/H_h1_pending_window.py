"""H1: does `python -m trader.journal pending` at 16:20 ET see decisions made today?"""
import os, sys, io, json, contextlib
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

SCRATCH = "docs/evidence/probes"
DB = os.path.join(SCRATCH, "H_h1_journal.db")
if os.path.exists(DB):
    os.remove(DB)
os.environ["TRADER_JOURNAL_PATH"] = DB
os.environ["TRADER_MODE"] = "review"

from trader import journal as J

ET = J.MARKET_TZ
FAKE = {"now": None}
J._now = lambda: FAKE["now"].astimezone(timezone.utc)   # patch module clock used by main()/pending


def at(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def run_pending(when):
    FAKE["now"] = when
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        J.main(["pending"])
    rows = [json.loads(l) for l in buf.getvalue().splitlines() if l.strip()]
    return rows


jr = J.Journal()

# Monday 2026-10-12 is a Monday? 2026-10-13 is today (Tuesday). Use a fixed week.
MON = at(2026, 10, 12, 10, 0)     # Monday 10:00 ET
TUE = at(2026, 10, 13, 10, 0)     # Tuesday 10:00 ET ("today")
FRI = at(2026, 10, 16, 10, 0)     # Friday 10:00 ET
print("weekdays:", MON.strftime("%a"), TUE.strftime("%a"), FRI.strftime("%a"))

id_mon = jr.add_decision("AAPL", "buy", 100.0, thesis="monday 10:00 ET", now=MON)
id_tue = jr.add_decision("MSFT", "buy", 200.0, thesis="tuesday 10:00 ET", now=TUE)
print("inserted decision ids:", id_mon, id_tue)
print("raw ts:", [dict(r)["ts"] for r in jr.decisions()])

print("\n--- pending at Tue 2026-10-13 16:20 ET (the cron score time, AFTER the 16:00 close) ---")
rows = run_pending(at(2026, 10, 13, 16, 20))
for r in rows:
    print("  PENDING:", r["id"], r["symbol"], r["ts"], r["thesis"])
print("  today's (MSFT, id=%d) present? %s" % (id_tue, any(r["id"] == id_tue for r in rows)))

print("\n--- pending at Wed 2026-10-14 16:20 ET (the NEXT day's score run) ---")
rows = run_pending(at(2026, 10, 14, 16, 20))
for r in rows:
    print("  PENDING:", r["id"], r["symbol"], r["ts"], r["thesis"])
print("  Tuesday's (MSFT, id=%d) present? %s" % (id_tue, any(r["id"] == id_tue for r in rows)))

# Friday decision -> is it lost over the weekend?
id_fri = jr.add_decision("NVDA", "buy", 300.0, thesis="friday 10:00 ET", now=FRI)
print("\n--- pending at Fri 2026-10-16 16:20 ET ---")
rows = run_pending(at(2026, 10, 16, 16, 20))
print("  ids:", [r["id"] for r in rows], " friday id %d present? %s" % (id_fri, any(r["id"] == id_fri for r in rows)))
print("--- pending at Mon 2026-10-19 16:20 ET (next weekday cron run) ---")
rows = run_pending(at(2026, 10, 19, 16, 20))
print("  ids:", [r["id"] for r in rows], " friday id %d present? %s" % (id_fri, any(r["id"] == id_fri for r in rows)))

# does a skipped cron day permanently lose rows?
print("\n--- cron never fires for 3 weeks; pending at 2026-11-09 16:20 ET ---")
rows = run_pending(at(2026, 11, 9, 16, 20))
print("  ids:", [r["id"] for r in rows])

# exact boundary: is the cut strictly today's ET midnight?
print("\n--- boundary probe ---")
b1 = jr.add_decision("BND1", "buy", 10.0, thesis="Tue 00:00:00 ET", now=at(2026, 10, 20, 0, 0))
b2 = jr.add_decision("BND2", "buy", 10.0, thesis="Mon 23:59:59 ET", now=at(2026, 10, 19, 23, 59) + timedelta(seconds=59))
rows = run_pending(at(2026, 10, 20, 16, 20))
got = {r["id"] for r in rows}
print("  Tue-00:00 id=%d in pending at Tue 16:20? %s" % (b1, b1 in got))
print("  Mon-23:59:59 id=%d in pending at Tue 16:20? %s" % (b2, b2 in got))
