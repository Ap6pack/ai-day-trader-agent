"""C1a across OS PROCESSES: the newsbot is spawned by trader/supervisor.py as
`python -m trader.newsbot run`, a separate process from the desk. No threading.Lock
in either process can cover the other. Shared broker is a local HTTP server.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

SCRATCH = Path("docs/evidence/probes")
sys.path.insert(0, str(SCRATCH))
import C_broker_server as broker  # noqa: E402

srv = broker.serve()
port = srv.server_address[1]
print(f"fake Alpaca listening on 127.0.0.1:{port} (nothing leaves loopback)")
print("ALPACA_LIVE_MAX_ORDERS_PER_DAY=1\n")

start = time.time() + 3.0
procs = []
for actor, symbol in (("newsbot-process", "ACME"), ("desk-process", "NVDA")):
    procs.append(subprocess.Popen(
        [sys.executable, str(SCRATCH / "C_child_actor.py"), str(port), actor, symbol, str(start)],
        cwd="/home/user/ai-day-trader-agent", stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**__import__("os").environ, "PYTHONPATH": "/home/user/ai-day-trader-agent"}, text=True))

for p in procs:
    out, err = p.communicate(timeout=120)
    if err.strip():
        print("stderr:", err.strip()[:800])
    for line in out.strip().splitlines():
        try:
            d = json.loads(line)
            print(f"  pid {d['pid']:>7} {d['actor']:<16} {d['result']} "
                  f"{d.get('order') or d.get('error','')} {d.get('symbol','')}")
        except json.JSONDecodeError:
            print("  ", line)

print("\nbroker wire trace:")
for line in broker.LOG:
    print("  ", line)
print(f"\nbroker order book holds {len(broker.ORDERS)} LIVE buy order(s) against a cap of 1:")
for o in broker.ORDERS:
    print(f"   {o['id']} {o['side']} {o['qty']} {o['symbol']}")
print("\nVERDICT:", "CAP BREACHED ACROSS PROCESSES" if len(broker.ORDERS) > 1 else "cap held")
