# Finding → regression test map

Regression tests for defects found in two independent reviews of commit
`d6e0768`. Every test here is **deterministic and offline**: `conftest.py`
installs an autouse kill switch that turns any real HTTP call into a failure,
and strips every trader environment variable so a developer's shell cannot
change an outcome. No test reads a credential, a real account or a `.env`.

## How to read a row

- **FIXED** — the test passes and guards the fix. A regression turns it red.
- **OPEN** — the test carries `xfail(strict=True)` and asserts the *intended
  safe behaviour*. It fails today for the reason named in the marker.

`strict=True` matters: when the defect is fixed the test reports **XPASS** and
the build goes **red** until the marker is removed. That is deliberate — it
forces the marker off in the same commit as the fix, so this table cannot drift
out of date silently.

Always verify a new xfail fails for the *right* reason:

```bash
python -m pytest tests/regressions/ --runxfail -q    # see every real failure message
```

## Open findings

| ID | Defect | Test | Severity |
|---|---|---|---|
| A1 | `stop()` cancels the asyncio task but the `to_thread` worker finishes the cycle and keeps submitting orders | `test_a1_no_order_is_placed_after_stop` | high |
| A1b | Two concurrent `start()`s orphan a `_loop` task that no `stop()` can reach | `test_a1b_concurrent_starts_leave_no_orphan_task` | high |
| B1 | `flatten_owned` sells a snapshot quantity after swallowing a cancel failure — double exit, or sells the operator's own shares | `test_b1_flatten_does_not_sell_an_already_exited_position` | medium |
| B2 | `flatten_owned` never cancels an unfilled BUY parent; it can still fill after "flatten" reports success | `test_b2_flatten_cancels_a_working_buy_entry` | medium |
| B3 | The flatten retry mints a new `client_order_id` per attempt, so a 5xx-after-acceptance duplicates the sell | `test_b3_flatten_retry_reuses_one_idempotency_key` | medium |
| C1a | `check_live_buy` and `submit_order` are separate operations with no shared reservation | `test_c1a_concurrent_live_buys_cannot_exceed_the_daily_cap` | medium |
| C1b | A buy cancelled after a partial fill drops out of the daily count | `test_c1b_a_partially_filled_then_cancelled_buy_still_counts` | medium |
| D1a | `analyze()` drops `get_bars()`'s `source`, so nothing downstream knows the provenance | `test_d1a_analysis_result_carries_its_market_data_provenance` | high |
| D1b | Demo prices build and send real bracket orders | `test_d1b_broker_modes_refuse_demo_sourced_prices` | high |
| D1c | `DemoMarket` stamps bars with the wall clock, so a timestamp-only fix misses demo data | `test_d1c_demo_bars_are_not_disguised_as_fresh_real_data` | medium |
| E1 | Two concurrent STARTs launch two news bots; STOP kills one and reports stopped | `test_e1_concurrent_starts_launch_one_process` | high |
| E2 | A failed respawn clears `_retry_at` before spawning, leaving a state no later `check()` can act on | `test_e2_a_failed_respawn_schedules_another_attempt` | medium |
| F1a | `/ws/desk` never validates `Origin`; any page reads the live account stream | `test_f1a_websocket_rejects_a_hostile_origin` | high |
| F1b | No `Host` validation anywhere — DNS rebinding reaches every endpoint | `test_f1b_requests_for_a_foreign_host_are_refused` | high |
| F1c | The bodyless cancel route is a CORS simple request; chained with F1a it strips a live stop-loss with no rebinding | `test_f1c_bodyless_cross_origin_cancel_is_refused` | high |
| H1 | `journal pending` excludes the session the 16:20 ET scoring run exists to score | `test_h1_same_day_decisions_are_scorable_after_the_close` | low |
| H2 | A 207 multi-status flatten counts failures as closed, then suppresses retries for the session | `test_h2_a_partially_failed_flatten_is_not_counted_as_closed` | medium |
| I1 | `fuse()`'s `elif confirming:` makes the conflict penalty unreachable whenever anything agrees | `test_i1_a_conflicting_signal_never_raises_confidence` | high |
| P1 | `Portfolios.record_fill` reads before the write lock — concurrent fills overspend the cash balance | `test_p1_concurrent_fills_cannot_overspend_the_cash_balance` | high |

## Fixed, guarded elsewhere

| ID | Defect | Fix | Test |
|---|---|---|---|
| T1 | Suite red: a hard-coded fixture date aged past the 24h judge window | `ca32922` | `tests/test_desk.py::test_judge_headline_age_is_measured_against_an_injected_clock` |
| S1 | `market_clock()` shipped the API key and secret to any configured host | `df4866a` | `tests/test_alpaca.py::test_market_clock_never_sends_credentials_to_an_unapproved_host` |
| N1 | `nan`/`inf` limits parsed fine and then never enforced | `8d282a7` | `tests/test_config_numbers.py` |
| O1 | Enabling options/crypto skipped the dollar cap, allowlist and sizeability | `f52a782` | `tests/test_guard.py` (`test_option_buy_respects_the_dollar_cap` and neighbours) |

## Deliberate properties pinned here

Not defects — behaviour that must not regress while the above are fixed:

- `test_live_sells_are_never_blocked_by_the_buy_limits` — an exit must always be possible.
- `test_portfolio_rejects_an_oversized_buy_when_uncontended` — the sequential ledger path is correct.
- `test_conflict_penalty_does_apply_when_nothing_confirms` — the penalty branch works when `confirming == 0`, so an I1 fix is visibly a change to the confirming case alone.
- `test_f1d_confirm_live_is_a_human_confirmation_not_an_access_control` — `confirm_live` is a fat-finger guard, not a security boundary.

## Not covered here

- **I1 sizing.** `position_size` applies confidence twice (tier threshold, then
  multiplier), so notional is roughly quadratic in confidence. The I1 test
  asserts only the *sign* — a conflicting signal must never raise confidence —
  because the intended sizing curve is an open product decision. Settle that
  before changing `position_size`.
- **Browser-level exploitability.** `TestClient` is not a browser. The F1 tests
  prove the *server* has no check; what a given browser permits (Private
  Network Access, preflight behaviour) varies and is not asserted.
- **`.claude/settings.json` hardening.** The deny-list and PostToolUse matcher
  changes cannot be made or tested from inside this repo's own tooling — the
  project denies edits to that file, correctly. See the handoff notes.
