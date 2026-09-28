from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from trader import alpaca, newsbot
from trader.journal import Journal
from trader.newsbot import Settings

NOW = datetime(2026, 9, 28, 15, 0, tzinfo=timezone.utc)  # 11:00 ET
PAPER = Settings(execution="paper")
OPEN_CLOCK = {"is_open": True, "next_open": "2026-09-29T09:30:00-04:00",
              "next_close": "2026-09-28T16:00:00-04:00"}


def iso(seconds_ago):
    return (NOW - timedelta(seconds=seconds_ago)).isoformat().replace("+00:00", "Z")


def article(seconds_ago=10, title="Acme wins $2B defense contract", symbols=("ACME",)):
    return {"title": title, "description": "", "url": f"https://x/{title}",
            "publishedAt": iso(seconds_ago), "symbols": list(symbols)}


def judgment(relevance=0.95, materiality=0.8, p_bullish=0.85, p_bearish=0.05, event="product"):
    return {"relevance": relevance, "materiality": materiality, "p_bullish": p_bullish,
            "p_bearish": p_bearish, "event_type": event, "direction": 0.8}


# --- decide -----------------------------------------------------------------


def test_decide_buy_on_bullish_mass():
    signal = newsbot.decide("ACME", article(), judgment(), PAPER, NOW)
    assert signal.action == "buy"
    assert signal.probability == 0.85


def test_decide_bearish_on_bearish_mass():
    signal = newsbot.decide("ACME", article(), judgment(p_bullish=0.05, p_bearish=0.8), PAPER, NOW)
    assert signal.action == "bearish"
    assert signal.probability == 0.8


@pytest.mark.parametrize("art,judg,reason", [
    (article(), None, "no Jev judgment"),
    ({**article(), "publishedAt": None}, judgment(), "no timestamp"),
    (article(seconds_ago=121), judgment(), "121s old"),
    (article(), judgment(relevance=0.79), "relevance"),
    (article(), judgment(materiality=0.59), "materiality"),
    (article(), judgment(p_bullish=0.74, p_bearish=0.2), "no clear direction"),
])
def test_decide_none_with_reason(art, judg, reason):
    signal = newsbot.decide("ACME", art, judg, PAPER, NOW)
    assert signal.action == "none"
    assert reason in signal.reason


def test_decide_thresholds_are_inclusive():
    j = judgment(relevance=0.8, materiality=0.6, p_bullish=0.75)
    assert newsbot.decide("ACME", article(seconds_ago=120), j, PAPER, NOW).action == "buy"


# --- symbols_to_judge -------------------------------------------------------


def test_roundups_are_skipped():
    assert newsbot.symbols_to_judge(["AAPL", "MSFT", "NVDA"], PAPER) == []
    assert newsbot.symbols_to_judge(["aapl", "MSFT"], PAPER) == ["AAPL", "MSFT"]
    assert newsbot.symbols_to_judge([], PAPER) == []


def test_allowlist_and_non_stock_symbols():
    settings = replace(PAPER, symbols=frozenset({"AAPL"}))
    assert newsbot.symbols_to_judge(["AAPL", "MSFT"], settings) == ["AAPL"]
    assert newsbot.symbols_to_judge(["BTCUSD"], PAPER) == []


# --- settings ---------------------------------------------------------------


def test_settings_defaults_and_validation():
    assert newsbot.load_settings({}) == Settings()
    loaded = newsbot.load_settings({"NEWSBOT_EXECUTION": "PAPER", "NEWSBOT_SYMBOLS": "aapl, msft"})
    assert loaded.execution == "paper"
    assert loaded.symbols == frozenset({"AAPL", "MSFT"})
    with pytest.raises(ValueError):
        newsbot.load_settings({"NEWSBOT_EXECUTION": "live"})
    with pytest.raises(ValueError):
        newsbot.load_settings({"NEWSBOT_STOP_LOSS_PCT": "0"})
    with pytest.raises(ValueError, match="FLATTEN"):
        newsbot.load_settings({"NEWSBOT_FLATTEN_MINUTES": "15", "NEWSBOT_ENTRY_CUTOFF_MINUTES": "15"})


# --- NewsBot.handle ---------------------------------------------------------


class FakeBroker:
    def __init__(self, price=50.0, clock=OPEN_CLOCK):
        self.price = price
        self.clock = clock
        self.orders = []
        self.clock_calls = 0

    def market_clock(self):
        self.clock_calls += 1
        return self.clock

    def latest_price(self, symbol):
        return self.price

    bracket_order_payload = staticmethod(alpaca.bracket_order_payload)

    flattens = 0

    def close_all_positions(self):
        self.flattens += 1
        return [{"symbol": "ACME", "status": 200}]

    def submit_order(self, payload):
        self.orders.append(payload)
        return {"id": f"order-{len(self.orders)}", "status": "accepted"}


class Clock:
    def __init__(self, now=NOW):
        self.value = now

    def __call__(self):
        return self.value


@pytest.fixture
def journal(tmp_path):
    return Journal(tmp_path / "journal.db")


def make_bot(journal, settings=PAPER, broker=None, judge=None, clock=None):
    broker = broker or FakeBroker()
    bot = newsbot.NewsBot(
        settings, journal=journal, broker=broker,
        judge=judge or (lambda symbol, art: judgment()),
        now=clock or Clock(), monotonic=lambda: 0.0,
    )
    return bot, broker


def events(journal, kind):
    return [dict(e) for e in journal.events(100) if e["kind"] == kind]


def test_passing_signal_submits_one_bracket_order_and_logs_latency(journal):
    bot, broker = make_bot(journal)
    received = NOW - timedelta(milliseconds=350)
    record = bot.handle("ACME", article(), received, news_id=42)

    assert record["kind"] == newsbot.ORDER_EVENT
    assert len(broker.orders) == 1
    order = broker.orders[0]
    assert order["order_class"] == "bracket"
    assert order["qty"] == "10"  # floor(500 / 50)
    assert order["take_profit"] == {"limit_price": "51.0"}
    assert order["stop_loss"] == {"stop_price": "49.5"}
    assert order["client_order_id"] == alpaca.client_order_id("42:ACME")

    [logged] = events(journal, newsbot.ORDER_EVENT)
    assert logged["symbol"] == "ACME" and logged["side"] == "buy" and logged["quantity"] == 10
    assert logged["notional"] == 500
    payload = json.loads(logged["payload"])
    assert payload["order_id"] == "order-1"
    assert payload["total_ms"] == 350
    assert "jev_ms" in payload

    [decision] = journal.decisions()
    assert decision["action"] == "buy" and decision["price"] == 50.0
    assert decision["confidence"] == 0.85
    assert decision["mode"] == "newsbot-paper"
    assert decision["thesis"].startswith("[newsbot] product: Acme wins")


def test_execution_off_journals_only(journal):
    bot, broker = make_bot(journal, settings=Settings(execution="off"))
    record = bot.handle("ACME", article(), NOW)
    assert broker.orders == []
    assert "execution off" in record["blocked"]
    assert len(events(journal, newsbot.SIGNAL_EVENT)) == 1
    assert journal.decisions()[0]["mode"] == "newsbot-off"


def test_bearish_is_journaled_as_sell_never_ordered(journal):
    bot, broker = make_bot(journal, judge=lambda s, a: judgment(p_bullish=0.0, p_bearish=0.9))
    record = bot.handle("ACME", article(), NOW)
    assert broker.orders == []
    assert "long only" in record["blocked"]
    assert journal.decisions()[0]["action"] == "sell"


def test_no_signal_logs_reason_without_decision(journal):
    bot, broker = make_bot(journal, judge=lambda s, a: judgment(relevance=0.1))
    record = bot.handle("ACME", article(), NOW)
    assert record["signal"] == "none"
    assert journal.decisions() == []
    [logged] = events(journal, newsbot.SIGNAL_EVENT)
    assert "relevance" in logged["detail"]


def test_jev_exception_is_no_signal(journal):
    def boom(symbol, art):
        raise RuntimeError("timeout")
    bot, broker = make_bot(journal, judge=boom)
    assert bot.handle("ACME", article(), NOW)["reason"] == "no Jev judgment"


@pytest.mark.parametrize("broker,settings,reason", [
    (FakeBroker(clock={**OPEN_CLOCK, "is_open": False}), PAPER, "market closed"),
    (FakeBroker(price=4.99), PAPER, "below NEWSBOT_MIN_PRICE"),
    (FakeBroker(price=600.0), PAPER, "less than one share"),
    (FakeBroker(clock={**OPEN_CLOCK, "next_close": "2026-09-28T11:10:00-04:00"}), PAPER,
     "minutes of the close"),
])
def test_blocks_with_reason(journal, broker, settings, reason):
    bot, _ = make_bot(journal, settings=settings, broker=broker)
    record = bot.handle("ACME", article(), NOW)
    assert broker.orders == []
    assert reason in record["blocked"]
    assert events(journal, newsbot.ORDER_EVENT) == []


def test_clock_failure_blocks(journal):
    class Down(FakeBroker):
        def market_clock(self):
            raise ConnectionError("down")
    bot, broker = make_bot(journal, broker=Down())
    assert "market clock unavailable" in bot.handle("ACME", article(), NOW)["blocked"]
    assert broker.orders == []


def test_order_rejection_is_not_counted(journal):
    class Rejects(FakeBroker):
        def submit_order(self, payload):
            raise RuntimeError("422 insufficient buying power")
    bot, _ = make_bot(journal, broker=Rejects())
    record = bot.handle("ACME", article(), NOW)
    assert "order rejected" in record["blocked"]
    assert journal.count_events_today(newsbot.ORDER_EVENT, now=NOW) == 0


def test_daily_cap(journal):
    settings = replace(PAPER, max_trades_per_day=2, cooldown_minutes=0)
    bot, broker = make_bot(journal, settings=settings)
    for symbol in ("AAA", "BBB", "CCC"):
        record = bot.handle(symbol, article(title=f"{symbol} news"), NOW)
    assert len(broker.orders) == 2
    assert "daily trade limit reached (2/2)" in record["blocked"]


def test_cooldown_per_symbol(journal):
    clock = Clock()
    bot, broker = make_bot(journal, settings=replace(PAPER, cooldown_minutes=30), clock=clock)
    bot.handle("ACME", article(title="first"), NOW)
    assert "cooldown" in bot.handle("ACME", article(title="second"), NOW)["blocked"]
    assert bot.handle("OTHR", article(title="other"), NOW)["kind"] == newsbot.ORDER_EVENT

    clock.value = NOW + timedelta(minutes=31)
    fresh = {**article(title="third"), "publishedAt": clock.value.isoformat()}
    assert bot.handle("ACME", fresh, clock.value)["kind"] == newsbot.ORDER_EVENT
    assert len(broker.orders) == 3


def test_market_clock_is_cached(journal):
    bot, broker = make_bot(journal, settings=replace(PAPER, cooldown_minutes=0))
    bot.handle("AAA", article(title="a"), NOW)
    bot.handle("BBB", article(title="b"), NOW)
    assert broker.clock_calls == 1


# --- stream parsing ---------------------------------------------------------


def news(news_id, symbols=("AAPL",), headline="Apple raises guidance"):
    return {"T": "n", "id": news_id, "headline": headline, "summary": "<p>More</p>",
            "created_at": iso(5), "updated_at": iso(5), "url": f"https://x/{news_id}",
            "symbols": list(symbols), "source": "benzinga", "author": "Benzinga Newsdesk"}


def test_news_tasks_dedupes_and_splits_symbols():
    seen = newsbot.SeenIds()
    raw = json.dumps([news(1, ("AAPL", "MSFT")), news(2, ("A", "B", "C")), {"T": "subscription", "news": ["*"]}])
    tasks = newsbot.news_tasks(newsbot.parse_messages(raw), seen, PAPER)
    assert [(s, i) for s, _, i in tasks] == [("AAPL", 1), ("MSFT", 1)]
    article_ = tasks[0][1]
    assert article_["title"] == "Apple raises guidance"
    assert article_["description"] == "More"
    assert article_["publishedAt"] == iso(5)

    again = json.dumps([news(1, ("AAPL",), headline="Apple raises guidance (updated)")])
    assert newsbot.news_tasks(newsbot.parse_messages(again), seen, PAPER) == []


def test_seen_ids_are_bounded():
    seen = newsbot.SeenIds(max_size=2)
    assert seen.add(1) and seen.add(2) and seen.add(3)
    assert seen.add(1)  # evicted, so accepted again
    assert not seen.add(3)


def test_control_messages():
    assert newsbot.check_control({"T": "success", "msg": "authenticated"}) == "authenticated"
    assert newsbot.check_control({"T": "subscription", "news": ["*"]}) == "subscription"
    assert newsbot.check_control(news(1)) is None
    with pytest.raises(newsbot.StreamError, match="402"):
        newsbot.check_control({"T": "error", "code": 402, "msg": "auth failed"})


class FakeWebSocket:
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def recv(self):
        return self.replies.pop(0)


def test_authenticate_then_subscribe(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    ws = FakeWebSocket([
        '[{"T":"success","msg":"connected"}]',
        '[{"T":"success","msg":"authenticated"}]',
    ])
    asyncio.run(newsbot._authenticate(ws, replace(PAPER, symbols=frozenset({"MSFT", "AAPL"}))))
    assert ws.sent == [
        {"action": "auth", "key": "k", "secret": "s"},
        {"action": "subscribe", "news": ["AAPL", "MSFT"]},
    ]


def test_auth_failure_stops(monkeypatch):
    ws = FakeWebSocket([
        '[{"T":"success","msg":"connected"}]',
        '[{"T":"error","code":402,"msg":"auth failed"}]',
    ])
    with pytest.raises(newsbot.StreamError):
        asyncio.run(newsbot._authenticate(ws, PAPER))
    assert len(ws.sent) == 1  # never subscribed


def test_run_refuses_paper_execution_on_live_url(monkeypatch, capsys):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("TYPESAFE_API_KEY", "t")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets")
    assert newsbot.run(PAPER) == 2
    assert "paper" in capsys.readouterr().err


# --- end-of-day flatten -----------------------------------------------------

HALF_DAY = {"is_open": True, "next_open": "2026-11-30T09:30:00-05:00",
            "next_close": "2026-11-27T13:00:00-05:00"}
HALF_DAY_1252 = datetime(2026, 11, 27, 17, 52, tzinfo=timezone.utc)  # 12:52 ET


def test_flatten_on_early_close_runs_once(journal):
    broker = FakeBroker(clock=HALF_DAY)
    bot, _ = make_bot(journal, broker=broker, clock=Clock(HALF_DAY_1252))
    assert bot.maybe_flatten() == {"closed": 1, "next_close": HALF_DAY["next_close"]}
    assert bot.maybe_flatten() is None
    assert broker.flattens == 1
    [logged] = events(journal, newsbot.FLATTEN_EVENT)
    assert "closed 1 positions" in logged["detail"]


def test_no_flatten_before_window_or_when_off_or_closed(journal):
    early = FakeBroker(clock=HALF_DAY)
    bot, _ = make_bot(journal, broker=early, clock=Clock(HALF_DAY_1252 - timedelta(minutes=5)))
    assert bot.maybe_flatten() is None  # 12:47, 13 minutes out

    off = FakeBroker(clock=HALF_DAY)
    bot, _ = make_bot(journal, settings=Settings(execution="off"), broker=off, clock=Clock(HALF_DAY_1252))
    assert bot.maybe_flatten() is None

    closed = FakeBroker(clock={**HALF_DAY, "is_open": False})
    bot, _ = make_bot(journal, broker=closed, clock=Clock(HALF_DAY_1252))
    assert bot.maybe_flatten() is None
    assert early.flattens == off.flattens == closed.flattens == 0


def test_failed_flatten_is_retried(journal):
    class Flaky(FakeBroker):
        def close_all_positions(self):
            self.flattens += 1
            if self.flattens == 1:
                raise ConnectionError("down")
            return []
    broker = Flaky(clock=HALF_DAY)
    bot, _ = make_bot(journal, broker=broker, clock=Clock(HALF_DAY_1252))
    with pytest.raises(ConnectionError):
        bot.maybe_flatten()
    assert bot.maybe_flatten() == {"closed": 0, "next_close": HALF_DAY["next_close"]}


def test_entry_blocked_before_flatten_window(journal):
    broker = FakeBroker(clock=HALF_DAY)
    bot, _ = make_bot(journal, broker=broker, clock=Clock(HALF_DAY_1252 - timedelta(minutes=5)))
    fresh = {**article(), "publishedAt": (HALF_DAY_1252 - timedelta(minutes=5)).isoformat()}
    assert "minutes of the close" in bot.handle("ACME", fresh, HALF_DAY_1252)["blocked"]


# --- message loop -----------------------------------------------------------


class FrameStream:
    def __init__(self, frames):
        self.frames = frames

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for frame in self.frames:
            yield frame


class RecordingExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, fn, *args):
        self.calls.append(args)


def test_consume_survives_errors_and_bad_frames():
    ws = FrameStream([
        '[{"T":"error","code":500,"msg":"internal error"}]',
        "not json",
        json.dumps([news(5)]),
    ])
    executor = RecordingExecutor()
    asyncio.run(newsbot.consume(ws, _SettingsOnly(), executor,
                                newsbot.SeenIds()))
    assert [(a[0], a[3]) for a in executor.calls] == [("AAPL", 5)]


class _SettingsOnly:
    settings = PAPER


# --- NEWSBOT_SYMBOLS=auto ------------------------------------------------------


def test_auto_symbols_setting():
    settings = newsbot.load_settings({"NEWSBOT_SYMBOLS": "auto", "NEWSBOT_UNIVERSE_REFRESH_MINUTES": "5"})
    assert settings.auto_symbols and settings.symbols == frozenset()
    assert settings.universe_refresh_minutes == 5
    assert not newsbot.load_settings({"NEWSBOT_SYMBOLS": "AAPL"}).auto_symbols
    with pytest.raises(ValueError):
        newsbot.load_settings({"NEWSBOT_SYMBOLS": "auto", "NEWSBOT_UNIVERSE_REFRESH_MINUTES": "0"})


def test_auto_symbols_judges_only_the_universe_and_fails_closed():
    auto = newsbot.load_settings({"NEWSBOT_SYMBOLS": "auto"})
    universe = frozenset({"NVDA", "TSLA"})
    assert newsbot.symbols_to_judge(["NVDA"], auto, universe) == ["NVDA"]
    assert newsbot.symbols_to_judge(["NVDA", "ZZZZ"], auto, universe) == ["NVDA"]
    assert newsbot.symbols_to_judge(["ZZZZ"], auto, universe) == []
    # No universe loaded yet: judge nothing rather than the whole market.
    assert newsbot.symbols_to_judge(["NVDA"], auto, None) == []
    assert newsbot.symbols_to_judge(["NVDA"], auto, frozenset()) == []


def test_news_tasks_use_the_universe():
    auto = newsbot.load_settings({"NEWSBOT_SYMBOLS": "auto"})
    raw = json.dumps([
        {"T": "n", "id": 1, "headline": "Nvidia wins deal", "symbols": ["NVDA"]},
        {"T": "n", "id": 2, "headline": "Tiny co news", "symbols": ["ZZZZ"]},
    ])
    tasks = newsbot.news_tasks(newsbot.parse_messages(raw), newsbot.SeenIds(), auto, frozenset({"NVDA"}))
    assert [t[0] for t in tasks] == ["NVDA"]


def test_universe_refresh_keeps_previous_on_failure_or_empty():
    results = [["nvda", "tsla"], RuntimeError("503"), [], ["AMD"]]

    def loader():
        value = results.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    universe = newsbot.Universe(loader)
    assert universe.symbols == frozenset()
    assert universe.refresh() == frozenset({"NVDA", "TSLA"})
    assert universe.refresh() == frozenset({"NVDA", "TSLA"})  # failure keeps the list
    assert universe.refresh() == frozenset({"NVDA", "TSLA"})  # empty keeps the list
    assert universe.refresh() == frozenset({"AMD"})


def test_bot_has_a_universe_only_in_auto_mode(journal):
    assert newsbot.NewsBot(newsbot.load_settings({"NEWSBOT_SYMBOLS": "auto"}), journal=journal).universe
    assert newsbot.NewsBot(newsbot.load_settings({"NEWSBOT_SYMBOLS": "AAPL"}), journal=journal).universe is None
