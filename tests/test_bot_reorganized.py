import asyncio
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from telegram.error import TimedOut

from sports_ai_bot.bot import telegram_bot as bot
from sports_ai_bot.utils.config import Settings


@pytest.fixture
def settings(monkeypatch, tmp_path):
    settings = Settings(
        _env_file=None, DATA_DIR=tmp_path, TELEGRAM_ADMIN_IDS="42",
        TELEGRAM_CHAT_ID="-1001", TELEGRAM_BOT_TOKEN="123:fake",
    )
    monkeypatch.setattr(bot, "get_settings", lambda: settings)
    return settings


@pytest.fixture
def update():
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=7),
        effective_message=SimpleNamespace(reply_text=AsyncMock()),
    )


@pytest.mark.parametrize("picks", [[], [object()]])
def test_picks_only_reads_cache(monkeypatch, settings, update, picks):
    cached = Mock(return_value=picks)
    generated = Mock(side_effect=AssertionError("Must not generate"))
    formatted = Mock(return_value="cached picks")
    status = Mock(return_value="No fresh manual quotes")
    monkeypatch.setattr(bot, "cached_picks", cached)
    monkeypatch.setattr(bot, "generate_picks", generated)
    monkeypatch.setattr(bot, "format_picks", formatted)
    monkeypatch.setattr(bot, "status_message", status)
    asyncio.run(bot.picks_command(update, None))
    cached.assert_called_once_with()
    generated.assert_not_called()
    update.effective_message.reply_text.assert_awaited_once_with(
        "cached picks" if picks else "No fresh manual quotes"
    )
    if picks:
        formatted.assert_called_once_with(picks)
        status.assert_not_called()
    else:
        formatted.assert_not_called()
        status.assert_called_once_with()


def test_quotes_and_performance(monkeypatch, settings, update):
    quotes = Mock(return_value="Manual quotes")
    report = Mock(return_value=object())
    formatted = Mock(return_value="Performance")
    monkeypatch.setattr(bot, "format_quotes", quotes)
    monkeypatch.setattr(bot, "build_performance_report", report)
    monkeypatch.setattr(bot, "format_performance_message", formatted)
    asyncio.run(bot.cuotas_command(update, None))
    update.effective_message.reply_text.assert_awaited_with("Manual quotes")
    quotes.assert_called_once_with()
    asyncio.run(bot.performance_command(update, None))
    formatted.assert_called_once_with(report.return_value)
    update.effective_message.reply_text.assert_awaited_with("Performance")


@pytest.mark.parametrize("command", [bot.help_command, bot.start_command])
@pytest.mark.parametrize("admin", [False, True])
def test_help_only_new_commands(settings, update, command, admin):
    update.effective_user.id = 42 if admin else 7
    asyncio.run(command(update, None))
    text = update.effective_message.reply_text.call_args.args[0]
    commands = {word for word in text.split() if word.startswith("/")}
    assert commands == {"/picks", "/cuotas", "/forebet", "/rendimiento", "/help", "/start"} | (
        {"/publishnow"} if admin else set()
    )
    assert "manualmente" in text


def test_registered_aliases_and_forebet_source(monkeypatch, settings, update):
    app = bot._build_application()
    handlers = {name: handler.callback for handler in app.handlers[0] for name in handler.commands}
    for alias in ("picks", "today", "top", "value", "best", "over", "btts", "over15", "under45"):
        assert handlers[alias] is bot.picks_command
    assert handlers["cuotas"] is bot.cuotas_command
    assert handlers["rendimiento"] is bot.performance_command
    work = AsyncMock(side_effect=AssertionError("No fetch or work for disabled commands"))
    monkeypatch.setattr(bot, "_work", work)
    asyncio.run(handlers["corners"](update, SimpleNamespace(args=[])))
    text = update.effective_message.reply_text.call_args.args[0]
    assert "Desactivado" in text
    for alias in ("forebet", "forebettop", "forebetvalue", "forebet48h"):
        assert handlers[alias] is bot.forebet_command
    assert work.await_count == 0


@pytest.mark.parametrize("picks", [[], [object()]])
@pytest.mark.parametrize("refresh", [False, True])
def test_daily_one_local_generation(monkeypatch, picks, refresh):
    generate = Mock(return_value=picks)
    monkeypatch.setattr(bot, "generate_picks", generate)
    monkeypatch.setattr(bot, "format_picks", Mock(return_value="Picks"))
    monkeypatch.setattr(bot, "status_message", Mock(return_value="Reasons"))
    monkeypatch.setattr(bot, "build_top_picks", Mock(side_effect=AssertionError("Legacy pipeline")))
    from sports_ai_bot.research import corners
    monkeypatch.setattr(corners, "build_corners_picks", Mock(side_effect=AssertionError("Corners")))
    assert bot._build_daily_pick_messages(refresh) == (["Picks" if picks else "Reasons"], picks)
    generate.assert_called_once_with()


@pytest.mark.parametrize("failure_at", [None, 1, 2])
def test_publication_marks_only_after_all_sends(monkeypatch, settings, failure_at):
    valid = SimpleNamespace(quote_id="quote", model_version="v1")
    picks = [valid, object(), SimpleNamespace(quote_id="quote", model_version=None)]
    events = []
    monkeypatch.setattr(bot, "_build_daily_pick_messages", lambda _: (["a" * 4001], picks))
    monkeypatch.setattr(bot, "persist_picks", lambda value: events.append(("persist", value)))
    store = Mock()
    store.mark_published.side_effect = lambda value: events.append(("mark", value))
    monkeypatch.setattr(bot, "Store", Mock(return_value=store))
    sent = 0

    async def send_message(**kwargs):
        nonlocal sent
        sent += 1
        events.append(("send", kwargs["text"]))
        if sent == failure_at:
            raise TimedOut()

    call = bot._send_daily_picks(SimpleNamespace(send_message=send_message), "-1001", False)
    if failure_at:
        with pytest.raises(TimedOut):
            asyncio.run(call)
        store.mark_published.assert_not_called()
    else:
        asyncio.run(call)
        store.mark_published.assert_called_once_with([valid])
        assert [event[0] for event in events] == ["persist", "send", "send", "mark"]
    assert events[0] == ("persist", picks)
    with sqlite3.connect(settings.data_dir / "publications.sqlite3") as db:
        assert db.execute("SELECT state FROM publications").fetchone()[0] == (
            "failed-uncertain" if failure_at else "completed"
        )


def test_refresh_job_schedule_and_offload(monkeypatch, settings):
    from telegram.ext import JobQueue

    repeating = Mock()
    monkeypatch.setattr(JobQueue, "run_repeating", repeating)
    bot._build_application()
    repeating.assert_called_once_with(
        bot.refresh_analysis, interval=900, first=10, name="refresh_analysis",
    )
    work = AsyncMock()
    monkeypatch.setattr(bot, "_work", work)
    asyncio.run(bot.refresh_analysis(None))
    work.assert_awaited_once_with(bot.generate_picks)
