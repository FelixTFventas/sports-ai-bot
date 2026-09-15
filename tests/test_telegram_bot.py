import asyncio
from types import SimpleNamespace
import pytest

from sports_ai_bot.bot import telegram_bot
from sports_ai_bot.external.forebet import ForebetError, ForebetResult
from sports_ai_bot.predict.pipeline import Pick
from sports_ai_bot.utils.config import Settings


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    settings = Settings(
        _env_file=None,
        DATA_DIR=tmp_path,
        TELEGRAM_ADMIN_IDS="42",
        TELEGRAM_CHAT_ID="-1001",
        THE_ODDS_API_KEY="",
    )
    monkeypatch.setattr(telegram_bot, "get_settings", lambda: settings)
    return settings


class _DummyBot:
    def __init__(self) -> None:
        self.sent_messages: list[tuple[str, str]] = []

    async def send_message(self, chat_id: str, text: str) -> None:
        self.sent_messages.append((chat_id, text))


def test_send_daily_picks_builds_and_sends_message(monkeypatch, isolated_settings) -> None:
    bot = _DummyBot()
    top_picks = [
        _pick("A vs B", "Over 1.5", 0.70, 0.03, 0.04),
        _pick("C vs D", "Over 2.5", 0.65, 0.03, 0.04),
        _pick("E vs F", "Under 4.5", 0.75, 0.03, 0.04),
        _pick("G vs H", "BTTS", 0.66, 0.03, 0.04),
    ]
    persisted: list[list[object]] = []
    generations: list[bool] = []

    monkeypatch.setattr(
        telegram_bot,
        "generate_picks",
        lambda: (generations.append(True) or top_picks),
    )
    monkeypatch.setattr(telegram_bot, "persist_picks", lambda value: persisted.append(value))
    monkeypatch.setattr(telegram_bot, "format_picks", lambda value: "Picks verificados")
    monkeypatch.setattr(telegram_bot, "build_prediction_message", lambda value: "Picks del dia")
    monkeypatch.setattr(
        telegram_bot,
        "build_market_message",
        lambda value, market: f"Picks {market}",
    )
    monkeypatch.setattr(telegram_bot, "build_value_message", lambda value: "Value picks")
    monkeypatch.setattr(telegram_bot, "build_best_message", lambda value: "Best picks")
    monkeypatch.setattr(
        telegram_bot,
        "get_settings",
        lambda: isolated_settings,
    )

    message = asyncio.run(
        telegram_bot._send_daily_picks(bot, chat_id="-1001", refresh_fixtures=True)
    )

    assert message == "Picks verificados"
    assert generations == [True]
    assert len(persisted) == 1
    assert persisted[0][:4] == top_picks
    assert len(persisted[0]) == 4
    assert bot.sent_messages == [("-1001", "Picks verificados")]


def _pick(
    match_label: str,
    market: str,
    probability: float,
    edge: float,
    expected_value: float,
) -> Pick:
    return Pick(
        match_date="2026-05-08",
        home_team=match_label.split(" vs ")[0],
        away_team=match_label.split(" vs ")[1],
        match_label=match_label,
        league="premier_league",
        market=market,
        probability=probability,
        confidence="Media-Alta",
        model_name="test",
        factors=[],
        odd=1.8,
        edge=edge,
        expected_value=expected_value,
    )


def test_send_daily_picks_now_requires_bot_env(monkeypatch) -> None:
    monkeypatch.setattr(
        telegram_bot,
        "get_settings",
        lambda: SimpleNamespace(missing_bot_env=lambda: ["TELEGRAM_BOT_TOKEN"]),
    )

    try:
        telegram_bot.send_daily_picks_now()
    except ValueError as exc:
        assert "TELEGRAM_BOT_TOKEN" in str(exc)
    else:
        raise AssertionError("Expected send_daily_picks_now to fail without bot env")


def test_forebet_command_sends_message(monkeypatch) -> None:
    replies: list[str] = []
    update = SimpleNamespace(message=SimpleNamespace(reply_text=lambda text: replies.append(text)))
    update.message.reply_text = _async_reply(update.message.reply_text)

    monkeypatch.setattr(
        telegram_bot,
        "get_forebet_predictions",
        lambda *args, **kwargs: ForebetResult([], "2026-09-15T12:00:00+00:00", False),
    )
    monkeypatch.setattr(
        telegram_bot,
        "format_forebet_message",
        lambda result, timezone_name: "Pronosticos Forebet",
    )

    asyncio.run(telegram_bot.forebet_command(update, None))

    assert replies == ["Pronosticos Forebet"]


def test_forebet_command_handles_failure(monkeypatch) -> None:
    replies: list[str] = []
    update = SimpleNamespace(message=SimpleNamespace(reply_text=lambda text: replies.append(text)))
    update.message.reply_text = _async_reply(update.message.reply_text)

    monkeypatch.setattr(
        telegram_bot,
        "get_forebet_predictions",
        lambda *args, **kwargs: (_ for _ in ()).throw(ForebetError("boom")),
    )

    asyncio.run(telegram_bot.forebet_command(update, None))

    assert replies == [
        "No se pudo consultar Forebet con el navegador local. "
        "Comprueba que Chrome o Edge esten disponibles e intentalo de nuevo."
    ]


def _async_reply(fn):
    async def wrapper(text: str) -> None:
        fn(text)

    return wrapper


@pytest.mark.parametrize("user_id", [None, 7, -42])
def test_publish_denied(monkeypatch, user_id):
    replies = []
    update = SimpleNamespace(
        message=SimpleNamespace(reply_text=_async_reply(replies.append)),
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=42),
    )
    asyncio.run(telegram_bot.publishnow_command(update, None))
    assert replies == ["No autorizado para publicar."]


def test_publish_admin_and_empty_allowlist(monkeypatch, isolated_settings):
    replies, calls = [], []
    update = SimpleNamespace(
        message=SimpleNamespace(reply_text=_async_reply(replies.append)),
        effective_user=SimpleNamespace(id=42),
    )

    async def send(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(telegram_bot, "_send_daily_picks", send)
    asyncio.run(telegram_bot.publishnow_command(update, SimpleNamespace(bot=object())))
    assert calls == [{"kind": "manual"}]
    isolated_settings.telegram_admin_ids = ""
    asyncio.run(telegram_bot.publishnow_command(update, None))
    assert replies[-1] == "No autorizado para publicar."


@pytest.mark.parametrize(
    "text",
    [
        "",
        "a" * 9001,
        "\U0001f600" * 4001,
        "a" * 3998 + "\n\n" + "b" * 5000,
        "abc\n" * 2000,
        "a" * 3999 + "\U0001f600",
    ],
    ids=["empty", "ascii", "emoji", "paragraph", "lines", "boundary"],
)
def test_fragmentation_lossless(text):
    chunks = telegram_bot.split_message(text)
    assert "".join(chunks) == text
    assert all(0 < len(chunk.encode("utf-16-le")) // 2 <= 4000 for chunk in chunks)


@pytest.mark.parametrize("failure_at", [1, 2])
def test_partial_timeout_not_retried(monkeypatch, failure_at):
    from telegram.error import TimedOut
    from sports_ai_bot.bot.publication import PublicationBlocked

    monkeypatch.setattr(telegram_bot, "_build_daily_pick_messages", lambda _: (["a", "b", "c"], []))
    monkeypatch.setattr(telegram_bot, "persist_picks", lambda _: None)
    calls = []

    async def send_message(**kwargs):
        calls.append(kwargs["text"])
        if len(calls) == failure_at:
            raise TimedOut()

    bot = SimpleNamespace(send_message=send_message)
    with pytest.raises(TimedOut):
        asyncio.run(telegram_bot._send_daily_picks(bot, "1", False))
    with pytest.raises(PublicationBlocked):
        asyncio.run(telegram_bot._send_daily_picks(bot, "1", False))
    assert calls == ["a", "b"][:failure_at]


def test_handlers_and_serial_updates(isolated_settings):
    isolated_settings.telegram_bot_token = "123:fake"
    app = telegram_bot._build_application()
    commands = {command for handler in app.handlers[0] for command in handler.commands}
    assert {"publishnow", "today", "help", "forebet48h"} <= commands
    assert not any("worldcup" in command for command in commands)
    assert app.concurrent_updates == 1


def test_reply_fragmentation_and_offload(monkeypatch):
    import threading

    main_thread = threading.get_ident()
    workers, replies = [], []

    def build():
        workers.append(threading.get_ident())
        return []

    monkeypatch.setattr(telegram_bot, "build_top_picks", build)
    monkeypatch.setattr(telegram_bot, "build_prediction_message", lambda _: "\U0001f600" * 5000)
    update = SimpleNamespace(message=SimpleNamespace(reply_text=_async_reply(replies.append)))
    asyncio.run(telegram_bot.today_command(update, None))
    assert workers[0] != main_thread
    assert "".join(replies) == "\U0001f600" * 5000
    assert len(replies) == 3


def test_cli_and_scheduler_share_daily_dedupe(monkeypatch, isolated_settings):
    from sports_ai_bot.bot.publication import PublicationBlocked

    isolated_settings.telegram_bot_token = "123:fake"
    sent = []

    class FakeBot:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def send_message(self, **kwargs):
            sent.append(kwargs["text"])

    monkeypatch.setattr(telegram_bot, "Bot", FakeBot)
    monkeypatch.setattr(telegram_bot, "HTTPXRequest", lambda **kwargs: None)
    monkeypatch.setattr(
        telegram_bot, "_build_daily_pick_messages", lambda _: (["\U0001f600" * 4001], [])
    )
    monkeypatch.setattr(telegram_bot, "persist_picks", lambda _: None)
    assert telegram_bot.send_daily_picks_now() == "\U0001f600" * 4001
    with pytest.raises(PublicationBlocked):
        asyncio.run(telegram_bot.publish_daily_picks(SimpleNamespace(bot=FakeBot())))
    assert len(sent) == 3
    assert "".join(sent) == "\U0001f600" * 4001


def test_daily_key_uses_local_date(monkeypatch, isolated_settings):
    import sqlite3
    from datetime import datetime, timezone

    class FixedDatetime:
        @staticmethod
        def now(tz):
            return datetime(2026, 9, 9, 2, tzinfo=timezone.utc).astimezone(tz)

    isolated_settings.bot_timezone = "America/Bogota"
    monkeypatch.setattr(telegram_bot, "datetime", FixedDatetime)
    monkeypatch.setattr(telegram_bot, "_build_daily_pick_messages", lambda _: (["test"], []))
    monkeypatch.setattr(telegram_bot, "persist_picks", lambda _: None)
    asyncio.run(telegram_bot._send_daily_picks(_DummyBot(), "1", False))
    with sqlite3.connect(isolated_settings.data_dir / "publications.sqlite3") as db:
        assert db.execute("SELECT day, state FROM publications").fetchone() == (
            "2026-09-08",
            "completed",
        )


def test_concurrent_async_batches_only_one_sends(monkeypatch):
    from sports_ai_bot.bot.publication import PublicationBlocked

    monkeypatch.setattr(telegram_bot, "_build_daily_pick_messages", lambda _: (["test"], []))
    monkeypatch.setattr(telegram_bot, "persist_picks", lambda _: None)

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def send_message(**kwargs):
            calls.append(kwargs)
            entered.set()
            await release.wait()

        bot = SimpleNamespace(send_message=send_message)
        first = asyncio.create_task(telegram_bot._send_daily_picks(bot, "1", False))
        await entered.wait()
        try:
            with pytest.raises(PublicationBlocked, match="curso"):
                await telegram_bot._send_daily_picks(bot, "1", False, kind="manual")
        finally:
            release.set()
            await first
        assert len(calls) == 1

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["daily", "manual"])
@pytest.mark.parametrize("failure", ["build", "persist"])
def test_failure_before_send_does_not_consume_publication(
    monkeypatch, isolated_settings, kind, failure
):
    import sqlite3

    def fail(*args):
        raise ValueError("pre-send failure")

    monkeypatch.setattr(telegram_bot, "_build_daily_pick_messages", lambda _: (["test"], []))
    monkeypatch.setattr(telegram_bot, "persist_picks", lambda _: None)
    target = "_build_daily_pick_messages" if failure == "build" else "persist_picks"
    original = getattr(telegram_bot, target)
    monkeypatch.setattr(telegram_bot, target, fail)
    bot = _DummyBot()
    with pytest.raises(ValueError, match="pre-send"):
        asyncio.run(telegram_bot._send_daily_picks(bot, "1", False, kind=kind))
    assert not bot.sent_messages
    monkeypatch.setattr(telegram_bot, target, original)
    asyncio.run(telegram_bot._send_daily_picks(bot, "1", False, kind=kind))
    assert bot.sent_messages == [("1", "test")]
    with sqlite3.connect(isolated_settings.data_dir / "publications.sqlite3") as db:
        assert db.execute("SELECT state FROM publications ORDER BY id").fetchall() == [
            ("failed-before-send",),
            ("completed",),
        ]


@pytest.mark.parametrize("kind", ["daily", "manual"])
def test_prepare_rejection_does_not_consume_daily_or_cooldown(monkeypatch, isolated_settings, kind):
    import hashlib
    from sports_ai_bot.bot.publication import Publication, PublicationBlocked

    previous = Publication(isolated_settings.data_dir, "1", kind, "2000-01-01", clock=lambda: 0)
    previous.start()
    previous.prepare(hashlib.sha256(b"same").hexdigest())
    previous.finish("failed-uncertain")
    previous.close()
    monkeypatch.setattr(telegram_bot, "_build_daily_pick_messages", lambda _: (["same"], []))
    monkeypatch.setattr(telegram_bot, "persist_picks", lambda _: None)
    bot = _DummyBot()
    with pytest.raises(PublicationBlocked, match="incierto"):
        asyncio.run(telegram_bot._send_daily_picks(bot, "1", False, kind=kind))
    assert not bot.sent_messages
    monkeypatch.setattr(telegram_bot, "_build_daily_pick_messages", lambda _: (["new"], []))
    asyncio.run(telegram_bot._send_daily_picks(bot, "1", False, kind=kind))
    assert bot.sent_messages == [("1", "new")]


def test_daily_waits_for_manual_to_finish(monkeypatch, isolated_settings):
    monkeypatch.setattr(telegram_bot, "_build_daily_pick_messages", lambda _: (["test"], []))
    monkeypatch.setattr(telegram_bot, "persist_picks", lambda _: None)

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        calls, waits = [], []

        async def send_message(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                entered.set()
                await release.wait()

        bot = SimpleNamespace(send_message=send_message)
        manual = asyncio.create_task(
            telegram_bot._send_daily_picks(
                bot, isolated_settings.telegram_chat_id, False, kind="manual"
            )
        )
        await entered.wait()

        async def sleep(delay):
            waits.append(delay)
            release.set()
            await manual

        monkeypatch.setattr(telegram_bot.asyncio, "sleep", sleep)
        try:
            await telegram_bot.publish_daily_picks(SimpleNamespace(bot=bot))
        finally:
            release.set()
            await manual
        assert waits == [5]
        assert len(calls) == 2

    asyncio.run(run())


def test_daily_busy_wait_is_bounded(monkeypatch):
    from sports_ai_bot.bot.publication import PublicationBusy

    attempts, waits = [], []

    async def send(*args, **kwargs):
        attempts.append(1)
        raise PublicationBusy("busy")

    async def sleep(delay):
        waits.append(delay)

    monkeypatch.setattr(telegram_bot, "_send_daily_picks", send)
    monkeypatch.setattr(telegram_bot.asyncio, "sleep", sleep)
    with pytest.raises(PublicationBusy):
        asyncio.run(telegram_bot.publish_daily_picks(SimpleNamespace(bot=object())))
    assert len(attempts) == 61
    assert waits == [5] * 60


@pytest.mark.parametrize("failure", ["duplicate", "uncertain", "timeout", "build"])
def test_daily_does_not_retry_non_busy_errors(monkeypatch, failure):
    from telegram.error import TimedOut
    from sports_ai_bot.bot.publication import PublicationBlocked

    error = {
        "duplicate": PublicationBlocked("duplicate"),
        "uncertain": PublicationBlocked("uncertain"),
        "timeout": TimedOut(),
        "build": ValueError("build"),
    }[failure]
    calls = []

    async def send(*args, **kwargs):
        calls.append(1)
        raise error

    async def sleep(delay):
        pytest.fail("Only busy may retry")

    monkeypatch.setattr(telegram_bot, "_send_daily_picks", send)
    monkeypatch.setattr(telegram_bot.asyncio, "sleep", sleep)
    with pytest.raises(type(error)):
        asyncio.run(telegram_bot.publish_daily_picks(SimpleNamespace(bot=object())))
    assert calls == [1]


@pytest.mark.parametrize(
    "user_id, admins, visible",
    [(42, "42", True), (7, "42", False), (None, "42", False), (42, "", False)],
)
def test_help_publishnow_only_for_admins(isolated_settings, user_id, admins, visible):
    isolated_settings.telegram_admin_ids = admins
    replies = []
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id) if user_id is not None else None,
        message=SimpleNamespace(reply_text=_async_reply(replies.append)),
    )
    asyncio.run(telegram_bot.help_command(update, None))
    assert ("/publishnow" in "".join(replies)) is visible
    if visible:
        assert "solo administradores" in "".join(replies)


def test_error_handler_logs_jobs_without_update(caplog):
    import logging

    error = RuntimeError("job failed")
    with caplog.at_level(logging.ERROR, logger=telegram_bot.__name__):
        asyncio.run(telegram_bot.error_handler(None, SimpleNamespace(error=error)))
    assert len(caplog.records) == 1
    assert caplog.records[0].exc_info[1] is error
    assert "job failed" in caplog.text
