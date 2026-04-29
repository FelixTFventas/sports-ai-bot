import asyncio
from types import SimpleNamespace

from sports_ai_bot.bot import telegram_bot
from sports_ai_bot.external.forebet import ForebetError, ForebetPick


class _DummyBot:
    def __init__(self) -> None:
        self.sent_messages: list[tuple[str, str]] = []

    async def send_message(self, chat_id: str, text: str) -> None:
        self.sent_messages.append((chat_id, text))


def test_send_daily_picks_builds_and_sends_message(monkeypatch) -> None:
    bot = _DummyBot()
    picks = [object()]
    persisted: list[list[object]] = []
    refresh_flags: list[bool] = []

    monkeypatch.setattr(
        telegram_bot,
        "build_top_picks",
        lambda refresh_fixtures: refresh_flags.append(refresh_fixtures) or picks,
    )
    monkeypatch.setattr(telegram_bot, "persist_picks", lambda value: persisted.append(value))
    monkeypatch.setattr(telegram_bot, "build_prediction_message", lambda value: "Picks del dia")

    message = asyncio.run(telegram_bot._send_daily_picks(bot, chat_id="-1001", refresh_fixtures=True))

    assert message == "Picks del dia"
    assert refresh_flags == [True]
    assert persisted == [picks]
    assert bot.sent_messages == [("-1001", "Picks del dia")]


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


def test_forebet_top_command_sends_message(monkeypatch) -> None:
    replies: list[str] = []
    update = SimpleNamespace(message=SimpleNamespace(reply_text=lambda text: replies.append(text)))
    update.message.reply_text = _async_reply(update.message.reply_text)

    monkeypatch.setattr(
        telegram_bot,
        "fetch_top_picks",
        lambda limit=5: [
            ForebetPick(
                match_label="A B",
                match_date="04/24/2026",
                match_time="5:00 PM",
                prediction="2",
                probability=80,
                predicted_score="1-3",
                source_url="https://example.com",
            )
        ],
    )
    monkeypatch.setattr(
        telegram_bot,
        "format_top_picks_message",
        lambda picks: "Top 5 Forebet del dia:\n1. A B",
    )

    asyncio.run(telegram_bot.forebet_top_command(update, None))

    assert replies == ["Top 5 Forebet del dia:\n1. A B"]


def test_forebet_top_command_handles_failure(monkeypatch) -> None:
    replies: list[str] = []
    update = SimpleNamespace(message=SimpleNamespace(reply_text=lambda text: replies.append(text)))
    update.message.reply_text = _async_reply(update.message.reply_text)

    monkeypatch.setattr(
        telegram_bot,
        "fetch_top_picks",
        lambda limit=5: (_ for _ in ()).throw(ForebetError("boom")),
    )

    asyncio.run(telegram_bot.forebet_top_command(update, None))

    assert replies == ["No se pudieron obtener las top predicciones de Forebet hoy."]


def test_forebet_value_command_without_url_uses_48h_value(monkeypatch) -> None:
    replies: list[str] = []
    update = SimpleNamespace(message=SimpleNamespace(reply_text=lambda text: replies.append(text)))
    update.message.reply_text = _async_reply(update.message.reply_text)
    context = SimpleNamespace(args=[])
    picks = [object()]

    monkeypatch.setattr(
        telegram_bot,
        "fetch_48h_value_picks",
        lambda limit_matches=30, limit=10, min_odd=1.50: picks,
    )
    monkeypatch.setattr(telegram_bot, "build_forebet_value_message", lambda value: "Forebet 48h value")

    asyncio.run(telegram_bot.forebet_value_command(update, context))

    assert replies == ["Forebet 48h value"]


def test_forebet_value_command_sends_message(monkeypatch) -> None:
    replies: list[str] = []
    update = SimpleNamespace(message=SimpleNamespace(reply_text=lambda text: replies.append(text)))
    update.message.reply_text = _async_reply(update.message.reply_text)
    context = SimpleNamespace(args=["https://www.forebet.com/es/football/matches/a-b-1"])
    picks = [object()]

    monkeypatch.setattr(
        telegram_bot,
        "fetch_match_value_picks",
        lambda url, limit=5, min_odd=1.50: picks,
    )
    monkeypatch.setattr(telegram_bot, "build_forebet_value_message", lambda value: "Forebet value")

    asyncio.run(telegram_bot.forebet_value_command(update, context))

    assert replies == ["Forebet value"]


def test_forebet_value_command_handles_failure(monkeypatch) -> None:
    replies: list[str] = []
    update = SimpleNamespace(message=SimpleNamespace(reply_text=lambda text: replies.append(text)))
    update.message.reply_text = _async_reply(update.message.reply_text)
    context = SimpleNamespace(args=["https://www.forebet.com/es/football/matches/a-b-1"])

    monkeypatch.setattr(
        telegram_bot,
        "fetch_match_value_picks",
        lambda url, limit=5, min_odd=1.50: (_ for _ in ()).throw(ForebetError("boom")),
    )

    asyncio.run(telegram_bot.forebet_value_command(update, context))

    assert replies == ["No se pudieron analizar value picks de Forebet."]


def test_forebet_48h_command_sends_message(monkeypatch) -> None:
    replies: list[str] = []
    update = SimpleNamespace(message=SimpleNamespace(reply_text=lambda text: replies.append(text)))
    update.message.reply_text = _async_reply(update.message.reply_text)
    picks = [object()]

    monkeypatch.setattr(
        telegram_bot,
        "fetch_48h_value_picks",
        lambda limit_matches=30, limit=10, min_odd=1.50: picks,
    )
    monkeypatch.setattr(telegram_bot, "build_forebet_value_message", lambda value: "Forebet 48h")

    asyncio.run(telegram_bot.forebet_48h_command(update, None))

    assert replies == ["Forebet 48h"]


def _async_reply(fn):
    async def wrapper(text: str) -> None:
        fn(text)

    return wrapper
