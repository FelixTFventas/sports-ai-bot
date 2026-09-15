from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.request import HTTPXRequest

from sports_ai_bot.evaluate.performance import build_performance_report, format_performance_message
from sports_ai_bot.explain.messages import (
    build_best_message,
    build_market_message,
    build_prediction_message,
    build_value_message,
)
from sports_ai_bot.external.forebet import (
    ForebetError,
    format_forebet_message,
    get_forebet_predictions,
)
from sports_ai_bot.predict.pipeline import (
    Pick,
    build_best_picks,
    build_market_picks,
    build_top_picks,
    build_value_picks,
    persist_picks,
)
from sports_ai_bot.utils.config import get_settings
from sports_ai_bot.predict.service import (
    generate_picks,
    cached_picks,
    format_picks,
    format_quotes,
    status_message,
)
from sports_ai_bot.storage import Store
from sports_ai_bot.bot.publication import (
    Publication,
    PublicationBlocked,
    PublicationBusy,
    shared_work,
)


logger = logging.getLogger(__name__)
DAILY_BUSY_RETRIES = 60
DAILY_BUSY_DELAY = 5


def split_message(message: str) -> list[str]:
    chunks = []
    while message:
        units = 0
        end = 0
        for char in message:
            size = 2 if ord(char) > 0xFFFF else 1
            if units + size > 4000:
                break
            units += size
            end += 1
        if end < len(message):
            paragraph = message.rfind("\n\n", 0, end)
            line = message.rfind("\n", 0, end)
            if paragraph >= 0:
                end = paragraph + 2
            elif line >= 0:
                end = line + 1
        chunks.append(message[:end])
        message = message[end:]
    return chunks


async def _reply(update: Update, message: str) -> None:
    target = getattr(update, "effective_message", None) or getattr(update, "message", None)
    if target is not None:
        for chunk in split_message(message):
            await target.reply_text(chunk)


async def _work(function, *args, **kwargs):
    def run():
        with shared_work(get_settings().data_dir):
            return function(*args, **kwargs)

    return await asyncio.to_thread(run)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await help_command(update, context)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = get_settings()
    user = getattr(update, "effective_user", None)
    publish_help = (
        "/publishnow - publica ahora en el chat configurado (solo administradores)\n"
        if user is not None and user.id in settings.telegram_admin_ids_set()
        else ""
    )
    await _reply(
        update,
        "Comandos disponibles:\n"
        "/picks - analisis local vigente en cache\n"
        "/cuotas - cuotas importadas manualmente, sin actualizacion automatica de casas\n"
        "/forebet - consulta privada y manual de pronosticos Forebet experimentales\n"
        "/rendimiento - estado de rendimiento\n"
        f"{publish_help}"
        "/help - ayuda\n"
        "/start - inicio",
    )


async def picks_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = await _work(cached_picks)
    message = format_picks(picks) if picks else await _work(status_message)
    await _reply(update, message)


async def cuotas_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _reply(update, await _work(format_quotes))


async def experimental_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _reply(
        update,
        "Desactivado: contenido experimental de referencia, no una prediccion validada. "
        "No se consultan fuentes externas. Usa /picks.",
    )


async def today_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = await _work(build_top_picks)
    message = build_prediction_message(picks)
    await _reply(update, message)


async def over_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = await _work(build_market_picks, "Over 2.5", limit=10, threshold=0.60)
    message = build_market_message(picks, "Over 2.5")
    await _reply(update, message)


async def over15_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = await _work(build_market_picks, "Over 1.5", limit=10, threshold=0.65)
    message = build_market_message(picks, "Over 1.5")
    await _reply(update, message)


async def btts_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = await _work(build_market_picks, "BTTS", limit=10, threshold=0.60)
    message = build_market_message(picks, "BTTS")
    await _reply(update, message)


async def under45_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = await _work(build_market_picks, "Under 4.5", limit=10, threshold=0.72)
    message = build_market_message(picks, "Under 4.5")
    await _reply(update, message)


async def corners_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await experimental_command(update, context)


async def top_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = await _work(build_top_picks, limit=10, threshold=0.60)
    message = build_prediction_message(picks)
    await _reply(update, message)


async def forebet_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = get_settings()
    try:
        result = await asyncio.to_thread(
            get_forebet_predictions,
            settings.data_dir,
            timezone_name=settings.bot_timezone,
            cache_minutes=settings.forebet_cache_minutes,
            min_probability=settings.forebet_min_probability,
            limit=settings.forebet_limit,
            browser_executable=settings.forebet_browser_executable,
        )
    except ForebetError as exc:
        logger.warning("Consulta Forebet no disponible: %s", exc)
        await _reply(
            update,
            "No se pudo consultar Forebet con el navegador local. "
            "Comprueba que Chrome o Edge esten disponibles e intentalo de nuevo.",
        )
        return
    await _reply(update, format_forebet_message(result, settings.bot_timezone))


async def value_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = await _work(build_value_picks, limit=5)
    message = build_value_message(picks)
    await _reply(update, message)


async def best_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = await _work(build_best_picks, limit=5)
    message = build_best_message(picks)
    await _reply(update, message)


async def performance_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    report = await _work(build_performance_report)
    await _reply(update, format_performance_message(report))


async def publishnow_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = get_settings()
    user = getattr(update, "effective_user", None)
    if user is None or user.id not in settings.telegram_admin_ids_set():
        await _reply(update, "No autorizado para publicar.")
        return
    try:
        await _send_daily_picks(context.bot, settings.telegram_chat_id, False, kind="manual")
    except PublicationBlocked as exc:
        await _reply(update, str(exc))
        return
    await _reply(update, "Publicacion manual enviada al chat configurado.")


async def _send_daily_picks(
    bot: Bot, chat_id: str, refresh_fixtures: bool, *, kind: str = "daily"
) -> str:
    settings = get_settings()
    publication = Publication(
        settings.data_dir,
        chat_id,
        kind,
        datetime.now(ZoneInfo(settings.bot_timezone)).date().isoformat(),
    )

    # Shield the worker's lifetime: cancellation must not release its gate while
    # a to_thread operation is still preparing or recording this batch.
    async def run():
        await asyncio.to_thread(publication.start)
        send_started = False
        try:

            def build():
                messages, picks = _build_daily_pick_messages(refresh_fixtures)
                persist_picks(picks)
                return messages, picks

            messages, picks = await _work(build)
            text = "\n\n".join(messages)
            await asyncio.to_thread(publication.prepare, hashlib.sha256(text.encode()).hexdigest())
            for message in messages:
                for chunk in split_message(message):
                    # TimedOut is uncertain delivery, never an automatic retry.
                    send_started = True
                    await bot.send_message(chat_id=chat_id, text=chunk)
            published_picks = [
                pick for pick in picks
                if getattr(pick, "quote_id", None) and getattr(pick, "model_version", None)
            ]
            if published_picks:
                def mark_published():
                    Store(settings.data_dir).mark_published(published_picks)

                await _work(mark_published)
            await asyncio.to_thread(publication.finish, "completed")
            return text
        except BaseException:
            await asyncio.to_thread(
                publication.finish,
                "failed-uncertain" if send_started else "failed-before-send",
            )
            raise
        finally:
            publication.close()

    task = asyncio.create_task(run())
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        finally:
            raise


def _build_daily_pick_messages(refresh_fixtures: bool) -> tuple[list[str], list[Pick]]:
    # The legacy refresh flag must never trigger a network refresh.
    picks = generate_picks()
    return [format_picks(picks) if picks else status_message()], picks


def _filter_market_picks(
    picks: list[Pick], market: str, limit: int, threshold: float
) -> list[Pick]:
    return [
        pick
        for pick in picks
        if pick.market.lower() == market.lower() and pick.probability >= threshold
    ][:limit]


def _filter_value_picks(picks: list[Pick], limit: int) -> list[Pick]:
    return [
        pick
        for pick in picks
        if pick.edge is not None
        and pick.edge >= 0.02
        and pick.expected_value is not None
        and pick.expected_value >= 0.0
    ][:limit]


def _filter_best_picks(picks: list[Pick], limit: int) -> list[Pick]:
    best_by_match: dict[str, Pick] = {}
    for pick in _filter_value_picks(picks, limit=limit * 6):
        if pick.expected_value is None or pick.expected_value < 0.02:
            continue
        current = best_by_match.get(pick.match_label)
        if current is None or _pick_message_score(pick) > _pick_message_score(current):
            best_by_match[pick.match_label] = pick
    return sorted(best_by_match.values(), key=_pick_message_score, reverse=True)[:limit]


def _pick_message_score(pick: Pick) -> tuple[float, float, float]:
    return (
        float(pick.expected_value or 0.0),
        float(pick.edge or 0.0),
        float(pick.probability),
    )


async def publish_daily_picks(context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = get_settings()
    for attempt in range(DAILY_BUSY_RETRIES + 1):
        try:
            await _send_daily_picks(
                context.bot,
                chat_id=settings.telegram_chat_id,
                refresh_fixtures=False,
            )
            return
        except PublicationBusy:
            if attempt == DAILY_BUSY_RETRIES:
                raise
            await asyncio.sleep(DAILY_BUSY_DELAY)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    error = context.error
    logger.error(
        "Error procesando solicitud o job de Telegram",
        exc_info=(type(error), error, error.__traceback__),
    )
    if isinstance(update, Update) and update.effective_message:
        await _reply(
            update,
            "Hubo un error procesando la solicitud; si hubo un envio, "
            "su estado puede ser incierto y no se reintentara automaticamente.",
        )


async def refresh_analysis(context: ContextTypes.DEFAULT_TYPE) -> None:
    await _work(generate_picks)


def _build_application() -> Application:
    settings = get_settings()
    missing = settings.missing_bot_env()
    if missing:
        raise ValueError(f"Faltan variables de entorno del bot: {', '.join(missing)}")

    application = Application.builder().token(settings.telegram_bot_token).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("picks", picks_command))
    application.add_handler(CommandHandler("cuotas", cuotas_command))
    application.add_handler(CommandHandler("rendimiento", performance_command))
    for alias in ("today", "top", "value", "best", "over", "btts", "over15", "under45"):
        application.add_handler(CommandHandler(alias, picks_command))
    application.add_handler(CommandHandler("corners", corners_command))
    for alias in ("forebet", "forebettop", "forebetvalue", "forebet48h"):
        application.add_handler(CommandHandler(alias, forebet_command))
    application.add_handler(CommandHandler("publishnow", publishnow_command))
    application.add_handler(CommandHandler("performance", performance_command))
    application.add_error_handler(error_handler)

    hour, minute = settings.post_hour_local.split(":")
    application.job_queue.run_daily(
        publish_daily_picks,
        time=_local_time(int(hour), int(minute), settings.bot_timezone),
        name="daily-picks",
    )
    application.job_queue.run_repeating(
        refresh_analysis, interval=15 * 60, first=10, name="refresh_analysis",
    )
    return application


def _local_time(hour: int, minute: int, timezone_name: str):
    from datetime import time

    return time(hour=hour, minute=minute, tzinfo=ZoneInfo(timezone_name))


def run_bot() -> None:
    application = _build_application()
    application.run_polling()


def send_daily_picks_now(refresh_fixtures: bool = True) -> str:
    settings = get_settings()
    missing = settings.missing_bot_env()
    if missing:
        raise ValueError(f"Faltan variables de entorno del bot: {', '.join(missing)}")

    async def _runner() -> str:
        request = HTTPXRequest(connect_timeout=30, read_timeout=60, write_timeout=30)
        async with Bot(token=settings.telegram_bot_token, request=request) as bot:
            return await _send_daily_picks(
                bot,
                chat_id=settings.telegram_chat_id,
                refresh_fixtures=refresh_fixtures,
            )

    return asyncio.run(_runner())
