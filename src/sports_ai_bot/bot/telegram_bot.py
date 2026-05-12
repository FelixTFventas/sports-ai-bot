from __future__ import annotations

import asyncio
from zoneinfo import ZoneInfo

import httpx
from telegram import Bot, Update
from telegram.error import TimedOut
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.request import HTTPXRequest

from sports_ai_bot.evaluate.performance import build_performance_report, format_performance_message
from sports_ai_bot.explain.messages import (
    build_best_message,
    build_forebet_value_message,
    build_market_message,
    build_prediction_message,
    build_value_message,
)
from sports_ai_bot.external.forebet import (
    ForebetError,
    fetch_48h_value_picks,
    fetch_match_value_picks,
    fetch_top_picks,
    format_top_picks_message,
)
from sports_ai_bot.research.corners import build_corners_picks
from sports_ai_bot.predict.pipeline import (
    Pick,
    build_best_picks,
    build_market_picks,
    build_top_picks,
    build_value_picks,
    persist_picks,
)
from sports_ai_bot.utils.config import get_settings


def _safe_message(message: str) -> str:
    return message[:4000]


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Bot listo. Usa /today, /over15, /over, /btts, /corners, /top, /forebettop, /forebetvalue, /forebet48h, /publishnow o /help."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = get_settings()
    await update.message.reply_text(
        "Comandos disponibles:\n"
        "/today - picks del dia\n"
        "/over15 - picks Over 1.5\n"
        "/over - picks Over 2.5\n"
        "/under45 - picks Under 4.5\n"
        "/btts - picks Ambos marcan\n"
        f"/corners - picks experimentales {settings.corners_pick_market_label()}\n"
        "/top - mejores picks disponibles\n"
        "/forebettop - top 5 publicados por Forebet\n"
        "/forebetvalue [url] - analiza value Forebet; sin url usa proximas 48h\n"
        "/forebet48h - value Forebet de partidos en proximas 48h\n"
        "/value - value picks con edge positivo\n"
        "/best - picks premium mas fuertes\n"
        "/publishnow - publica ahora en el chat configurado\n"
        "/performance - estado de rendimiento"
    )


async def today_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = build_top_picks()
    message = build_prediction_message(picks)
    await update.message.reply_text(_safe_message(message))


async def over_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = build_market_picks("Over 2.5", limit=10, threshold=0.60)
    message = build_market_message(picks, "Over 2.5")
    await update.message.reply_text(_safe_message(message))


async def over15_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = build_market_picks("Over 1.5", limit=10, threshold=0.65)
    message = build_market_message(picks, "Over 1.5")
    await update.message.reply_text(_safe_message(message))


async def btts_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = build_market_picks("BTTS", limit=10, threshold=0.60)
    message = build_market_message(picks, "BTTS")
    await update.message.reply_text(_safe_message(message))


async def under45_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = build_market_picks("Under 4.5", limit=10, threshold=0.72)
    message = build_market_message(picks, "Under 4.5")
    await update.message.reply_text(_safe_message(message))


async def corners_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = get_settings()
    picks = build_corners_picks(
        limit=6,
        days_ahead=2,
        limit_per_league=2,
        target_point=settings.corners_pick_point,
        selection=settings.corners_pick_selection,
    )
    persist_picks(picks)
    message = build_market_message(picks, settings.corners_pick_market_label())
    await update.message.reply_text(_safe_message(message))


async def top_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = build_top_picks(limit=10, threshold=0.60)
    message = build_prediction_message(picks)
    await update.message.reply_text(_safe_message(message))


async def forebet_top_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        picks = fetch_top_picks(limit=5)
    except (ForebetError, httpx.HTTPError):
        await update.message.reply_text(
            "No se pudieron obtener las top predicciones de Forebet hoy."
        )
        return

    await update.message.reply_text(_safe_message(format_top_picks_message(picks)))


async def forebet_value_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = getattr(context, "args", []) if context is not None else []
    try:
        if args:
            picks = fetch_match_value_picks(args[0], limit=5, min_odd=1.50)
        else:
            picks = fetch_48h_value_picks(limit_matches=30, limit=10, min_odd=1.50)
    except (ForebetError, httpx.HTTPError):
        await update.message.reply_text("No se pudieron analizar value picks de Forebet.")
        return

    await update.message.reply_text(_safe_message(build_forebet_value_message(picks)))


async def forebet_48h_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        picks = fetch_48h_value_picks(limit_matches=30, limit=10, min_odd=1.50)
    except (ForebetError, httpx.HTTPError):
        await update.message.reply_text("No se pudieron analizar value picks Forebet de 48h.")
        return

    await update.message.reply_text(_safe_message(build_forebet_value_message(picks)))


async def value_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = build_value_picks(limit=5)
    message = build_value_message(picks)
    await update.message.reply_text(_safe_message(message))


async def best_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = build_best_picks(limit=5)
    message = build_best_message(picks)
    await update.message.reply_text(_safe_message(message))


async def performance_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    report = build_performance_report()
    await update.message.reply_text(_safe_message(format_performance_message(report)))


async def publishnow_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await publish_daily_picks(context)
    await update.message.reply_text("Publicacion manual enviada al chat configurado.")


async def _send_daily_picks(bot: Bot, chat_id: str, refresh_fixtures: bool) -> str:
    messages, picks = _build_daily_pick_messages(refresh_fixtures=refresh_fixtures)
    persist_picks(picks)
    for message in messages:
        await _send_message_with_retry(bot, chat_id=chat_id, text=_safe_message(message))
    return "\n\n".join(messages)


async def _send_message_with_retry(bot: Bot, chat_id: str, text: str) -> None:
    try:
        await bot.send_message(chat_id=chat_id, text=text)
    except TimedOut:
        await asyncio.sleep(2)
        await bot.send_message(chat_id=chat_id, text=text)


def _build_daily_pick_messages(refresh_fixtures: bool) -> tuple[list[str], list[Pick]]:
    settings = get_settings()
    messages: list[str] = []
    all_picks: list[Pick] = []

    candidate_picks = build_top_picks(
        limit=120,
        threshold=0.50,
        refresh_fixtures=refresh_fixtures,
    )
    top_picks = candidate_picks[:20]
    messages.append(build_prediction_message(top_picks))
    all_picks.extend(top_picks)

    market_specs = [
        ("Over 1.5", 10, 0.65),
        ("Over 2.5", 10, 0.60),
        ("Under 4.5", 10, 0.72),
        ("BTTS", 10, 0.60),
    ]
    for market, limit, threshold in market_specs:
        picks = _filter_market_picks(candidate_picks, market, limit=limit, threshold=threshold)
        messages.append(build_market_message(picks, market))
        all_picks.extend(picks)

    value_picks = _filter_value_picks(candidate_picks, limit=10)
    messages.append(build_value_message(value_picks))
    all_picks.extend(value_picks)

    best_picks = _filter_best_picks(candidate_picks, limit=5)
    messages.append(build_best_message(best_picks))
    all_picks.extend(best_picks)

    if settings.has_the_odds_api():
        corners_picks = build_corners_picks(
            limit=6,
            days_ahead=2,
            limit_per_league=2,
            target_point=settings.corners_pick_point,
            selection=settings.corners_pick_selection,
        )
        messages.append(build_market_message(corners_picks, settings.corners_pick_market_label()))
        all_picks.extend(corners_picks)

    return messages, all_picks


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
    await _send_daily_picks(
        context.bot,
        chat_id=settings.telegram_chat_id,
        refresh_fixtures=False,
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text("Hubo un error procesando la solicitud.")


def _build_application() -> Application:
    settings = get_settings()
    missing = settings.missing_bot_env()
    if missing:
        raise ValueError(f"Faltan variables de entorno del bot: {', '.join(missing)}")

    application = Application.builder().token(settings.telegram_bot_token).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("today", today_command))
    application.add_handler(CommandHandler("over15", over15_command))
    application.add_handler(CommandHandler("over", over_command))
    application.add_handler(CommandHandler("under45", under45_command))
    application.add_handler(CommandHandler("btts", btts_command))
    application.add_handler(CommandHandler("corners", corners_command))
    application.add_handler(CommandHandler("top", top_command))
    application.add_handler(CommandHandler("forebettop", forebet_top_command))
    application.add_handler(CommandHandler("forebetvalue", forebet_value_command))
    application.add_handler(CommandHandler("forebet48h", forebet_48h_command))
    application.add_handler(CommandHandler("value", value_command))
    application.add_handler(CommandHandler("best", best_command))
    application.add_handler(CommandHandler("publishnow", publishnow_command))
    application.add_handler(CommandHandler("performance", performance_command))
    application.add_error_handler(error_handler)

    hour, minute = settings.post_hour_local.split(":")
    application.job_queue.run_daily(
        publish_daily_picks,
        time=_local_time(int(hour), int(minute), settings.bot_timezone),
        name="daily-picks",
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
