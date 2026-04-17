from __future__ import annotations

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from sports_ai_bot.evaluate.performance import build_performance_report, format_performance_message
from sports_ai_bot.explain.messages import (
    build_market_message,
    build_prediction_message,
    build_value_message,
)
from sports_ai_bot.predict.pipeline import build_top_picks, build_value_picks, persist_picks
from sports_ai_bot.utils.config import get_settings


def _safe_message(message: str) -> str:
    return message[:4000]


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Bot listo. Usa /today, /over, /btts, /top, /publishnow o /help."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Comandos disponibles:\n"
        "/today - picks del dia\n"
        "/over - picks Over 2.5\n"
        "/btts - picks BTTS\n"
        "/top - mejores picks disponibles\n"
        "/value - value picks con edge positivo\n"
        "/publishnow - publica ahora en el chat configurado\n"
        "/performance - estado de rendimiento"
    )


async def today_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = build_top_picks()
    message = build_prediction_message(picks)
    await update.message.reply_text(_safe_message(message))


async def over_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = build_top_picks(limit=10, threshold=0.6)
    message = build_market_message(picks, "Over 2.5")
    await update.message.reply_text(_safe_message(message))


async def btts_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = build_top_picks(limit=10, threshold=0.6)
    message = build_market_message(picks, "BTTS")
    await update.message.reply_text(_safe_message(message))


async def top_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = build_top_picks(limit=5, threshold=0.6)
    message = build_prediction_message(picks)
    await update.message.reply_text(_safe_message(message))


async def value_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = build_value_picks(limit=5)
    message = build_value_message(picks)
    await update.message.reply_text(_safe_message(message))


async def performance_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    report = build_performance_report()
    await update.message.reply_text(_safe_message(format_performance_message(report)))


async def publishnow_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await publish_daily_picks(context)
    await update.message.reply_text("Publicacion manual enviada al chat configurado.")


async def publish_daily_picks(context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = get_settings()
    picks = build_top_picks()
    persist_picks(picks)
    message = build_prediction_message(picks)
    await context.bot.send_message(chat_id=settings.telegram_chat_id, text=_safe_message(message))


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
    application.add_handler(CommandHandler("over", over_command))
    application.add_handler(CommandHandler("btts", btts_command))
    application.add_handler(CommandHandler("top", top_command))
    application.add_handler(CommandHandler("value", value_command))
    application.add_handler(CommandHandler("publishnow", publishnow_command))
    application.add_handler(CommandHandler("performance", performance_command))
    application.add_error_handler(error_handler)

    hour, minute = settings.post_hour_local.split(":")
    application.job_queue.run_daily(
        publish_daily_picks,
        time=_local_time(int(hour), int(minute)),
        name="daily-picks",
    )
    return application


def _local_time(hour: int, minute: int):
    from datetime import time

    return time(hour=hour, minute=minute)


def run_bot() -> None:
    application = _build_application()
    application.run_polling()
