from __future__ import annotations

import argparse
import sys

from sports_ai_bot.bot.telegram_bot import run_bot, send_daily_picks_now
from sports_ai_bot.collect.historical import download_historical_data
from sports_ai_bot.evaluate.performance import (
    build_performance_report,
    format_performance_message,
    settle_picks,
)
from sports_ai_bot.explain.messages import (
    build_best_message,
    build_market_message,
    build_prediction_message,
    build_value_message,
)
from sports_ai_bot.research.corners import (
    build_corners_picks,
    build_corners_odds_preview,
    build_corners_research_report,
    build_target_corners_odds_preview,
    format_corners_odds_preview,
    format_corners_research_report,
    format_target_corners_odds_preview,
)
from sports_ai_bot.features.build import build_fixture_features, build_training_dataset
from sports_ai_bot.predict.pipeline import (
    build_best_picks,
    build_market_picks,
    build_top_picks,
    build_value_picks,
)
from sports_ai_bot.train.train_models import train_models
from sports_ai_bot.utils.config import get_settings


def _configure_console_encoding() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def main() -> None:
    _configure_console_encoding()
    parser = argparse.ArgumentParser(description="Sports AI Bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("fetch-data")
    subparsers.add_parser("build-dataset")
    subparsers.add_parser("build-fixtures")
    subparsers.add_parser("train")
    subparsers.add_parser("run-bot")
    subparsers.add_parser("send-today")
    subparsers.add_parser("preview-message")
    subparsers.add_parser("preview-over15")
    subparsers.add_parser("preview-under45")
    subparsers.add_parser("preview-corners-odds")
    subparsers.add_parser("preview-corners")
    subparsers.add_parser("preview-corners-picks")
    subparsers.add_parser("save-corners-picks")
    subparsers.add_parser("research-corners")
    subparsers.add_parser("check-config")
    subparsers.add_parser("update-results")
    subparsers.add_parser("report-performance")
    subparsers.add_parser("preview-value")
    subparsers.add_parser("preview-best")

    args = parser.parse_args()

    if args.command == "fetch-data":
        download_historical_data()
    elif args.command == "build-dataset":
        dataset = build_training_dataset()
        print(f"Dataset generado con {len(dataset)} filas")
    elif args.command == "build-fixtures":
        fixtures = build_fixture_features()
        print(f"Fixtures con features generados: {len(fixtures)}")
    elif args.command == "train":
        summary = train_models()
        print(summary)
    elif args.command == "preview-message":
        picks = build_top_picks()
        message = build_prediction_message(picks)
        print(message)
    elif args.command == "preview-over15":
        picks = build_market_picks("Over 1.5", limit=10, threshold=0.65)
        print(build_market_message(picks, "Over 1.5"))
    elif args.command == "preview-under45":
        picks = build_market_picks("Under 4.5", limit=10, threshold=0.72)
        print(build_market_message(picks, "Under 4.5"))
    elif args.command == "preview-corners-odds":
        report = build_corners_odds_preview(days_ahead=2, limit_per_league=3)
        print(format_corners_odds_preview(report))
    elif args.command == "preview-corners":
        settings = get_settings()
        report = build_target_corners_odds_preview(
            days_ahead=2,
            limit_per_league=3,
            target_point=settings.corners_pick_point,
            selection=settings.corners_pick_selection,
        )
        print(format_target_corners_odds_preview(report))
    elif args.command == "preview-corners-picks":
        settings = get_settings()
        picks = build_corners_picks(
            limit=6,
            days_ahead=2,
            limit_per_league=2,
            target_point=settings.corners_pick_point,
            selection=settings.corners_pick_selection,
        )
        print(build_market_message(picks, settings.corners_pick_market_label()))
    elif args.command == "save-corners-picks":
        settings = get_settings()
        picks = build_corners_picks(
            limit=6,
            days_ahead=2,
            limit_per_league=2,
            target_point=settings.corners_pick_point,
            selection=settings.corners_pick_selection,
        )
        from sports_ai_bot.predict.pipeline import persist_picks

        persist_picks(picks)
        print(build_market_message(picks, settings.corners_pick_market_label()))
    elif args.command == "research-corners":
        report = build_corners_research_report(days_ahead=2)
        print(format_corners_research_report(report))
    elif args.command == "check-config":
        settings = get_settings()
        print(
            {
                "missing_bot_env": settings.missing_bot_env(),
                "post_hour_local": settings.post_hour_local,
                "the_odds_api_configured": settings.has_the_odds_api(),
                "the_odds_api_bookmakers": settings.the_odds_api_bookmakers_list(),
                "corners_pick_selection": settings.corners_pick_selection,
                "corners_pick_point": settings.corners_pick_point,
                "corners_pick_min_price": settings.corners_pick_min_price,
                "corners_pick_max_price": settings.corners_pick_max_price,
            }
        )
    elif args.command == "run-bot":
        run_bot()
    elif args.command == "send-today":
        print(send_daily_picks_now())
    elif args.command == "update-results":
        print(settle_picks())
    elif args.command == "report-performance":
        report = build_performance_report()
        print(format_performance_message(report))
    elif args.command == "preview-value":
        picks = build_value_picks()
        print(build_value_message(picks))
    elif args.command == "preview-best":
        picks = build_best_picks()
        print(build_best_message(picks))


if __name__ == "__main__":
    main()
