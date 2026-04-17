from __future__ import annotations

import argparse

from sports_ai_bot.bot.telegram_bot import run_bot
from sports_ai_bot.collect.historical import download_historical_data
from sports_ai_bot.evaluate.performance import (
    build_performance_report,
    format_performance_message,
    settle_picks,
)
from sports_ai_bot.explain.messages import build_prediction_message, build_value_message
from sports_ai_bot.features.build import build_fixture_features, build_training_dataset
from sports_ai_bot.predict.pipeline import build_top_picks, build_value_picks
from sports_ai_bot.train.train_models import train_models
from sports_ai_bot.utils.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Sports AI Bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("fetch-data")
    subparsers.add_parser("build-dataset")
    subparsers.add_parser("build-fixtures")
    subparsers.add_parser("train")
    subparsers.add_parser("run-bot")
    subparsers.add_parser("preview-message")
    subparsers.add_parser("check-config")
    subparsers.add_parser("update-results")
    subparsers.add_parser("report-performance")
    subparsers.add_parser("preview-value")

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
    elif args.command == "check-config":
        settings = get_settings()
        print(
            {
                "missing_bot_env": settings.missing_bot_env(),
                "post_hour_local": settings.post_hour_local,
            }
        )
    elif args.command == "run-bot":
        run_bot()
    elif args.command == "update-results":
        print(settle_picks())
    elif args.command == "report-performance":
        report = build_performance_report()
        print(format_performance_message(report))
    elif args.command == "preview-value":
        picks = build_value_picks()
        print(build_value_message(picks))


if __name__ == "__main__":
    main()
