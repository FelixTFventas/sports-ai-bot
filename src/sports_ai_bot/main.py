from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sports_ai_bot.bot.publication import shared_work
from sports_ai_bot.bot.telegram_bot import run_bot, send_daily_picks_now
from sports_ai_bot.collect.historical import LEAGUES, download_historical_data
from sports_ai_bot.collect.local import import_history
from sports_ai_bot.evaluate.performance import build_performance_report, format_performance_message, settle_picks
from sports_ai_bot.features.build import build_training_dataset
from sports_ai_bot.predict.service import generate_picks, format_picks, format_quotes, status_message
from sports_ai_bot.storage import BOOKMAKERS, Store
from sports_ai_bot.train.train_models import train_models
from sports_ai_bot.utils.config import get_settings


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Picks auditables con cuotas manuales de Colombia")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("coverage", "check-config", "fetch-data", "build-dataset", "train",
                 "run-bot", "send-today", "quotes", "status", "update-results", "report-performance",
                 "preview-message", "preview-value", "preview-best"):
        commands.add_parser(name)
    generate = commands.add_parser("generate", help="Analiza cuotas locales, sin red")
    generate.add_argument("--observation", action="store_true", help="Registro experimental, nunca publicado")
    quotes = commands.add_parser("import-quotes", help="Importa cuotas observadas manualmente")
    quotes.add_argument("file", type=Path)
    history = commands.add_parser("import-history", help="Importa historicos/resultados autorizados")
    history.add_argument("file", type=Path)
    history.add_argument("--league", required=True, choices=list(LEAGUES.values()))
    history.add_argument("--source-url", required=True)
    history.add_argument("--permission-note", required=True)
    args = parser.parse_args()
    settings = get_settings()
    try:
        if args.command == "run-bot":
            run_bot()
            return
        if args.command == "send-today":
            print(send_daily_picks_now(refresh_fixtures=False))
            return
        with shared_work(settings.data_dir):
            if args.command == "coverage":
                print("Casas: entrada manual, integracion automatica colombiana NO verificada.")
                for key, title in BOOKMAKERS.items():
                    print(f"{key}: {title} | manual | mercados Over 2.5 y BTTS Si, 90 minutos")
                print("Ligas admitidas para importacion (no implica modelo validado):")
                print(", ".join(LEAGUES.values()))
            elif args.command == "check-config":
                print(json.dumps({
                    "missing_bot_env": settings.missing_bot_env(),
                    "admin_configured": bool(settings.telegram_admin_ids_set()),
                    "data_dir": str(settings.data_dir),
                    "quote_max_age_minutes": settings.quote_max_age_minutes,
                    "history_max_age_days": settings.history_max_age_days,
                    "historical_download_authorized": settings.historical_download_authorized,
                }, indent=2))
            elif args.command == "fetch-data":
                download_historical_data()
            elif args.command == "import-history":
                count = import_history(args.file, args.league, args.source_url, args.permission_note)
                print(f"Importados {count} resultados. {settle_picks()}")
            elif args.command == "import-quotes":
                count = Store(settings.data_dir).import_quotes(args.file)
                print(f"Importadas {count} observaciones nuevas. No son cuotas en vivo.")
                picks = generate_picks()
                print(format_picks(picks))
            elif args.command == "build-dataset":
                print(f"Dataset: {len(build_training_dataset())} filas")
            elif args.command == "train":
                print(json.dumps(train_models(), indent=2))
            elif args.command in {"generate", "preview-message", "preview-value", "preview-best"}:
                print(format_picks(generate_picks(observation=getattr(args, "observation", False))))
            elif args.command == "quotes":
                print(format_quotes())
            elif args.command == "status":
                print(status_message())
            elif args.command == "update-results":
                print("Liquidacion con resultados locales. Para nuevos resultados usa import-history.")
                print(settle_picks())
            elif args.command == "report-performance":
                print(format_performance_message(build_performance_report()))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
