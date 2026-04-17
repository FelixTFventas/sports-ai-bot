from datetime import date

from sports_ai_bot.collect.api_football import _season_year_for_date
from sports_ai_bot.collect.odds import _extract_api_football_over25


def test_season_year_uses_previous_year_before_july() -> None:
    assert _season_year_for_date(date(2026, 4, 17)) == 2025
    assert _season_year_for_date(date(2026, 8, 1)) == 2026


def test_extract_api_football_over25_decimal() -> None:
    bookmakers = [
        {
            "bets": [
                {
                    "name": "Goals Over/Under",
                    "values": [
                        {"value": "Over 2.5", "odd": "1.83"},
                        {"value": "Under 2.5", "odd": "1.95"},
                    ],
                }
            ]
        }
    ]
    assert _extract_api_football_over25(bookmakers) == 1.83
