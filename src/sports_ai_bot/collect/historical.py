from __future__ import annotations

from datetime import datetime
from pathlib import Path

import httpx

from sports_ai_bot.utils.config import get_settings
from sports_ai_bot.utils.logging import get_logger


LOGGER = get_logger(__name__)

LEAGUES = {
    "E0": "premier_league",
    "SP1": "la_liga",
    "I1": "serie_a",
    "D1": "bundesliga",
    "F1": "ligue_1",
}

BASE_URL = "https://www.football-data.co.uk/mmz4281"


def current_season_code(today: datetime | None = None) -> str:
    today = today or datetime.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    end_year = start_year + 1
    return f"{str(start_year)[-2:]}{str(end_year)[-2:]}"


def training_season_codes(depth: int = 5) -> list[str]:
    current_code = current_season_code()
    current_start = int(current_code[:2])
    seasons: list[str] = []
    for offset in range(depth):
        start = current_start - offset
        end = (start + 1) % 100
        seasons.append(f"{start:02d}{end:02d}")
    return seasons


def _download_csv(client: httpx.Client, season: str, league_code: str, output_file: Path) -> None:
    url = f"{BASE_URL}/{season}/{league_code}.csv"
    response = client.get(url, timeout=30.0)
    response.raise_for_status()
    output_file.write_bytes(response.content)


def download_historical_data() -> None:
    settings = get_settings()
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    seasons = training_season_codes(depth=5)

    with httpx.Client(follow_redirects=True) as client:
        for season in seasons:
            for league_code, league_name in LEAGUES.items():
                output_file = settings.raw_dir / f"{league_name}_{season}.csv"
                LOGGER.info("Descargando %s %s", league_name, season)
                _download_csv(client, season, league_code, output_file)
