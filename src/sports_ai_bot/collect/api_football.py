from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import httpx

from sports_ai_bot.utils.config import get_settings
from sports_ai_bot.utils.logging import get_logger


LOGGER = get_logger(__name__)

LEAGUE_IDS = {
    "premier_league": 39,
    "la_liga": 140,
    "serie_a": 135,
    "bundesliga": 78,
    "ligue_1": 61,
    "eredivisie": 88,
    "primeira_liga": 94,
    "belgian_pro_league": 144,
    "super_lig": 203,
    "primera_division_argentina": 128,
    "serie_a_brasil": 71,
    "liga_mx": 262,
    "mls": 253,
}


@dataclass(frozen=True)
class ApiFootballLeague:
    league_name: str
    league_id: int
    season_year: int


class ApiFootballError(RuntimeError):
    """Raised when API-Football returns a non-usable payload."""


def is_configured() -> bool:
    return get_settings().has_api_football()


def build_headers() -> dict[str, str]:
    settings = get_settings()
    headers = {
        "x-apisports-key": settings.api_football_key,
    }
    if settings.api_football_host:
        headers["x-rapidapi-host"] = settings.api_football_host
    return headers


def get_json(path: str, params: dict[str, object]) -> dict[str, object]:
    settings = get_settings()
    if not settings.has_api_football():
        raise RuntimeError("API-Football no configurada")

    url = f"{settings.api_football_base_url.rstrip('/')}/{path.lstrip('/')}"
    with httpx.Client(follow_redirects=True, timeout=30.0, headers=build_headers()) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()

    errors = payload.get("errors") or {}
    if errors:
        raise ApiFootballError(_format_errors(errors))

    results = payload.get("results")
    LOGGER.info("API-Football %s results=%s", path, results)
    return payload


def league_context(league_name: str, target_day: date) -> ApiFootballLeague:
    league_id = LEAGUE_IDS[league_name]
    return ApiFootballLeague(
        league_name=league_name,
        league_id=league_id,
        season_year=_season_year_for_date(target_day),
    )


def _season_year_for_date(target_day: date) -> int:
    return target_day.year if target_day.month >= 7 else target_day.year - 1


def _format_errors(errors: dict[str, object]) -> str:
    parts = [f"{key}: {value}" for key, value in errors.items() if value]
    return "; ".join(parts) if parts else "unknown API-Football error"
