from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from sports_ai_bot.utils.config import get_settings
from sports_ai_bot.utils.logging import get_logger


LOGGER = get_logger(__name__)

SOCCER_SPORT_KEYS = {
    "premier_league": "soccer_epl",
    "la_liga": "soccer_spain_la_liga",
    "serie_a": "soccer_italy_serie_a",
    "bundesliga": "soccer_germany_bundesliga",
    "ligue_1": "soccer_france_ligue_one",
    "eredivisie": "soccer_netherlands_eredivisie",
    "primeira_liga": "soccer_portugal_primeira_liga",
    "belgian_pro_league": "soccer_belgium_first_div",
    "super_lig": "soccer_turkey_super_league",
    "primera_division_argentina": "soccer_argentina_primera_division",
    "serie_a_brasil": "soccer_brazil_campeonato",
    "liga_mx": "soccer_mexico_ligamx",
    "mls": "soccer_usa_mls",
}


class TheOddsApiError(RuntimeError):
    """Raised when The Odds API returns a non-usable payload."""


def is_configured() -> bool:
    return get_settings().has_the_odds_api()


def get_events_for_league(league_name: str, days_ahead: int = 7) -> list[dict[str, Any]]:
    payload = get_json(f"/sports/{sport_key_for_league(league_name)}/events", {})
    return _filter_events_in_window(payload, days_ahead)


def get_odds_for_league(
    league_name: str,
    markets: str = "totals",
    days_ahead: int = 7,
) -> list[dict[str, Any]]:
    payload = get_json(
        f"/sports/{sport_key_for_league(league_name)}/odds",
        {
            "regions": get_settings().the_odds_api_region,
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
            "bookmakers": get_settings().the_odds_api_bookmaker,
        },
    )
    return _filter_events_in_window(payload, days_ahead)


def get_json(path: str, params: dict[str, object]) -> list[dict[str, Any]]:
    settings = get_settings()
    if not settings.has_the_odds_api():
        raise RuntimeError("The Odds API no configurada")

    query = {"apiKey": settings.the_odds_api_key}
    query.update(params)
    url = f"{settings.the_odds_api_base_url.rstrip('/')}/{path.lstrip('/')}"
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        response = client.get(url, params=query)
        response.raise_for_status()
        _log_quota(response)
        payload = response.json()

    if not isinstance(payload, list):
        raise TheOddsApiError("Respuesta inesperada de The Odds API")
    return payload


def sport_key_for_league(league_name: str) -> str:
    try:
        return SOCCER_SPORT_KEYS[league_name]
    except KeyError as exc:
        raise TheOddsApiError(f"Liga no soportada en The Odds API: {league_name}") from exc


def _filter_events_in_window(
    payload: list[dict[str, Any]], days_ahead: int
) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days_ahead)
    filtered: list[dict[str, Any]] = []
    for item in payload:
        commence_time = item.get("commence_time")
        event_time = _parse_commence_time(commence_time)
        if event_time is None:
            continue
        if now <= event_time <= end:
            filtered.append(item)
    return filtered


def _parse_commence_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _log_quota(response: httpx.Response) -> None:
    remaining = response.headers.get("x-requests-remaining")
    used = response.headers.get("x-requests-used")
    last = response.headers.get("x-requests-last")
    LOGGER.info(
        "The Odds API quota remaining=%s used=%s last=%s",
        remaining or "n/d",
        used or "n/d",
        last or "n/d",
    )
