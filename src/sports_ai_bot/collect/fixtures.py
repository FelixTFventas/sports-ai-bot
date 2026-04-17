from __future__ import annotations

from datetime import datetime, timedelta

import httpx
import pandas as pd

from sports_ai_bot.collect.api_football import (
    ApiFootballError,
    get_json,
    is_configured,
    league_context,
)
from sports_ai_bot.utils.team_names import canonical_team_name
from sports_ai_bot.utils.logging import get_logger


LOGGER = get_logger(__name__)

LEAGUE_FEEDS = {
    "premier_league": "eng.1",
    "la_liga": "esp.1",
    "serie_a": "ita.1",
    "bundesliga": "ger.1",
    "ligue_1": "fra.1",
}


def fetch_upcoming_fixtures(days_ahead: int = 7) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    if is_configured():
        try:
            rows = _fetch_api_football_fixtures(days_ahead)
        except (ApiFootballError, httpx.HTTPError, RuntimeError, ValueError) as exc:
            LOGGER.warning("API-Football fixtures fallback to ESPN: %s", exc)
    if not rows:
        rows = _fetch_espn_fixtures(days_ahead)

    fixtures = pd.DataFrame(rows)
    if fixtures.empty:
        return fixtures

    fixtures = fixtures.drop_duplicates(subset=["Date", "League", "HomeTeam", "AwayTeam"])
    fixtures = fixtures.sort_values(["Date", "League", "HomeTeam"])
    return fixtures


def _fetch_api_football_fixtures(days_ahead: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    today = datetime.today().date()

    for league_name in LEAGUE_FEEDS:
        for offset in range(days_ahead + 1):
            target_day = today + timedelta(days=offset)
            rows.extend(_fetch_api_football_league_day(league_name, target_day))

    return rows


def _fetch_api_football_league_day(league_name: str, target_day) -> list[dict[str, str]]:
    context = league_context(league_name, target_day)
    payload = get_json(
        "/fixtures",
        {
            "league": context.league_id,
            "season": context.season_year,
            "date": target_day.isoformat(),
            "timezone": "UTC",
        },
    )

    rows: list[dict[str, str]] = []
    for item in payload.get("response", []):
        fixture = item.get("fixture", {})
        status = fixture.get("status", {})
        if status.get("short") in {"FT", "AET", "PEN", "CANC", "PST", "ABD"}:
            continue

        teams = item.get("teams", {})
        home_name = teams.get("home", {}).get("name", "")
        away_name = teams.get("away", {}).get("name", "")
        if not home_name or not away_name:
            continue

        rows.append(
            {
                "Date": fixture.get("date", ""),
                "League": league_name,
                "HomeTeam": canonical_team_name(league_name, home_name) or home_name,
                "AwayTeam": canonical_team_name(league_name, away_name) or away_name,
                "Status": status.get("long", status.get("short", "")),
            }
        )

    if rows:
        LOGGER.info("Fixtures API-Football %s %s: %s", league_name, target_day, len(rows))
    return rows


def _fetch_espn_fixtures(days_ahead: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    today = datetime.today().date()

    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        for league_name, slug in LEAGUE_FEEDS.items():
            for offset in range(days_ahead + 1):
                target_day = today + timedelta(days=offset)
                rows.extend(_fetch_league_day(client, league_name, slug, target_day))
    return rows


def _fetch_league_day(
    client: httpx.Client,
    league_name: str,
    slug: str,
    target_day,
) -> list[dict[str, str]]:
    date_value = target_day.strftime("%Y%m%d")
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates={date_value}"
    )
    response = client.get(url)
    response.raise_for_status()
    payload = response.json()

    rows: list[dict[str, str]] = []
    for event in payload.get("events", []):
        competitions = event.get("competitions", [])
        if not competitions:
            continue
        competition = competitions[0]
        status_type = competition.get("status", {}).get("type", {})
        if status_type.get("completed"):
            continue

        competitors = competition.get("competitors", [])
        if len(competitors) != 2:
            continue

        home_team = next((item for item in competitors if item.get("homeAway") == "home"), None)
        away_team = next((item for item in competitors if item.get("homeAway") == "away"), None)
        if not home_team or not away_team:
            continue

        rows.append(
            {
                "Date": competition.get("date", event.get("date", "")),
                "League": league_name,
                "HomeTeam": canonical_team_name(
                    league_name, home_team.get("team", {}).get("displayName", "")
                )
                or home_team.get("team", {}).get("displayName", ""),
                "AwayTeam": canonical_team_name(
                    league_name, away_team.get("team", {}).get("displayName", "")
                )
                or away_team.get("team", {}).get("displayName", ""),
                "Status": status_type.get("description", ""),
            }
        )

    if rows:
        LOGGER.info("Fixtures ESPN %s %s: %s", league_name, date_value, len(rows))
    return rows
