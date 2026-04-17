from __future__ import annotations

from datetime import datetime, timedelta

import httpx
import pandas as pd

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
    today = datetime.today().date()

    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        for league_name, slug in LEAGUE_FEEDS.items():
            for offset in range(days_ahead + 1):
                target_day = today + timedelta(days=offset)
                rows.extend(_fetch_league_day(client, league_name, slug, target_day))

    fixtures = pd.DataFrame(rows)
    if fixtures.empty:
        return fixtures

    fixtures = fixtures.drop_duplicates(subset=["Date", "League", "HomeTeam", "AwayTeam"])
    fixtures = fixtures.sort_values(["Date", "League", "HomeTeam"])
    return fixtures


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
                "HomeTeam": home_team.get("team", {}).get("displayName", ""),
                "AwayTeam": away_team.get("team", {}).get("displayName", ""),
                "Status": status_type.get("description", ""),
            }
        )

    if rows:
        LOGGER.info("Fixtures ESPN %s %s: %s", league_name, date_value, len(rows))
    return rows
