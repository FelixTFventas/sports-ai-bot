from __future__ import annotations

from datetime import datetime, timedelta

import httpx
import pandas as pd

from sports_ai_bot.collect.the_odds_api import (
    TheOddsApiError,
    get_odds_for_league,
    is_configured,
)
from sports_ai_bot.utils.logging import get_logger
from sports_ai_bot.collect.fixtures import LEAGUE_FEEDS
from sports_ai_bot.collect.historical import LEAGUES
from sports_ai_bot.utils.config import get_settings
from sports_ai_bot.utils.team_names import canonical_team_name


LOGGER = get_logger(__name__)


def load_upcoming_market_odds() -> pd.DataFrame:
    the_odds_api_odds = pd.DataFrame()
    if is_configured():
        try:
            the_odds_api_odds = _fetch_the_odds_api_market_odds(days_ahead=7)
        except (TheOddsApiError, httpx.HTTPError, RuntimeError, ValueError) as exc:
            the_odds_api_odds = pd.DataFrame()
            LOGGER.warning("The Odds API odds fallback to ESPN/CSV: %s", exc)
    espn_odds = _fetch_espn_market_odds(days_ahead=7)
    csv_odds = _load_csv_market_odds()
    combined = pd.concat([the_odds_api_odds, espn_odds, csv_odds], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=["match_day", "League", "HomeTeam", "AwayTeam", "odd_over25"])
    return combined.drop_duplicates(
        subset=["match_day", "League", "HomeTeam", "AwayTeam"], keep="first"
    )


def _fetch_the_odds_api_market_odds(days_ahead: int = 7) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for league_name in LEAGUE_FEEDS:
        rows.extend(_fetch_the_odds_api_league_odds(league_name, days_ahead))

    if not rows:
        return pd.DataFrame(columns=["match_day", "League", "HomeTeam", "AwayTeam", "odd_over25"])

    frame = pd.DataFrame(rows)
    return frame.drop_duplicates(
        subset=["match_day", "League", "HomeTeam", "AwayTeam"], keep="first"
    )


def _fetch_the_odds_api_league_odds(league_name: str, days_ahead: int) -> list[dict[str, object]]:
    payload = get_odds_for_league(league_name, markets="totals", days_ahead=days_ahead)
    rows: list[dict[str, object]] = []
    for item in payload:
        home_name = str(item.get("home_team") or "")
        away_name = str(item.get("away_team") or "")
        if not home_name or not away_name:
            continue

        odd_over25 = _extract_the_odds_totals_price(item.get("bookmakers", []), point=2.5)
        if odd_over25 is None:
            continue

        match_day = pd.to_datetime(item.get("commence_time"), errors="coerce")
        if pd.isna(match_day):
            continue

        rows.append(
            {
                "match_day": match_day.date(),
                "League": league_name,
                "HomeTeam": canonical_team_name(league_name, home_name) or home_name,
                "AwayTeam": canonical_team_name(league_name, away_name) or away_name,
                "odd_over25": odd_over25,
            }
        )
    if rows:
        LOGGER.info("Odds The Odds API %s: %s", league_name, len(rows))
    return rows


def _fetch_espn_market_odds(days_ahead: int = 7) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    today = datetime.today().date()

    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        for league_name, slug in LEAGUE_FEEDS.items():
            for offset in range(days_ahead + 1):
                target_day = today + timedelta(days=offset)
                url = (
                    f"https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/scoreboard?dates="
                    f"{target_day.strftime('%Y%m%d')}"
                )
                response = client.get(url)
                response.raise_for_status()
                payload = response.json()
                rows.extend(_extract_event_odds(payload, league_name))

    if not rows:
        return pd.DataFrame(columns=["match_day", "League", "HomeTeam", "AwayTeam", "odd_over25"])

    frame = pd.DataFrame(rows)
    return frame.drop_duplicates(
        subset=["match_day", "League", "HomeTeam", "AwayTeam"], keep="first"
    )


def _load_csv_market_odds() -> pd.DataFrame:
    settings = get_settings()
    frames: list[pd.DataFrame] = []

    for league_name in LEAGUES.values():
        for file_path in sorted(settings.raw_dir.glob(f"{league_name}_*.csv")):
            frame = pd.read_csv(file_path)
            frame = frame.rename(
                columns={
                    "Home": "HomeTeam",
                    "Away": "AwayTeam",
                    "HG": "FTHG",
                    "AG": "FTAG",
                }
            )
            required = {"Date", "HomeTeam", "AwayTeam"}
            if not required.issubset(frame.columns):
                continue

            working = frame.copy()
            working["Date"] = pd.to_datetime(working["Date"], dayfirst=True, errors="coerce")
            working = working.dropna(subset=["Date", "HomeTeam", "AwayTeam"])

            if {"FTHG", "FTAG"}.issubset(working.columns):
                working = working[working["FTHG"].isna() & working["FTAG"].isna()].copy()

            working = working[working["Date"] >= pd.Timestamp.today().normalize()].copy()
            if working.empty:
                continue

            working["match_day"] = working["Date"].dt.date
            working["League"] = league_name
            working["odd_over25"] = _first_available_decimal(
                working, ["B365>2.5", "Avg>2.5", "Max>2.5"]
            )
            frames.append(working[["match_day", "League", "HomeTeam", "AwayTeam", "odd_over25"]])

    if not frames:
        return pd.DataFrame(columns=["match_day", "League", "HomeTeam", "AwayTeam", "odd_over25"])

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["odd_over25"])
    return merged.drop_duplicates(
        subset=["match_day", "League", "HomeTeam", "AwayTeam"], keep="last"
    )


def _extract_event_odds(payload: dict[str, object], league_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in payload.get("events", []):
        competitions = event.get("competitions", [])
        if not competitions:
            continue
        competition = competitions[0]
        status_type = competition.get("status", {}).get("type", {})
        if status_type.get("completed"):
            continue

        competitors = competition.get("competitors", [])
        home_team = next((item for item in competitors if item.get("homeAway") == "home"), None)
        away_team = next((item for item in competitors if item.get("homeAway") == "away"), None)
        if not home_team or not away_team:
            continue

        odd_over25 = _extract_over25_decimal(competition.get("odds", []))
        if odd_over25 is None:
            continue

        match_day = pd.to_datetime(competition.get("date", event.get("date")), errors="coerce")
        if pd.isna(match_day):
            continue

        rows.append(
            {
                "match_day": match_day.date(),
                "League": league_name,
                "HomeTeam": canonical_team_name(
                    league_name, home_team.get("team", {}).get("displayName", "")
                )
                or home_team.get("team", {}).get("displayName", ""),
                "AwayTeam": canonical_team_name(
                    league_name, away_team.get("team", {}).get("displayName", "")
                )
                or away_team.get("team", {}).get("displayName", ""),
                "odd_over25": odd_over25,
            }
        )
    return rows


def _extract_over25_decimal(odds_payload: list[dict[str, object]]) -> float | None:
    for entry in odds_payload:
        if not isinstance(entry, dict):
            continue
        total = entry.get("total", {})
        over = total.get("over", {})
        american = over.get("close", {}).get("odds") or over.get("open", {}).get("odds")
        decimal = _american_to_decimal(american)
        if decimal is not None:
            return decimal
    return None


def _extract_the_odds_totals_price(
    bookmakers: list[dict[str, object]], point: float
) -> float | None:
    for bookmaker in bookmakers:
        markets = bookmaker.get("markets", [])
        for market in markets:
            if market.get("key") != "totals":
                continue
            for outcome in market.get("outcomes", []):
                if str(outcome.get("name", "")).lower() != "over":
                    continue
                outcome_point = pd.to_numeric(outcome.get("point"), errors="coerce")
                outcome_price = pd.to_numeric(outcome.get("price"), errors="coerce")
                if (
                    pd.notna(outcome_point)
                    and pd.notna(outcome_price)
                    and float(outcome_point) == point
                ):
                    return float(outcome_price)
    return None


def _american_to_decimal(value: str | int | float | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        american = int(str(value))
    except ValueError:
        return None

    if american > 0:
        return round(1 + (american / 100), 4)
    if american < 0:
        return round(1 + (100 / abs(american)), 4)
    return None


def _first_available_decimal(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series([None] * len(frame), index=frame.index, dtype="float64")
    for column in columns:
        if column not in frame.columns:
            continue
        result = result.fillna(pd.to_numeric(frame[column], errors="coerce"))
    return result
