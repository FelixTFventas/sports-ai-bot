from __future__ import annotations

from datetime import datetime, timedelta

import httpx
import pandas as pd

from sports_ai_bot.collect.fixtures import LEAGUE_FEEDS
from sports_ai_bot.collect.historical import LEAGUES
from sports_ai_bot.collect.the_odds_api import TheOddsApiError, get_odds_for_league, is_configured
from sports_ai_bot.utils.config import get_settings
from sports_ai_bot.utils.logging import get_logger
from sports_ai_bot.utils.team_names import canonical_team_name


LOGGER = get_logger(__name__)
ODDS_COLUMNS = [
    "odd_home_win",
    "odd_draw",
    "odd_away_win",
    "odd_over15",
    "odd_over25",
    "odd_under45",
    "odd_btts",
]
BASE_COLUMNS = ["match_day", "League", "HomeTeam", "AwayTeam"]
ALL_COLUMNS = BASE_COLUMNS + ODDS_COLUMNS


def load_upcoming_market_odds() -> pd.DataFrame:
    if is_configured():
        try:
            the_odds_api_odds = _fetch_the_odds_api_market_odds(days_ahead=7)
            if not the_odds_api_odds.empty:
                LOGGER.info("Odds source=the_odds count=%s", len(the_odds_api_odds))
                return the_odds_api_odds
            LOGGER.info("Odds source=the_odds count=0 fallback=espn_csv")
        except (TheOddsApiError, httpx.HTTPError, RuntimeError, ValueError) as exc:
            LOGGER.warning("The Odds API odds fallback to ESPN/CSV: %s", _safe_error_message(exc))
    else:
        LOGGER.info("Odds source=the_odds disabled fallback=espn_csv")

    espn_odds = _fetch_espn_market_odds(days_ahead=7)
    if not espn_odds.empty:
        LOGGER.info("Odds source=espn count=%s", len(espn_odds))
        return espn_odds

    csv_odds = _load_csv_market_odds()
    if not csv_odds.empty:
        LOGGER.info("Odds source=csv count=%s", len(csv_odds))
        return csv_odds

    return pd.DataFrame(columns=ALL_COLUMNS)


def _fetch_the_odds_api_market_odds(days_ahead: int = 7) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for league_name in LEAGUE_FEEDS:
        rows.extend(_fetch_the_odds_api_league_odds(league_name, days_ahead))

    if not rows:
        return pd.DataFrame(columns=ALL_COLUMNS)

    frame = pd.DataFrame(rows)
    return frame.drop_duplicates(subset=BASE_COLUMNS, keep="first")


def _fetch_the_odds_api_league_odds(league_name: str, days_ahead: int) -> list[dict[str, object]]:
    payloads: dict[str, list[dict[str, object]]] = {}
    for market_key in ("h2h", "totals", "btts"):
        try:
            payloads[market_key] = get_odds_for_league(
                league_name,
                markets=market_key,
                days_ahead=days_ahead,
            )
        except (TheOddsApiError, httpx.HTTPError, RuntimeError, ValueError) as exc:
            LOGGER.info(
                "Odds The Odds API %s skipped market=%s reason=%s",
                league_name,
                market_key,
                _safe_error_message(exc),
            )
            payloads[market_key] = []

    if not any(payloads.values()):
        return []

    merged_events: dict[tuple[str, str, str], dict[str, object]] = {}
    for payload in payloads.values():
        for item in payload:
            event_key = _event_key(item)
            if event_key is None:
                continue
            merged_events.setdefault(event_key, item)

    rows: list[dict[str, object]] = []
    for event_key, item in merged_events.items():
        home_name = str(item.get("home_team") or "")
        away_name = str(item.get("away_team") or "")
        if not home_name or not away_name:
            continue

        match_day = pd.to_datetime(item.get("commence_time"), errors="coerce")
        if pd.isna(match_day):
            continue

        h2h_bookmakers = _bookmakers_for_event(payloads["h2h"], event_key)
        totals_bookmakers = _bookmakers_for_event(payloads["totals"], event_key)
        btts_bookmakers = _bookmakers_for_event(payloads["btts"], event_key)
        odds_row = {
            "odd_home_win": _extract_the_odds_h2h_price(h2h_bookmakers, home_name),
            "odd_draw": _extract_the_odds_h2h_price(h2h_bookmakers, "draw"),
            "odd_away_win": _extract_the_odds_h2h_price(h2h_bookmakers, away_name),
            "odd_over15": _extract_the_odds_totals_price(
                totals_bookmakers, point=1.5, selection="over"
            ),
            "odd_over25": _extract_the_odds_totals_price(
                totals_bookmakers, point=2.5, selection="over"
            ),
            "odd_under45": _extract_the_odds_totals_price(
                totals_bookmakers, point=4.5, selection="under"
            ),
            "odd_btts": _extract_the_odds_btts_price(btts_bookmakers, selection="yes"),
        }
        if not any(value is not None for value in odds_row.values()):
            continue

        rows.append(
            {
                "match_day": match_day.date(),
                "League": league_name,
                "HomeTeam": canonical_team_name(league_name, home_name) or home_name,
                "AwayTeam": canonical_team_name(league_name, away_name) or away_name,
                **odds_row,
            }
        )
    if rows:
        LOGGER.info("Odds The Odds API %s: %s", league_name, len(rows))
    return rows


def _event_key(item: dict[str, object]) -> tuple[str, str, str] | None:
    commence_time = str(item.get("commence_time") or "").strip()
    home_team = str(item.get("home_team") or "").strip()
    away_team = str(item.get("away_team") or "").strip()
    if not commence_time or not home_team or not away_team:
        return None
    return commence_time, home_team, away_team


def _bookmakers_for_event(
    payload: list[dict[str, object]], event_key: tuple[str, str, str]
) -> list[dict[str, object]]:
    for item in payload:
        if _event_key(item) == event_key:
            return item.get("bookmakers", [])
    return []


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
        return pd.DataFrame(columns=ALL_COLUMNS)

    frame = pd.DataFrame(rows)
    return frame.drop_duplicates(subset=BASE_COLUMNS, keep="first")


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
            working["odd_home_win"] = _first_available_decimal(working, ["B365H", "AvgH", "MaxH"])
            working["odd_draw"] = _first_available_decimal(working, ["B365D", "AvgD", "MaxD"])
            working["odd_away_win"] = _first_available_decimal(working, ["B365A", "AvgA", "MaxA"])
            working["odd_over15"] = _first_available_decimal(working, ["B365>1.5", "Avg>1.5", "Max>1.5"])
            working["odd_over25"] = _first_available_decimal(working, ["B365>2.5", "Avg>2.5", "Max>2.5"])
            working["odd_under45"] = _first_available_decimal(working, ["B365<4.5", "Avg<4.5", "Max<4.5"])
            working["odd_btts"] = _first_available_decimal(
                working,
                ["B365BTSY", "AvgBTSY", "MaxBTSY", "B365BTTS", "AvgBTTS", "MaxBTTS"],
            )
            frames.append(working[ALL_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=ALL_COLUMNS)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=ODDS_COLUMNS, how="all")
    return merged.drop_duplicates(subset=BASE_COLUMNS, keep="last")


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
                "odd_home_win": None,
                "odd_draw": None,
                "odd_away_win": None,
                "odd_over15": None,
                "odd_over25": odd_over25,
                "odd_under45": None,
                "odd_btts": None,
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


def _extract_the_odds_h2h_price(bookmakers: list[dict[str, object]], selection_name: str) -> float | None:
    normalized_selection = selection_name.strip().lower()
    for bookmaker in bookmakers:
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                outcome_name = str(outcome.get("name", "")).strip().lower()
                if outcome_name != normalized_selection:
                    continue
                outcome_price = pd.to_numeric(outcome.get("price"), errors="coerce")
                if pd.notna(outcome_price):
                    return float(outcome_price)
    return None


def _extract_the_odds_totals_price(
    bookmakers: list[dict[str, object]], point: float, selection: str = "over"
) -> float | None:
    normalized_selection = selection.strip().lower()
    for bookmaker in bookmakers:
        for market in bookmaker.get("markets", []):
            if market.get("key") != "totals":
                continue
            for outcome in market.get("outcomes", []):
                if str(outcome.get("name", "")).strip().lower() != normalized_selection:
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


def _extract_the_odds_btts_price(bookmakers: list[dict[str, object]], selection: str = "yes") -> float | None:
    normalized_selection = selection.strip().lower()
    for bookmaker in bookmakers:
        for market in bookmaker.get("markets", []):
            if market.get("key") != "btts":
                continue
            for outcome in market.get("outcomes", []):
                if str(outcome.get("name", "")).strip().lower() != normalized_selection:
                    continue
                outcome_price = pd.to_numeric(outcome.get("price"), errors="coerce")
                if pd.notna(outcome_price):
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


def _safe_error_message(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return str(exc)
