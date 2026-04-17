from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from sports_ai_bot.collect.fixtures import fetch_upcoming_fixtures
from sports_ai_bot.collect.historical import LEAGUES, current_season_code
from sports_ai_bot.utils.config import get_settings
from sports_ai_bot.utils.team_names import canonical_team_name


REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
ELO_BASE = 1500.0
ELO_K = 20.0
WINDOW = 5


@dataclass
class TeamState:
    overall_history: list[dict[str, float]] = field(default_factory=list)
    home_history: list[dict[str, float]] = field(default_factory=list)
    away_history: list[dict[str, float]] = field(default_factory=list)
    elo: float = ELO_BASE
    last_match_date: pd.Timestamp | None = None


def _load_raw_csv(file_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(file_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {file_path.name}: {missing}")
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], dayfirst=True, errors="coerce")
    frame = frame.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    frame["FTHG"] = pd.to_numeric(frame["FTHG"], errors="coerce")
    frame["FTAG"] = pd.to_numeric(frame["FTAG"], errors="coerce")
    frame = frame.dropna(subset=["FTHG", "FTAG"])
    return frame


def _load_fixture_csv(file_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(file_path)
    missing = [column for column in ["Date", "HomeTeam", "AwayTeam"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {file_path.name}: {missing}")
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], dayfirst=True, errors="coerce")
    frame["FTHG"] = pd.to_numeric(frame.get("FTHG"), errors="coerce")
    frame["FTAG"] = pd.to_numeric(frame.get("FTAG"), errors="coerce")
    frame = frame.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    return frame


def _rolling_avg(history: list[dict[str, float]], key: str, window: int = WINDOW) -> float | None:
    if len(history) < window:
        return None
    sample = history[-window:]
    return sum(item[key] for item in sample) / window


def _get_team_state(states: dict[str, TeamState], team_name: str) -> TeamState:
    return states.setdefault(team_name, TeamState())


def _days_rest(last_match_date: pd.Timestamp | None, current_date: pd.Timestamp) -> float | None:
    if last_match_date is None:
        return None
    current_ts = pd.Timestamp(current_date)
    last_ts = pd.Timestamp(last_match_date)
    if current_ts.tzinfo is not None:
        current_ts = current_ts.tz_convert(None)
    if last_ts.tzinfo is not None:
        last_ts = last_ts.tz_convert(None)
    return float((current_ts.normalize() - last_ts.normalize()).days)


def _build_feature_row(
    league: str,
    match_date: pd.Timestamp,
    home_team: str,
    away_team: str,
    states: dict[str, TeamState],
) -> dict[str, object]:
    home_state = _get_team_state(states, home_team)
    away_state = _get_team_state(states, away_team)

    home_goals_for_avg_5 = _rolling_avg(home_state.overall_history, "goals_for")
    home_goals_against_avg_5 = _rolling_avg(home_state.overall_history, "goals_against")
    home_points_avg_5 = _rolling_avg(home_state.overall_history, "points")
    home_over25_rate_5 = _rolling_avg(home_state.overall_history, "over25")
    home_btts_rate_5 = _rolling_avg(home_state.overall_history, "btts")
    away_goals_for_avg_5 = _rolling_avg(away_state.overall_history, "goals_for")
    away_goals_against_avg_5 = _rolling_avg(away_state.overall_history, "goals_against")
    away_points_avg_5 = _rolling_avg(away_state.overall_history, "points")
    away_over25_rate_5 = _rolling_avg(away_state.overall_history, "over25")
    away_btts_rate_5 = _rolling_avg(away_state.overall_history, "btts")

    home_home_goals_for_avg_5 = _rolling_avg(home_state.home_history, "goals_for")
    home_home_goals_against_avg_5 = _rolling_avg(home_state.home_history, "goals_against")
    home_home_points_avg_5 = _rolling_avg(home_state.home_history, "points")
    away_away_goals_for_avg_5 = _rolling_avg(away_state.away_history, "goals_for")
    away_away_goals_against_avg_5 = _rolling_avg(away_state.away_history, "goals_against")
    away_away_points_avg_5 = _rolling_avg(away_state.away_history, "points")

    return {
        "Date": match_date,
        "League": league,
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "home_goals_for_avg_5": home_goals_for_avg_5,
        "home_goals_against_avg_5": home_goals_against_avg_5,
        "home_points_avg_5": home_points_avg_5,
        "home_over25_rate_5": home_over25_rate_5,
        "home_btts_rate_5": home_btts_rate_5,
        "away_goals_for_avg_5": away_goals_for_avg_5,
        "away_goals_against_avg_5": away_goals_against_avg_5,
        "away_points_avg_5": away_points_avg_5,
        "away_over25_rate_5": away_over25_rate_5,
        "away_btts_rate_5": away_btts_rate_5,
        "home_home_goals_for_avg_5": home_home_goals_for_avg_5,
        "home_home_goals_against_avg_5": home_home_goals_against_avg_5,
        "home_home_points_avg_5": home_home_points_avg_5,
        "away_away_goals_for_avg_5": away_away_goals_for_avg_5,
        "away_away_goals_against_avg_5": away_away_goals_against_avg_5,
        "away_away_points_avg_5": away_away_points_avg_5,
        "home_rest_days": _days_rest(home_state.last_match_date, match_date),
        "away_rest_days": _days_rest(away_state.last_match_date, match_date),
        "elo_home": home_state.elo,
        "elo_away": away_state.elo,
        "elo_diff": home_state.elo - away_state.elo,
        "attack_diff": None
        if home_goals_for_avg_5 is None or away_goals_for_avg_5 is None
        else home_goals_for_avg_5 - away_goals_for_avg_5,
        "defense_diff": None
        if home_goals_against_avg_5 is None or away_goals_against_avg_5 is None
        else away_goals_against_avg_5 - home_goals_against_avg_5,
        "form_diff": None
        if home_points_avg_5 is None or away_points_avg_5 is None
        else home_points_avg_5 - away_points_avg_5,
        "goal_balance_diff": None
        if home_goals_for_avg_5 is None
        or home_goals_against_avg_5 is None
        or away_goals_for_avg_5 is None
        or away_goals_against_avg_5 is None
        else (home_goals_for_avg_5 - home_goals_against_avg_5)
        - (away_goals_for_avg_5 - away_goals_against_avg_5),
    }


def _expected_score(home_elo: float, away_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400.0))


def _update_team_states(
    states: dict[str, TeamState],
    match_date: pd.Timestamp,
    home_team: str,
    away_team: str,
    home_goals: float,
    away_goals: float,
) -> None:
    home_state = _get_team_state(states, home_team)
    away_state = _get_team_state(states, away_team)

    home_points = 3 if home_goals > away_goals else 1 if home_goals == away_goals else 0
    away_points = 3 if away_goals > home_goals else 1 if home_goals == away_goals else 0
    total_goals = home_goals + away_goals
    over25 = float(total_goals > 2.5)
    btts = float(home_goals > 0 and away_goals > 0)

    home_record = {
        "goals_for": float(home_goals),
        "goals_against": float(away_goals),
        "points": float(home_points),
        "over25": over25,
        "btts": btts,
    }
    away_record = {
        "goals_for": float(away_goals),
        "goals_against": float(home_goals),
        "points": float(away_points),
        "over25": over25,
        "btts": btts,
    }

    home_state.overall_history.append(home_record)
    home_state.home_history.append(home_record)
    away_state.overall_history.append(away_record)
    away_state.away_history.append(away_record)

    home_actual = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
    away_actual = 1.0 - home_actual
    expected_home = _expected_score(home_state.elo, away_state.elo)
    expected_away = 1.0 - expected_home
    home_state.elo = home_state.elo + ELO_K * (home_actual - expected_home)
    away_state.elo = away_state.elo + ELO_K * (away_actual - expected_away)

    home_state.last_match_date = match_date
    away_state.last_match_date = match_date


def _build_state_from_completed_matches(matches: pd.DataFrame) -> dict[str, TeamState]:
    states: dict[str, TeamState] = {}
    completed = matches.dropna(subset=["FTHG", "FTAG"]).sort_values("Date")
    for match in completed.itertuples(index=False):
        _update_team_states(
            states,
            match.Date,
            match.HomeTeam,
            match.AwayTeam,
            float(match.FTHG),
            float(match.FTAG),
        )
    return states


def _attach_team_history_features(matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    states: dict[str, TeamState] = {}

    for match in matches.sort_values("Date").itertuples(index=False):
        row = _build_feature_row(match.League, match.Date, match.HomeTeam, match.AwayTeam, states)
        row["target_over25"] = int((match.FTHG + match.FTAG) > 2.5)
        row["target_btts"] = int(match.FTHG > 0 and match.FTAG > 0)
        rows.append(row)
        _update_team_states(
            states,
            match.Date,
            match.HomeTeam,
            match.AwayTeam,
            float(match.FTHG),
            float(match.FTAG),
        )

    dataset = pd.DataFrame(rows)
    dataset = dataset.dropna()
    return dataset


def build_training_dataset() -> pd.DataFrame:
    settings = get_settings()
    frames: list[pd.DataFrame] = []

    for league_name in LEAGUES.values():
        for file_path in sorted(settings.raw_dir.glob(f"{league_name}_*.csv")):
            frame = _load_raw_csv(file_path)
            frame["League"] = league_name
            frames.append(frame)

    if not frames:
        raise FileNotFoundError("No hay historicos descargados en data/raw")

    matches = pd.concat(frames, ignore_index=True)
    matches = matches.sort_values("Date")
    dataset = _attach_team_history_features(matches)
    output_file = settings.processed_dir / "training_dataset.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_file, index=False)
    return dataset


def build_fixture_features() -> pd.DataFrame:
    settings = get_settings()
    current_season = current_season_code()
    historical_frames: list[pd.DataFrame] = []
    fixture_frames: list[pd.DataFrame] = []

    for league_name in LEAGUES.values():
        for file_path in sorted(settings.raw_dir.glob(f"{league_name}_*.csv")):
            fixture_frame = _load_fixture_csv(file_path)
            fixture_frame["League"] = league_name
            fixture_frames.append(fixture_frame)

            completed_frame = fixture_frame.dropna(subset=["FTHG", "FTAG"]).copy()
            if not completed_frame.empty:
                completed_frame["League"] = league_name
                historical_frames.append(completed_frame)

    if not historical_frames:
        raise FileNotFoundError("No hay historicos completos para generar features de fixtures")

    completed_matches = pd.concat(historical_frames, ignore_index=True)
    states = _build_state_from_completed_matches(completed_matches)

    all_fixtures = pd.concat(fixture_frames, ignore_index=True)
    source_upcoming = all_fixtures[
        (all_fixtures["Date"] >= pd.Timestamp.today().normalize())
        & (all_fixtures["FTHG"].isna())
        & (all_fixtures["FTAG"].isna())
    ].copy()
    espn_upcoming = fetch_upcoming_fixtures(days_ahead=7)
    upcoming = _merge_fixture_sources(source_upcoming, espn_upcoming)
    if upcoming.empty:
        output_file = settings.processed_dir / "fixture_features.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["Date", "League", "HomeTeam", "AwayTeam"]).to_csv(
            output_file,
            index=False,
        )
        return upcoming

    rows: list[dict[str, object]] = []
    for fixture in upcoming.sort_values("Date").itertuples(index=False):
        home_team = canonical_team_name(fixture.League, fixture.HomeTeam) or fixture.HomeTeam
        away_team = canonical_team_name(fixture.League, fixture.AwayTeam) or fixture.AwayTeam
        row = _build_feature_row(fixture.League, fixture.Date, home_team, away_team, states)
        row["season_code"] = current_season
        rows.append(row)

    fixtures = pd.DataFrame(rows)
    fixtures = fixtures.dropna()
    output_file = settings.processed_dir / "fixture_features.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fixtures.to_csv(output_file, index=False)
    return fixtures


def _merge_fixture_sources(csv_upcoming: pd.DataFrame, espn_upcoming: pd.DataFrame) -> pd.DataFrame:
    csv_copy = csv_upcoming.copy()
    if not csv_copy.empty:
        csv_copy = csv_copy[["Date", "League", "HomeTeam", "AwayTeam"]]

    espn_copy = espn_upcoming.copy()
    if not espn_copy.empty:
        espn_copy["Date"] = pd.to_datetime(espn_copy["Date"], errors="coerce")
        espn_copy = espn_copy[["Date", "League", "HomeTeam", "AwayTeam"]]

    frames = [frame for frame in [csv_copy, espn_copy] if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["Date", "League", "HomeTeam", "AwayTeam"])

    upcoming = pd.concat(frames, ignore_index=True)
    upcoming = upcoming.dropna(subset=["Date", "League", "HomeTeam", "AwayTeam"])
    upcoming = upcoming.drop_duplicates(subset=["Date", "League", "HomeTeam", "AwayTeam"])
    return upcoming
