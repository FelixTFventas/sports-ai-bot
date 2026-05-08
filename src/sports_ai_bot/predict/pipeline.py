from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json

import joblib
import pandas as pd
from pandas.errors import EmptyDataError

from sports_ai_bot.collect.odds import ODDS_COLUMNS, load_upcoming_market_odds
from sports_ai_bot.features.build import build_fixture_features
from sports_ai_bot.train.train_models import FEATURE_COLUMNS
from sports_ai_bot.utils.config import get_settings


DEFAULT_MARKETS = ("1X2", "Over 1.5", "Over 2.5", "Under 4.5", "BTTS")


@dataclass
class Pick:
    match_date: str
    home_team: str
    away_team: str
    match_label: str
    league: str
    market: str
    probability: float
    confidence: str
    model_name: str
    factors: list[str]
    selection: str | None = None
    line: float | None = None
    odd: float | None = None
    implied_probability: float | None = None
    edge: float | None = None
    expected_value: float | None = None
    stake_units: int | None = None
    rating: str | None = None
    bookmaker: str | None = None
    score: float | None = None
    is_experimental: bool = False


def _confidence_label(probability: float) -> str:
    if probability >= 0.72:
        return "Alta"
    if probability >= 0.65:
        return "Media-Alta"
    return "Media"


def build_top_picks(
    limit: int = 20,
    threshold: float = 0.50,
    refresh_fixtures: bool = True,
    markets: tuple[str, ...] = DEFAULT_MARKETS,
    horizon_hours: int = 48,
    include_odds: bool = True,
    min_odd: float = 1.40,
    min_edge: float = 0.0,
    include_draw: bool = False,
) -> list[Pick]:
    latest = _prepare_fixtures(
        limit=limit,
        refresh_fixtures=refresh_fixtures,
        horizon_hours=horizon_hours,
        include_odds=include_odds,
    )
    if latest.empty:
        return []

    latest, model_names = _attach_probabilities(latest)
    if latest.empty:
        return []

    picks: list[Pick] = []
    for row in latest.itertuples(index=False):
        picks.extend(
            _build_row_picks(
                row,
                model_names=model_names,
                threshold=threshold,
                min_odd=min_odd,
                min_edge=min_edge,
                markets=markets,
                include_draw=include_draw,
            )
        )

    picks.sort(key=_pick_sort_key, reverse=True)
    return picks[:limit]


def build_value_picks(
    limit: int = 10,
    min_edge: float = 0.02,
    min_ev: float = 0.0,
    min_odd: float = 1.40,
    include_draw: bool = False,
    refresh_fixtures: bool = True,
) -> list[Pick]:
    picks = build_top_picks(
        limit=limit * 6,
        threshold=0.50,
        refresh_fixtures=refresh_fixtures,
        include_odds=True,
        min_odd=min_odd,
        min_edge=min_edge,
        include_draw=include_draw,
    )
    value_picks = [
        pick
        for pick in picks
        if pick.expected_value is not None and pick.expected_value >= min_ev
    ]
    for pick in value_picks:
        pick.stake_units = _stake_units(pick.edge, pick.expected_value)
        pick.rating = _rating(pick.edge, pick.expected_value)
        pick.score = _score_pick(pick)
    value_picks.sort(key=_pick_sort_key, reverse=True)
    return value_picks[:limit]


def build_best_picks(limit: int = 5, refresh_fixtures: bool = True) -> list[Pick]:
    picks = build_value_picks(
        limit=limit * 6,
        min_edge=0.02,
        min_ev=0.02,
        refresh_fixtures=refresh_fixtures,
    )
    best_by_match: dict[str, Pick] = {}
    for pick in picks:
        current = best_by_match.get(pick.match_label)
        if current is None or _pick_sort_key(pick) > _pick_sort_key(current):
            best_by_match[pick.match_label] = pick
    best = list(best_by_match.values())
    best.sort(key=_pick_sort_key, reverse=True)
    return best[:limit]


def build_market_picks(
    market: str,
    limit: int = 10,
    threshold: float = 0.55,
    refresh_fixtures: bool = True,
    horizon_hours: int = 48,
) -> list[Pick]:
    include_draw = market.strip().lower() in {"1x2", "draw", "empate"}
    picks = build_top_picks(
        limit=limit * 6,
        threshold=threshold,
        refresh_fixtures=refresh_fixtures,
        markets=(market,),
        horizon_hours=horizon_hours,
        include_draw=include_draw,
    )
    filtered = [pick for pick in picks if _matches_market(pick, market)]
    filtered.sort(key=_pick_sort_key, reverse=True)
    return filtered[:limit]


def persist_picks(picks: list[Pick]) -> pd.DataFrame:
    settings = get_settings()
    settings.predictions_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "prediction_date": date.today().isoformat(),
                "match_date": pick.match_date,
                "home_team": pick.home_team,
                "away_team": pick.away_team,
                "match_label": pick.match_label,
                "league": pick.league,
                "market": pick.market,
                "selection": pick.selection,
                "line": pick.line,
                "probability": round(pick.probability, 4),
                "confidence": pick.confidence,
                "model_name": pick.model_name,
                "odd": round(pick.odd, 4) if pick.odd is not None else None,
                "implied_probability": round(pick.implied_probability, 4)
                if pick.implied_probability is not None
                else None,
                "edge": round(pick.edge, 4) if pick.edge is not None else None,
                "expected_value": round(pick.expected_value, 4)
                if pick.expected_value is not None
                else None,
                "stake_units": pick.stake_units,
                "rating": pick.rating,
                "score": round(pick.score, 4) if pick.score is not None else None,
                "is_experimental": pick.is_experimental,
                "factors": " | ".join(pick.factors),
                "status": "pending",
                "outcome": "pending",
                "result_home_goals": None,
                "result_away_goals": None,
            }
            for pick in picks
        ]
    )
    output_file = settings.predictions_dir / f"picks_{date.today().isoformat()}.csv"
    frame.to_csv(output_file, index=False)
    return frame


def _prepare_fixtures(
    limit: int,
    refresh_fixtures: bool,
    horizon_hours: int,
    include_odds: bool,
) -> pd.DataFrame:
    latest = _load_latest_fixtures(refresh=refresh_fixtures)
    if latest.empty:
        return latest

    latest["Date"] = pd.to_datetime(latest["Date"], errors="coerce", utc=True)
    latest = latest.dropna(subset=FEATURE_COLUMNS + ["Date", "HomeTeam", "AwayTeam", "League"])
    window_start = _now_utc()
    window_end = window_start + pd.Timedelta(hours=horizon_hours)
    latest = latest[(latest["Date"] >= window_start) & (latest["Date"] <= window_end)]
    latest = latest.sort_values("Date").head(limit * 8).copy()
    if latest.empty:
        return latest
    if include_odds:
        latest = _attach_market_odds(latest)
    return latest


def _attach_probabilities(fixtures: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    settings = get_settings()
    model_files = {
        "prob_over15": settings.models_dir / "target_over15.joblib",
        "prob_over25": settings.models_dir / "target_over25.joblib",
        "prob_under45": settings.models_dir / "target_under45.joblib",
        "prob_btts": settings.models_dir / "target_btts.joblib",
        "prob_home_win": settings.models_dir / "target_home_win.joblib",
        "prob_draw": settings.models_dir / "target_draw.joblib",
        "prob_away_win": settings.models_dir / "target_away_win.joblib",
    }
    model_names = _model_names_by_market()
    enriched = fixtures.copy()
    available_probabilities: list[str] = []
    for probability_column, model_path in model_files.items():
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        enriched[probability_column] = model.predict_proba(enriched[FEATURE_COLUMNS])[:, 1]
        available_probabilities.append(probability_column)
    if not available_probabilities:
        return pd.DataFrame(), model_names
    return enriched, model_names


def _build_row_picks(
    row: object,
    model_names: dict[str, str],
    threshold: float,
    min_odd: float,
    min_edge: float,
    markets: tuple[str, ...],
    include_draw: bool,
) -> list[Pick]:
    picks: list[Pick] = []
    requested_markets = {market.strip().lower() for market in markets}
    candidates = [
        {
            "market": "1X2",
            "selection": "Local",
            "probability_column": "prob_home_win",
            "odd_column": "odd_home_win",
            "model_name": model_names["1X2 Local"],
            "line": None,
            "preference": 2,
        },
        {
            "market": "1X2",
            "selection": "Empate",
            "probability_column": "prob_draw",
            "odd_column": "odd_draw",
            "model_name": model_names["1X2 Empate"],
            "line": None,
            "preference": 0,
        },
        {
            "market": "1X2",
            "selection": "Visitante",
            "probability_column": "prob_away_win",
            "odd_column": "odd_away_win",
            "model_name": model_names["1X2 Visitante"],
            "line": None,
            "preference": 2,
        },
        {
            "market": "Over 1.5",
            "selection": None,
            "probability_column": "prob_over15",
            "odd_column": "odd_over15",
            "model_name": model_names["Over 1.5"],
            "line": 1.5,
            "preference": 1,
        },
        {
            "market": "Over 2.5",
            "selection": None,
            "probability_column": "prob_over25",
            "odd_column": "odd_over25",
            "model_name": model_names["Over 2.5"],
            "line": 2.5,
            "preference": 1,
        },
        {
            "market": "Under 4.5",
            "selection": None,
            "probability_column": "prob_under45",
            "odd_column": "odd_under45",
            "model_name": model_names["Under 4.5"],
            "line": 4.5,
            "preference": 1,
        },
        {
            "market": "BTTS",
            "selection": "Si",
            "probability_column": "prob_btts",
            "odd_column": "odd_btts",
            "model_name": model_names["BTTS"],
            "line": None,
            "preference": 1,
        },
    ]

    for candidate in candidates:
        if candidate["market"].lower() not in requested_markets and not _market_alias_requested(
            requested_markets, candidate["market"], candidate["selection"]
        ):
            continue
        if candidate["selection"] == "Empate" and not include_draw:
            continue
        probability = _row_value(row, candidate["probability_column"])
        odd = _row_value(row, candidate["odd_column"])
        if probability is None or odd is None:
            continue
        if probability < threshold or odd < min_odd:
            continue
        edge = _edge(probability, odd)
        expected_value = _expected_value(probability, odd)
        if edge is None or expected_value is None or edge <= min_edge:
            continue
        pick = Pick(
            match_date=row.Date.date().isoformat(),
            home_team=row.HomeTeam,
            away_team=row.AwayTeam,
            match_label=f"{row.HomeTeam} vs {row.AwayTeam}",
            league=row.League,
            market=candidate["market"],
            selection=candidate["selection"],
            line=candidate["line"],
            probability=probability,
            confidence=_confidence_label(probability),
            model_name=candidate["model_name"],
            odd=odd,
            implied_probability=_implied_probability(odd),
            edge=edge,
            expected_value=expected_value,
            factors=_build_factors(row, market=candidate["market"], selection=candidate["selection"]),
        )
        pick.stake_units = _stake_units(pick.edge, pick.expected_value)
        pick.rating = _rating(pick.edge, pick.expected_value)
        pick.score = _score_pick(pick, preference=int(candidate["preference"]))
        picks.append(pick)
    return picks


def _market_alias_requested(requested_markets: set[str], market: str, selection: str | None) -> bool:
    if market.lower() in requested_markets:
        return True
    if selection is None:
        return False
    aliases = {
        "Local": {"local", "home", "1x2 local", "1x2 home"},
        "Visitante": {"visitante", "away", "1x2 visitante", "1x2 away"},
        "Empate": {"empate", "draw", "1x2 empate", "1x2 draw"},
    }
    return bool(requested_markets.intersection(aliases.get(selection, set())))


def _matches_market(pick: Pick, market: str) -> bool:
    normalized = market.strip().lower()
    if pick.market.lower() == normalized:
        return True
    if pick.selection and f"{pick.market} {pick.selection}".lower() == normalized:
        return True
    aliases = {
        "local": ("1x2", "local"),
        "home": ("1x2", "local"),
        "visitante": ("1x2", "visitante"),
        "away": ("1x2", "visitante"),
        "empate": ("1x2", "empate"),
        "draw": ("1x2", "empate"),
    }
    alias = aliases.get(normalized)
    return bool(alias and pick.market.lower() == alias[0] and (pick.selection or "").lower() == alias[1])


def _build_factors(row: object, market: str, selection: str | None = None) -> list[str]:
    if market == "1X2":
        return [
            f"Diferencia ELO local: {row.elo_diff:+.0f}",
            f"Forma local: {row.home_points_avg_5:.2f} pts de media reciente",
            f"Forma visitante: {row.away_points_avg_5:.2f} pts de media reciente",
            f"Resultado objetivo: {selection}",
        ]
    if market in {"Over 1.5", "Over 2.5"}:
        return [
            f"Mercado objetivo: {market}",
            f"Local en casa anota {row.home_home_goals_for_avg_5:.2f} de media reciente",
            f"Visitante fuera anota {row.away_away_goals_for_avg_5:.2f} de media reciente",
            f"Diferencial ofensivo estimado: {row.attack_diff:+.2f}",
        ]
    if market == "Under 4.5":
        return [
            "Mercado objetivo: Under 4.5",
            f"Local concede {row.home_goals_against_avg_5:.2f} goles de media reciente",
            f"Visitante concede {row.away_goals_against_avg_5:.2f} goles de media reciente",
            f"Balance de gol estimado: {row.goal_balance_diff:+.2f}",
        ]
    return [
        f"Local llega con {row.home_btts_rate_5:.0%} de BTTS en sus ultimos 5",
        f"Visitante fuera tiene {row.away_away_goals_for_avg_5:.2f} goles de media",
        f"La diferencia ELO es {row.elo_diff:+.0f} para el local",
    ]


def _model_names_by_market() -> dict[str, str]:
    settings = get_settings()
    report_file = settings.reports_dir / "training_summary.json"
    defaults = {
        "Over 1.5": "unknown",
        "Over 2.5": "unknown",
        "Under 4.5": "unknown",
        "BTTS": "unknown",
        "1X2 Local": "unknown",
        "1X2 Empate": "unknown",
        "1X2 Visitante": "unknown",
    }
    if not report_file.exists():
        return defaults

    try:
        summary = json.loads(report_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults

    return {
        "Over 1.5": summary.get("target_over15", {}).get("selected_model", "unknown"),
        "Over 2.5": summary.get("target_over25", {}).get("selected_model", "unknown"),
        "Under 4.5": summary.get("target_under45", {}).get("selected_model", "unknown"),
        "BTTS": summary.get("target_btts", {}).get("selected_model", "unknown"),
        "1X2 Local": summary.get("target_home_win", {}).get("selected_model", "unknown"),
        "1X2 Empate": summary.get("target_draw", {}).get("selected_model", "unknown"),
        "1X2 Visitante": summary.get("target_away_win", {}).get("selected_model", "unknown"),
    }


def _load_latest_fixtures(refresh: bool = False) -> pd.DataFrame:
    settings = get_settings()
    fixtures_file = settings.processed_dir / "fixture_features.csv"
    required_fixture_columns = set(FEATURE_COLUMNS + ["Date", "HomeTeam", "AwayTeam", "League"])
    if refresh:
        return build_fixture_features()

    if fixtures_file.exists():
        try:
            cached = pd.read_csv(fixtures_file)
        except EmptyDataError:
            cached = pd.DataFrame()
        else:
            if required_fixture_columns.issubset(set(cached.columns)):
                cached["Date"] = pd.to_datetime(cached["Date"], errors="coerce", utc=True)
                cached = cached.dropna(subset=["Date"])
                if not cached.empty and (cached["Date"] >= _now_utc()).any():
                    return cached

    return build_fixture_features()


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _attach_market_odds(fixtures: pd.DataFrame) -> pd.DataFrame:
    odds = load_upcoming_market_odds()
    fixture_copy = fixtures.copy()
    if odds.empty:
        for column in ODDS_COLUMNS:
            fixture_copy[column] = None
        return fixture_copy

    fixture_copy["match_day"] = pd.to_datetime(fixture_copy["Date"], errors="coerce").dt.date
    merged = fixture_copy.merge(
        odds,
        on=["match_day", "League", "HomeTeam", "AwayTeam"],
        how="left",
    )
    for column in ODDS_COLUMNS:
        if column not in merged.columns:
            merged[column] = None
    return merged.drop(columns=["match_day"])


def _implied_probability(odd: float | None) -> float | None:
    if odd is None or odd <= 1:
        return None
    return 1.0 / odd


def _edge(model_probability: float, odd: float | None) -> float | None:
    implied = _implied_probability(odd)
    if implied is None:
        return None
    return model_probability - implied


def _expected_value(model_probability: float, odd: float | None) -> float | None:
    if odd is None or odd <= 1:
        return None
    return (model_probability * (odd - 1.0)) - (1.0 - model_probability)


def _row_value(row: object, attribute: str) -> float | None:
    value = getattr(row, attribute, None)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _stake_units(edge: float | None, expected_value: float | None) -> int | None:
    if edge is None or expected_value is None:
        return None
    if edge >= 0.10 and expected_value >= 0.08:
        return 3
    if edge >= 0.06 and expected_value >= 0.05:
        return 2
    if edge >= 0.02 and expected_value >= 0.02:
        return 1
    return None


def _rating(edge: float | None, expected_value: float | None) -> str | None:
    if edge is None or expected_value is None:
        return None
    if edge >= 0.10 and expected_value >= 0.08:
        return "A"
    if edge >= 0.05 and expected_value >= 0.04:
        return "B"
    if edge >= 0.02 and expected_value >= 0.02:
        return "C"
    return None


def _score_pick(pick: Pick, preference: int = 1) -> float:
    return (
        float(pick.expected_value or 0.0) * 1000
        + float(pick.edge or 0.0) * 100
        + float(pick.probability) * 10
        + preference
    )


def _pick_sort_key(pick: Pick) -> tuple[float, float, float, float]:
    preference = 1.0
    if pick.market == "1X2" and pick.selection in {"Local", "Visitante"}:
        preference = 2.0
    return (
        float(pick.expected_value or 0.0),
        float(pick.edge or 0.0),
        float(pick.probability),
        preference,
    )


def _best_pick_score(pick: Pick) -> tuple[float, float, float, float]:
    return _pick_sort_key(pick)
