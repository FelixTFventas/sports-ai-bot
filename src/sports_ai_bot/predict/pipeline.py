from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json

import joblib
import pandas as pd
from pandas.errors import EmptyDataError

from sports_ai_bot.collect.odds import load_upcoming_market_odds
from sports_ai_bot.features.build import build_fixture_features
from sports_ai_bot.train.train_models import FEATURE_COLUMNS
from sports_ai_bot.utils.config import get_settings


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
    odd: float | None = None
    implied_probability: float | None = None
    edge: float | None = None
    expected_value: float | None = None
    stake_units: int | None = None
    rating: str | None = None


def _confidence_label(probability: float) -> str:
    if probability >= 0.72:
        return "Alta"
    if probability >= 0.65:
        return "Media-Alta"
    return "Media"


def build_top_picks(
    limit: int = 10,
    threshold: float = 0.60,
    refresh_fixtures: bool = True,
    markets: tuple[str, ...] = ("Over 2.5", "BTTS"),
    horizon_hours: int = 48,
) -> list[Pick]:
    settings = get_settings()
    over15_model_file = settings.models_dir / "target_over15.joblib"
    over_model_file = settings.models_dir / "target_over25.joblib"
    btts_model_file = settings.models_dir / "target_btts.joblib"
    model_names = _model_names_by_market()
    if (
        not over15_model_file.exists()
        or not over_model_file.exists()
        or not btts_model_file.exists()
    ):
        return []

    latest = _load_latest_fixtures(refresh=refresh_fixtures)

    if latest.empty:
        return []

    latest["Date"] = pd.to_datetime(latest["Date"], errors="coerce", utc=True)
    latest = latest.dropna(subset=FEATURE_COLUMNS + ["Date", "HomeTeam", "AwayTeam", "League"])
    window_start = _now_utc()
    window_end = window_start + pd.Timedelta(hours=horizon_hours)
    latest = latest[(latest["Date"] >= window_start) & (latest["Date"] <= window_end)]
    latest = latest.sort_values("Date").head(limit * 5).copy()
    if latest.empty:
        return []

    latest = _attach_market_odds(latest)

    over15_model = joblib.load(over15_model_file)
    over_model = joblib.load(over_model_file)
    btts_model = joblib.load(btts_model_file)
    latest["prob_over15"] = over15_model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]
    latest["prob_over25"] = over_model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]
    latest["prob_btts"] = btts_model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]

    picks: list[Pick] = []
    for row in latest.itertuples(index=False):
        if "Over 1.5" in markets and row.prob_over15 >= threshold:
            picks.append(
                Pick(
                    match_date=row.Date.date().isoformat(),
                    home_team=row.HomeTeam,
                    away_team=row.AwayTeam,
                    match_label=f"{row.HomeTeam} vs {row.AwayTeam}",
                    league=row.League,
                    market="Over 1.5",
                    probability=float(row.prob_over15),
                    confidence=_confidence_label(float(row.prob_over15)),
                    model_name=model_names["Over 1.5"],
                    factors=_build_factors(row, market="over15"),
                )
            )
        if "Over 2.5" in markets and row.prob_over25 >= threshold:
            picks.append(
                Pick(
                    match_date=row.Date.date().isoformat(),
                    home_team=row.HomeTeam,
                    away_team=row.AwayTeam,
                    match_label=f"{row.HomeTeam} vs {row.AwayTeam}",
                    league=row.League,
                    market="Over 2.5",
                    probability=float(row.prob_over25),
                    confidence=_confidence_label(float(row.prob_over25)),
                    model_name=model_names["Over 2.5"],
                    odd=_row_value(row, "odd_over25"),
                    implied_probability=_implied_probability(_row_value(row, "odd_over25")),
                    edge=_edge(float(row.prob_over25), _row_value(row, "odd_over25")),
                    expected_value=_expected_value(
                        float(row.prob_over25), _row_value(row, "odd_over25")
                    ),
                    factors=_build_factors(row, market="over"),
                )
            )
        if "BTTS" in markets and row.prob_btts >= threshold:
            picks.append(
                Pick(
                    match_date=row.Date.date().isoformat(),
                    home_team=row.HomeTeam,
                    away_team=row.AwayTeam,
                    match_label=f"{row.HomeTeam} vs {row.AwayTeam}",
                    league=row.League,
                    market="BTTS",
                    probability=float(row.prob_btts),
                    confidence=_confidence_label(float(row.prob_btts)),
                    model_name=model_names["BTTS"],
                    factors=_build_factors(row, market="btts"),
                )
            )

    picks.sort(key=lambda item: item.probability, reverse=True)
    return picks[:limit]


def build_value_picks(limit: int = 5, min_edge: float = 0.02, min_ev: float = 0.02) -> list[Pick]:
    picks = build_top_picks(limit=limit * 4, threshold=0.6)
    value_picks = [
        pick
        for pick in picks
        if pick.odd is not None
        and pick.edge is not None
        and pick.expected_value is not None
        and pick.edge >= min_edge
        and pick.expected_value >= min_ev
    ]
    for pick in value_picks:
        pick.stake_units = _stake_units(pick.edge, pick.expected_value)
        pick.rating = _rating(pick.edge, pick.expected_value)
    value_picks.sort(key=lambda item: (item.expected_value or 0.0, item.edge or 0.0), reverse=True)
    return value_picks[:limit]


def build_best_picks(limit: int = 5) -> list[Pick]:
    picks = build_value_picks(limit=limit * 3)
    for pick in picks:
        if pick.stake_units is None:
            pick.stake_units = _stake_units(pick.edge, pick.expected_value)
        if pick.rating is None:
            pick.rating = _rating(pick.edge, pick.expected_value)
    picks.sort(key=_best_pick_score, reverse=True)
    return picks[:limit]


def _build_factors(row: object, market: str) -> list[str]:
    if market in {"over", "over15"}:
        threshold_label = "1.5" if market == "over15" else "2.5"
        return [
            f"Mercado objetivo: Over {threshold_label}",
            f"Local en casa anota {row.home_home_goals_for_avg_5:.2f} de media reciente",
            f"Visitante fuera anota {row.away_away_goals_for_avg_5:.2f} de media reciente",
            f"Diferencial ofensivo estimado: {row.attack_diff:+.2f}",
        ]
    return [
        f"Local llega con {row.home_btts_rate_5:.0%} de BTTS en sus ultimos 5",
        f"Visitante fuera tiene {row.away_away_goals_for_avg_5:.2f} goles de media",
        f"La diferencia ELO es {row.elo_diff:+.0f} para el local",
    ]


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
                "factors": " | ".join(pick.factors),
                "status": "pending",
                "outcome": "pending",
                "result_home_goals": None,
                "result_away_goals": None,
            }
            for pick in picks
        ],
        columns=[
            "prediction_date",
            "match_date",
            "home_team",
            "away_team",
            "match_label",
            "league",
            "market",
            "probability",
            "confidence",
            "model_name",
            "odd",
            "implied_probability",
            "edge",
            "expected_value",
            "stake_units",
            "rating",
            "factors",
            "status",
            "outcome",
            "result_home_goals",
            "result_away_goals",
        ],
    )
    output_file = settings.predictions_dir / f"picks_{date.today().isoformat()}.csv"
    frame.to_csv(output_file, index=False)
    return frame


def _model_names_by_market() -> dict[str, str]:
    settings = get_settings()
    report_file = settings.reports_dir / "training_summary.json"
    defaults = {"Over 1.5": "unknown", "Over 2.5": "unknown", "BTTS": "unknown"}
    if not report_file.exists():
        return defaults

    try:
        summary = json.loads(report_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults

    return {
        "Over 1.5": summary.get("target_over15", {}).get("selected_model", "unknown"),
        "Over 2.5": summary.get("target_over25", {}).get("selected_model", "unknown"),
        "BTTS": summary.get("target_btts", {}).get("selected_model", "unknown"),
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


def build_market_picks(
    market: str,
    limit: int = 10,
    threshold: float = 0.60,
    refresh_fixtures: bool = True,
    horizon_hours: int = 48,
) -> list[Pick]:
    return build_top_picks(
        limit=limit,
        threshold=threshold,
        refresh_fixtures=refresh_fixtures,
        markets=(market,),
        horizon_hours=horizon_hours,
    )


def _attach_market_odds(fixtures: pd.DataFrame) -> pd.DataFrame:
    odds = load_upcoming_market_odds()
    if odds.empty:
        fixtures["odd_over25"] = None
        return fixtures

    fixture_copy = fixtures.copy()
    fixture_copy["match_day"] = pd.to_datetime(fixture_copy["Date"], errors="coerce").dt.date
    merged = fixture_copy.merge(
        odds, on=["match_day", "League", "HomeTeam", "AwayTeam"], how="left"
    )
    if "odd_over25" not in merged.columns:
        merged["odd_over25"] = None
    merged = merged.drop(columns=["match_day"])
    return merged


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


def _best_pick_score(pick: Pick) -> tuple[float, float, float, float]:
    return (
        float(pick.stake_units or 0),
        float(pick.expected_value or 0.0),
        float(pick.edge or 0.0),
        float(pick.probability),
    )
