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


def _confidence_label(probability: float) -> str:
    if probability >= 0.72:
        return "Alta"
    if probability >= 0.65:
        return "Media-Alta"
    return "Media"


def build_top_picks(limit: int = 5, threshold: float = 0.65) -> list[Pick]:
    settings = get_settings()
    over_model_file = settings.models_dir / "target_over25.joblib"
    btts_model_file = settings.models_dir / "target_btts.joblib"
    model_names = _model_names_by_market()
    if not over_model_file.exists() or not btts_model_file.exists():
        return []

    fixtures_file = settings.processed_dir / "fixture_features.csv"
    required_fixture_columns = set(FEATURE_COLUMNS + ["Date", "HomeTeam", "AwayTeam", "League"])
    if fixtures_file.exists():
        try:
            latest = pd.read_csv(fixtures_file)
        except EmptyDataError:
            latest = build_fixture_features()
        else:
            if not required_fixture_columns.issubset(set(latest.columns)):
                latest = build_fixture_features()
    else:
        latest = build_fixture_features()

    if latest.empty:
        return []

    latest["Date"] = pd.to_datetime(latest["Date"], errors="coerce")
    latest = latest.dropna(subset=FEATURE_COLUMNS + ["Date", "HomeTeam", "AwayTeam", "League"])
    latest = latest.sort_values("Date").head(limit * 5).copy()
    if latest.empty:
        return []

    latest = _attach_market_odds(latest)

    over_model = joblib.load(over_model_file)
    btts_model = joblib.load(btts_model_file)
    latest["prob_over25"] = over_model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]
    latest["prob_btts"] = btts_model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]

    picks: list[Pick] = []
    for row in latest.itertuples(index=False):
        if row.prob_over25 >= threshold:
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
        if row.prob_btts >= threshold:
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
    value_picks.sort(key=lambda item: (item.expected_value or 0.0, item.edge or 0.0), reverse=True)
    return value_picks[:limit]


def _build_factors(row: object, market: str) -> list[str]:
    if market == "over":
        return [
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
    defaults = {"Over 2.5": "unknown", "BTTS": "unknown"}
    if not report_file.exists():
        return defaults

    try:
        summary = json.loads(report_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults

    return {
        "Over 2.5": summary.get("target_over25", {}).get("selected_model", "unknown"),
        "BTTS": summary.get("target_btts", {}).get("selected_model", "unknown"),
    }


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
