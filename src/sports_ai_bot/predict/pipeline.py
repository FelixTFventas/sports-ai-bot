from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import joblib
import pandas as pd
from pandas.errors import EmptyDataError

from sports_ai_bot.features.build import build_fixture_features
from sports_ai_bot.train.train_models import FEATURE_COLUMNS
from sports_ai_bot.utils.config import get_settings


@dataclass
class Pick:
    match_label: str
    league: str
    market: str
    probability: float
    confidence: str
    factors: list[str]


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

    over_model = joblib.load(over_model_file)
    btts_model = joblib.load(btts_model_file)
    latest["prob_over25"] = over_model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]
    latest["prob_btts"] = btts_model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]

    picks: list[Pick] = []
    for row in latest.itertuples(index=False):
        if row.prob_over25 >= threshold:
            picks.append(
                Pick(
                    match_label=f"{row.HomeTeam} vs {row.AwayTeam}",
                    league=row.League,
                    market="Over 2.5",
                    probability=float(row.prob_over25),
                    confidence=_confidence_label(float(row.prob_over25)),
                    factors=_build_factors(row, market="over"),
                )
            )
        if row.prob_btts >= threshold:
            picks.append(
                Pick(
                    match_label=f"{row.HomeTeam} vs {row.AwayTeam}",
                    league=row.League,
                    market="BTTS",
                    probability=float(row.prob_btts),
                    confidence=_confidence_label(float(row.prob_btts)),
                    factors=_build_factors(row, market="btts"),
                )
            )

    picks.sort(key=lambda item: item.probability, reverse=True)
    return picks[:limit]


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
                "match_label": pick.match_label,
                "league": pick.league,
                "market": pick.market,
                "probability": round(pick.probability, 4),
                "confidence": pick.confidence,
                "factors": " | ".join(pick.factors),
            }
            for pick in picks
        ],
        columns=[
            "prediction_date",
            "match_label",
            "league",
            "market",
            "probability",
            "confidence",
            "factors",
        ],
    )
    output_file = settings.predictions_dir / f"picks_{date.today().isoformat()}.csv"
    frame.to_csv(output_file, index=False)
    return frame
