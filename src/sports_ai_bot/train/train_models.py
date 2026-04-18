from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sports_ai_bot.utils.config import get_settings


FEATURE_COLUMNS = [
    "home_goals_for_avg_5",
    "home_goals_against_avg_5",
    "home_points_avg_5",
    "home_over25_rate_5",
    "home_btts_rate_5",
    "away_goals_for_avg_5",
    "away_goals_against_avg_5",
    "away_points_avg_5",
    "away_over25_rate_5",
    "away_btts_rate_5",
    "home_home_goals_for_avg_5",
    "home_home_goals_against_avg_5",
    "home_home_points_avg_5",
    "away_away_goals_for_avg_5",
    "away_away_goals_against_avg_5",
    "away_away_points_avg_5",
    "home_rest_days",
    "away_rest_days",
    "elo_home",
    "elo_away",
    "elo_diff",
    "attack_diff",
    "defense_diff",
    "form_diff",
    "goal_balance_diff",
]


def _logistic_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500)),
        ]
    )


def _boosting_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=250,
        min_samples_leaf=30,
        random_state=42,
    )


def _split_temporally(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    ordered = frame.sort_values("Date").reset_index(drop=True)
    split_index = int(len(ordered) * 0.8)
    X_train = ordered.iloc[:split_index][FEATURE_COLUMNS]
    X_test = ordered.iloc[split_index:][FEATURE_COLUMNS]
    return X_train, X_test, ordered.iloc[:split_index], ordered.iloc[split_index:]


def _evaluate_model(
    model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
):
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "log_loss": round(float(log_loss(y_test, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_test, probabilities)), 4),
    }
    return model, metrics


def _train_single_target(frame: pd.DataFrame, target_column: str) -> dict[str, object]:
    settings = get_settings()
    X_train, X_test, train_frame, test_frame = _split_temporally(frame)
    y_train = train_frame[target_column]
    y_test = test_frame[target_column]

    candidates = {
        "logistic_regression": _logistic_model(),
        "hist_gradient_boosting": _boosting_model(),
    }

    evaluations: dict[str, dict[str, float]] = {}
    best_name = ""
    best_model = None
    best_log_loss = float("inf")

    for model_name, model in candidates.items():
        trained_model, metrics = _evaluate_model(model, X_train, y_train, X_test, y_test)
        evaluations[model_name] = metrics
        if metrics["log_loss"] < best_log_loss:
            best_name = model_name
            best_model = trained_model
            best_log_loss = metrics["log_loss"]

    if best_model is None:
        raise RuntimeError(f"No se pudo entrenar el target {target_column}")

    joblib.dump(best_model, settings.models_dir / f"{target_column}.joblib")
    return {
        "selected_model": best_name,
        "models": evaluations,
    }


def train_models() -> dict[str, dict[str, object]]:
    settings = get_settings()
    dataset_file = settings.processed_dir / "training_dataset.csv"
    if not dataset_file.exists():
        raise FileNotFoundError("No existe training_dataset.csv. Ejecuta build-dataset primero.")

    frame = pd.read_csv(dataset_file)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=FEATURE_COLUMNS + ["Date"])
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "target_over15": _train_single_target(frame, "target_over15"),
        "target_over25": _train_single_target(frame, "target_over25"),
        "target_btts": _train_single_target(frame, "target_btts"),
    }

    report_file = settings.reports_dir / "training_summary.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
