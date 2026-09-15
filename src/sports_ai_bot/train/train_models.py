from __future__ import annotations

import json
import hashlib

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sports_ai_bot.utils.config import get_settings
from sports_ai_bot.features.build import DATA_SCHEMA_VERSION


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
    "home_corners_for_avg_5",
    "home_corners_against_avg_5",
    "away_corners_for_avg_5",
    "away_corners_against_avg_5",
    "home_home_corners_for_avg_5",
    "home_home_corners_against_avg_5",
    "away_away_corners_for_avg_5",
    "away_away_corners_against_avg_5",
    "home_rest_days",
    "away_rest_days",
    "elo_home",
    "elo_away",
    "elo_diff",
    "attack_diff",
    "defense_diff",
    "corners_balance_diff",
    "corners_total_avg_5",
    "form_diff",
    "goal_balance_diff",
]


def _logistic_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500)),
        ]
    )


def _split_temporally(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("Date", kind="stable").reset_index(drop=True)
    days = ordered["Date"].dt.normalize()
    unique_days = days.drop_duplicates().tolist()
    if len(unique_days) < 3:
        return ordered, ordered.iloc[:0], ordered.iloc[:0]
    first = max(1, int(len(unique_days) * 0.6))
    second = min(len(unique_days) - 1, max(first + 1, int(len(unique_days) * 0.8)))
    return (ordered[days < unique_days[first]],
            ordered[(days >= unique_days[first]) & (days < unique_days[second])],
            ordered[days >= unique_days[second]])


def _metrics(y_test: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
    }


MIN_SAMPLES = {"train": 200, "calibration": 100, "test": 200}
MIN_CLASS_SAMPLES = 20
LEAGUE_MIN_SAMPLES = {"train": 100, "calibration": 50, "test": 100}
LEAGUE_MIN_TEST_CLASS_SAMPLES = 10
MODEL_RECIPE_VERSION = "logistic-sigmoid-temporal-v2"


def _calibration_bins(labels: pd.Series, probabilities: np.ndarray) -> list[dict[str, object]]:
    # Left-closed intervals, with probability 1 included in the last bin.
    assignments = np.searchsorted([index / 5 for index in range(1, 5)], probabilities, side="right")
    bins = []
    for index in range(5):
        mask = assignments == index
        bins.append({
            "lower": index / 5, "upper": (index + 1) / 5,
            "n": int(mask.sum()),
            "mean_probability": float(probabilities[mask].mean()) if mask.any() else None,
            "observed_rate": float(labels.to_numpy()[mask].mean()) if mask.any() else None,
        })
    return bins


def _per_league_report(splits, target_column: str, model, global_status: str) -> dict[str, object]:
    if "League" not in splits["train"]:
        return {}
    leagues = pd.concat([part["League"] for part in splits.values()]).dropna().unique()
    result = {}
    for league in sorted(leagues):
        parts = {name: part[part["League"].eq(league)] for name, part in splits.items()}
        samples = {}
        reasons = []
        for name, part in parts.items():
            counts = part[target_column].value_counts()
            samples[name] = {
                "n": len(part), "negative": int(counts.get(0, 0)), "positive": int(counts.get(1, 0)),
                "start": part["Date"].min().isoformat() if len(part) else None,
                "end": part["Date"].max().isoformat() if len(part) else None,
            }
            if len(part) < LEAGUE_MIN_SAMPLES[name]:
                reasons.append(f"insufficient_{name}_sample")
        if min(samples["test"]["negative"], samples["test"]["positive"]) < LEAGUE_MIN_TEST_CLASS_SAMPLES:
            reasons.append("insufficient_test_class_sample")
        if global_status != "validated":
            reasons.append("global_not_validated")
        if model is None:
            reasons.append("model_unavailable")
        prevalence = float(parts["train"][target_column].mean()) if len(parts["train"]) else None
        entry = {
            "validation_status": "experimental", "validation_reasons": reasons,
            "samples": samples, "test_sample": samples["test"],
            "test_metrics": None, "calibration_metrics": None,
            "test_calibration_bins": [], "calibration_bins": [],
            "baseline_train_prevalence": prevalence, "baseline_metrics": {},
            "validation_policy": {
                "min_samples": dict(LEAGUE_MIN_SAMPLES),
                "min_test_class_samples": LEAGUE_MIN_TEST_CLASS_SAMPLES,
                "requires_global_validated": True,
                "required_metrics": ["brier_score", "log_loss"],
                "required_baselines": ["train_prevalence", "constant_0_5"],
                "comparison": "strictly_lower_on_test",
            },
        }
        if model is not None:
            for name in ("calibration", "test"):
                part = parts[name]
                if not part.empty:
                    probabilities = model.predict_proba(part[FEATURE_COLUMNS])[:, 1]
                    entry[f"{name}_metrics"] = _metrics(part[target_column], probabilities)
                    bin_key = "test_calibration_bins" if name == "test" else "calibration_bins"
                    entry[bin_key] = _calibration_bins(part[target_column], probabilities)
            if prevalence is not None and not parts["test"].empty:
                entry["baseline_metrics"] = {
                    name: _metrics(parts["test"][target_column], np.full(len(parts["test"]), probability))
                    for name, probability in (("train_prevalence", prevalence), ("constant_0_5", 0.5))
                }
                if not all(entry["test_metrics"][metric] < baseline[metric]
                           for baseline in entry["baseline_metrics"].values()
                           for metric in ("brier_score", "log_loss")):
                    reasons.append("does_not_beat_baselines")
        entry["validation_status"] = "experimental" if reasons else "validated"
        result[str(league)] = entry
    return result


def _fit_calibrated(train_frame, calibration_frame, target_column):
    base_model = _logistic_model()
    base_model.fit(train_frame[FEATURE_COLUMNS], train_frame[target_column].astype(int))
    # FrozenEstimator (sklearn >= 1.6) never refits the base on calibration data.
    model = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    model.fit(calibration_frame[FEATURE_COLUMNS], calibration_frame[target_column].astype(int))
    return model


def _walk_forward_research(frame: pd.DataFrame, target_column: str) -> dict[str, object]:
    research = {"used_for_gate": False, "scope": "train_only", "windows": []}
    days = frame["Date"].dt.normalize().drop_duplicates().sort_values().tolist()
    # Two expanding prefixes of production train; calibration and final test stay untouched.
    for fraction in (0.65, 1.0):
        count = int(len(days) * fraction)
        prefix = frame[frame["Date"].dt.normalize().isin(days[:count])]
        train_part, cal_part, test_part = _split_temporally(prefix)
        parts = {"train": train_part, "calibration": cal_part, "test": test_part}
        window = {"train_fraction": fraction, "status": "skipped_insufficient_sample",
                  "samples": {name: {"n": len(part),
                                     "start": part.Date.min().isoformat() if len(part) else None,
                                     "end": part.Date.max().isoformat() if len(part) else None}
                              for name, part in parts.items()},
                  "test_metrics": None, "baseline_metrics": {}}
        if all(len(part) >= 20 and part[target_column].nunique() == 2 for part in parts.values()):
            model = _fit_calibrated(train_part, cal_part, target_column)
            window["status"] = "evaluated"
            window["test_metrics"] = _metrics(test_part[target_column], model.predict_proba(test_part[FEATURE_COLUMNS])[:, 1])
            window["baseline_metrics"] = {
                name: _metrics(test_part[target_column], np.full(len(test_part), probability))
                for name, probability in (("train_prevalence", float(train_part[target_column].mean())),
                                          ("constant_0_5", 0.5))
            }
        research["windows"].append(window)
    return research


def _train_single_target(frame: pd.DataFrame, target_column: str) -> dict[str, object]:
    settings = get_settings()
    frame = frame.copy()
    input_rows = len(frame)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", utc=True).dt.tz_convert(None)
    frame[target_column] = pd.to_numeric(frame.get(target_column), errors="coerce")
    frame = frame[frame["Date"].notna() & frame[target_column].isin([0, 1])].copy()
    eligible_rows = len(frame)
    keys = ["League", "HomeTeam", "AwayTeam"]
    if all(key in frame for key in keys):
        frame["_match_day"] = frame["Date"].dt.normalize()
        frame = frame.drop_duplicates(keys + ["_match_day"], keep="last").drop(columns="_match_day")
    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = frame.sort_values("Date", kind="stable")
    version_columns = ["Date"] + FEATURE_COLUMNS + [target_column]
    if "League" in frame:
        version_columns.append("League")
    data_version = hashlib.sha256(pd.util.hash_pandas_object(
        frame[version_columns], index=False
    ).values.tobytes()).hexdigest()
    train_frame, calibration_frame, test_frame = _split_temporally(frame)
    splits = {"train": train_frame, "calibration": calibration_frame, "test": test_frame}
    samples = {}
    reasons = []
    for name, part in splits.items():
        counts = part[target_column].value_counts()
        samples[name] = {
            "n": len(part), "negative": int(counts.get(0, 0)), "positive": int(counts.get(1, 0)),
            "start": part["Date"].min().isoformat() if len(part) else None,
            "end": part["Date"].max().isoformat() if len(part) else None,
        }
        if len(part) < MIN_SAMPLES[name] or min(counts.get(0, 0), counts.get(1, 0)) < MIN_CLASS_SAMPLES:
            reasons.append(f"insufficient_{name}_sample")
    if "data_schema_version" not in frame or not frame["data_schema_version"].eq(DATA_SCHEMA_VERSION).all():
        reasons.append("unverified_data_schema")
    report = {
        "report_schema_version": 2,
        "validation_status": "experimental",
        "validation_reasons": reasons,
        "model_version": None,
        "model_recipe_version": MODEL_RECIPE_VERSION,
        "sklearn_version": sklearn.__version__,
        "data_version": data_version,
        "data_schema_version": DATA_SCHEMA_VERSION if "unverified_data_schema" not in reasons else None,
        "feature_columns": list(FEATURE_COLUMNS),
        "trained_through": None,
        "base_trained_through": None,
        "artifact_available": False,
        "selected_model": "logistic_regression_sigmoid",
        "selection_policy": "fixed_a_priori_no_test_selection",
        "samples": samples,
        "test_sample": samples["test"],
        "test_metrics": None,
        "baseline_metrics": {},
        "baseline_train_prevalence": None,
        "per_league": {},
        "research": {"walk_forward": {"used_for_gate": False, "scope": "train_only", "windows": []}},
        "quality": {"input_rows": input_rows, "invalid_date_or_target_rows": input_rows - eligible_rows,
                    "duplicate_rows_removed": eligible_rows - len(frame), "target_rows": len(frame),
                    "rows_with_missing_features": int(frame[FEATURE_COLUMNS].isna().any(axis=1).sum())},
        "validation_policy": {"min_samples": dict(MIN_SAMPLES), "min_class_samples_per_split": MIN_CLASS_SAMPLES,
                              "required_metrics": ["brier_score", "log_loss"],
                              "required_baselines": ["train_prevalence", "constant_0_5"],
                              "comparison": "strictly_lower_on_test"},
    }
    if any(part.empty for part in splits.values()) or any(
        part[target_column].nunique() != 2 for part in (train_frame, calibration_frame)
    ):
        reasons.append("untrainable_split")
        report["per_league"] = _per_league_report(splits, target_column, None, "experimental")
        return report

    model = _fit_calibrated(train_frame, calibration_frame, target_column)
    test_metrics = _metrics(test_frame[target_column], model.predict_proba(test_frame[FEATURE_COLUMNS])[:, 1])
    prevalence = float(train_frame[target_column].mean())
    baselines = {name: _metrics(test_frame[target_column], np.full(len(test_frame), probability))
                 for name, probability in (("train_prevalence", prevalence), ("constant_0_5", 0.5))}
    if not all(test_metrics[metric] < baseline[metric]
               for baseline in baselines.values() for metric in ("brier_score", "log_loss")):
        reasons.append("does_not_beat_baselines")
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    artifact = settings.models_dir / f"{target_column}.joblib"
    joblib.dump(model, artifact)
    report.update({
        "validation_status": "experimental" if reasons else "validated",
        "model_version": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "artifact_available": True,
        "trained_through": samples["calibration"]["end"],
        "base_trained_through": samples["train"]["end"],
        "test_metrics": test_metrics, "baseline_metrics": baselines,
        "baseline_train_prevalence": prevalence,
    })
    report["per_league"] = _per_league_report(splits, target_column, model, report["validation_status"])
    report["research"]["walk_forward"] = _walk_forward_research(train_frame, target_column)
    return report


def train_models() -> dict[str, dict[str, object]]:
    settings = get_settings()
    dataset_file = settings.processed_dir / "training_dataset.csv"
    if not dataset_file.exists():
        raise FileNotFoundError("No existe training_dataset.csv. Ejecuta build-dataset primero.")

    frame = pd.read_csv(dataset_file)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "target_over15": _train_single_target(frame, "target_over15"),
        "target_over25": _train_single_target(frame, "target_over25"),
        "target_under45": _train_single_target(frame, "target_under45"),
        "target_btts": _train_single_target(frame, "target_btts"),
        "target_home_win": _train_single_target(frame, "target_home_win"),
        "target_draw": _train_single_target(frame, "target_draw"),
        "target_away_win": _train_single_target(frame, "target_away_win"),
        "target_corners_over95": _train_single_target(frame, "target_corners_over95"),
    }

    report_file = settings.reports_dir / "training_summary.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
