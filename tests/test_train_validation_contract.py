import hashlib
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from sports_ai_bot.train import train_models as train


def _frame(n=1000, signal=True):
    labels = np.arange(n) % 2
    frame = pd.DataFrame({column: labels.astype(float) if signal else np.ones(n)
                          for column in train.FEATURE_COLUMNS})
    frame["Date"] = pd.date_range("2000-01-01", periods=n)
    frame["target_over25"] = labels
    frame["target_corners_over95"] = np.nan
    frame["data_schema_version"] = train.DATA_SCHEMA_VERSION
    for column in train.FEATURE_COLUMNS:
        if "corners" in column:
            frame[column] = np.nan
    return frame


@pytest.fixture
def settings(monkeypatch, tmp_path):
    settings = SimpleNamespace(models_dir=tmp_path / "models", processed_dir=tmp_path / "processed",
                               reports_dir=tmp_path / "reports")
    monkeypatch.setattr(train, "get_settings", lambda: settings)
    return settings


def test_temporal_split_keeps_whole_days():
    frame = _frame(103)
    frame.loc[59:63, "Date"] = frame.loc[59, "Date"]
    parts = train._split_temporally(frame)
    assert sum(map(len, parts)) == len(frame)
    assert parts[0].Date.max().normalize() < parts[1].Date.min().normalize()
    assert parts[1].Date.max().normalize() < parts[2].Date.min().normalize()


def test_validated_report_and_binary_artifact(settings):
    frame = _frame()
    report = train._train_single_target(frame, "target_over25")
    assert report["validation_status"] == "validated"
    assert report["validation_reasons"] == []
    assert report["per_league"] == {}
    assert report["feature_columns"] == train.FEATURE_COLUMNS
    assert report["test_sample"]["n"] == 200
    assert report["trained_through"] < report["test_sample"]["start"]
    artifact = settings.models_dir / "target_over25.joblib"
    assert report["model_version"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    model = joblib.load(artifact)
    assert model.method == "sigmoid"
    assert list(model.classes_) == [0, 1]
    probabilities = model.predict_proba(frame[train.FEATURE_COLUMNS])
    assert probabilities.shape == (1000, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    for baseline in report["baseline_metrics"].values():
        for metric in ("brier_score", "log_loss"):
            assert report["test_metrics"][metric] < baseline[metric]
    assert report["quality"]["rows_with_missing_features"] == 1000
    json.dumps(report, allow_nan=False)


def test_test_labels_never_change_fitted_model(settings):
    frame = _frame()
    first = train._train_single_target(frame, "target_over25")
    frame.loc[800:, "target_over25"] = 1 - frame.loc[800:, "target_over25"]
    second = train._train_single_target(frame, "target_over25")
    assert first["model_version"] == second["model_version"]
    assert first["data_version"] != second["data_version"]
    assert second["validation_status"] == "experimental"
    assert "does_not_beat_baselines" in second["validation_reasons"]


@pytest.mark.parametrize("n,signal,reason", [(100, True, "insufficient_test_sample"),
                                           (1000, False, "does_not_beat_baselines")])
def test_validation_gates(settings, n, signal, reason):
    report = train._train_single_target(_frame(n, signal), "target_over25")
    assert report["validation_status"] == "experimental"
    assert reason in report["validation_reasons"]


def test_missing_target_and_single_class_fail_closed_preserve_existing(settings):
    settings.models_dir.mkdir()
    artifact = settings.models_dir / "target_corners_over95.joblib"
    artifact.write_bytes(b"existing artifact")
    report = train._train_single_target(_frame(), "target_corners_over95")
    assert report["quality"]["target_rows"] == 0
    assert not report["artifact_available"]
    assert report["model_version"] is None
    assert report["validation_status"] == "experimental"
    assert artifact.read_bytes() == b"existing artifact"
    report = train._train_single_target(_frame().assign(target_over25=1), "target_over25")
    assert "untrainable_split" in report["validation_reasons"]
    json.dumps(report, allow_nan=False)


def test_dedup_target_filter_and_unverified_schema(settings):
    frame = _frame().assign(League="synthetic", HomeTeam="A", AwayTeam="B")
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True).drop(columns="data_schema_version")
    frame.loc[1, "target_over25"] = np.nan
    report = train._train_single_target(frame, "target_over25")
    assert report["quality"]["duplicate_rows_removed"] == 1
    assert report["quality"]["invalid_date_or_target_rows"] == 1
    assert report["quality"]["target_rows"] == 999
    assert "unverified_data_schema" in report["validation_reasons"]
    assert report["validation_status"] == "experimental"


def test_imputer_fitted_on_train_only(settings):
    frame = _frame()
    frame.loc[:599, "home_rest_days"] = 7
    frame.loc[600:, "home_rest_days"] = 900
    train._train_single_target(frame, "target_over25")
    model = joblib.load(settings.models_dir / "target_over25.joblib")
    imputer = model.estimator.estimator.named_steps["imputer"]
    assert imputer.statistics_[train.FEATURE_COLUMNS.index("home_rest_days")] == 7


def test_train_entrypoint_persists_per_target_report_without_global_dropna(settings):
    frame = _frame(100)
    for target in ("target_over15", "target_under45", "target_btts", "target_home_win",
                   "target_draw", "target_away_win"):
        frame[target] = frame.target_over25
    settings.processed_dir.mkdir()
    frame.to_csv(settings.processed_dir / "training_dataset.csv", index=False)
    summary = train.train_models()
    persisted = json.loads((settings.reports_dir / "training_summary.json").read_text())
    assert summary == persisted
    assert len(summary) == 8
    assert summary["target_over25"]["quality"]["target_rows"] == 100
    assert summary["target_over25"]["artifact_available"]
    assert summary["target_corners_over95"]["quality"]["target_rows"] == 0
    assert not summary["target_corners_over95"]["artifact_available"]
    assert all(report["validation_status"] == "experimental" for report in summary.values())


class _SignalModel:
    def predict_proba(self, frame):
        probabilities = 0.1 + 0.8 * frame[train.FEATURE_COLUMNS[0]].to_numpy()
        return np.column_stack([1 - probabilities, probabilities])


def _league_splits(train_n=100, cal_n=50, test_n=100):
    return {name: _frame(n).assign(League="colombia")
            for name, n in (("train", train_n), ("calibration", cal_n), ("test", test_n))}


def test_per_league_exact_thresholds_and_local_prevalence():
    splits = _league_splits()
    splits["train"]["target_over25"] = (np.arange(100) < 80).astype(int)
    league = train._per_league_report(splits, "target_over25", _SignalModel(), "validated")["colombia"]
    assert league["validation_status"] == "validated"
    assert league["baseline_train_prevalence"] == 0.8
    assert [league["samples"][name]["n"] for name in splits] == [100, 50, 100]
    expected = train._metrics(splits["test"].target_over25, np.full(100, 0.8))
    assert league["baseline_metrics"]["train_prevalence"] == expected
    for key, n in (("calibration_bins", 50), ("test_calibration_bins", 100)):
        bins = league[key]
        assert len(bins) == 5
        assert sum(item["n"] for item in bins) == n
        assert bins[0]["observed_rate"] == 0
        assert bins[-1]["observed_rate"] == 1
        assert bins[1]["mean_probability"] is None
    json.dumps(league, allow_nan=False)


@pytest.mark.parametrize("train_n,cal_n,test_n,reason", [
    (99, 50, 100, "insufficient_train_sample"),
    (100, 49, 100, "insufficient_calibration_sample"),
    (100, 50, 99, "insufficient_test_sample"),
])
def test_per_league_sample_gates(train_n, cal_n, test_n, reason):
    splits = _league_splits(train_n, cal_n, test_n)
    league = train._per_league_report(splits, "target_over25", _SignalModel(), "validated")["colombia"]
    assert league["validation_status"] == "experimental"
    assert reason in league["validation_reasons"]


@pytest.mark.parametrize("positives,expected", [(9, "experimental"), (10, "validated"),
                                               (90, "validated"), (91, "experimental")])
def test_per_league_test_class_gate(positives, expected):
    splits = _league_splits()
    labels = (np.arange(100) < positives).astype(int)
    splits["test"]["target_over25"] = labels
    splits["test"][train.FEATURE_COLUMNS[0]] = labels
    league = train._per_league_report(splits, "target_over25", _SignalModel(), "validated")["colombia"]
    assert league["validation_status"] == expected


def test_per_league_global_gate_and_bad_local_predictions():
    splits = _league_splits()
    league = train._per_league_report(splits, "target_over25", _SignalModel(), "experimental")["colombia"]
    assert league["validation_reasons"] == ["global_not_validated"]
    assert league["validation_status"] == "experimental"
    splits["test"]["target_over25"] = 1 - splits["test"].target_over25
    league = train._per_league_report(splits, "target_over25", _SignalModel(), "validated")["colombia"]
    assert league["validation_status"] == "experimental"
    assert "does_not_beat_baselines" in league["validation_reasons"]


def test_global_validation_does_not_cover_unseen_colombia(settings):
    frame = _frame().assign(League="established")
    colombia = frame.iloc[800:].assign(League="colombia")
    report = train._train_single_target(pd.concat([frame, colombia]), "target_over25")
    assert report["validation_status"] == "validated"
    assert report["per_league"]["established"]["validation_status"] == "validated"
    league = report["per_league"]["colombia"]
    assert league["validation_status"] == "experimental"
    assert league["samples"]["train"]["n"] == 0
    assert league["samples"]["calibration"]["n"] == 0
    assert league["test_sample"]["n"] == 200
    assert league["baseline_train_prevalence"] is None
    assert league["test_metrics"] is not None
    json.dumps(report, allow_nan=False)


def test_per_league_untrainable_and_missing_league(settings):
    frame = _frame().assign(League="colombia", target_over25=1)
    report = train._train_single_target(frame, "target_over25")
    league = report["per_league"]["colombia"]
    assert league["validation_status"] == "experimental"
    assert "model_unavailable" in league["validation_reasons"]
    assert league["test_metrics"] is None
    assert league["test_calibration_bins"] == []
    frame["League"] = None
    assert train._train_single_target(frame, "target_over25")["per_league"] == {}


def test_bins_include_boundaries_and_empty_bins():
    probabilities = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bins = train._calibration_bins(pd.Series([0, 0, 0, 1, 1, 1]), probabilities)
    assert [item["n"] for item in bins] == [1, 1, 1, 1, 2]
    assert bins[-1]["mean_probability"] == 0.9
    assert bins[-1]["observed_rate"] == 1


def test_walk_forward_is_train_only_and_never_changes_artifact(settings, monkeypatch):
    frame = _frame()
    report = train._train_single_target(frame, "target_over25")
    research = report["research"]["walk_forward"]
    assert research["used_for_gate"] is False
    assert research["scope"] == "train_only"
    assert len(research["windows"]) == 2
    for window in research["windows"]:
        assert window["status"] == "evaluated"
        samples = window["samples"]
        assert samples["train"]["end"] < samples["calibration"]["start"]
        assert samples["calibration"]["end"] < samples["test"]["start"]
        assert samples["test"]["end"] <= report["base_trained_through"]
    monkeypatch.setattr(train, "_walk_forward_research", lambda *args: {})
    without_research = train._train_single_target(frame, "target_over25")
    assert without_research["model_version"] == report["model_version"]
    assert without_research["validation_status"] == report["validation_status"]


def test_league_identity_changes_data_version_not_model(settings):
    frame = _frame().assign(League="established")
    first = train._train_single_target(frame, "target_over25")
    second = train._train_single_target(frame.assign(League="colombia"), "target_over25")
    assert first["data_version"] != second["data_version"]
    assert first["model_version"] == second["model_version"]
