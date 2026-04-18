from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sports_ai_bot.predict import pipeline
from sports_ai_bot.predict.pipeline import _confidence_label


def test_confidence_label_thresholds() -> None:
    assert _confidence_label(0.80) == "Alta"
    assert _confidence_label(0.66) == "Media-Alta"
    assert _confidence_label(0.50) == "Media"


def test_build_top_picks_filters_started_matches(monkeypatch, tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    fixtures = pd.DataFrame(
        [
            _fixture_row("2026-04-18 08:00:00+00:00", "Past FC", "Old Town"),
            _fixture_row("2026-04-18 12:00:00+00:00", "Future FC", "Next Town"),
        ]
    )

    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    monkeypatch.setattr(pipeline, "build_fixture_features", lambda: fixtures)
    monkeypatch.setattr(pipeline, "_attach_market_odds", lambda frame: frame)
    monkeypatch.setattr(
        pipeline,
        "joblib",
        SimpleNamespace(load=_fake_model_loader),
    )
    monkeypatch.setattr(
        pipeline,
        "_now_utc",
        lambda: pd.Timestamp("2026-04-18 09:00:00+00:00"),
    )

    picks = pipeline.build_top_picks(limit=5, threshold=0.65)

    assert [pick.match_label for pick in picks] == ["Future FC vs Next Town"]


def test_build_top_picks_uses_default_markets_and_48_hour_window(
    monkeypatch, tmp_path: Path
) -> None:
    settings = _make_settings(tmp_path)
    fixtures = pd.DataFrame(
        [
            _fixture_row("2026-04-18 12:00:00+00:00", "Near FC", "Soon Town"),
            _fixture_row("2026-04-20 10:00:01+00:00", "Far FC", "Later Town"),
        ]
    )

    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    monkeypatch.setattr(pipeline, "build_fixture_features", lambda: fixtures)
    monkeypatch.setattr(pipeline, "_attach_market_odds", lambda frame: frame)
    monkeypatch.setattr(pipeline, "joblib", SimpleNamespace(load=_fake_model_loader))
    monkeypatch.setattr(
        pipeline,
        "_now_utc",
        lambda: pd.Timestamp("2026-04-18 10:00:00+00:00"),
    )

    picks = pipeline.build_top_picks(limit=10, threshold=0.65)

    assert [(pick.match_label, pick.market) for pick in picks] == [
        ("Near FC vs Soon Town", "Over 2.5")
    ]


def test_build_market_picks_returns_over15_when_requested(monkeypatch, tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    fixtures = pd.DataFrame([_fixture_row("2026-04-18 12:00:00+00:00", "Near FC", "Soon Town")])

    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    monkeypatch.setattr(pipeline, "build_fixture_features", lambda: fixtures)
    monkeypatch.setattr(pipeline, "_attach_market_odds", lambda frame: frame)
    monkeypatch.setattr(pipeline, "joblib", SimpleNamespace(load=_fake_model_loader))
    monkeypatch.setattr(
        pipeline,
        "_now_utc",
        lambda: pd.Timestamp("2026-04-18 10:00:00+00:00"),
    )

    picks = pipeline.build_market_picks("Over 1.5", limit=10, threshold=0.65)

    assert [(pick.match_label, pick.market) for pick in picks] == [
        ("Near FC vs Soon Town", "Over 1.5")
    ]


def test_build_top_picks_refreshes_fixtures_instead_of_using_cached_file(
    monkeypatch, tmp_path: Path
) -> None:
    settings = _make_settings(tmp_path)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_fixture_row("2026-04-18 12:00:00+00:00", "Cached FC", "Cached Town")]).to_csv(
        settings.processed_dir / "fixture_features.csv", index=False
    )

    refreshed = pd.DataFrame([_fixture_row("2026-04-18 13:00:00+00:00", "Fresh FC", "Fresh Town")])

    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    monkeypatch.setattr(pipeline, "build_fixture_features", lambda: refreshed)
    monkeypatch.setattr(pipeline, "_attach_market_odds", lambda frame: frame)
    monkeypatch.setattr(
        pipeline,
        "joblib",
        SimpleNamespace(load=_fake_model_loader),
    )
    monkeypatch.setattr(
        pipeline,
        "_now_utc",
        lambda: pd.Timestamp("2026-04-18 09:00:00+00:00"),
    )

    picks = pipeline.build_top_picks(limit=5, threshold=0.65)

    assert [pick.match_label for pick in picks] == ["Fresh FC vs Fresh Town"]


class _FakeModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [np.full(len(frame), 1 - self.probability), np.full(len(frame), self.probability)]
        )


def _fake_model_loader(path: Path) -> _FakeModel:
    if "over15" in path.name:
        return _FakeModel(0.85)
    if "btts" in path.name:
        return _FakeModel(0.2)
    return _FakeModel(0.8)


def _make_settings(tmp_path: Path) -> SimpleNamespace:
    models_dir = tmp_path / "models"
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "target_over15.joblib").touch()
    (models_dir / "target_over25.joblib").touch()
    (models_dir / "target_btts.joblib").touch()
    return SimpleNamespace(
        models_dir=models_dir, processed_dir=processed_dir, reports_dir=reports_dir
    )


def _fixture_row(match_date: str, home_team: str, away_team: str) -> dict[str, object]:
    row = {
        "Date": match_date,
        "League": "premier_league",
        "HomeTeam": home_team,
        "AwayTeam": away_team,
    }
    row.update({column: 1.0 for column in pipeline.FEATURE_COLUMNS})
    return row
