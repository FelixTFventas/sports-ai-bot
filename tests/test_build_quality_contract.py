from types import SimpleNamespace

import pandas as pd

from sports_ai_bot.features import build
from sports_ai_bot.train.train_models import FEATURE_COLUMNS


def _history(n=12):
    return pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n), "League": "synthetic",
        "HomeTeam": "Alpha", "AwayTeam": "Beta", "FTHG": 2, "FTAG": 1,
    })


def test_iso_and_day_first_dates_have_same_meaning():
    dates = build._parse_source_dates(pd.Series(["2026-09-02", "02/09/2026", "bad"]))
    assert dates.iloc[0] == dates.iloc[1] == pd.Timestamp("2026-09-02")
    assert pd.isna(dates.iloc[2])


def test_missing_corners_preserves_goal_targets_and_feature_contract():
    dataset = build._attach_team_history_features(_history())
    assert len(dataset) == 12
    assert dataset.target_over25.eq(1).all()
    assert dataset.target_corners_over95.isna().all()
    assert dataset.home_corners_for_avg_5.isna().all()
    row = dataset.iloc[-1]
    assert row.home_goals_for_avg_5 == 2
    assert row.quality_home_corners_observed_5 == 0
    assert row.quality_missing_features == row[FEATURE_COLUMNS].isna().sum()
    assert row.quality_history_ready
    assert row.history_as_of < row.Date
    assert row.data_schema_version == build.DATA_SCHEMA_VERSION


def test_dedup_before_history_and_same_day_exclusion():
    history = _history()
    duplicate = history.iloc[[0]].assign(HC=0, AC=0)
    dataset = build._attach_team_history_features(pd.concat([history, duplicate]))
    assert len(dataset) == len(history)
    assert dataset.iloc[0].target_corners_over95 == 0
    assert dataset.iloc[-1].quality_home_history_matches == 11
    same_day = history.iloc[[0]].assign(AwayTeam="Gamma")
    dataset = build._attach_team_history_features(pd.concat([history, same_day]))
    assert dataset.iloc[:2].quality_home_history_matches.eq(0).all()


def test_unknown_corners_not_zero_and_leagues_isolated():
    history = _history(6).assign(HC=0, AC=0)
    history.loc[3, "HC"] = -1
    second_league = _history(1).assign(League="another", Date=pd.Timestamp("2020-02-01"))
    dataset = build._attach_team_history_features(pd.concat([history, second_league]))
    assert pd.isna(dataset.iloc[3].target_corners_over95)
    assert pd.isna(dataset.iloc[5].home_corners_for_avg_5)
    assert dataset.iloc[5].quality_home_corners_observed_5 == 4
    assert dataset.iloc[-1].quality_home_history_matches == 0


def test_fixture_quality_persisted_and_future_results_excluded(monkeypatch, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    processed = tmp_path / "processed"
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    history = _history().assign(Date=pd.date_range(today - pd.Timedelta(days=12), periods=12))
    future_result = history.iloc[[0]].assign(Date=today + pd.Timedelta(days=2), FTHG=99)
    pd.concat([history, history.iloc[[0]], future_result]).to_csv(raw / "synthetic_2020.csv", index=False)
    fixtures = pd.DataFrame({"Date": [today + pd.Timedelta(days=1)], "League": ["synthetic"],
                             "HomeTeam": ["Alpha"], "AwayTeam": ["Beta"]})
    monkeypatch.setattr(build, "get_settings", lambda: SimpleNamespace(raw_dir=raw, processed_dir=processed))
    monkeypatch.setattr(build, "LEAGUES", {"S": "synthetic"})
    monkeypatch.setattr(build, "fetch_upcoming_fixtures", lambda **kwargs: fixtures)
    result = build.build_fixture_features()
    assert len(result) == 1
    assert result.iloc[0].quality_home_history_matches == 12
    assert result.iloc[0].home_goals_for_avg_5 == 2
    assert result.iloc[0].history_as_of < today
    persisted = pd.read_csv(processed / "fixture_features.csv")
    assert persisted.iloc[0].quality_home_corners_observed_5 == 0
    assert persisted.iloc[0].quality_missing_features > 0
