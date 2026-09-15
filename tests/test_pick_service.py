from dataclasses import asdict, make_dataclass
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
import socket
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from sports_ai_bot.predict import pipeline, service
from sports_ai_bot.storage import Store


class SyntheticModel:
    feature_names_in_ = np.array(service.FEATURE_COLUMNS)
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        assert list(frame.columns) == service.FEATURE_COLUMNS
        assert frame.filter(like="corners").isna().all().all()
        return np.array([[0.3, 0.7]])


@pytest.fixture
def local(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network/fixture access forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(service.build, "fetch_upcoming_fixtures", forbidden)
    monkeypatch.setattr(service.build, "build_fixture_features", forbidden)
    settings = SimpleNamespace(
        data_dir=tmp_path, raw_dir=tmp_path / "raw", models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports", predictions_dir=tmp_path / "predictions",
        quote_max_age_minutes=30, history_max_age_days=45, picks_min_edge=0.03,
        picks_min_ev=0.03,
    )
    for path in (settings.raw_dir, settings.models_dir, settings.reports_dir):
        path.mkdir()
    monkeypatch.setattr(service, "get_settings", lambda: settings)
    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    # The Pick extension belongs to the caller's parallel change, not this task.
    fields = [(name, str | None, None) for name in (
        "event_id", "quote_id", "kickoff", "quoted_at", "source_url", "source",
        "model_version", "generated_at",
    ) if name not in pipeline.Pick.__dataclass_fields__]
    monkeypatch.setattr(service, "Pick", make_dataclass("ServicePick", fields, bases=(pipeline.Pick,)))
    now = datetime.now(timezone.utc)
    kickoff = now + timedelta(days=1)
    monkeypatch.setattr(service, "_now", lambda: now)
    history = pd.DataFrame([
        {"Date": (now - timedelta(days=days)).date().isoformat(), "HomeTeam": "Arsenal",
         "AwayTeam": "Chelsea", "FTHG": 2, "FTAG": 1}
        for days in range(2, 12)
    ])
    history.to_csv(settings.raw_dir / "premier_league_2627.csv", index=False)
    summary = {}
    for target in service.MARKETS.values():
        path = settings.models_dir / f"{target}.joblib"
        joblib.dump(SyntheticModel(), path)
        summary[target] = {
            "report_schema_version": 2, "data_schema_version": "features-v2",
            "feature_columns": service.FEATURE_COLUMNS.copy(), "artifact_available": True,
            "validation_status": "validated", "selected_model": "synthetic",
            "model_version": hashlib.sha256(path.read_bytes()).hexdigest(),
            "per_league": {"premier_league": {
                "validation_status": "validated", "samples": {"train": {"n": 200}, "test": {"n": 100}},
            }},
        }
    report_path = settings.reports_dir / "training_summary.json"
    report_path.write_text(json.dumps(summary), encoding="utf-8")
    store = Store(tmp_path)

    def quote(odd=2.0, market="Over 2.5", bookmaker="betplay", age=2, **overrides):
        row = {
            "league": "premier_league", "home_team": "Arsenal", "away_team": "Chelsea",
            "kickoff": kickoff.isoformat(), "bookmaker": bookmaker, "market": market,
            "selection": "Over" if market == "Over 2.5" else "Si",
            "line": "2.5" if market == "Over 2.5" else "", "odd": odd,
            "observed_at": (now - timedelta(minutes=age)).isoformat(),
            "source_url": f"https://{bookmaker}.{'com.co' if bookmaker == 'betplay' else 'co'}/sports",
            **overrides,
        }
        path = tmp_path / "quotes.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        store.import_quotes(path)

    return SimpleNamespace(settings=settings, store=store, quote=quote, summary=summary,
                           report_path=report_path, now=now)


def test_local_generation_persists_and_formats(local):
    local.quote()
    picks = service.generate_picks()
    assert len(picks) == 1
    pick = picks[0]
    assert pick.probability == 0.7
    assert pick.edge == pytest.approx(0.2)
    assert pick.expected_value == pytest.approx(0.4)
    assert pick.stake_units is None and pick.rating is None
    assert pick.model_version in pick.model_name
    assert local.store.predictions()[0]["snapshot"] == asdict(pick)
    assert local.store.get_status("picks") == [asdict(pick)]
    assert service.cached_picks() == picks
    csv_path, = local.settings.predictions_dir.glob("*.csv")
    assert pd.read_csv(csv_path).iloc[0].model_name == pick.model_name
    text = service.format_picks(picks)
    for word in ("Probabilidad estimada", "BetPlay", "Cuota observada manual", "Bogota",
                 "EV teorico", "Advertencia", "https://betplay"):
        assert word in text
    assert "Premium" not in text and "stake" not in text.lower()
    assert "no en vivo" in service.format_quotes()
    assert local.store.get_status("generation") == {
        "code": "ready", "observation": False, "generated_at": local.now.isoformat(),
        "quotes": 1, "eligible_quotes": 1, "data_ready": 1, "evaluated": 1, "picks": 1,
    }


def test_latest_not_highest_historical_and_one_per_event(local):
    local.quote(odd=5, age=10)
    local.quote(odd=1.1, age=1)
    local.quote(odd=2.1, bookmaker="wplay")
    local.quote(odd=2.3, market="BTTS")
    picks = service.generate_picks()
    assert len(picks) == 1
    assert picks[0].market == "BTTS" and picks[0].odd == 2.3


@pytest.mark.parametrize("field,value", [
    ("report_schema_version", 1), ("data_schema_version", "features-v1"),
    ("feature_columns", list(reversed(service.FEATURE_COLUMNS))),
    ("artifact_available", False), ("model_version", "wrong"),
    ("validation_status", "rejected"),
])
def test_incompatible_report_never_observation(local, field, value):
    local.quote()
    local.summary["target_over25"][field] = value
    local.report_path.write_text(json.dumps(local.summary), encoding="utf-8")
    assert service.generate_picks(observation=True) == []
    assert local.store.get_status("generation")["code"] == "no_model"


@pytest.mark.parametrize("change", ["global", "missing_league", "experimental_league"])
def test_observation_is_explicit_and_separate(local, change):
    local.quote()
    report = local.summary["target_over25"]
    if change == "global":
        report["validation_status"] = "experimental"
    elif change == "missing_league":
        del report["per_league"]
    else:
        report["per_league"]["premier_league"]["validation_status"] = "experimental"
    local.report_path.write_text(json.dumps(local.summary), encoding="utf-8")
    assert service.generate_picks() == []
    picks = service.generate_picks(observation=True)
    assert len(picks) == 1 and picks[0].is_experimental
    assert local.store.get_status("observation_picks") == [asdict(picks[0])]
    assert service.cached_picks() == []
    assert "OBSERVACION EXPERIMENTAL" in service.format_picks(picks)


@pytest.mark.parametrize("reason", ["quote_age", "kickoff", "hash", "version", "league", "new_quote"])
def test_cache_revalidates(local, monkeypatch, reason):
    local.quote()
    assert service.generate_picks()
    if reason in ("quote_age", "kickoff"):
        delta = timedelta(minutes=31) if reason == "quote_age" else timedelta(days=2)
        monkeypatch.setattr(service, "_now", lambda: local.now + delta)
    elif reason == "hash":
        (local.settings.models_dir / "target_over25.joblib").write_bytes(b"replaced")
    elif reason == "version":
        local.summary["target_over25"]["model_version"] = "new"
        local.report_path.write_text(json.dumps(local.summary), encoding="utf-8")
    elif reason == "league":
        local.summary["target_over25"]["per_league"] = {}
        local.report_path.write_text(json.dumps(local.summary), encoding="utf-8")
    else:
        local.quote(odd=1.1, age=1)
    assert service.cached_picks() == []
    assert local.store.get_status("generation")["code"] == "cache_invalid"


@pytest.mark.parametrize("mutation", ["unready", "away_old", "home_old", "zero", "nan", "schema"])
def test_feature_quality_gates(local, monkeypatch, mutation):
    local.quote()
    original = service.build._build_feature_row

    def altered(*args, **kwargs):
        row = original(*args, **kwargs)
        key, value = {
            "unready": ("quality_away_venue_matches", 4),
            "away_old": ("away_rest_days", 46), "home_old": ("home_rest_days", 46),
            "zero": ("away_rest_days", 0), "nan": ("home_points_avg_5", np.nan),
            "schema": ("data_schema_version", "features-v1"),
        }[mutation]
        row[key] = value
        return row

    monkeypatch.setattr(service.build, "_build_feature_row", altered)
    assert service.generate_picks(observation=True) == []
    assert local.store.get_status("generation")["code"] == "no_data"


def test_empty_states_and_limit(local):
    assert "Todavia" in service.status_message()
    assert service.generate_picks() == []
    assert local.store.get_status("generation")["code"] == "no_quotes"
    assert "Sin cuotas" in service.format_picks([])
    local.quote(odd=1.1)
    assert service.generate_picks() == []
    assert local.store.get_status("generation")["code"] == "no_value"
    for limit in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            service.generate_picks(limit=limit)


def test_unknown_team_and_no_local_history(local):
    local.quote(home_team="Unknown team")
    assert service.generate_picks() == []
    assert local.store.get_status("generation")["code"] == "no_data"


def test_history_excludes_today_and_future_results(local):
    local.quote()
    path = local.settings.raw_dir / "premier_league_2627.csv"
    frame = pd.read_csv(path).iloc[:3]
    extra = pd.DataFrame([
        {"Date": (local.now + timedelta(days=days)).date().isoformat(), "HomeTeam": "Arsenal",
         "AwayTeam": "Chelsea", "FTHG": 9, "FTAG": 9}
        for days in range(5)
    ])
    pd.concat([frame, extra]).to_csv(path, index=False)
    assert service.generate_picks() == []
    assert local.store.get_status("generation")["code"] == "no_data"


def test_missing_history(local):
    local.quote()
    (local.settings.raw_dir / "premier_league_2627.csv").unlink()
    assert service.generate_picks() == []
    assert local.store.get_status("generation")["code"] == "no_data"


@pytest.mark.parametrize("age", [31, 120])
def test_stale_quotes_are_not_generated(local, age):
    local.quote(age=age)
    assert service.generate_picks() == []
    assert local.store.get_status("generation")["code"] == "no_quotes"


def test_observation_cannot_replace_validated_cache(local):
    local.quote()
    normal = service.generate_picks()
    observation = service.generate_picks(observation=True)
    assert observation[0].is_experimental
    assert service.cached_picks() == normal


def test_inference_failure_is_model_not_no_value(local, monkeypatch):
    local.quote()
    monkeypatch.setattr(SyntheticModel, "predict_proba", lambda *args: np.array([[np.nan, np.nan]]))
    assert service.generate_picks() == []
    assert local.store.get_status("generation")["code"] == "no_model"


def test_artifact_feature_order_is_checked(local, monkeypatch):
    local.quote()
    monkeypatch.setattr(SyntheticModel, "feature_names_in_", np.array(["wrong"]))
    assert service.generate_picks(observation=True) == []
    assert local.store.get_status("generation")["code"] == "no_model"


def test_multiple_events_limit_and_ev_order(local):
    local.quote(odd=2)
    local.quote(odd=2.5, kickoff=(local.now + timedelta(days=2)).isoformat())
    picks = service.generate_picks(limit=1)
    assert len(picks) == 1 and picks[0].odd == 2.5


@pytest.mark.parametrize("split", ["train", "test"])
def test_league_requires_train_and_test_representation(local, split):
    local.quote()
    report = local.summary["target_over25"]["per_league"]["premier_league"]
    report["samples"][split]["n"] = 0
    local.report_path.write_text(json.dumps(local.summary), encoding="utf-8")
    assert service.generate_picks() == []
    assert local.store.get_status("generation")["code"] == "no_model"
    assert service.generate_picks(observation=True)[0].is_experimental
