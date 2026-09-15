import json

import pandas as pd
import pytest

from sports_ai_bot.collect import historical, local
from sports_ai_bot.evaluate import performance
from sports_ai_bot.features import build
from sports_ai_bot.storage import Store
from sports_ai_bot.utils.config import Settings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, DATA_DIR=tmp_path / "data")
    for module in (local, historical, performance, build):
        monkeypatch.setattr(module, "get_settings", lambda: settings)
    return settings


def source(tmp_path, **changes):
    path = tmp_path / "results.csv"
    pd.DataFrame([{
        "Date": "2025-09-12T22:00:00-05:00", "HomeTeam": "Equipo A",
        "AwayTeam": "Equipo B", "FTHG": 2, "FTAG": 1, "HC": "", "AC": "",
        **changes,
    }]).to_csv(path, index=False)
    return path


def test_import_colombia_and_settle_preserves_trace(settings, tmp_path):
    path = source(tmp_path)
    store = Store(settings.data_dir)
    store.set_status("picks", [{"old": True}])
    assert local.import_history(path, "liga_colombia", "https://example.org/results", "Test permission") == 1
    assert store.get_status("picks") == []
    imported, = settings.raw_dir.glob("*.csv")
    data = pd.read_csv(imported)
    assert data.iloc[0].Date == "2025-09-13T03:00:00+00:00"
    assert pd.isna(data.iloc[0].HC)
    manifest, = (settings.data_dir / "imports").glob("*.json")
    assert json.loads(manifest.read_text())["rows"] == 1
    assert local.import_history(path, "liga_colombia", "https://example.org/results", "Test permission") == 1
    assert len(list(settings.raw_dir.glob("*.csv"))) == 1
    settings.predictions_dir.mkdir()
    picks_path = settings.predictions_dir / "picks_test.csv"
    pd.DataFrame([{
        "match_date": "2025-09-12", "prediction_date": "2025-09-12",
        "kickoff": "2025-09-13T03:00:00+00:00", "home_team": "Equipo A",
        "away_team": "Equipo B", "league": "liga_colombia", "market": "Over 2.5",
        "selection": "Over", "line": 2.5, "odd": 1.9, "probability": 0.65,
        "is_experimental": False, "model_version": "v2", "quote_id": "q1",
        "source": "manual", "bookmaker": "betplay", "source_url": "https://betplay.com.co/event",
        "custom_future_field": "preserved",
    }]).to_csv(picks_path, index=False)
    report = performance.build_performance_report()
    assert report["summary_scope"] == "validated"
    assert report["summary"]["wins"] == 1
    assert report["summary"]["total_profit"] == pytest.approx(0.9)
    after = pd.read_csv(picks_path).iloc[0]
    assert after.match_date == "2025-09-12"
    assert after.quote_id == "q1"
    assert after.bookmaker == "betplay"
    assert after.custom_future_field == "preserved"
    assert performance.build_performance_report() == report


@pytest.mark.parametrize("changes", [
    {"Date": "2025-01-01"}, {"Date": "2099-01-01T00:00:00Z"},
    {"FTHG": -1}, {"FTAG": 1.5}, {"FTHG": ""}, {"HC": "unknown"},
    {"HomeTeam": ""}, {"HomeTeam": "Equipo B"}, {"AC": float("inf")},
])
def test_invalid_history_never_writes(settings, tmp_path, changes):
    with pytest.raises(ValueError):
        local.import_history(source(tmp_path, **changes), "liga_colombia", "https://example.org", "Permission")
    assert not settings.raw_dir.exists()


def test_requires_provenance(settings, tmp_path):
    path = source(tmp_path)
    for league, url, note in [
        ("unknown", "https://example.org", "yes"),
        ("liga_colombia", "http://example.org", "yes"),
        ("liga_colombia", "https://secret@example.org", "yes"),
        ("liga_colombia", "https://example.org", ""),
    ]:
        with pytest.raises(ValueError):
            local.import_history(path, league, url, note)


def test_download_disabled_without_permission(settings, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No network without permission")
    monkeypatch.setattr(historical.httpx, "Client", forbidden)
    with pytest.raises(ValueError, match="autorizacion"):
        historical.download_historical_data()
