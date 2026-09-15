import csv
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone

import pytest

import sports_ai_bot.storage as storage
from sports_ai_bot.storage import BOOKMAKERS, Store


NOW = datetime(2026, 9, 14, 12, tzinfo=timezone.utc)
COLUMNS = (
    "league", "home_team", "away_team", "kickoff", "bookmaker", "market",
    "selection", "line", "odd", "observed_at", "source_url",
)


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz)

    monkeypatch.setattr(storage, "datetime", Clock)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


def quote(**changes):
    return {
        "league": "premier_league", "home_team": "Manchester City", "away_team": "Arsenal",
        "kickoff": (NOW + timedelta(hours=2)).isoformat(), "bookmaker": "betplay",
        "market": "Over 2.5", "selection": "Over", "line": "2.5", "odd": "2.1",
        "observed_at": (NOW - timedelta(minutes=5)).isoformat(),
        "source_url": "https://apuestas.betplay.com.co/event/123", **changes,
    }


def write_csv(tmp_path, rows, columns=COLUMNS):
    path = tmp_path / "quotes.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_import_normalizes_and_deduplicates(store, tmp_path):
    row = quote(kickoff="2026-09-14T09:00:00-05:00",
                observed_at="2026-09-14T06:55:00-05:00")
    path = write_csv(tmp_path, [row, row])
    assert store.import_quotes(path) == 1
    assert store.import_quotes(path) == 0
    result, = Store(store.data_dir).quotes()
    assert result["home_team"] == "Man City"
    assert result["kickoff"] == "2026-09-14T14:00:00.000000+00:00"
    assert result["observed_at"] == result["quoted_at"] == "2026-09-14T11:55:00.000000+00:00"
    assert result["source"] == result["method"] == "manual"
    assert result["odd"] == 2.1
    assert result["line"] == 2.5
    assert len(result["quote_id"]) == len(result["event_id"]) == 64
    assert set(COLUMNS).issubset(result)
    assert store.import_quotes(write_csv(tmp_path, [quote(home_team="Man City")])) == 0


@pytest.mark.parametrize("bookmaker,domain", [
    ("betplay", "betplay.com.co"), ("wplay", "wplay.co"), ("rushbet", "rushbet.co"),
    ("codere", "codere.com.co"), ("betano", "betano.co"),
])
@pytest.mark.parametrize("prefix", ["", "sports.live."])
def test_authorized_bookmakers(store, tmp_path, bookmaker, domain, prefix):
    assert set(BOOKMAKERS) == {"betplay", "wplay", "rushbet", "codere", "betano"}
    assert BOOKMAKERS[bookmaker]
    row = quote(bookmaker=bookmaker, source_url=f"https://{prefix}{domain}/event",
                market="BTTS", selection="Si", line="")
    assert store.import_quotes(write_csv(tmp_path, [row])) == 1
    assert store.quotes()[0]["line"] is None


@pytest.mark.parametrize("changes", [
    {"odd": value} for value in ("nan", "inf", "-inf", "1", "0", "-2", "oops")
] + [
    {"source_url": value} for value in (
        "http://betplay.com.co", "https://betplay.com.co.evil.test", "https://evilbetplay.com.co",
        "https://betplay.com", "https://wplay.co", "https://betplay.com.co@evil.test",
        "https://evil.test@betplay.com.co", "https://betplay.com.co:bad",
        "https://betplay.com.co:8080", "https://-bad.betplay.com.co",
        "https://betplay.com.co\\@evil.test", "https://betplay.com.co\n.evil.test",
    )
] + [
    {"kickoff": "2026-09-14T14:00:00"}, {"observed_at": "2026-09-14T11:55:00"},
    {"observed_at": "2026-09-14T12:00:01Z"}, {"observed_at": "not-a-date"},
    {"kickoff": "2026-09-14T11:55:00Z"}, {"kickoff": "2026-09-14T11:54:59Z"},
    {"bookmaker": "other"}, {"market": "1X2"}, {"selection": "Under"},
    {"line": "2.50x"}, {"line": "3.5"}, {"line": "nan"}, {"line": ""},
    {"market": "BTTS", "selection": "Si", "line": "2.5"},
    {"market": "BTTS", "selection": "No", "line": ""},
    {"away_team": " man city "},
] + [{key: " \t "} for key in COLUMNS if key != "line"])
def test_invalid_import_is_atomic(store, tmp_path, changes):
    existing = quote(odd="1.9")
    store.import_quotes(write_csv(tmp_path, [existing]))
    before = store.quotes()
    with pytest.raises(ValueError, match="CSV row 3"):
        store.import_quotes(write_csv(tmp_path, [quote(), quote(**changes)]))
    assert store.quotes() == before
    with closing(sqlite3.connect(store.db_path)) as db:
        assert db.execute("SELECT COUNT(*) FROM quotes").fetchone()[0] == 1


@pytest.mark.parametrize("missing", COLUMNS)
def test_missing_headers(store, tmp_path, missing):
    with pytest.raises(ValueError, match="headers"):
        store.import_quotes(write_csv(tmp_path, [], [key for key in COLUMNS if key != missing]))


def test_duplicate_headers(store, tmp_path):
    with pytest.raises(ValueError, match="headers"):
        store.import_quotes(write_csv(tmp_path, [], [*COLUMNS, "odd"]))


def test_only_latest_not_best_preserves_history(store, tmp_path):
    old = quote(odd="4", observed_at="2026-09-14T11:40:00Z")
    latest = quote(odd="1.8")
    other_market = quote(market="BTTS", selection="Si", line="")
    other_bookmaker = quote(bookmaker="wplay", source_url="https://wplay.co")
    # Importing older observations later must not make them the latest.
    assert store.import_quotes(write_csv(tmp_path, [latest, old, other_market, other_bookmaker])) == 4
    results = store.quotes()
    assert len(results) == 3
    assert len({row["event_id"] for row in results}) == 1
    assert next(row for row in results if row["bookmaker"] == "betplay"
                and row["market"] == "Over 2.5")["odd"] == 1.8
    same_time = quote(odd="1.7")
    assert store.import_quotes(write_csv(tmp_path, [same_time, latest])) == 1
    assert next(row for row in store.quotes() if row["bookmaker"] == "betplay"
                and row["market"] == "Over 2.5")["odd"] == 1.7
    with closing(sqlite3.connect(store.db_path)) as db:
        assert db.execute("SELECT COUNT(*) FROM quotes").fetchone()[0] == 5


def test_age_and_kickoff_boundaries(store, tmp_path):
    rows = [
        quote(observed_at="2026-09-14T11:30:00Z"),
        quote(away_team="Chelsea", observed_at="2026-09-14T11:29:59.999999Z"),
        quote(away_team="Liverpool", kickoff="2026-09-14T12:00:00Z"),
        quote(away_team="Everton", kickoff="2026-09-14T11:59:00Z"),
        quote(away_team="Fulham", observed_at="2026-09-14T12:00:00Z"),
    ]
    assert store.import_quotes(write_csv(tmp_path, rows)) == 5
    assert {row["away_team"] for row in store.quotes()} == {"Arsenal", "Fulham"}
    assert {row["away_team"] for row in store.quotes(31)} == {"Arsenal", "Chelsea", "Fulham"}
    assert [row["away_team"] for row in store.quotes(0)] == ["Fulham"]


@pytest.mark.parametrize("age", [-1, float("inf"), float("nan")])
def test_invalid_age(store, age):
    with pytest.raises(ValueError):
        store.quotes(age)


def test_manual_cannot_be_overridden(store, tmp_path):
    row = quote(source="scraper", method="automatic")
    store.import_quotes(write_csv(tmp_path, [row], [*COLUMNS, "source", "method"]))
    assert store.quotes()[0]["source"] == store.quotes()[0]["method"] == "manual"


def test_unknown_teams_and_canonical_aliases(store, tmp_path):
    rows = [quote(league="custom", home_team="  Local   Club  "),
            quote(home_team="Brighton & Hove Albion")]
    store.import_quotes(write_csv(tmp_path, rows))
    assert {row["home_team"] for row in store.quotes()} == {"Local Club", "Brighton"}


@dataclass
class Details:
    factors: list[str] = field(default_factory=lambda: ["form", "goals"])


@dataclass
class Pick:
    quote_id: str = "quote-1"
    model_version: str = "model-v1"
    probability: float = 0.65
    details: Details = field(default_factory=Details)


@dataclass
class ExternalObservation:
    event_id: str = "event-1"
    market: str = "BTTS"
    selection: str = "Si"
    observed_at: str = "2026-09-14T12:00:00+00:00"
    probability: float = 0.65


def test_prediction_snapshots_are_complete_immutable_and_persistent(store):
    pick = Pick()
    original = asdict(pick)
    store.record_predictions([pick, pick])
    initial, = store.predictions()
    assert initial["snapshot"] == original
    assert initial["published_at"] is None
    pick.details.factors.append("changed")
    pick.probability = 0.9
    store.record_predictions([pick])
    assert store.predictions() == [initial]
    store.record_predictions([replace(pick, model_version="v2"), replace(pick, quote_id="quote-2")])
    records = Store(store.data_dir).predictions()
    assert len({row["prediction_id"] for row in records}) == 3
    assert records[0]["snapshot"] == original
    records[0]["snapshot"]["details"]["factors"].append("external mutation")
    assert store.predictions()[0]["snapshot"] == original


def test_publication_is_separate_idempotent_and_atomic(store):
    pick = Pick()
    store.record_predictions([pick])
    before = store.predictions()[0]
    with pytest.raises(ValueError, match="unrecorded"):
        store.mark_published([pick, replace(pick, quote_id="missing")])
    assert store.predictions()[0] == before
    store.mark_published([pick])
    after = store.predictions()[0]
    assert after["published_at"] == "2026-09-14T12:00:00.000000+00:00"
    assert after["snapshot"] == before["snapshot"]
    store.mark_published([pick])
    assert store.predictions()[0] == after


@pytest.mark.parametrize("bad", [Pick(quote_id=""), Pick(model_version=" "),
                                  Pick(probability=float("nan")), {"quote_id": "q"}])
def test_bad_prediction_rolls_back_entire_batch(store, bad):
    with pytest.raises((ValueError, TypeError)):
        store.record_predictions([Pick(), bad])
    assert store.predictions() == []


def test_status_and_empty_operations(store):
    assert store.get_status("generation") is None
    assert store.get_status("missing", "idle") == "idle"
    store.set_status("generation", {"state": "running", "count": 0})
    assert Store(store.data_dir).get_status("generation") == {"state": "running", "count": 0}
    store.set_status("generation", "complete")
    assert store.get_status("generation") == "complete"
    with pytest.raises(ValueError):
        store.set_status(" ", "bad")
    store.record_predictions([])
    store.mark_published([])
    assert store.predictions() == store.quotes() == []


def test_external_observations_are_separate_and_deduplicated(store):
    observation = ExternalObservation()
    assert store.record_external_observations("forebet", [observation, observation]) == 1
    stored, = Store(store.data_dir).external_observations("forebet")
    assert stored["snapshot"] == asdict(observation)
    assert stored["source"] == "forebet"
    assert store.predictions() == []


def test_external_observation_batch_is_atomic(store):
    with pytest.raises(ValueError, match="incomplete"):
        store.record_external_observations(
            "forebet", [ExternalObservation(), replace(ExternalObservation(), selection="")]
        )
    assert store.external_observations() == []


def test_connections_close_and_busy_timeout(store, monkeypatch):
    connections = []
    connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        db = connect(*args, **kwargs)
        connections.append(db)
        return db

    monkeypatch.setattr(storage.sqlite3, "connect", tracking_connect)
    with store._connection() as db:
        assert db.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
    with pytest.raises(RuntimeError):
        with store._connection():
            raise RuntimeError("rollback")
    store.quotes()
    store.predictions()
    for db in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            db.execute("SELECT 1")
