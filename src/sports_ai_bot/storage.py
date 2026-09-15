"""Local, manual quote imports and immutable prediction snapshots."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from sports_ai_bot.utils.team_names import TEAM_NAME_ALIASES, normalize_team_name


BOOKMAKERS = {
    "betplay": "BetPlay",
    "wplay": "Wplay",
    "rushbet": "Rushbet",
    "codere": "Codere",
    "betano": "Betano",
}
_DOMAINS = {
    "betplay": "betplay.com.co",
    "wplay": "wplay.co",
    "rushbet": "rushbet.co",
    "codere": "codere.com.co",
    "betano": "betano.co",
}
_COLUMNS = (
    "league", "home_team", "away_team", "kickoff", "bookmaker", "market",
    "selection", "line", "odd", "observed_at", "source_url",
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False,
                      separators=(",", ":"))


def _hash(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Dates must include an explicit timezone")
    return result.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="microseconds")


def _prediction(pick: object) -> tuple[str, dict]:
    snapshot = asdict(pick)
    for key in ("quote_id", "model_version"):
        if not isinstance(snapshot.get(key), str) or not snapshot[key].strip():
            raise ValueError(f"Prediction requires a nonempty {key}")
    return _hash([snapshot["quote_id"], snapshot["model_version"],
                  bool(snapshot.get("is_experimental", False))]), snapshot


class Store:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "analysis.sqlite3"
        with self._connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS quotes (
                    quote_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    league TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    kickoff TEXT NOT NULL,
                    bookmaker TEXT NOT NULL,
                    market TEXT NOT NULL,
                    selection TEXT NOT NULL,
                    line REAL,
                    odd REAL NOT NULL,
                    observed_at TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    source TEXT NOT NULL CHECK (source = 'manual'),
                    method TEXT NOT NULL CHECK (method = 'manual')
                );
                CREATE INDEX IF NOT EXISTS quotes_latest ON quotes
                    (event_id, bookmaker, market, selection, line, observed_at);
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    snapshot TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    published_at TEXT
                );
                CREATE TABLE IF NOT EXISTS status (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS external_observations (
                    observation_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    market TEXT NOT NULL,
                    selection TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    snapshot TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS external_observations_lookup ON external_observations
                    (source, event_id, market, selection, observed_at);
            """)

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA busy_timeout = 30000")
            with db:
                yield db
        finally:
            db.close()

    def import_quotes(self, path: Path) -> int:
        """Import a CSV atomically; return the number of new snapshots inserted.

        Exact duplicates are ignored, not replaced. Unknown team names retain
        their spelling with whitespace collapsed; known aliases are canonicalized.
        BTTS uses None for its empty line. Extra CSV fields are not trusted.
        """
        now = datetime.now(timezone.utc)
        inserted = 0
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            headers = reader.fieldnames or []
            if len(headers) != len(set(headers)) or not set(_COLUMNS).issubset(headers):
                raise ValueError("CSV requires unique headers and all required columns")
            with self._connection() as db:
                for number, raw in enumerate(reader, start=2):
                    try:
                        if None in raw or any(value is None for value in raw.values()):
                            raise ValueError("Malformed CSV row")
                        row = {key: raw[key].strip() for key in _COLUMNS}
                        if any(not value for key, value in row.items() if key != "line"):
                            raise ValueError("Required fields must not be empty")
                        row["league"] = row["league"].casefold()
                        aliases = TEAM_NAME_ALIASES.get(row["league"], {})
                        names = {
                            normalize_team_name(name): canonical
                            for name, canonical in aliases.items()
                        }
                        names.update({normalize_team_name(name): name for name in aliases.values()})
                        for key in ("home_team", "away_team"):
                            name = " ".join(row[key].split())
                            row[key] = names.get(normalize_team_name(name), name)
                        if row["home_team"].casefold() == row["away_team"].casefold():
                            raise ValueError("Home and away teams must differ")
                        bookmaker = row["bookmaker"].casefold()
                        if bookmaker not in BOOKMAKERS:
                            raise ValueError("Unsupported bookmaker")
                        row["bookmaker"] = bookmaker
                        url = urlsplit(row["source_url"])
                        host = url.hostname or ""
                        domain = _DOMAINS[bookmaker]
                        if (
                            url.scheme != "https" or url.username is not None
                            or url.password is not None or url.port not in (None, 443)
                            or not re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)*", host)
                            or any(label.startswith("-") or label.endswith("-")
                                   for label in host.split("."))
                            or not (host == domain or host.endswith("." + domain))
                            or any(char.isspace() or ord(char) < 32 for char in row["source_url"])
                            or "\\" in row["source_url"]
                        ):
                            raise ValueError("Source URL must use the bookmaker's local HTTPS domain")
                        if row["market"] == "Over 2.5" and row["selection"] == "Over":
                            if float(row["line"]) != 2.5:
                                raise ValueError("Over 2.5 requires line 2.5")
                            row["line"] = 2.5
                        elif (row["market"] == "BTTS" and row["selection"] == "Si"
                              and row["line"] == ""):
                            row["line"] = None
                        else:
                            raise ValueError("Only Over 2.5 / Over and BTTS / Si are supported")
                        row["odd"] = float(row["odd"])
                        if not math.isfinite(row["odd"]) or row["odd"] <= 1:
                            raise ValueError("Odd must be finite and greater than 1")
                        kickoff = _timestamp(row["kickoff"])
                        observed = _timestamp(row["observed_at"])
                        if observed > now or observed >= kickoff:
                            raise ValueError("Observation must be at or before now and before kickoff")
                        row["kickoff"] = _iso(kickoff)
                        row["observed_at"] = _iso(observed)
                        row["source"] = row["method"] = "manual"
                        row["event_id"] = _hash([
                            row["league"], row["home_team"].casefold(),
                            row["away_team"].casefold(), row["kickoff"],
                        ])
                        row["quote_id"] = _hash(row)
                        columns = ", ".join(row)
                        placeholders = ", ".join("?" for _ in row)
                        result = db.execute(
                            f"INSERT INTO quotes ({columns}) VALUES ({placeholders}) "
                            "ON CONFLICT(quote_id) DO NOTHING", tuple(row.values()),
                        )
                        inserted += result.rowcount
                    except (ValueError, OverflowError) as exc:
                        raise ValueError(f"CSV row {number}: {exc}") from exc
        return inserted

    def quotes(self, max_age_minutes: float = 30) -> list[dict]:
        """Return fresh, pre-match latest quotes, never the best historical price.

        Equal observation times are resolved by last insertion, not by odd.
        """
        if not math.isfinite(max_age_minutes) or max_age_minutes < 0:
            raise ValueError("max_age_minutes must be finite and nonnegative")
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=max_age_minutes)
        with self._connection() as db:
            rows = db.execute("""
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY event_id, bookmaker, market, selection, line
                        ORDER BY observed_at DESC, rowid DESC
                    ) AS position FROM quotes
                ) WHERE position = 1 AND kickoff > ? AND observed_at >= ? AND observed_at <= ?
                ORDER BY kickoff, event_id, bookmaker, market, selection, line
            """, (_iso(now), _iso(cutoff), _iso(now))).fetchall()
        result = []
        for row in rows:
            quote = dict(row)
            del quote["position"]
            quote["quoted_at"] = quote["observed_at"]
            result.append(quote)
        return result

    def record_predictions(self, picks: list) -> None:
        """Persist complete JSON-compatible dataclass snapshots, first write wins."""
        now = _iso(datetime.now(timezone.utc))
        with self._connection() as db:
            for pick in picks:
                prediction_id, snapshot = _prediction(pick)
                db.execute(
                    "INSERT INTO predictions (prediction_id, snapshot, recorded_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(prediction_id) DO NOTHING",
                    (prediction_id, _json(snapshot), now),
                )

    def mark_published(self, picks: list) -> None:
        """Mark previously recorded picks after sending; preserve the first timestamp."""
        now = _iso(datetime.now(timezone.utc))
        with self._connection() as db:
            for pick in picks:
                prediction_id, _ = _prediction(pick)
                result = db.execute(
                    "UPDATE predictions SET published_at = COALESCE(published_at, ?) "
                    "WHERE prediction_id = ?", (now, prediction_id),
                )
                if not result.rowcount:
                    raise ValueError("Cannot publish an unrecorded prediction")

    def predictions(self) -> list[dict]:
        """Return snapshot fields plus prediction_id, recorded_at and published_at.

        The `snapshot` entry also preserves the complete original asdict payload,
        including any fields whose names coincide with storage metadata.
        """
        with self._connection() as db:
            rows = db.execute("SELECT * FROM predictions ORDER BY rowid").fetchall()
        return [
            {**json.loads(row["snapshot"]), "prediction_id": row["prediction_id"],
             "recorded_at": row["recorded_at"], "published_at": row["published_at"],
             "snapshot": json.loads(row["snapshot"])}
            for row in rows
        ]

    def record_external_observations(self, source: str, observations: list) -> int:
        """Store private source snapshots without treating them as validated predictions."""
        if not isinstance(source, str) or not source.strip():
            raise ValueError("External source must be nonempty")
        inserted = 0
        with self._connection() as db:
            for observation in observations:
                snapshot = asdict(observation)
                required = ("event_id", "market", "selection", "observed_at")
                if any(not isinstance(snapshot.get(key), str) or not snapshot[key].strip()
                       for key in required):
                    raise ValueError("External observation is incomplete")
                _timestamp(snapshot["observed_at"])
                observation_id = _hash([
                    source, snapshot["event_id"], snapshot["market"],
                    snapshot["selection"], snapshot["observed_at"],
                ])
                result = db.execute(
                    "INSERT INTO external_observations "
                    "(observation_id, source, event_id, market, selection, observed_at, snapshot) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(observation_id) DO NOTHING",
                    (
                        observation_id, source, snapshot["event_id"], snapshot["market"],
                        snapshot["selection"], snapshot["observed_at"], _json(snapshot),
                    ),
                )
                inserted += result.rowcount
        return inserted

    def external_observations(self, source: str | None = None) -> list[dict]:
        query = "SELECT * FROM external_observations"
        parameters: tuple[str, ...] = ()
        if source is not None:
            query += " WHERE source = ?"
            parameters = (source,)
        query += " ORDER BY rowid"
        with self._connection() as db:
            rows = db.execute(query, parameters).fetchall()
        return [
            {
                **json.loads(row["snapshot"]),
                "observation_id": row["observation_id"],
                "source": row["source"],
                "snapshot": json.loads(row["snapshot"]),
            }
            for row in rows
        ]

    def set_status(self, key: str, value: object) -> None:
        """Set a JSON-compatible status value (for example, key='generation')."""
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Status key must be nonempty")
        with self._connection() as db:
            db.execute(
                "INSERT INTO status (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, _json(value)),
            )

    def get_status(self, key: str, default: object = None) -> object:
        with self._connection() as db:
            row = db.execute("SELECT value FROM status WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])
