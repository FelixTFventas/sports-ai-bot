"""Durable publication journal; a separate SQLite transaction gates live workers.

The gate rolls back on process death. Journal entries survive, so recovery never
assumes that an unacknowledged Telegram request was not delivered.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing, contextmanager
from pathlib import Path


class PublicationBlocked(RuntimeError):
    pass


class PublicationBusy(PublicationBlocked):
    """Only live gate contention is safe to wait for automatically."""


@contextmanager
def shared_work(data_dir: Path):
    """Serialize synchronous pipeline writes across bot threads and publishing CLI."""
    data_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(data_dir / "work-lock.sqlite3", timeout=120)
    try:
        connection.execute("BEGIN IMMEDIATE")
        yield
    finally:
        connection.close()


class Publication:
    def __init__(self, data_dir: Path, chat: str, kind: str, day: str, clock=time.time):
        if kind not in {"daily", "manual"}:
            raise ValueError("Unknown publication kind")
        self.data_dir = data_dir
        self.chat, self.kind, self.day = str(chat), kind, day
        self.clock = clock
        self.gate = None
        self.row_id = None

    def start(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.gate = sqlite3.connect(
            self.data_dir / "publication-lock.sqlite3", timeout=0, check_same_thread=False
        )
        try:
            self.gate.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            self.close()
            if exc.sqlite_errorcode not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                raise
            raise PublicationBusy("Publicacion en curso; no se ha enviado otro lote.") from exc
        try:
            with self.connect() as db:
                db.execute("""CREATE TABLE IF NOT EXISTS publications (
                    id INTEGER PRIMARY KEY, chat TEXT NOT NULL, kind TEXT NOT NULL,
                    day TEXT NOT NULL, started REAL NOT NULL, finished REAL,
                    state TEXT NOT NULL, fingerprint TEXT)""")
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "UPDATE publications SET state=CASE WHEN fingerprint IS NULL "
                    "THEN 'failed-before-send' ELSE 'failed-uncertain' END "
                    "WHERE state IN ('in-progress', 'failed-uncertain')"
                )
                rows = db.execute(
                    "SELECT day, started, finished, state FROM publications "
                    "WHERE chat=? AND kind=? AND state != 'failed-before-send'",
                    (self.chat, self.kind),
                ).fetchall()
                now = self.clock()
                blocked = None
                if self.kind == "daily" and any(row[0] == self.day for row in rows):
                    blocked = "Publicacion diaria ya registrada (completada o incierta)."
                elif self.kind == "manual" and any(
                    now - (row[2] if row[2] is not None else row[1]) < 60 for row in rows
                ):
                    blocked = "Publicacion manual bloqueada: cooldown de 60 segundos."
                if not blocked:
                    self.row_id = db.execute(
                        "INSERT INTO publications(chat,kind,day,started,state) "
                        "VALUES(?,?,?,?,'in-progress')",
                        (self.chat, self.kind, self.day, now),
                    ).lastrowid
            if blocked:
                raise PublicationBlocked(blocked)
        except BaseException:
            self.close()
            raise

    @contextmanager
    def connect(self):
        # Explicit closing matters on Windows as well as for crash tests.
        with closing(sqlite3.connect(self.data_dir / "publications.sqlite3")) as db:
            with db:
                yield db

    def prepare(self, fingerprint: str):
        with self.connect() as db:
            uncertain = db.execute(
                "SELECT 1 FROM publications WHERE chat=? AND kind=? "
                "AND state='failed-uncertain' AND fingerprint=?",
                (self.chat, self.kind, fingerprint),
            ).fetchone()
            if uncertain:
                raise PublicationBlocked("Lote incierto previo: requiere revision, no se reenvia.")
            db.execute(
                "UPDATE publications SET fingerprint=? WHERE id=?", (fingerprint, self.row_id)
            )

    def finish(self, state: str):
        with self.connect() as db:
            db.execute(
                "UPDATE publications SET state=?, finished=? WHERE id=?",
                (state, self.clock(), self.row_id),
            )

    def close(self):
        if self.gate is not None:
            self.gate.close()
            self.gate = None
