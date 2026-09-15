import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from sports_ai_bot.bot.publication import Publication, PublicationBlocked, shared_work


def test_shared_writes_are_transactionally_serialized(tmp_path):
    from threading import Event

    entered = Event()
    release = Event()
    second_started = Event()
    order = []

    def first():
        with shared_work(tmp_path):
            order.append("first-start")
            entered.set()
            assert release.wait(5)
            order.append("first-end")

    def second():
        second_started.set()
        with shared_work(tmp_path):
            order.append("second")

    with ThreadPoolExecutor(2) as pool:
        one = pool.submit(first)
        assert entered.wait(5)
        two = pool.submit(second)
        try:
            assert second_started.wait(5)
            assert order == ["first-start"]
        finally:
            release.set()
        one.result()
        two.result()
    assert order == ["first-start", "first-end", "second"]


def test_process_crash_releases_gate_without_replaying(tmp_path):
    script = (
        "import os, sys; from pathlib import Path; "
        "from sports_ai_bot.bot.publication import Publication; "
        "p=Publication(Path(sys.argv[1]), '1', 'daily', '2026-09-08'); "
        "p.start(); p.prepare('batch'); os._exit(0)"
    )
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, timeout=20)
    p = Publication(tmp_path, "1", "daily", "2026-09-08")
    with pytest.raises(PublicationBlocked, match="diaria"):
        p.start()
    p = Publication(tmp_path, "1", "daily", "2026-09-09")
    p.start()
    p.finish("completed")
    p.close()


def test_other_process_cannot_publish_while_gate_held(tmp_path):
    p = Publication(tmp_path, "1", "daily", "2026-09-08")
    p.start()
    script = (
        "import sys; from pathlib import Path; "
        "from sports_ai_bot.bot.publication import Publication, PublicationBlocked\n"
        "try: Publication(Path(sys.argv[1]), '1', 'manual', '2026-09-08').start()\n"
        "except PublicationBlocked: sys.exit(0)\n"
        "sys.exit(1)"
    )
    try:
        subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, timeout=20)
    finally:
        p.close()


def test_concurrent_claims(tmp_path):
    barrier = Barrier(2)

    def claim():
        p = Publication(tmp_path, "1", "daily", "2026-09-08")
        barrier.wait()
        try:
            p.start()
            p.finish("completed")
            return True
        except PublicationBlocked:
            return False
        finally:
            p.close()

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: claim(), range(2)))
    assert sorted(results) == [False, True]


def test_restart_daily_and_manual_independent(tmp_path):
    def publication(kind="daily", day="2026-09-08", chat="1", now=100):
        return Publication(tmp_path, chat, kind, day, clock=lambda: now)

    first = publication()
    first.start()
    first.prepare("batch")
    first.close()  # Simulate process death: SQLite releases the live transaction.
    with pytest.raises(PublicationBlocked):
        publication().start()
    with sqlite3.connect(tmp_path / "publications.sqlite3") as db:
        assert db.execute("SELECT state FROM publications").fetchone()[0] == "failed-uncertain"
    for p in [publication("manual"), publication(day="2026-09-09"), publication(chat="2")]:
        p.start()
        p.finish("completed")
        p.close()


def test_manual_cooldown_and_uncertain_fingerprint(tmp_path):
    def publication(now):
        return Publication(tmp_path, "1", "manual", "2026-09-08", clock=lambda: now)

    p = publication(100)
    p.start()
    p.prepare("same")
    p.finish("completed")
    p.close()
    with pytest.raises(PublicationBlocked, match="cooldown"):
        publication(159).start()
    p = publication(160)
    p.start()
    p.prepare("same")
    p.close()
    p = publication(220)
    p.start()
    with pytest.raises(PublicationBlocked, match="incierto"):
        p.prepare("same")
    p.finish("failed-before-send")
    p.close()
    p = publication(220)
    p.start()
    p.prepare("different")
    p.finish("completed")
    p.close()


@pytest.mark.parametrize("kind", ["daily", "manual"])
def test_crash_before_prepare_can_retry_immediately(tmp_path, kind):
    script = (
        "import os, sys; from pathlib import Path; "
        "from sports_ai_bot.bot.publication import Publication; "
        "p=Publication(Path(sys.argv[1]), '1', sys.argv[2], '2026-09-08'); "
        "p.start(); os._exit(0)"
    )
    subprocess.run([sys.executable, "-c", script, str(tmp_path), kind], check=True, timeout=20)
    p = Publication(tmp_path, "1", kind, "2026-09-08")
    try:
        p.start()
        with p.connect() as db:
            assert db.execute("SELECT state FROM publications ORDER BY id").fetchall() == [
                ("failed-before-send",),
                ("in-progress",),
            ]
    finally:
        p.close()
