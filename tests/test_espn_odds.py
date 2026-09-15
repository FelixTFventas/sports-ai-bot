import pandas as pd
import pytest

from sports_ai_bot.collect import odds


@pytest.mark.parametrize("snapshot", ["close", "open"])
@pytest.mark.parametrize("line", [2.5, "2.5", "2.50"])
@pytest.mark.parametrize("price, expected", [("+120", 2.2), (120, 2.2), ("-125", 1.8), (-125, 1.8)])
def test_over25_requires_matching_snapshot(snapshot, line, price, expected) -> None:
    payload = [{"total": {"over": {snapshot: {"line": line, "odds": price}}}}]

    assert odds._extract_over25_decimal(payload) == expected


@pytest.mark.parametrize("snapshot", ["close", "open"])
@pytest.mark.parametrize(
    "line",
    [None, "", "bad", "o2.5", "2,5", 1.5, "3.5", 2, True, [], {}, "nan", "inf"],
)
def test_over25_rejects_other_or_malformed_lines(snapshot, line) -> None:
    payload = [{"total": {"over": {snapshot: {"line": line, "odds": "-125"}}}}]

    assert odds._extract_over25_decimal(payload) is None


@pytest.mark.parametrize(
    "payload",
    [
        None, {}, "bad", 12, [], [None, [], "bad"], [{}],
        [{"total": None}], [{"total": []}], [{"total": "bad"}],
        [{"total": {"over": None}}], [{"total": {"over": []}}],
        [{"total": {"over": "bad"}}],
        [{"total": {"over": {"close": None, "open": []}}}],
        [{"total": {"over": {"close": "bad", "open": 2.5}}}],
    ],
)
def test_over25_handles_malformed_structures(payload) -> None:
    assert odds._extract_over25_decimal(payload) is None


@pytest.mark.parametrize("price", [None, "", "bad", 0, True, [], {}, "nan", "inf"])
def test_over25_rejects_invalid_prices(price) -> None:
    payload = [{"total": {"over": {"close": {"line": 2.5, "odds": price}}}}]

    assert odds._extract_over25_decimal(payload) is None


@pytest.mark.parametrize(
    "close, opening, expected",
    [
        ({"line": 2.5, "odds": 120}, {"line": 2.5, "odds": -125}, 2.2),
        ({"line": 3.5, "odds": 120}, {"line": 2.5, "odds": -125}, 1.8),
        ({"line": 2.5, "odds": 120}, {"line": 3.5, "odds": -125}, 2.2),
        ({"line": 2.5}, {"line": 3.5, "odds": -125}, None),
        ({"line": 3.5, "odds": 120}, {"line": 2.5}, None),
        ({"odds": 120}, {"line": 2.5}, None),
        ({"line": 2.5}, {"odds": -125}, None),
        ({"odds": 120}, {"odds": -125}, None),
        ({"line": "bad", "odds": 120}, {"line": 2.5, "odds": -125}, 1.8),
        ({"line": 2.5, "odds": "bad"}, {"line": 2.5, "odds": -125}, 1.8),
        (None, {"line": 2.5, "odds": -125}, 1.8),
    ],
)
def test_over25_never_combines_snapshots(close, opening, expected) -> None:
    payload = [{"total": {"over": {"close": close, "open": opening}}}]

    assert odds._extract_over25_decimal(payload) == expected


def test_over25_does_not_infer_line_from_other_fields() -> None:
    payload = [{
        "overUnder": 2.5,
        "total": {
            "line": 2.5,
            "under": {"close": {"line": 2.5}},
            "over": {"line": 2.5, "close": {"odds": 120}},
        },
    }]

    assert odds._extract_over25_decimal(payload) is None


def test_over25_continues_to_next_valid_provider() -> None:
    payload = [
        None,
        {"total": None},
        {"total": {"over": {"close": {"line": 3.5, "odds": 120}}}},
        {"total": {"over": {"close": {"line": 2.5}}}},
        {"total": {"over": {"close": {"odds": 120}}}},
        {"total": {"over": {"open": {"line": "2.5", "odds": -125}}}},
    ]

    assert odds._extract_over25_decimal(payload) == 1.8


@pytest.mark.parametrize(
    "over, expected",
    [
        ({"close": {"line": "2.5", "odds": 120}}, 2.2),
        ({"open": {"line": 2.5, "odds": -125}}, 1.8),
        ({"close": {"line": 3.5, "odds": 120}}, None),
        ({"close": {"odds": 120}}, None),
        ({"close": {"line": 2.5}, "open": {"line": 3.5, "odds": 120}}, None),
        (None, None),
    ],
)
def test_extract_event_odds_only_emits_verified_over25(over, expected) -> None:
    payload = {"events": [{"competitions": [{
        "date": "2026-04-18T15:00:00Z",
        "status": {"type": {"completed": False}},
        "competitors": [
            {"homeAway": "home", "team": {"displayName": "A"}},
            {"homeAway": "away", "team": {"displayName": "B"}},
        ],
        "odds": [{"total": {"over": over}}],
    }]}]}

    rows = odds._extract_event_odds(payload, "premier_league")

    if expected is None:
        assert rows == []
    else:
        assert len(rows) == 1
        assert rows[0]["odd_over25"] == expected
        assert rows[0]["match_day"] == pd.Timestamp("2026-04-18").date()
        assert rows[0]["HomeTeam"] == "A"
        assert rows[0]["AwayTeam"] == "B"
        assert rows[0]["League"] == "premier_league"
        assert all(rows[0][key] is None for key in odds.ODDS_COLUMNS if key != "odd_over25")
