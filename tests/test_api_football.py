from datetime import datetime, timezone

from sports_ai_bot.collect.odds import _extract_the_odds_totals_price
from sports_ai_bot.collect.the_odds_api import _filter_events_in_window, sport_key_for_league


def test_sport_key_for_supported_league() -> None:
    assert sport_key_for_league("premier_league") == "soccer_epl"
    assert sport_key_for_league("mls") == "soccer_usa_mls"


def test_extract_the_odds_totals_price_for_over25() -> None:
    bookmakers = [
        {
            "markets": [
                {
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "point": 2.5, "price": 1.83},
                        {"name": "Under", "point": 2.5, "price": 1.95},
                    ],
                }
            ]
        }
    ]
    assert _extract_the_odds_totals_price(bookmakers, point=2.5) == 1.83


def test_filter_events_in_window_keeps_only_upcoming_range(monkeypatch) -> None:
    now = datetime(2026, 4, 18, 10, 0, tzinfo=timezone.utc)
    payload = [
        {"commence_time": "2026-04-18T12:00:00Z"},
        {"commence_time": "2026-04-27T12:00:00Z"},
        {"commence_time": "2026-04-18T09:00:00Z"},
    ]

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now if tz is None else now.astimezone(tz)

    monkeypatch.setattr("sports_ai_bot.collect.the_odds_api.datetime", _FakeDatetime)

    filtered = _filter_events_in_window(payload, days_ahead=7)

    assert filtered == [{"commence_time": "2026-04-18T12:00:00Z"}]
