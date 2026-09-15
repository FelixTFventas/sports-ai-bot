from sports_ai_bot.utils.config import Settings
import pytest


def test_the_odds_api_bookmakers_query_combines_and_deduplicates() -> None:
    settings = Settings(
        _env_file=None,
        THE_ODDS_API_BOOKMAKER="bet365,pinnacle",
        THE_ODDS_API_EXTRA_BOOKMAKERS="pinnacle,mybookieag",
    )

    assert settings.the_odds_api_bookmakers_list() == ["bet365", "pinnacle", "mybookieag"]
    assert settings.the_odds_api_bookmakers_query() == "bet365,pinnacle,mybookieag"


def test_corners_pick_market_label_uses_configured_line() -> None:
    settings = Settings(_env_file=None, CORNERS_PICK_SELECTION="Under", CORNERS_PICK_POINT=10.5)

    assert settings.corners_pick_market_label() == "Under 10.5 Corners"


def test_corners_price_range_uses_configured_values() -> None:
    settings = Settings(_env_file=None, CORNERS_PICK_MIN_PRICE=1.7, CORNERS_PICK_MAX_PRICE=2.05)

    assert settings.corners_pick_min_price == 1.7
    assert settings.corners_pick_max_price == 2.05


@pytest.mark.parametrize("value, expected", [("", set()), ("  ", set()), ("1, 42,001", {1, 42})])
def test_admin_csv(value, expected):
    assert Settings(_env_file=None, TELEGRAM_ADMIN_IDS=value).telegram_admin_ids_set() == expected


@pytest.mark.parametrize("value", ["0", "-1", "1,", ",1", "abc", "1.5", "1,,2", "+1"])
def test_invalid_admin_csv(value):
    with pytest.raises(ValueError, match="IDs positivos"):
        Settings(_env_file=None, TELEGRAM_ADMIN_IDS=value)


def test_admin_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ADMIN_IDS", "99,100")
    assert Settings(_env_file=None).telegram_admin_ids_set() == {99, 100}


def test_forebet_defaults_and_bounds():
    settings = Settings(_env_file=None)
    assert settings.forebet_cache_minutes == 45
    assert settings.forebet_min_probability == 0.60
    assert settings.forebet_limit == 10
    with pytest.raises(ValueError):
        Settings(_env_file=None, FOREBET_MIN_PROBABILITY=1.1)
