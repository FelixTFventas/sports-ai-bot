from sports_ai_bot.utils.config import Settings


def test_the_odds_api_bookmakers_query_combines_and_deduplicates() -> None:
    settings = Settings(
        THE_ODDS_API_BOOKMAKER="bet365,pinnacle",
        THE_ODDS_API_EXTRA_BOOKMAKERS="pinnacle,mybookieag",
    )

    assert settings.the_odds_api_bookmakers_list() == ["bet365", "pinnacle", "mybookieag"]
    assert settings.the_odds_api_bookmakers_query() == "bet365,pinnacle,mybookieag"


def test_corners_pick_market_label_uses_configured_line() -> None:
    settings = Settings(CORNERS_PICK_SELECTION="Under", CORNERS_PICK_POINT=10.5)

    assert settings.corners_pick_market_label() == "Under 10.5 Corners"


def test_corners_price_range_uses_configured_values() -> None:
    settings = Settings(CORNERS_PICK_MIN_PRICE=1.7, CORNERS_PICK_MAX_PRICE=2.05)

    assert settings.corners_pick_min_price == 1.7
    assert settings.corners_pick_max_price == 2.05
