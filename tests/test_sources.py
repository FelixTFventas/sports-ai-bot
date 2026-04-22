import pandas as pd

from sports_ai_bot.collect import fixtures, odds


def test_fetch_upcoming_fixtures_prefers_the_odds(monkeypatch) -> None:
    monkeypatch.setattr(fixtures, "is_configured", lambda: True)
    monkeypatch.setattr(
        fixtures,
        "_fetch_the_odds_api_fixtures",
        lambda days_ahead: [{"Date": "2026-04-18", "League": "premier_league", "HomeTeam": "A", "AwayTeam": "B"}],
    )
    monkeypatch.setattr(
        fixtures,
        "_fetch_espn_fixtures",
        lambda days_ahead: (_ for _ in ()).throw(AssertionError("ESPN should not be called")),
    )

    frame = fixtures.fetch_upcoming_fixtures()

    assert len(frame) == 1
    assert frame.iloc[0]["HomeTeam"] == "A"


def test_fetch_upcoming_fixtures_falls_back_to_espn(monkeypatch) -> None:
    monkeypatch.setattr(fixtures, "is_configured", lambda: True)
    monkeypatch.setattr(fixtures, "_fetch_the_odds_api_fixtures", lambda days_ahead: [])
    monkeypatch.setattr(
        fixtures,
        "_fetch_espn_fixtures",
        lambda days_ahead: [{"Date": "2026-04-18", "League": "premier_league", "HomeTeam": "C", "AwayTeam": "D"}],
    )

    frame = fixtures.fetch_upcoming_fixtures()

    assert len(frame) == 1
    assert frame.iloc[0]["HomeTeam"] == "C"


def test_load_upcoming_market_odds_prefers_the_odds(monkeypatch) -> None:
    source_frame = pd.DataFrame([_odds_row("A", "B", odd_over25=1.9, odd_home_win=1.7)])
    monkeypatch.setattr(odds, "is_configured", lambda: True)
    monkeypatch.setattr(odds, "_fetch_the_odds_api_market_odds", lambda days_ahead=7: source_frame)
    monkeypatch.setattr(
        odds,
        "_fetch_espn_market_odds",
        lambda days_ahead=7: (_ for _ in ()).throw(AssertionError("ESPN should not be called")),
    )

    frame = odds.load_upcoming_market_odds()

    assert frame.equals(source_frame)


def test_load_upcoming_market_odds_falls_back_to_csv(monkeypatch) -> None:
    csv_frame = pd.DataFrame([_odds_row("E", "F", odd_over25=2.0, odd_away_win=2.2)])
    monkeypatch.setattr(odds, "is_configured", lambda: True)
    monkeypatch.setattr(odds, "_fetch_the_odds_api_market_odds", lambda days_ahead=7: pd.DataFrame())
    monkeypatch.setattr(odds, "_fetch_espn_market_odds", lambda days_ahead=7: pd.DataFrame())
    monkeypatch.setattr(odds, "_load_csv_market_odds", lambda: csv_frame)

    frame = odds.load_upcoming_market_odds()

    assert frame.equals(csv_frame)


def _odds_row(home_team: str, away_team: str, **updates: float) -> dict[str, object]:
    row = {
        "match_day": pd.Timestamp("2026-04-18").date(),
        "League": "premier_league",
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "odd_home_win": None,
        "odd_draw": None,
        "odd_away_win": None,
        "odd_over15": None,
        "odd_over25": None,
        "odd_under45": None,
        "odd_btts": None,
    }
    row.update(updates)
    return row
