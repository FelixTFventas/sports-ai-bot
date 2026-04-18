from sports_ai_bot.explain.messages import build_prediction_message
from sports_ai_bot.predict.pipeline import Pick


def test_build_prediction_message_shows_premium_and_standard() -> None:
    picks = [
        Pick(
            match_date="2026-04-18",
            home_team="A",
            away_team="B",
            match_label="A vs B",
            league="premier_league",
            market="Over 2.5",
            probability=0.66,
            confidence="Media-Alta",
            model_name="test",
            factors=[],
        ),
        Pick(
            match_date="2026-04-18",
            home_team="C",
            away_team="D",
            match_label="C vs D",
            league="premier_league",
            market="BTTS",
            probability=0.61,
            confidence="Media",
            model_name="test",
            factors=[],
        ),
    ]

    message = build_prediction_message(picks)

    assert "A vs B | Over 2.5 | 66% | Premium | Media-Alta" in message
    assert "C vs D | Ambos marcan | 61% | Standard | Media" in message
