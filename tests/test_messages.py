from sports_ai_bot.explain.messages import build_forebet_value_message, build_prediction_message
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
            odd=1.8,
            edge=0.104,
            expected_value=0.188,
        ),
        Pick(
            match_date="2026-04-18",
            home_team="C",
            away_team="D",
            match_label="C vs D",
            league="premier_league",
            market="1X2",
            selection="Local",
            probability=0.61,
            confidence="Media",
            model_name="test",
            factors=[],
            odd=1.7,
            edge=0.022,
            expected_value=0.037,
        ),
    ]

    message = build_prediction_message(picks)

    assert "A vs B | Over 2.5 | Prob 66% | Cuota 1.80 | Edge 10.4% | EV 18.8% | Premium | Media-Alta" in message
    assert "C vs D | 1X2 Local | Prob 61% | Cuota 1.70 | Edge 2.2% | EV 3.7% | Standard | Media" in message


def test_build_forebet_value_message_includes_selection() -> None:
    message = build_forebet_value_message(
        [
            Pick(
                match_date="2026-04-29",
                home_team="A",
                away_team="B",
                match_label="A vs B",
                league="forebet",
                market="Doble oportunidad",
                selection="A o Empate",
                probability=0.658,
                confidence="Media-Alta",
                model_name="forebet_market_blend",
                factors=[],
                odd=1.6,
                edge=0.033,
                expected_value=0.053,
            )
        ]
    )

    assert message.startswith("🎯 Forebet Value Picks")
    assert "1. 🟡 A vs B" in message
    assert "📌 Pick: Doble oportunidad A o Empate" in message
    assert "📊 Probabilidad: 65.8%" in message
    assert "📈 Edge: +3.3%" in message
    assert "🔥 EV: +5.3%" in message
    assert "60% Forebet + 40% mercado" in message
