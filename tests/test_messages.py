from sports_ai_bot.explain.messages import (
    build_best_message,
    build_forebet_value_message,
    build_prediction_message,
    build_value_message,
)
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

    assert message.startswith("🎯 Picks del dia")
    assert "1. 🟢 A vs B" in message
    assert "📌 Pick: Over 2.5" in message
    assert "📊 Probabilidad: 66.0%" in message
    assert "📈 Edge: +10.4%" in message
    assert "🔥 EV: +18.8%" in message
    assert "⭐ Nivel: Premium | Confianza: Media-Alta" in message
    assert "2. 🔵 C vs D" in message
    assert "📌 Pick: 1X2 Local" in message
    assert "⭐ Nivel: Standard | Confianza: Media" in message


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


def test_build_value_message_uses_multiline_card_format() -> None:
    message = build_value_message(
        [
            Pick(
                match_date="2026-04-18",
                home_team="A",
                away_team="B",
                match_label="A vs B",
                league="premier_league",
                market="1X2",
                selection="Local",
                probability=0.635,
                confidence="Media",
                model_name="test",
                factors=[],
                odd=2.1,
                edge=0.159,
                expected_value=0.333,
                stake_units=3,
                rating="A",
            )
        ]
    )

    assert message.startswith("💎 Value picks del dia")
    assert "1. 🟢 A vs B" in message
    assert "📌 Pick: 1X2 Local" in message
    assert "⭐ Rating: A | Stake: 3u | Nivel: Standard" in message


def test_build_best_message_uses_multiline_card_format() -> None:
    message = build_best_message(
        [
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
                odd=1.87,
                edge=0.032,
                expected_value=0.059,
                stake_units=1,
                rating="C",
            )
        ]
    )

    assert message.startswith("🏆 Best picks del dia")
    assert "1. 🟡 A vs B" in message
    assert "📌 Pick: Over 2.5" in message
    assert "⭐ Rating: C | Stake: 1u | Nivel: Premium" in message
