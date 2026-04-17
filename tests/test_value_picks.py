from sports_ai_bot.predict.pipeline import _edge, _expected_value, _implied_probability


def test_implied_probability_from_odd() -> None:
    assert round(_implied_probability(2.0), 4) == 0.5


def test_edge_uses_model_minus_implied() -> None:
    assert round(_edge(0.6, 2.0), 4) == 0.1


def test_expected_value_is_positive_when_model_has_value() -> None:
    assert round(_expected_value(0.6, 2.0), 4) == 0.2
