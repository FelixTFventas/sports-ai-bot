from sports_ai_bot.predict.pipeline import (
    Pick,
    _best_pick_score,
    _edge,
    _expected_value,
    _implied_probability,
    _rating,
    _stake_units,
)


def test_implied_probability_from_odd() -> None:
    assert round(_implied_probability(2.0), 4) == 0.5


def test_edge_uses_model_minus_implied() -> None:
    assert round(_edge(0.6, 2.0), 4) == 0.1


def test_expected_value_is_positive_when_model_has_value() -> None:
    assert round(_expected_value(0.6, 2.0), 4) == 0.2


def test_stake_units_thresholds() -> None:
    assert _stake_units(0.11, 0.09) == 3
    assert _stake_units(0.07, 0.06) == 2
    assert _stake_units(0.03, 0.03) == 1


def test_rating_thresholds() -> None:
    assert _rating(0.11, 0.09) == "A"
    assert _rating(0.06, 0.05) == "B"
    assert _rating(0.03, 0.03) == "C"


def test_best_pick_score_prioritizes_stake_then_ev() -> None:
    stronger = Pick(
        match_date="2026-04-17",
        home_team="A",
        away_team="B",
        match_label="A vs B",
        league="premier_league",
        market="Over 2.5",
        probability=0.65,
        confidence="Media-Alta",
        model_name="test",
        factors=[],
        odd=2.1,
        implied_probability=0.4762,
        edge=0.12,
        expected_value=0.10,
        stake_units=3,
        rating="A",
    )
    weaker = Pick(
        match_date="2026-04-17",
        home_team="C",
        away_team="D",
        match_label="C vs D",
        league="premier_league",
        market="Over 2.5",
        probability=0.70,
        confidence="Media-Alta",
        model_name="test",
        factors=[],
        odd=1.9,
        implied_probability=0.5263,
        edge=0.05,
        expected_value=0.04,
        stake_units=1,
        rating="C",
    )
    assert _best_pick_score(stronger) > _best_pick_score(weaker)
