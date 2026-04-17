from sports_ai_bot.evaluate.performance import _determine_outcome, _probability_bucket


def test_determine_outcome_for_over25() -> None:
    assert _determine_outcome("Over 2.5", 2, 1) == "won"
    assert _determine_outcome("Over 2.5", 1, 1) == "lost"


def test_determine_outcome_for_btts() -> None:
    assert _determine_outcome("BTTS", 1, 1) == "won"
    assert _determine_outcome("BTTS", 2, 0) == "lost"


def test_probability_bucket_ranges() -> None:
    assert _probability_bucket(0.72) == "0.70+"
    assert _probability_bucket(0.67) == "0.65-0.69"
    assert _probability_bucket(0.61) == "0.60-0.64"
    assert _probability_bucket(0.55) == "<0.60"
