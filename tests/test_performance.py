import pandas as pd

from sports_ai_bot.evaluate.performance import _determine_outcome, _probability_bucket, _summarize_frame


def test_determine_outcome_for_over25() -> None:
    assert _determine_outcome("Over 2.5", 2, 1) == "won"
    assert _determine_outcome("Over 2.5", 1, 1) == "lost"


def test_determine_outcome_for_over15() -> None:
    assert _determine_outcome("Over 1.5", 1, 1) == "won"
    assert _determine_outcome("Over 1.5", 1, 0) == "lost"


def test_determine_outcome_for_under45() -> None:
    assert _determine_outcome("Under 4.5", 2, 1) == "won"
    assert _determine_outcome("Under 4.5", 3, 2) == "lost"


def test_determine_outcome_for_btts() -> None:
    assert _determine_outcome("BTTS", 1, 1) == "won"
    assert _determine_outcome("BTTS", 2, 0) == "lost"


def test_probability_bucket_ranges() -> None:
    assert _probability_bucket(0.72) == "0.70+"
    assert _probability_bucket(0.67) == "0.65-0.69"
    assert _probability_bucket(0.61) == "0.60-0.64"
    assert _probability_bucket(0.55) == "<0.60"


def test_summarize_frame_includes_profit_and_roi() -> None:
    frame = pd.DataFrame(
        [
            {"status": "settled", "outcome": "won", "odd": 2.0},
            {"status": "settled", "outcome": "lost", "odd": 1.8},
            {"status": "pending", "outcome": "pending", "odd": None},
        ]
    )

    summary = _summarize_frame(frame)

    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["total_profit"] == 0.0
    assert summary["roi"] == 0.0
    assert summary["yield"] == 0.0
