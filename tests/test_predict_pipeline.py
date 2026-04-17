from sports_ai_bot.predict.pipeline import _confidence_label


def test_confidence_label_thresholds() -> None:
    assert _confidence_label(0.80) == "Alta"
    assert _confidence_label(0.66) == "Media-Alta"
    assert _confidence_label(0.50) == "Media"
