from sports_ai_bot.external.forebet import ForebetError, format_top_picks_message, parse_top_picks


SAMPLE_MARKDOWN = """
# Mejores predicciones de fútbol del algoritmo

Principales predicciones

[IFK Berga Oskarshamns AIK 04/24/2026 5:00 PM](https://www.forebet.com/es/football/matches/ifk-berga-oskarshamns-aik-2445858)

11 9 80

2 1-3

1 - 3

[TPS Turku IF Gnistan 04/24/2026 3:00 PM](https://www.forebet.com/es/football/matches/tps-turku-if-gnistan-2448570)

74 14 12

1 2-0

2 - 0

[Tiverton Town Farnham Town 04/25/2026 2:00 PM](https://www.forebet.com/es/football/matches/tiverton-town-farnham-town-2457940)

14 10 76

2 0-3

0 - 3

[Touring Zarautz 04/25/2026 2:30 PM](https://www.forebet.com/es/football/matches/touring-zarautz-2375688)

74 14 12

1 3-0

3 - 0

[Adamstown FC Lake Macquarie FC 04/25/2026 5:00 AM](https://www.forebet.com/es/football/matches/adamstown-fc-lake-macquarie-fc-2426286)

13 15 72

2 1-2

1 - 2

[GW Micheldorf Vortuna Bad Leonfelden 04/24/2026 5:30 PM](https://www.forebet.com/es/football/matches/gw-micheldorf-vortuna-bad-leonfelden-2333619)

20 10 71

2 1-3

1 - 3

Ver más
"""


def test_parse_top_picks_orders_by_highest_probability() -> None:
    picks = parse_top_picks(SAMPLE_MARKDOWN, limit=5)

    assert [pick.match_label for pick in picks] == [
        "IFK Berga Oskarshamns AIK",
        "Tiverton Town Farnham Town",
        "TPS Turku IF Gnistan",
        "Touring Zarautz",
        "Adamstown FC Lake Macquarie FC",
    ]
    assert picks[0].prediction == "2"
    assert picks[0].probability == 80
    assert picks[0].predicted_score == "1-3"


def test_format_top_picks_message_includes_reference_note() -> None:
    picks = parse_top_picks(SAMPLE_MARKDOWN, limit=2)

    message = format_top_picks_message(picks)

    assert message.startswith("Top 5 Forebet del dia:")
    assert "1. IFK Berga Oskarshamns AIK | Pick 2 | Prob 80%" in message
    assert message.endswith("Fuente externa de referencia: Forebet.")


def test_parse_top_picks_raises_when_section_is_missing() -> None:
    try:
        parse_top_picks("sin datos")
    except ForebetError as exc:
        assert "Forebet" in str(exc)
    else:
        raise AssertionError("Expected parse_top_picks to raise ForebetError")
