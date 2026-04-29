from datetime import datetime

from sports_ai_bot.external import forebet
from sports_ai_bot.external.forebet import (
    ForebetError,
    ForebetPick,
    _american_to_decimal,
    format_top_picks_message,
    parse_match_links,
    parse_match_value_picks,
    parse_list_value_picks,
    parse_top_picks,
)
from sports_ai_bot.predict.pipeline import Pick


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

MATCH_MARKDOWN = """
# Atlético Madrid vs Arsenal Predicciones de Fútbol y Estadística - 29/04/2026

# [Atlético Madrid](http://www.forebet.com/es/teams/atl%C3%A9tico-madrid) - [Arsenal](http://www.forebet.com/es/teams/arsenal)

04/29/2026 7:00 PM

Equipo local

 Equipo visitante

Probabilidad %

1 X 2

[Atlético Madrid Arsenal 04/29/2026 7:00 PM](http://www.forebet.com/es/football/matches/atl%C3%A9tico-madrid-arsenal-2457943)

33 35 32

X 2-2

+225

+210+225+135

Equipo local

 Equipo visitante

Probabilidad %

Menos/Más

2.5

[Atlético Madrid Arsenal 04/29/2026 7:00 PM](http://www.forebet.com/es/football/matches/atl%C3%A9tico-madrid-arsenal-2457943)

43 57

Más 2.83

+115

-149+115

Equipo local

 Equipo visitante

Probabilidad de marcador en el medio tiempo %

1 X 2

[Atlético Madrid Arsenal 04/29/2026 7:00 PM](http://www.forebet.com/es/football/matches/atl%C3%A9tico-madrid-arsenal-2457943)

31 47 22

X

+105

Equipo local

 Equipo visitante

Probabilidad MT/FT %

Pred.

[Atlético Madrid Arsenal 04/29/2026 7:00 PM](http://www.forebet.com/es/football/matches/atl%C3%A9tico-madrid-arsenal-2457943)

16%

X

X 2-2

+350

Equipo local

 Equipo visitante

Probabilidad %

No Sí

[Atlético Madrid Arsenal 04/29/2026 7:00 PM](http://www.forebet.com/es/football/matches/atl%C3%A9tico-madrid-arsenal-2457943)

36 64

Sí 2-2

-118

-118-111

Equipo local

 Equipo visitante

Probabilidad %

1X/2X/12

[Atlético Madrid Arsenal 04/29/2026 7:00 PM](http://www.forebet.com/es/football/matches/atl%C3%A9tico-madrid-arsenal-2457943)

68%

X1 2-2

-167
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


def test_parse_top_picks_accepts_24_hour_times() -> None:
    markdown = """
Principales predicciones

[Barito Putera Aceh United 03/05/2026 10:30](http://www.forebet.com/es/football/matches/barito-putera-aceh-united-2458488)

82 15 4

1 3-0

3 - 0

### Partido destacado
"""

    picks = parse_top_picks(markdown, limit=1)

    assert picks[0].match_label == "Barito Putera Aceh United"
    assert picks[0].match_time == "10:30"


def test_parse_top_picks_raises_when_section_is_missing() -> None:
    try:
        parse_top_picks("sin datos")
    except ForebetError as exc:
        assert "Forebet" in str(exc)
    else:
        raise AssertionError("Expected parse_top_picks to raise ForebetError")


def test_parse_match_links_filters_next_48h_and_dedupes() -> None:
    markdown = """
[Past FC Old Town 29/04/2026 09:00](http://www.forebet.com/es/football/matches/past-old-1)
[Today FC Rival 29/04/2026 20:00](http://www.forebet.com/es/football/matches/today-rival-1)
[Today FC Rival 29/04/2026 20:00](http://www.forebet.com/es/football/matches/today-rival-1)
[US Format Team Opponent 04/29/2026 7:00 PM](http://www.forebet.com/es/football/matches/us-format-1)
[Paren Team Opponent 29/04/2026 21:00](http://www.forebet.com/es/football/matches/paren-team-(x)-1)
[Far FC Future 03/05/2026 10:30](http://www.forebet.com/es/football/matches/far-future-1)
"""

    links = parse_match_links(markdown, now=datetime(2026, 4, 29, 10, 0), horizon_hours=48)

    assert [(link.match_label, link.match_datetime.isoformat()) for link in links] == [
        ("US Format Team Opponent", "2026-04-29T19:00:00"),
        ("Today FC Rival", "2026-04-29T20:00:00"),
        ("Paren Team Opponent", "2026-04-29T21:00:00"),
    ]
    assert links[-1].source_url.endswith("paren-team-(x)-1")


def test_american_to_decimal_converts_common_prices() -> None:
    assert _american_to_decimal("+210") == 3.1
    assert _american_to_decimal("-167") == 1.5988
    assert _american_to_decimal("0") is None


def test_parse_match_value_picks_ranks_forebet_market_blend() -> None:
    picks = parse_match_value_picks(MATCH_MARKDOWN, source_url="https://www.forebet.com/match", limit=3)

    assert [(pick.market, pick.selection) for pick in picks] == [
        ("Doble oportunidad", "Atlético Madrid o Empate"),
        ("BTTS", "Si"),
        ("Over 2.5", None),
    ]
    assert picks[0].match_label == "Atlético Madrid vs Arsenal"
    assert picks[0].match_date == "2026-04-29"
    assert picks[0].odd == 1.5988
    assert round(picks[0].probability, 3) == 0.658
    assert picks[0].model_name == "forebet_market_blend"


def test_parse_match_value_picks_filters_by_min_odd() -> None:
    picks = parse_match_value_picks(MATCH_MARKDOWN, limit=20, min_odd=2.0)

    assert all((pick.odd or 0.0) >= 2.0 for pick in picks)
    assert ("Doble oportunidad", "Atlético Madrid o Empate") not in {
        (pick.market, pick.selection) for pick in picks
    }


def test_fetch_top_value_picks_keeps_best_pick_per_match(monkeypatch) -> None:
    monkeypatch.setattr(
        forebet,
        "fetch_top_picks",
        lambda limit=5, timeout=30.0: [
            ForebetPick("A B", "04/29/2026", "7:00 PM", "1", 70, "2-0", "url-a"),
            ForebetPick("C D", "04/29/2026", "8:00 PM", "2", 68, "1-2", "url-b"),
        ],
    )

    def fetch_match_value_picks(url, limit=5, min_odd=1.50, timeout=30.0):
        if url == "url-a":
            return [
                _pick("A vs B", "Over 2.5", None, 0.55),
                _pick("A vs B", "BTTS", "Si", 0.70),
            ]
        return [_pick("C vs D", "1X2", "Visitante", 0.65)]

    monkeypatch.setattr(forebet, "fetch_match_value_picks", fetch_match_value_picks)

    picks = forebet.fetch_top_value_picks(limit_matches=2, limit=5)

    assert [(pick.match_label, pick.market, pick.selection) for pick in picks] == [
        ("A vs B", "BTTS", "Si"),
        ("C vs D", "1X2", "Visitante"),
    ]


def test_parse_list_value_picks_extracts_1x2_rows() -> None:
    markdown = """
[Home FC Away FC 29/04/2026 20:00](http://www.forebet.com/es/football/matches/home-away-1)

40 30 30

1 2-0

+150

+150+230+300 no no no

[Past FC Old FC 29/04/2026 09:00](http://www.forebet.com/es/football/matches/past-old-1)

80 10 10

1 3-0

+110

+110+300+800 no no no
"""

    picks = parse_list_value_picks(markdown, now=datetime(2026, 4, 29, 10, 0), min_odd=1.50)

    assert {(pick.match_label, pick.market, pick.selection) for pick in picks} == {
        ("Home FC Away FC", "1X2", "Local"),
        ("Home FC Away FC", "1X2", "Empate"),
        ("Home FC Away FC", "1X2", "Visitante"),
    }
    assert {pick.match_date for pick in picks} == {"2026-04-29"}
    assert all(pick.odd is not None and pick.odd >= 1.50 for pick in picks)


def _pick(
    match_label: str,
    market: str,
    selection: str | None,
    probability: float,
    edge: float = 0.05,
) -> Pick:
    return Pick(
        match_date="2026-04-29",
        home_team=match_label.split(" vs ")[0],
        away_team=match_label.split(" vs ")[1],
        match_label=match_label,
        league="forebet",
        market=market,
        selection=selection,
        probability=probability,
        confidence="Media",
        model_name="forebet_market_blend",
        factors=[],
        odd=1.8,
        edge=edge,
        expected_value=0.08,
    )
