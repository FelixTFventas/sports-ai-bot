from dataclasses import asdict
from datetime import datetime, timezone

import pytest

from sports_ai_bot.external import forebet
from sports_ai_bot.external.forebet import (
    ForebetPrediction,
    ForebetResult,
    format_forebet_message,
    get_forebet_predictions,
    parse_predictions,
)
from sports_ai_bot.storage import Store


SAMPLE_HTML = """
<div class="rcnt tr_0">
  <div class="shortagDiv"><img onclick="getstag(this,123,'Colombia','Primera A','x','co')"></div>
  <a class="tnmscn" href="/en/football/matches/america-cali-123">
    <span class="homeTeam"><span itemprop="name">America de Cali</span></span>
    <span class="awayTeam"><span itemprop="name">Deportivo Cali</span></span>
    <span class="date_bah">15/09/2026 10:00</span>
  </a>
  <div class="fprc"><span>35</span><span>65</span></div>
  <div class="predict"><span class="forepr"><span>Over</span></span></div>
  <div class="ex_sc tabonly">2 - 1</div>
  <div class="avg_sc">3.10</div>
  <div class="bigOnly prmod"><span class="lscrsp">1.90</span></div>
</div>
<div class="rcnt tr_1">
  <a class="tnmscn" href="/en/football/matches/old-game-999">
    <span class="homeTeam"><span itemprop="name">Old</span></span>
    <span class="awayTeam"><span itemprop="name">Game</span></span>
    <span class="date_bah">15/09/2026 06:00</span>
  </a>
  <div class="fprc"><span>20</span><span>80</span></div>
  <div class="predict"><span class="forepr">Over</span></div>
</div>
"""


NOW = datetime(2026, 9, 15, 12, tzinfo=timezone.utc)


def prediction(**changes) -> ForebetPrediction:
    values = {
        "event_id": "123",
        "league": "Primera A",
        "home_team": "America de Cali",
        "away_team": "Deportivo Cali",
        "kickoff": "2026-09-15T10:00-05:00",
        "market": "Over 2.5",
        "selection": "Over",
        "probability": 0.65,
        "predicted_score": "2 - 1",
        "average_goals": 3.1,
        "reference_odd": 1.9,
        "source_url": "https://www.forebet.com/en/football/matches/america-cali-123",
        "observed_at": NOW.isoformat(timespec="seconds"),
        "scope": "colombia",
    }
    values.update(changes)
    return ForebetPrediction(**values)


def test_parse_totals_uses_structured_teams_and_bogota_time() -> None:
    parsed = parse_predictions(
        SAMPLE_HTML,
        market_kind="totals",
        scope="colombia",
        timezone_name="America/Bogota",
        observed_at=NOW.isoformat(),
        now=NOW,
    )

    assert parsed == [prediction(observed_at=NOW.isoformat())]


def test_parse_btts_maps_yes_to_si() -> None:
    html = SAMPLE_HTML.replace("35</span><span>65", "30</span><span>70").replace(
        ">Over<", ">Yes<"
    )
    parsed = parse_predictions(
        html,
        market_kind="btts",
        scope="global",
        timezone_name="America/Bogota",
        observed_at=NOW.isoformat(),
        now=NOW,
    )

    assert parsed[0].market == "BTTS"
    assert parsed[0].selection == "Si"
    assert parsed[0].probability == 0.70


def test_parse_rejects_unknown_market() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        parse_predictions(
            SAMPLE_HTML,
            market_kind="corners",
            scope="global",
            timezone_name="UTC",
            observed_at=NOW.isoformat(),
            now=NOW,
        )


def test_fresh_cache_avoids_browser_fetch(monkeypatch, tmp_path) -> None:
    item = prediction()
    Store(tmp_path).set_status(
        "forebet_cache", {"observed_at": item.observed_at, "predictions": [asdict(item)]}
    )
    monkeypatch.setattr(
        forebet,
        "fetch_predictions",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("browser must not open")),
    )

    result = get_forebet_predictions(tmp_path, now=NOW)

    assert result.from_cache is True
    assert result.predictions == [item]


def test_fetch_records_private_observation_and_cache(monkeypatch, tmp_path) -> None:
    item = prediction()
    monkeypatch.setattr(forebet, "fetch_predictions", lambda **kwargs: [item])

    result = get_forebet_predictions(tmp_path, now=NOW)

    assert result.from_cache is False
    stored, = Store(tmp_path).external_observations("forebet")
    assert stored["snapshot"] == asdict(item)
    assert Store(tmp_path).get_status("forebet_cache")["predictions"] == [asdict(item)]


def test_format_message_labels_external_experimental_data() -> None:
    message = format_forebet_message(ForebetResult([prediction()], NOW.isoformat(), False))

    assert message.startswith("PRONOSTICOS FOREBET EXPERIMENTALES")
    assert "Probabilidad publicada por Forebet: 65%" in message
    assert "Cuota de referencia Forebet: 1.90" in message
    assert "no es una prediccion propia ni una garantia" in message
