from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from urllib.parse import urlparse

import httpx

from sports_ai_bot.predict.pipeline import (
    Pick,
    _confidence_label,
    _edge,
    _expected_value,
    _implied_probability,
    _rating,
    _stake_units,
)


FOREBET_TOP_URL = "https://www.forebet.com/es/top-predicciones"
FOREBET_MIRROR_URL = f"https://r.jina.ai/http://{FOREBET_TOP_URL.removeprefix('https://')}"
FOREBET_DAILY_1X2_URL_TEMPLATE = (
    "https://www.forebet.com/es/predicciones-de-futbol/predicciones-1x2/{date}"
)
FOREBET_VALUE_MODEL_NAME = "forebet_market_blend"
FOREBET_WEIGHT = 0.60
MARKET_WEIGHT = 0.40

_SECTION_PATTERN = re.compile(
    r"Principales predicciones\s*(?P<section>.+?)(?:\nVer m[aá]s|\nEl f[uú]tbol|\nDeportes|\n### )",
    re.DOTALL,
)
_BLOCK_PATTERN = re.compile(
    r"(?m)^(?<!!)\[(?P<label>.+?)\]\((?P<url>https?://www\.forebet\.com/es/football/matches/.+?)\)\s+"
    r"(?P<home>\d{1,2})\s+(?P<draw>\d{1,2})\s+(?P<away>\d{1,2})\s+"
    r"(?P<pick>[12X])\s+(?P<line_score>\d+-\d+|\d+ - \d+)",
    re.DOTALL,
)
_LABEL_PATTERN = re.compile(
    r"^(?P<match>.+?)\s*(?P<date>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<time>\d{1,2}:\d{2}(?:\s+[AP]M)?)$"
)
_MATCH_TITLE_PATTERN = re.compile(
    r"# \[(?P<home>.+?)\]\(.+?\)\s+-\s+\[(?P<away>.+?)\]\(.+?\)",
    re.DOTALL,
)
_MATCH_DATE_PATTERN = re.compile(
    r"\b(?P<date>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<time>\d{1,2}:\d{2}(?:\s+[AP]M)?)\b"
)
_MATCH_LINK_PATTERN = re.compile(
    r"(?<!!)\[(?P<label>[^\]\n]*?\d{2}/\d{2}/\d{4}\s+\d{1,2}:\d{2}(?:\s+[AP]M)?)\]"
    r"\((?P<url>https?://www\.forebet\.com/es/football/matches/[^\s]+)\)",
    re.DOTALL,
)
_ODD_PATTERN = re.compile(r"[+-]\d{3,4}")


@dataclass(frozen=True)
class ForebetPick:
    match_label: str
    match_date: str
    match_time: str
    prediction: str
    probability: int
    predicted_score: str
    source_url: str


@dataclass(frozen=True)
class ForebetMatchLink:
    match_label: str
    match_datetime: datetime
    source_url: str


class ForebetError(RuntimeError):
    pass


@dataclass(frozen=True)
class _ForebetOutcome:
    market: str
    selection: str | None
    forebet_probability: float
    odd: float
    market_probability: float
    line: float | None = None


def fetch_top_picks(limit: int = 5, timeout: float = 30.0) -> list[ForebetPick]:
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(FOREBET_MIRROR_URL)
        response.raise_for_status()
    return parse_top_picks(response.text, limit=limit)


def fetch_match_value_picks(
    url: str,
    limit: int = 5,
    min_odd: float = 1.50,
    timeout: float = 30.0,
) -> list[Pick]:
    mirror_url = _forebet_match_mirror_url(url)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(mirror_url)
        response.raise_for_status()
    return parse_match_value_picks(response.text, source_url=url, limit=limit, min_odd=min_odd)


def fetch_top_value_picks(
    limit_matches: int = 5,
    limit: int = 5,
    min_odd: float = 1.50,
    timeout: float = 30.0,
) -> list[Pick]:
    top_picks = fetch_top_picks(limit=limit_matches, timeout=timeout)
    best_by_match: dict[str, Pick] = {}
    for top_pick in top_picks:
        try:
            match_picks = fetch_match_value_picks(
                top_pick.source_url,
                limit=limit,
                min_odd=min_odd,
                timeout=timeout,
            )
        except (ForebetError, httpx.HTTPError):
            continue
        for pick in match_picks:
            current = best_by_match.get(pick.match_label)
            if current is None or _value_pick_sort_key(pick) > _value_pick_sort_key(current):
                best_by_match[pick.match_label] = pick

    picks = list(best_by_match.values())
    picks.sort(key=_value_pick_sort_key, reverse=True)
    return picks[:limit]


def fetch_48h_value_picks(
    limit_matches: int = 30,
    limit: int = 10,
    min_odd: float = 1.50,
    min_edge: float = 0.0,
    min_probability: float = 0.60,
    horizon_hours: int = 48,
    timeout: float = 30.0,
) -> list[Pick]:
    now = datetime.now()
    best_by_match: dict[str, Pick] = {}
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for source_url in _forebet_48h_urls(now, horizon_hours=horizon_hours):
            response = client.get(_forebet_mirror_url(source_url))
            response.raise_for_status()
            match_picks = parse_list_value_picks(
                response.text,
                horizon_hours=horizon_hours,
                now=now,
                min_odd=min_odd,
                limit_matches=limit_matches,
            )
            for pick in match_picks:
                if pick.probability < min_probability:
                    continue
                if pick.edge is None or pick.edge < min_edge:
                    continue
                current = best_by_match.get(pick.match_label)
                if current is None or _value_pick_sort_key(pick) > _value_pick_sort_key(current):
                    best_by_match[pick.match_label] = pick

    picks = list(best_by_match.values())
    picks.sort(key=_value_pick_sort_key, reverse=True)
    return picks[:limit]


def fetch_48h_match_links(
    limit: int = 30,
    horizon_hours: int = 48,
    timeout: float = 30.0,
) -> list[ForebetMatchLink]:
    now = datetime.now()
    links: list[ForebetMatchLink] = []
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for source_url in _forebet_48h_urls(now, horizon_hours=horizon_hours):
            response = client.get(_forebet_mirror_url(source_url))
            response.raise_for_status()
            links.extend(
                parse_match_links(
                    response.text,
                    horizon_hours=horizon_hours,
                    now=now,
                    limit=limit * 2,
                )
            )
    return _dedupe_match_links(links)[:limit]


def parse_top_picks(markdown: str, limit: int = 5) -> list[ForebetPick]:
    section_match = _SECTION_PATTERN.search(markdown)
    if section_match is None:
        raise ForebetError("No se encontro la seccion de principales predicciones de Forebet.")

    picks: list[ForebetPick] = []
    for match in _BLOCK_PATTERN.finditer(section_match.group("section")):
        label_match = _LABEL_PATTERN.match(_normalize_spaces(match.group("label")))
        if label_match is None:
            continue

        probabilities = {
            "1": int(match.group("home")),
            "X": int(match.group("draw")),
            "2": int(match.group("away")),
        }
        prediction = match.group("pick")
        picks.append(
            ForebetPick(
                match_label=label_match.group("match"),
                match_date=label_match.group("date"),
                match_time=label_match.group("time"),
                prediction=prediction,
                probability=probabilities[prediction],
                predicted_score=match.group("line_score").replace(" ", ""),
                source_url=match.group("url"),
            )
        )

    if not picks:
        raise ForebetError("No se pudieron parsear predicciones de Forebet.")

    picks.sort(key=lambda item: item.probability, reverse=True)
    return picks[:limit]


def parse_match_links(
    markdown: str,
    horizon_hours: int = 48,
    now: datetime | None = None,
    limit: int = 30,
) -> list[ForebetMatchLink]:
    current_time = now or datetime.now()
    window_end = current_time + timedelta(hours=horizon_hours)
    links: list[ForebetMatchLink] = []
    for match in _MATCH_LINK_PATTERN.finditer(markdown):
        label = _normalize_spaces(match.group("label"))
        label_match = _LABEL_PATTERN.match(label)
        if label_match is None:
            continue
        match_time = _parse_forebet_datetime(
            label_match.group("date"),
            label_match.group("time"),
        )
        if match_time is None or match_time < current_time or match_time > window_end:
            continue
        links.append(
            ForebetMatchLink(
                match_label=label_match.group("match").strip(),
                match_datetime=match_time,
                source_url=match.group("url"),
            )
        )
    links.sort(key=lambda item: item.match_datetime)
    return _dedupe_match_links(links)[:limit]


def parse_list_value_picks(
    markdown: str,
    horizon_hours: int = 48,
    now: datetime | None = None,
    min_odd: float = 1.50,
    limit_matches: int = 30,
) -> list[Pick]:
    current_time = now or datetime.now()
    window_end = current_time + timedelta(hours=horizon_hours)
    link_matches = list(_MATCH_LINK_PATTERN.finditer(markdown))
    picks: list[Pick] = []
    parsed_matches = 0
    for index, match in enumerate(link_matches):
        if parsed_matches >= limit_matches:
            break
        label = _normalize_spaces(match.group("label"))
        label_match = _LABEL_PATTERN.match(label)
        if label_match is None:
            continue
        match_time = _parse_forebet_datetime(label_match.group("date"), label_match.group("time"))
        if match_time is None or match_time < current_time or match_time > window_end:
            continue

        next_start = link_matches[index + 1].start() if index + 1 < len(link_matches) else len(markdown)
        row_section = markdown[match.start():next_start]
        probabilities = _first_probability_row(row_section, count=3)
        odds = _last_decimal_odds(row_section, count=3)
        if probabilities is None or odds is None:
            continue
        parsed_matches += 1

        home_team, away_team = _split_match_label_guess(label_match.group("match"))
        match_label = label_match.group("match").strip()
        market_probabilities = _normalized_market_probabilities(odds)
        outcomes = [
            _ForebetOutcome("1X2", "Local", probabilities[0], odds[0], market_probabilities[0]),
            _ForebetOutcome("1X2", "Empate", probabilities[1], odds[1], market_probabilities[1]),
            _ForebetOutcome("1X2", "Visitante", probabilities[2], odds[2], market_probabilities[2]),
        ]
        for outcome in outcomes:
            if outcome.odd < min_odd:
                continue
            picks.append(
                _build_value_pick(
                    outcome,
                    match_date=match_time.date().isoformat(),
                    home_team=home_team,
                    away_team=away_team,
                    match_label=match_label,
                    source_url=match.group("url"),
                )
            )

    picks.sort(key=_value_pick_sort_key, reverse=True)
    return picks


def parse_match_value_picks(
    markdown: str,
    source_url: str = "",
    limit: int = 5,
    min_odd: float = 1.50,
) -> list[Pick]:
    home_team, away_team = _extract_match_teams(markdown)
    match_date = _extract_match_date(markdown)
    match_label = f"{home_team} vs {away_team}"
    outcomes = _extract_match_outcomes(markdown, home_team=home_team)
    picks = [
        _build_value_pick(
            outcome,
            match_date=match_date,
            home_team=home_team,
            away_team=away_team,
            match_label=match_label,
            source_url=source_url,
        )
        for outcome in outcomes
        if outcome.odd >= min_odd
    ]
    if not picks:
        raise ForebetError("No se encontraron eventos Forebet con cuota suficiente.")

    picks.sort(
        key=_value_pick_sort_key,
        reverse=True,
    )
    return picks[:limit]


def format_top_picks_message(picks: list[ForebetPick]) -> str:
    if not picks:
        return "No hay predicciones disponibles en Forebet."

    lines = ["Top 5 Forebet del dia:"]
    for index, pick in enumerate(picks, start=1):
        lines.append(
            f"{index}. {pick.match_label} | Pick {pick.prediction} | Prob {pick.probability}% | Marcador {pick.predicted_score} | {pick.match_date} {pick.match_time}"
        )
    lines.append("Fuente externa de referencia: Forebet.")
    return "\n".join(lines)


def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _parse_forebet_datetime(date_value: str, time_value: str) -> datetime | None:
    normalized_time = _normalize_spaces(time_value).upper()
    has_meridiem = "AM" in normalized_time or "PM" in normalized_time
    formats = ["%m/%d/%Y %I:%M %p", "%d/%m/%Y %I:%M %p"] if has_meridiem else [
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M",
    ]
    raw_value = f"{date_value} {normalized_time}"
    for date_format in formats:
        try:
            return datetime.strptime(raw_value, date_format)
        except ValueError:
            continue
    return None


def _dedupe_match_links(links: list[ForebetMatchLink]) -> list[ForebetMatchLink]:
    deduped: list[ForebetMatchLink] = []
    seen_urls: set[str] = set()
    for link in links:
        if link.source_url in seen_urls:
            continue
        seen_urls.add(link.source_url)
        deduped.append(link)
    return deduped


def _forebet_48h_urls(now: datetime, horizon_hours: int) -> list[str]:
    end_date = (now + timedelta(hours=horizon_hours)).date()
    dates = []
    current_date = now.date()
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    return [
        FOREBET_DAILY_1X2_URL_TEMPLATE.format(date=match_date.isoformat())
        for match_date in dates
    ]


def _split_match_label_guess(match_label: str) -> tuple[str, str]:
    parts = _normalize_spaces(match_label).split()
    if len(parts) <= 1:
        return match_label, ""
    midpoint = max(1, len(parts) // 2)
    return " ".join(parts[:midpoint]), " ".join(parts[midpoint:])


def _extract_match_teams(markdown: str) -> tuple[str, str]:
    title_match = _MATCH_TITLE_PATTERN.search(markdown)
    if title_match is not None:
        return _normalize_spaces(title_match.group("home")), _normalize_spaces(title_match.group("away"))

    heading_match = re.search(r"^# (?P<home>.+?)\s+vs\s+(?P<away>.+?)\s+Pred", markdown, re.MULTILINE)
    if heading_match is not None:
        return _normalize_spaces(heading_match.group("home")), _normalize_spaces(heading_match.group("away"))

    link_match = _MATCH_LINK_PATTERN.search(markdown)
    if link_match is not None:
        label = _normalize_spaces(link_match.group("label"))
        label = _MATCH_DATE_PATTERN.sub("", label).strip()
        parts = label.split()
        midpoint = max(1, len(parts) // 2)
        return " ".join(parts[:midpoint]), " ".join(parts[midpoint:])

    raise ForebetError("No se pudieron identificar los equipos del partido Forebet.")


def _extract_match_date(markdown: str) -> str:
    match = _MATCH_DATE_PATTERN.search(markdown)
    if match is None:
        return ""
    parsed = _parse_forebet_datetime(match.group("date"), match.group("time"))
    if parsed is None:
        return match.group("date")
    return parsed.date().isoformat()


def _extract_match_outcomes(markdown: str, home_team: str) -> list[_ForebetOutcome]:
    outcomes: list[_ForebetOutcome] = []
    outcomes.extend(_extract_1x2_outcomes(markdown))
    outcomes.extend(_extract_total25_outcomes(markdown))
    outcomes.extend(_extract_btts_outcomes(markdown))
    ht_outcome = _extract_halftime_outcome(markdown)
    if ht_outcome is not None:
        outcomes.append(ht_outcome)
    htft_outcome = _extract_htft_outcome(markdown)
    if htft_outcome is not None:
        outcomes.append(htft_outcome)
    double_chance_outcome = _extract_double_chance_outcome(markdown, home_team=home_team)
    if double_chance_outcome is not None:
        outcomes.append(double_chance_outcome)
    return outcomes


def _extract_1x2_outcomes(markdown: str) -> list[_ForebetOutcome]:
    section = _market_section(markdown, r"Probabilidad %\s+1 X 2")
    if not section:
        return []
    probabilities = _first_probability_row(section, count=3)
    odds = _last_decimal_odds(section, count=3)
    if probabilities is None or odds is None:
        return []
    market_probabilities = _normalized_market_probabilities(odds)
    return [
        _ForebetOutcome("1X2", "Local", probabilities[0], odds[0], market_probabilities[0]),
        _ForebetOutcome("1X2", "Empate", probabilities[1], odds[1], market_probabilities[1]),
        _ForebetOutcome("1X2", "Visitante", probabilities[2], odds[2], market_probabilities[2]),
    ]


def _extract_total25_outcomes(markdown: str) -> list[_ForebetOutcome]:
    section = _market_section(markdown, r"Probabilidad %\s+Menos/Más\s+2\.5")
    if not section:
        return []
    probabilities = _first_probability_row(section, count=2)
    odds = _last_decimal_odds(section, count=2)
    if probabilities is None or odds is None:
        return []
    market_probabilities = _normalized_market_probabilities(odds)
    return [
        _ForebetOutcome("Under 2.5", None, probabilities[0], odds[0], market_probabilities[0], 2.5),
        _ForebetOutcome("Over 2.5", None, probabilities[1], odds[1], market_probabilities[1], 2.5),
    ]


def _extract_btts_outcomes(markdown: str) -> list[_ForebetOutcome]:
    section = _market_section(markdown, r"Probabilidad %\s+No Sí")
    if not section:
        return []
    probabilities = _first_probability_row(section, count=2)
    odds = _last_decimal_odds(section, count=2)
    if probabilities is None or odds is None:
        return []
    market_probabilities = _normalized_market_probabilities(odds)
    return [
        _ForebetOutcome("BTTS", "Si", probabilities[1], odds[0], market_probabilities[0]),
        _ForebetOutcome("BTTS", "No", probabilities[0], odds[1], market_probabilities[1]),
    ]


def _extract_halftime_outcome(markdown: str) -> _ForebetOutcome | None:
    section = _market_section(markdown, r"Probabilidad de marcador en el medio tiempo %")
    if not section:
        return None
    probabilities = _first_probability_row(section, count=3)
    odds = _last_decimal_odds(section, count=1)
    selection = _prediction_token_after_probabilities(section, allowed={"1", "X", "2"})
    if probabilities is None or odds is None or selection is None:
        return None
    probability_by_selection = {"1": probabilities[0], "X": probabilities[1], "2": probabilities[2]}
    selection_label = _x12_label(selection)
    return _ForebetOutcome(
        "Medio Tiempo",
        selection_label,
        probability_by_selection[selection],
        odds[0],
        _implied_probability(odds[0]) or 0.0,
    )


def _extract_htft_outcome(markdown: str) -> _ForebetOutcome | None:
    section = _market_section(markdown, r"Probabilidad MT/FT %")
    if not section:
        return None
    probability = _single_percentage_probability(section)
    odds = _last_decimal_odds(section, count=1)
    selection = _htft_prediction(section)
    if probability is None or odds is None or selection is None:
        return None
    return _ForebetOutcome(
        "HT/FT",
        f"{_x12_label(selection[0])}/{_x12_label(selection[1])}",
        probability,
        odds[0],
        _implied_probability(odds[0]) or 0.0,
    )


def _extract_double_chance_outcome(markdown: str, home_team: str) -> _ForebetOutcome | None:
    section = _market_section(markdown, r"Probabilidad %\s+1X/2X/12")
    if not section:
        return None
    probability = _single_percentage_probability(section)
    odds = _last_decimal_odds(section, count=1)
    selection = _prediction_token_after_probabilities(section, allowed={"1X", "X1", "2X", "X2", "12"})
    if probability is None or odds is None or selection is None:
        return None
    return _ForebetOutcome(
        "Doble oportunidad",
        _double_chance_label(selection, home_team=home_team),
        probability,
        odds[0],
        _implied_probability(odds[0]) or 0.0,
    )


def _build_value_pick(
    outcome: _ForebetOutcome,
    match_date: str,
    home_team: str,
    away_team: str,
    match_label: str,
    source_url: str,
) -> Pick:
    probability = _blended_probability(outcome.forebet_probability, outcome.market_probability)
    edge = _edge(probability, outcome.odd)
    expected_value = _expected_value(probability, outcome.odd)
    pick = Pick(
        match_date=match_date,
        home_team=home_team,
        away_team=away_team,
        match_label=match_label,
        league="forebet",
        market=outcome.market,
        selection=outcome.selection,
        line=outcome.line,
        probability=probability,
        confidence=_confidence_label(probability),
        model_name=FOREBET_VALUE_MODEL_NAME,
        odd=outcome.odd,
        implied_probability=_implied_probability(outcome.odd),
        edge=edge,
        expected_value=expected_value,
        factors=[
            f"Prob Forebet: {outcome.forebet_probability:.1%}",
            f"Prob mercado: {outcome.market_probability:.1%}",
            "Formula: 60% Forebet + 40% mercado",
            f"Fuente: {source_url}" if source_url else "Fuente: Forebet",
        ],
        is_experimental=True,
    )
    pick.stake_units = _stake_units(pick.edge, pick.expected_value)
    pick.rating = _rating(pick.edge, pick.expected_value)
    pick.score = probability
    return pick


def _market_section(markdown: str, marker_pattern: str) -> str:
    marker_match = re.search(marker_pattern, markdown, re.IGNORECASE)
    if marker_match is None:
        return ""
    start = marker_match.start()
    next_section = re.search(r"\nEquipo local\s+\n\s*Equipo visitante", markdown[marker_match.end():])
    if next_section is None:
        return markdown[start:]
    end = marker_match.end() + next_section.start()
    return markdown[start:end]


def _first_probability_row(section: str, count: int) -> list[float] | None:
    link_match = _MATCH_LINK_PATTERN.search(section)
    search_area = section[link_match.end():] if link_match is not None else section
    pattern = r"(?m)^\s*" + r"\s+".join([r"(\d{1,3})"] * count) + r"\s*$"
    match = re.search(pattern, search_area)
    if match is None:
        return None
    probabilities = [int(value) / 100 for value in match.groups()]
    if any(probability > 1 for probability in probabilities):
        return None
    return probabilities


def _single_percentage_probability(section: str) -> float | None:
    link_match = _MATCH_LINK_PATTERN.search(section)
    search_area = section[link_match.end():] if link_match is not None else section
    match = re.search(r"(?m)^\s*(\d{1,3})%\s*$", search_area)
    if match is None:
        return None
    probability = int(match.group(1)) / 100
    return probability if probability <= 1 else None


def _prediction_token_after_probabilities(section: str, allowed: set[str]) -> str | None:
    probability_row = re.search(r"(?m)^\s*(?:\d{1,3}\s+){1,2}\d{1,3}\s*$", section)
    percentage_row = re.search(r"(?m)^\s*\d{1,3}%\s*$", section)
    start = 0
    if probability_row is not None:
        start = probability_row.end()
    elif percentage_row is not None:
        start = percentage_row.end()
    for line in section[start:].splitlines():
        normalized = line.strip()
        if not normalized:
            continue
        token = normalized.split()[0].upper()
        if token in allowed:
            return token
    return None


def _htft_prediction(section: str) -> tuple[str, str] | None:
    percentage_row = re.search(r"(?m)^\s*\d{1,3}%\s*$", section)
    if percentage_row is None:
        return None
    tokens: list[str] = []
    for line in section[percentage_row.end():].splitlines():
        normalized = line.strip()
        if not normalized:
            continue
        token = normalized.split()[0].upper()
        if token in {"1", "X", "2"}:
            tokens.append(token)
        if len(tokens) == 2:
            return tokens[0], tokens[1]
    return None


def _last_decimal_odds(section: str, count: int) -> list[float] | None:
    decimal_odds = [_american_to_decimal(value) for value in _ODD_PATTERN.findall(section)]
    decimal_odds = [value for value in decimal_odds if value is not None]
    if len(decimal_odds) < count:
        return None
    return decimal_odds[-count:]


def _american_to_decimal(value: str | int | float | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        american = int(str(value).strip())
    except ValueError:
        return None
    if american > 0:
        return round(1 + (american / 100), 4)
    if american < 0:
        return round(1 + (100 / abs(american)), 4)
    return None


def _normalized_market_probabilities(odds: list[float]) -> list[float]:
    implied_probabilities = [1 / odd for odd in odds]
    total = sum(implied_probabilities)
    if total <= 0:
        return implied_probabilities
    return [probability / total for probability in implied_probabilities]


def _blended_probability(forebet_probability: float, market_probability: float) -> float:
    return (FOREBET_WEIGHT * forebet_probability) + (MARKET_WEIGHT * market_probability)


def _value_pick_sort_key(pick: Pick) -> tuple[float, float, float]:
    return (
        float(pick.probability),
        float(pick.edge or 0.0),
        float(pick.expected_value or 0.0),
    )


def _x12_label(selection: str) -> str:
    return {"1": "Local", "X": "Empate", "2": "Visitante"}.get(selection, selection)


def _double_chance_label(selection: str, home_team: str) -> str:
    normalized = selection.upper()
    if normalized in {"1X", "X1"}:
        return f"{home_team} o Empate"
    if normalized in {"2X", "X2"}:
        return "Visitante o Empate"
    if normalized == "12":
        return "Local o Visitante"
    return selection


def _forebet_match_mirror_url(url: str) -> str:
    return _forebet_mirror_url(url)


def _forebet_mirror_url(url: str) -> str:
    normalized_url = url.strip()
    if not normalized_url:
        raise ForebetError("URL de Forebet vacia.")
    if normalized_url.startswith("https://r.jina.ai/http://"):
        return normalized_url
    if not normalized_url.startswith(("http://", "https://")):
        normalized_url = f"https://{normalized_url}"
    parsed = urlparse(normalized_url)
    if "forebet.com" not in parsed.netloc:
        raise ForebetError("La URL no pertenece a Forebet.")
    return f"https://r.jina.ai/http://{normalized_url.removeprefix('https://').removeprefix('http://')}"
