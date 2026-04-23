from __future__ import annotations

from dataclasses import dataclass
import re

import httpx


FOREBET_TOP_URL = "https://www.forebet.com/es/top-predicciones"
FOREBET_MIRROR_URL = f"https://r.jina.ai/http://{FOREBET_TOP_URL.removeprefix('https://')}"

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
    r"^(?P<match>.+?)\s+(?P<date>\d{2}/\d{2}/\d{4})\s+(?P<time>\d{1,2}:\d{2}\s+[AP]M)$"
)


@dataclass(frozen=True)
class ForebetPick:
    match_label: str
    match_date: str
    match_time: str
    prediction: str
    probability: int
    predicted_score: str
    source_url: str


class ForebetError(RuntimeError):
    pass


def fetch_top_picks(limit: int = 5, timeout: float = 30.0) -> list[ForebetPick]:
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(FOREBET_MIRROR_URL)
        response.raise_for_status()
    return parse_top_picks(response.text, limit=limit)


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
