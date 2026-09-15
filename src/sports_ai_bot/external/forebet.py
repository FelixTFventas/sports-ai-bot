from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import re
import shutil
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from sports_ai_bot.storage import Store


FOREBET_BASE_URL = "https://www.forebet.com"
FOREBET_PAGES = (
    (
        "global",
        "totals",
        f"{FOREBET_BASE_URL}/en/football-tips-and-predictions-for-today/"
        "predictions-under-over-goals",
    ),
    (
        "global",
        "btts",
        f"{FOREBET_BASE_URL}/en/football-tips-and-predictions-for-today/"
        "predictions-both-to-score",
    ),
    (
        "colombia",
        "totals",
        f"{FOREBET_BASE_URL}/en/football-tips-and-predictions-for-colombia/"
        "primera-a/under-over",
    ),
    (
        "colombia",
        "btts",
        f"{FOREBET_BASE_URL}/en/football-tips-and-predictions-for-colombia/"
        "primera-a/bothtoscore",
    ),
)
_BROWSER_CANDIDATES = (
    Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
)
_LEAGUE_PATTERN = re.compile(
    r"getstag\(this,\s*\d+,\s*'[^']*',\s*'(?P<league>[^']*)'"
)


class ForebetError(RuntimeError):
    pass


@dataclass(frozen=True)
class ForebetPrediction:
    event_id: str
    league: str
    home_team: str
    away_team: str
    kickoff: str
    market: str
    selection: str
    probability: float
    predicted_score: str
    average_goals: float | None
    reference_odd: float | None
    source_url: str
    observed_at: str
    scope: str


@dataclass(frozen=True)
class ForebetResult:
    predictions: list[ForebetPrediction]
    observed_at: str
    from_cache: bool


def get_forebet_predictions(
    data_dir: Path,
    *,
    timezone_name: str = "America/Bogota",
    cache_minutes: int = 45,
    min_probability: float = 0.60,
    limit: int = 10,
    browser_executable: str = "",
    now: datetime | None = None,
) -> ForebetResult:
    store = Store(data_dir)
    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cached = _read_cache(store)
    if cached is not None:
        age = current_time - datetime.fromisoformat(cached.observed_at).astimezone(timezone.utc)
        if timedelta(0) <= age <= timedelta(minutes=cache_minutes):
            valid = [
                prediction
                for prediction in cached.predictions
                if prediction.probability >= min_probability
                and datetime.fromisoformat(prediction.kickoff).astimezone(timezone.utc)
                >= current_time
            ]
            if valid:
                return ForebetResult(valid[:limit], cached.observed_at, True)

    predictions = fetch_predictions(
        timezone_name=timezone_name,
        min_probability=min_probability,
        limit=limit,
        browser_executable=browser_executable,
        now=current_time,
    )
    if not predictions:
        raise ForebetError("Forebet no devolvio pronosticos futuros completos.")
    store.record_external_observations("forebet", predictions)
    payload = {
        "observed_at": predictions[0].observed_at,
        "predictions": [asdict(prediction) for prediction in predictions],
    }
    store.set_status("forebet_cache", payload)
    return ForebetResult(predictions, predictions[0].observed_at, False)


def fetch_predictions(
    *,
    timezone_name: str = "America/Bogota",
    min_probability: float = 0.60,
    limit: int = 10,
    browser_executable: str = "",
    now: datetime | None = None,
) -> list[ForebetPrediction]:
    if not 0 <= min_probability <= 1:
        raise ValueError("min_probability must be between 0 and 1")
    if limit < 1:
        raise ValueError("limit must be positive")

    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    observed_at = current_time.isoformat(timespec="seconds")
    executable = _browser_executable(browser_executable)
    predictions: list[ForebetPrediction] = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=False, executable_path=executable)
            try:
                for scope, market_kind, url in FOREBET_PAGES:
                    context = browser.new_context(timezone_id=timezone_name)
                    try:
                        page = context.new_page()
                        page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                        page.wait_for_selector(".rcnt a.tnmscn", timeout=45_000)
                        predictions.extend(
                            parse_predictions(
                                page.content(),
                                market_kind=market_kind,
                                scope=scope,
                                timezone_name=timezone_name,
                                observed_at=observed_at,
                                now=current_time,
                            )
                        )
                    finally:
                        context.close()
            finally:
                browser.close()
    except ForebetError:
        raise
    except PlaywrightError as exc:
        raise ForebetError(f"No se pudo consultar Forebet con el navegador local: {exc}") from exc

    best_by_event: dict[str, ForebetPrediction] = {}
    for prediction in predictions:
        if prediction.probability < min_probability:
            continue
        current = best_by_event.get(prediction.event_id)
        if current is None or _prediction_key(prediction) > _prediction_key(current):
            best_by_event[prediction.event_id] = prediction
    return sorted(best_by_event.values(), key=_prediction_key, reverse=True)[:limit]


def parse_predictions(
    html: str,
    *,
    market_kind: str,
    scope: str,
    timezone_name: str,
    observed_at: str,
    now: datetime,
    horizon_hours: int = 48,
) -> list[ForebetPrediction]:
    if market_kind not in {"totals", "btts"}:
        raise ValueError("Unsupported Forebet market")
    zone = ZoneInfo(timezone_name)
    window_start = now.astimezone(timezone.utc)
    window_end = window_start + timedelta(hours=horizon_hours)
    soup = BeautifulSoup(html, "html.parser")
    predictions: list[ForebetPrediction] = []
    for row in soup.select(".rcnt"):
        link = row.select_one("a.tnmscn[href]")
        home = row.select_one(".homeTeam [itemprop='name']")
        away = row.select_one(".awayTeam [itemprop='name']")
        date_node = row.select_one(".date_bah")
        probability_nodes = row.select(".fprc > span")
        prediction_node = row.select_one(".predict .forepr")
        if not all((link, home, away, date_node, prediction_node)) or len(probability_nodes) < 2:
            continue

        kickoff = _parse_kickoff(date_node.get_text(" ", strip=True), zone)
        if kickoff is None or not window_start <= kickoff.astimezone(timezone.utc) <= window_end:
            continue
        probabilities = [_probability(node.get_text(strip=True)) for node in probability_nodes[:2]]
        if any(value is None for value in probabilities):
            continue
        raw_selection = prediction_node.get_text(" ", strip=True).casefold()
        if market_kind == "totals" and raw_selection in {"over", "under"}:
            index = 1 if raw_selection == "over" else 0
            market = f"{raw_selection.title()} 2.5"
            selection = raw_selection.title()
        elif market_kind == "btts" and raw_selection in {"yes", "no"}:
            index = 1 if raw_selection == "yes" else 0
            market = "BTTS"
            selection = "Si" if raw_selection == "yes" else "No"
        else:
            continue

        source_url = urljoin(FOREBET_BASE_URL, link.get("href", ""))
        event_match = re.search(r"-(\d+)(?:/)?$", source_url)
        if event_match is None:
            continue
        score_node = row.select_one(".ex_sc.tabonly")
        league_image = row.select_one(".shortagDiv img[onclick]")
        league = _league_name(league_image.get("onclick", "") if league_image else "")
        average_node = row.select_one(".avg_sc")
        odd_node = row.select_one(".prmod > .lscrsp")
        predictions.append(
            ForebetPrediction(
                event_id=event_match.group(1),
                league="Primera A" if scope == "colombia" else league,
                home_team=home.get_text(" ", strip=True),
                away_team=away.get_text(" ", strip=True),
                kickoff=kickoff.isoformat(timespec="minutes"),
                market=market,
                selection=selection,
                probability=probabilities[index] or 0.0,
                predicted_score=score_node.get_text(" ", strip=True) if score_node else "",
                average_goals=_decimal(average_node.get_text(strip=True) if average_node else ""),
                reference_odd=_decimal(odd_node.get_text(strip=True) if odd_node else ""),
                source_url=source_url,
                observed_at=observed_at,
                scope=scope,
            )
        )
    return predictions


def format_forebet_message(
    result: ForebetResult, timezone_name: str = "America/Bogota"
) -> str:
    if not result.predictions:
        return "No hay pronosticos Forebet futuros que cumplan el filtro."
    observed = datetime.fromisoformat(result.observed_at).astimezone(ZoneInfo(timezone_name))
    lines = ["PRONOSTICOS FOREBET EXPERIMENTALES", ""]
    for index, prediction in enumerate(result.predictions, start=1):
        scope = "Colombia" if prediction.scope == "colombia" else prediction.league
        lines.extend(
            [
                f"{index}. {prediction.home_team} vs {prediction.away_team}",
                f"Liga: {scope}",
                f"Inicio: {prediction.kickoff}",
                f"Mercado: {prediction.market} | Seleccion: {prediction.selection}",
                f"Probabilidad publicada por Forebet: {prediction.probability:.0%}",
                f"Marcador previsto: {prediction.predicted_score or 'No disponible'}",
                f"Promedio de goles: {_display_number(prediction.average_goals)}",
                f"Cuota de referencia Forebet: {_display_number(prediction.reference_odd)}",
                f"Fuente: {prediction.source_url}",
                "",
            ]
        )
    cache_note = " (cache local)" if result.from_cache else ""
    lines.extend(
        [
            f"Consultado: {observed:%Y-%m-%d %H:%M} {timezone_name}{cache_note}",
            "Fuente externa para uso privado; no es una prediccion propia ni una garantia.",
            "Verifica la cuota vigente en tu casa de apuestas.",
        ]
    )
    return "\n".join(lines)


def _read_cache(store: Store) -> ForebetResult | None:
    payload = store.get_status("forebet_cache")
    if not isinstance(payload, dict) or not isinstance(payload.get("predictions"), list):
        return None
    try:
        predictions = [ForebetPrediction(**item) for item in payload["predictions"]]
        observed_at = str(payload["observed_at"])
        parsed_observed_at = datetime.fromisoformat(observed_at)
        if parsed_observed_at.tzinfo is None or parsed_observed_at.utcoffset() is None:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return ForebetResult(predictions, observed_at, True)


def _browser_executable(configured: str) -> str | None:
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise ForebetError(f"No existe el navegador configurado: {path}")
        return str(path)
    for name in ("chrome", "msedge", "chromium"):
        executable = shutil.which(name)
        if executable:
            return executable
    for path in _BROWSER_CANDIDATES:
        if path.is_file():
            return str(path)
    return None


def _parse_kickoff(value: str, zone: ZoneInfo) -> datetime | None:
    for date_format in ("%d/%m/%Y %H:%M", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, date_format).replace(tzinfo=zone)
        except ValueError:
            continue
    return None


def _probability(value: str) -> float | None:
    try:
        probability = int(value) / 100
    except ValueError:
        return None
    return probability if 0 <= probability <= 1 else None


def _decimal(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) and number > 0 else None


def _league_name(onclick: str) -> str:
    match = _LEAGUE_PATTERN.search(onclick)
    return match.group("league") if match else "Liga no identificada"


def _prediction_key(prediction: ForebetPrediction) -> tuple[int, float, str]:
    return (prediction.scope == "colombia", prediction.probability, prediction.kickoff)


def _display_number(value: float | None) -> str:
    return "No disponible" if value is None else f"{value:.2f}"
