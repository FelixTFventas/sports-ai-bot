from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

from sports_ai_bot.predict.pipeline import Pick
from sports_ai_bot.collect.the_odds_api import (
    get_event_markets_for_league,
    get_event_odds_for_league,
    get_events_for_league,
)
from sports_ai_bot.utils.config import get_settings


TARGET_LEAGUES = (
    "premier_league",
    "la_liga",
    "serie_a",
    "bundesliga",
    "liga_mx",
)
CORNERS_MARKET_KEYS = (
    "alternate_totals_corners",
    "alternate_spreads_corners",
    "totals_corners",
    "spreads_corners",
)


@dataclass(frozen=True)
class CornersResearchSummary:
    league: str
    event_count_48h: int
    events_with_corners: int
    coverage_pct: float
    market_keys: list[str]
    bookmakers: list[str]
    sample_event: str | None
    sample_points: list[float]
    sample_prices: list[float]


@dataclass(frozen=True)
class CornersOddsPreview:
    league: str
    event_label: str
    market_key: str
    bookmaker: str
    lines: list[str]


@dataclass(frozen=True)
class TargetCornersOddsPreview:
    league: str
    match_date: str
    home_team: str
    away_team: str
    event_label: str
    market_key: str
    bookmaker: str
    selection: str
    point: float
    price: float


def build_corners_picks(
    limit: int = 6,
    days_ahead: int = 2,
    limit_per_league: int = 2,
    target_point: float | None = None,
    selection: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
) -> list[Pick]:
    settings = get_settings()
    target_point = settings.corners_pick_point if target_point is None else target_point
    selection = settings.corners_pick_selection if selection is None else selection
    min_price = settings.corners_pick_min_price if min_price is None else min_price
    max_price = settings.corners_pick_max_price if max_price is None else max_price
    picks: list[Pick] = []
    market_label = f"{selection} {target_point:g} Corners"

    for preview in build_target_corners_odds_preview(
        days_ahead=days_ahead,
        limit_per_league=limit_per_league,
        target_point=target_point,
        selection=selection,
    )["previews"]:
        odd = float(preview["price"])
        if odd < min_price or odd > max_price:
            continue
        implied_probability = round(1.0 / odd, 4)
        picks.append(
            Pick(
                match_date=str(preview["match_date"]),
                home_team=str(preview["home_team"]),
                away_team=str(preview["away_team"]),
                match_label=str(preview["event_label"]),
                league=str(preview["league"]),
                market=market_label,
                probability=implied_probability,
                confidence=_corners_confidence_label(implied_probability),
                model_name="market_corners_experimental",
                odd=odd,
                implied_probability=implied_probability,
                factors=[
                    f"Bookmaker: {preview['bookmaker']}",
                    f"Mercado: {preview['market_key']}",
                    f"Linea objetivo: {selection} {target_point:g}",
                    "Pick experimental basado en cobertura y precio de mercado",
                ],
            )
        )

    picks.sort(key=lambda item: (item.probability, -(item.odd or 0.0)), reverse=True)
    return picks[:limit]


def build_corners_research_report(days_ahead: int = 2) -> dict[str, object]:
    settings = get_settings()
    summaries = [
        _research_league(league_name, days_ahead=days_ahead) for league_name in TARGET_LEAGUES
    ]
    report = {
        "days_ahead": days_ahead,
        "region": settings.the_odds_api_region,
        "configured_bookmakers": settings.the_odds_api_bookmakers_list(),
        "target_market_keys": list(CORNERS_MARKET_KEYS),
        "leagues": [summary.__dict__ for summary in summaries],
    }
    output_file = settings.reports_dir / "corners_research.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def format_corners_research_report(report: dict[str, object]) -> str:
    lines = [
        "Investigacion corners (The Odds API):",
        f"Ventana: {report['days_ahead']} dias | Region: {report['region']} | Bookmakers config: {', '.join(report['configured_bookmakers'])}",
    ]
    for league_report in report["leagues"]:
        lines.append(
            f"{league_report['league']}: {league_report['events_with_corners']}/{league_report['event_count_48h']} con corners | cobertura {league_report['coverage_pct']:.1%}"
        )
        if league_report["market_keys"]:
            lines.append(f"Mercados: {', '.join(league_report['market_keys'])}")
        if league_report["bookmakers"]:
            lines.append(f"Bookmakers: {', '.join(league_report['bookmakers'][:3])}")
        if league_report["sample_event"]:
            sample_points = ", ".join(f"{point:g}" for point in league_report["sample_points"][:5])
            sample_prices = ", ".join(
                f"{price:.2f}" for price in league_report["sample_prices"][:5]
            )
            lines.append(
                f"Muestra: {league_report['sample_event']} | puntos {sample_points or 'n/d'} | cuotas {sample_prices or 'n/d'}"
            )
    return "\n".join(lines)


def build_corners_odds_preview(
    days_ahead: int = 2,
    limit_per_league: int = 3,
) -> dict[str, object]:
    settings = get_settings()
    previews: list[dict[str, object]] = []
    for league_name in TARGET_LEAGUES:
        previews.extend(
            preview.__dict__
            for preview in _preview_league_corner_odds(
                league_name,
                days_ahead=days_ahead,
                limit=limit_per_league,
            )
        )

    report = {
        "days_ahead": days_ahead,
        "limit_per_league": limit_per_league,
        "region": settings.the_odds_api_region,
        "configured_bookmakers": settings.the_odds_api_bookmakers_list(),
        "previews": previews,
    }
    output_file = settings.reports_dir / "corners_odds_preview.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def format_corners_odds_preview(report: dict[str, object]) -> str:
    lines = [
        "Preview cuotas corners (The Odds API):",
        f"Ventana: {report['days_ahead']} dias | Limite por liga: {report['limit_per_league']} | Bookmakers config: {', '.join(report['configured_bookmakers'])}",
    ]
    previews = report["previews"]
    if not previews:
        lines.append("No se encontraron cuotas de corners en la ventana consultada.")
        return "\n".join(lines)

    for preview in previews:
        lines.append(
            f"{preview['league']} | {preview['event_label']} | {preview['market_key']} | {preview['bookmaker']}"
        )
        lines.append(f"Lineas: {' ; '.join(preview['lines'])}")
    return "\n".join(lines)


def build_target_corners_odds_preview(
    days_ahead: int = 2,
    limit_per_league: int = 3,
    target_point: float = 9.5,
    selection: str = "Over",
) -> dict[str, object]:
    settings = get_settings()
    previews: list[dict[str, object]] = []
    for league_name in TARGET_LEAGUES:
        previews.extend(
            preview.__dict__
            for preview in _preview_league_target_corner_odds(
                league_name,
                days_ahead=days_ahead,
                limit=limit_per_league,
                target_point=target_point,
                selection=selection,
            )
        )

    report = {
        "days_ahead": days_ahead,
        "limit_per_league": limit_per_league,
        "region": settings.the_odds_api_region,
        "configured_bookmakers": settings.the_odds_api_bookmakers_list(),
        "selection": selection,
        "target_point": target_point,
        "previews": previews,
    }
    output_file = settings.reports_dir / "target_corners_odds_preview.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def format_target_corners_odds_preview(report: dict[str, object]) -> str:
    lines = [
        "Preview corners objetivo (The Odds API):",
        (
            f"Ventana: {report['days_ahead']} dias | Limite por liga: {report['limit_per_league']} | "
            f"Objetivo: {report['selection']} {report['target_point']:g} | "
            f"Bookmakers config: {', '.join(report['configured_bookmakers'])}"
        ),
    ]
    previews = report["previews"]
    if not previews:
        lines.append("No se encontraron cuotas para la linea objetivo de corners.")
        return "\n".join(lines)

    for preview in previews:
        lines.append(
            f"{preview['league']} | {preview['event_label']} | {preview['market_key']} | {preview['bookmaker']} | {preview['selection']} {preview['point']:g} @ {preview['price']:.2f}"
        )
    return "\n".join(lines)


def _corners_confidence_label(probability: float) -> str:
    if probability >= 0.58:
        return "Media-Alta"
    if probability >= 0.50:
        return "Media"
    return "Media-Baja"


def _research_league(league_name: str, days_ahead: int) -> CornersResearchSummary:
    events = get_events_for_league(league_name, days_ahead=days_ahead)
    event_count = len(events)
    events_with_corners = 0
    market_counter: Counter[str] = Counter()
    bookmaker_counter: Counter[str] = Counter()
    sample_event = None
    sample_points: list[float] = []
    sample_prices: list[float] = []

    for event in events:
        event_id = str(event.get("id") or "")
        if not event_id:
            continue
        markets_payload = get_event_markets_for_league(league_name, event_id)
        corner_markets = _extract_corner_markets(markets_payload)
        if not corner_markets:
            continue

        events_with_corners += 1
        market_counter.update(corner_markets)
        bookmaker_counter.update(_extract_corner_bookmakers(markets_payload))
        if sample_event is None:
            sample_event = f"{event.get('home_team', '')} vs {event.get('away_team', '')}"
            sample_points, sample_prices = _load_sample_corner_prices(
                league_name,
                event_id,
                preferred_market=_preferred_corner_market(corner_markets),
            )

    coverage_pct = (events_with_corners / event_count) if event_count else 0.0
    return CornersResearchSummary(
        league=league_name,
        event_count_48h=event_count,
        events_with_corners=events_with_corners,
        coverage_pct=round(coverage_pct, 4),
        market_keys=sorted(market_counter),
        bookmakers=[name for name, _ in bookmaker_counter.most_common(5)],
        sample_event=sample_event,
        sample_points=sample_points,
        sample_prices=sample_prices,
    )


def _preview_league_corner_odds(
    league_name: str,
    days_ahead: int,
    limit: int,
) -> list[CornersOddsPreview]:
    events = get_events_for_league(league_name, days_ahead=days_ahead)
    previews: list[CornersOddsPreview] = []

    for event in events:
        if len(previews) >= limit:
            break
        event_id = str(event.get("id") or "")
        if not event_id:
            continue
        markets_payload = get_event_markets_for_league(league_name, event_id)
        corner_markets = _extract_corner_markets(markets_payload)
        if not corner_markets:
            continue
        preferred_market = _preferred_corner_market(corner_markets)
        bookmaker, lines = _load_preview_lines(league_name, event_id, preferred_market)
        if not lines:
            continue
        previews.append(
            CornersOddsPreview(
                league=league_name,
                event_label=f"{event.get('home_team', '')} vs {event.get('away_team', '')}",
                market_key=preferred_market,
                bookmaker=bookmaker,
                lines=lines,
            )
        )
    return previews


def _preview_league_target_corner_odds(
    league_name: str,
    days_ahead: int,
    limit: int,
    target_point: float,
    selection: str,
) -> list[TargetCornersOddsPreview]:
    events = get_events_for_league(league_name, days_ahead=days_ahead)
    previews: list[TargetCornersOddsPreview] = []

    for event in events:
        if len(previews) >= limit:
            break
        event_id = str(event.get("id") or "")
        if not event_id:
            continue
        markets_payload = get_event_markets_for_league(league_name, event_id)
        corner_markets = _extract_corner_markets(markets_payload)
        if not corner_markets:
            continue
        preferred_market = _preferred_corner_market(corner_markets)
        bookmaker, selected_outcome = _load_target_line(
            league_name,
            event_id,
            preferred_market,
            target_point=target_point,
            selection=selection,
        )
        if selected_outcome is None:
            continue
        previews.append(
            TargetCornersOddsPreview(
                league=league_name,
                match_date=_event_match_date(event),
                home_team=str(event.get('home_team', '') or ''),
                away_team=str(event.get('away_team', '') or ''),
                event_label=f"{event.get('home_team', '')} vs {event.get('away_team', '')}",
                market_key=preferred_market,
                bookmaker=bookmaker,
                selection=selection,
                point=selected_outcome[0],
                price=selected_outcome[1],
            )
        )
    return previews


def _event_match_date(event: dict[str, object]) -> str:
    commence_time = event.get("commence_time")
    if not commence_time:
        return ""
    try:
        return str(commence_time).split("T", 1)[0]
    except Exception:
        return ""


def _extract_corner_markets(payload: dict[str, object]) -> list[str]:
    keys: set[str] = set()
    for bookmaker in payload.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            key = str(market.get("key") or "")
            if key in CORNERS_MARKET_KEYS:
                keys.add(key)
    return sorted(keys)


def _extract_corner_bookmakers(payload: dict[str, object]) -> list[str]:
    names: list[str] = []
    for bookmaker in payload.get("bookmakers", []):
        has_corner_market = any(
            str(market.get("key") or "") in CORNERS_MARKET_KEYS
            for market in bookmaker.get("markets", [])
        )
        if has_corner_market:
            title = str(bookmaker.get("title") or bookmaker.get("key") or "")
            if title:
                names.append(title)
    return names


def _preferred_corner_market(market_keys: list[str]) -> str:
    for market_key in ("alternate_totals_corners", "totals_corners", "alternate_spreads_corners"):
        if market_key in market_keys:
            return market_key
    return market_keys[0]


def _load_sample_corner_prices(
    league_name: str,
    event_id: str,
    preferred_market: str,
) -> tuple[list[float], list[float]]:
    payload = get_event_odds_for_league(league_name, event_id, markets=preferred_market)
    for bookmaker in payload.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if str(market.get("key") or "") != preferred_market:
                continue
            points = []
            prices = []
            for outcome in market.get("outcomes", []):
                point = outcome.get("point")
                price = outcome.get("price")
                if point is not None:
                    try:
                        points.append(float(point))
                    except (TypeError, ValueError):
                        pass
                if price is not None:
                    try:
                        prices.append(float(price))
                    except (TypeError, ValueError):
                        pass
            return sorted(set(points)), prices[:10]
    return [], []


def _load_preview_lines(
    league_name: str,
    event_id: str,
    preferred_market: str,
) -> tuple[str, list[str]]:
    payload = get_event_odds_for_league(league_name, event_id, markets=preferred_market)
    for bookmaker in payload.get("bookmakers", []):
        bookmaker_title = str(bookmaker.get("title") or bookmaker.get("key") or "")
        for market in bookmaker.get("markets", []):
            if str(market.get("key") or "") != preferred_market:
                continue
            lines = []
            for outcome in market.get("outcomes", [])[:8]:
                name = str(outcome.get("name") or "")
                point = outcome.get("point")
                price = outcome.get("price")
                if point is None or price is None:
                    continue
                try:
                    point_value = float(point)
                    price_value = float(price)
                except (TypeError, ValueError):
                    continue
                lines.append(f"{name} {point_value:g} @ {price_value:.2f}")
            if lines:
                return bookmaker_title, lines
    return "", []


def _load_target_line(
    league_name: str,
    event_id: str,
    preferred_market: str,
    target_point: float,
    selection: str,
) -> tuple[str, tuple[float, float] | None]:
    payload = get_event_odds_for_league(league_name, event_id, markets=preferred_market)
    desired_selection = selection.strip().lower()
    for bookmaker in payload.get("bookmakers", []):
        bookmaker_title = str(bookmaker.get("title") or bookmaker.get("key") or "")
        for market in bookmaker.get("markets", []):
            if str(market.get("key") or "") != preferred_market:
                continue
            for outcome in market.get("outcomes", []):
                if str(outcome.get("name") or "").strip().lower() != desired_selection:
                    continue
                point = outcome.get("point")
                price = outcome.get("price")
                try:
                    point_value = float(point)
                    price_value = float(price)
                except (TypeError, ValueError):
                    continue
                if point_value == float(target_point):
                    return bookmaker_title, (point_value, price_value)
    return "", None
