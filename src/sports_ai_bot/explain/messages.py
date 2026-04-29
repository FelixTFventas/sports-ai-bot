from __future__ import annotations

from sports_ai_bot.predict.pipeline import Pick


def _display_market_name(market: str) -> str:
    if market == "BTTS":
        return "Ambos marcan"
    return market


def _display_pick_name(pick: Pick) -> str:
    market = _display_market_name(pick.market)
    if pick.market == "1X2" and pick.selection:
        return f"{market} {pick.selection}"
    return market


def _display_pick_tier(pick: Pick) -> str:
    return "Premium" if pick.probability >= 0.65 else "Standard"


def build_prediction_message(picks: list[Pick]) -> str:
    if not picks:
        return "No hay picks con confianza suficiente para hoy."

    lines = ["Picks del dia:"]
    for pick in picks:
        details = [
            pick.match_label,
            _display_pick_name(pick),
            f"Prob {pick.probability:.0%}",
        ]
        if pick.odd is not None:
            details.append(f"Cuota {pick.odd:.2f}")
        if pick.edge is not None:
            details.append(f"Edge {pick.edge:.1%}")
        if pick.expected_value is not None:
            details.append(f"EV {pick.expected_value:.1%}")
        details.append(_display_pick_tier(pick))
        details.append(pick.confidence)
        if pick.is_experimental:
            details.append("Experimental")
        lines.append(" | ".join(details))
    lines.append("Pronosticos basados en modelo estadistico, sin garantia de resultado.")
    return "\n".join(lines)


def build_market_message(picks: list[Pick], market: str) -> str:
    filtered = [pick for pick in picks if pick.market.lower() == market.lower()]
    if not filtered:
        return (
            f"No hay picks disponibles para {_display_market_name(market)} con la confianza actual."
        )
    return build_prediction_message(filtered)


def build_value_message(picks: list[Pick]) -> str:
    if not picks:
        return "No hay value picks disponibles con el edge minimo actual."

    lines = ["Value picks del dia:"]
    for pick in picks:
        stake = f"{pick.stake_units}u" if pick.stake_units is not None else "n/d"
        rating = pick.rating or "n/d"
        odd = f"{pick.odd:.2f}" if pick.odd is not None else "n/d"
        edge = f"{pick.edge:.1%}" if pick.edge is not None else "n/d"
        ev = f"{pick.expected_value:.1%}" if pick.expected_value is not None else "n/d"
        lines.append(
            f"{rating} | {stake} | {pick.match_label} | {_display_pick_name(pick)} | {_display_pick_tier(pick)} | Prob {pick.probability:.0%} | Cuota {odd} | Edge {edge} | EV {ev}"
        )
    lines.append("Value = probabilidad del modelo por encima de la implicita de la cuota.")
    return "\n".join(lines)


def build_forebet_value_message(picks: list[Pick]) -> str:
    if not picks:
        return "No hay eventos Forebet con cuota suficiente para analizar."

    lines = ["🎯 Forebet Value Picks", ""]
    for index, pick in enumerate(picks, start=1):
        odd = f"{pick.odd:.2f}" if pick.odd is not None else "n/d"
        lines.extend(
            [
                f"{index}. {_forebet_value_icon(pick)} {pick.match_label}",
                f"📌 Pick: {_display_forebet_pick_name(pick)}",
                f"📊 Probabilidad: {pick.probability:.1%}",
                f"💰 Cuota: {odd}",
                f"📈 Edge: {_signed_percent(pick.edge)}",
                f"🔥 EV: {_signed_percent(pick.expected_value)}",
            ]
        )
        if index < len(picks):
            lines.append("")
    lines.extend(
        [
            "",
            "📐 Modelo: 60% Forebet + 40% mercado",
            "⚠️ Pronosticos estadisticos, sin garantia de resultado.",
        ]
    )
    return "\n".join(lines)


def build_best_message(picks: list[Pick]) -> str:
    if not picks:
        return "No hay best picks disponibles con el criterio premium actual."

    lines = ["Best picks del dia:"]
    for pick in picks:
        stake = f"{pick.stake_units}u" if pick.stake_units is not None else "n/d"
        rating = pick.rating or "n/d"
        odd = f"{pick.odd:.2f}" if pick.odd is not None else "n/d"
        edge = f"{pick.edge:.1%}" if pick.edge is not None else "n/d"
        lines.append(
            f"{rating} | {stake} | {pick.match_label} | {_display_pick_name(pick)} | {_display_pick_tier(pick)} | Cuota {odd} | Edge {edge}"
        )
    return "\n".join(lines)


def _display_forebet_pick_name(pick: Pick) -> str:
    market = _display_market_name(pick.market)
    if pick.selection:
        return f"{market} {pick.selection}"
    return market


def _signed_percent(value: float | None) -> str:
    if value is None:
        return "n/d"
    return f"{value:+.1%}"


def _forebet_value_icon(pick: Pick) -> str:
    edge = pick.edge or 0.0
    if edge >= 0.10:
        return "🟢"
    if edge >= 0.03:
        return "🟡"
    return "🔴"
