from __future__ import annotations

from sports_ai_bot.predict.pipeline import Pick


def build_prediction_message(picks: list[Pick]) -> str:
    if not picks:
        return "No hay picks con confianza suficiente para hoy."

    lines = ["Picks del dia:"]
    for pick in picks:
        lines.append(
            f"{pick.match_label} | {pick.market} | {pick.probability:.0%} | {pick.confidence}"
        )
    lines.append("Pronosticos basados en modelo estadistico, sin garantia de resultado.")
    return "\n".join(lines)


def build_market_message(picks: list[Pick], market: str) -> str:
    filtered = [pick for pick in picks if pick.market.lower() == market.lower()]
    if not filtered:
        return f"No hay picks disponibles para {market} con la confianza actual."
    return build_prediction_message(filtered)


def build_value_message(picks: list[Pick]) -> str:
    if not picks:
        return "No hay value picks disponibles con el edge minimo actual."

    lines = ["Value picks del dia:"]
    for pick in picks:
        odd = f"{pick.odd:.2f}" if pick.odd is not None else "n/d"
        edge = f"{pick.edge:.1%}" if pick.edge is not None else "n/d"
        ev = f"{pick.expected_value:.1%}" if pick.expected_value is not None else "n/d"
        lines.append(
            f"{pick.match_label} | {pick.market} | Prob {pick.probability:.0%} | Cuota {odd} | Edge {edge} | EV {ev}"
        )
    lines.append("Value = probabilidad del modelo por encima de la implicita de la cuota.")
    return "\n".join(lines)
