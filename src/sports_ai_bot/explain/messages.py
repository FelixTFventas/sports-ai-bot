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
