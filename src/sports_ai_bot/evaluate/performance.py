from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from sports_ai_bot.utils.config import get_settings
from sports_ai_bot.features.build import _parse_source_dates
from sports_ai_bot.utils.team_names import canonical_team_name


PICK_COLUMNS = [
    "prediction_date",
    "match_date",
    "home_team",
    "away_team",
    "match_label",
    "league",
    "market",
    "selection",
    "line",
    "probability",
    "confidence",
    "model_name",
    "model_version",
    "event_id",
    "quote_id",
    "kickoff",
    "quoted_at",
    "source_url",
    "source",
    "generated_at",
    "bookmaker",
    "odd",
    "implied_probability",
    "edge",
    "expected_value",
    "stake_units",
    "rating",
    "score",
    "is_experimental",
    "factors",
    "status",
    "outcome",
    "result_home_goals",
    "result_away_goals",
    "result_home_corners",
    "result_away_corners",
]


def settle_picks() -> dict[str, int]:
    settings = get_settings()
    pick_files = sorted(settings.predictions_dir.glob("picks_*.csv"))
    if not pick_files:
        return {"files": 0, "settled": 0, "pending": 0}

    results = _load_completed_results()
    settled_count = 0
    pending_count = 0

    for pick_file in pick_files:
        picks = _read_pick_file(pick_file)
        if picks.empty:
            continue

        updated = _settle_pick_frame(picks, results)
        settled_count += int((updated["status"] == "settled").sum())
        pending_count += int((updated["status"] == "pending").sum())
        updated.to_csv(pick_file, index=False)

    return {"files": len(pick_files), "settled": settled_count, "pending": pending_count}


def build_performance_report() -> dict[str, object]:
    settings = get_settings()
    settle_picks()
    all_picks = _dedupe_picks(_load_all_pick_files())
    if all_picks.empty:
        report = {
            "summary": {
                "total": 0,
                "settled": 0,
                "pending": 0,
                "wins": 0,
                "losses": 0,
                "hit_rate": 0.0,
                "total_profit": 0.0,
                "risked_units": 0.0,
                "priced_settled": 0,
                "unpriced_settled": 0,
                "roi": 0.0,
                "yield": 0.0,
            },
            "by_market": {},
            "by_league": {},
            "by_probability_bucket": {},
            "by_odd_bucket": {},
            "by_edge_bucket": {},
            "by_ev_bucket": {},
            "by_rating": {},
            "by_strategy": {},
            "summary_scope": "unclassified",
            "quality": {},
        }
        _write_report(report, settings.reports_dir / "performance_summary.json")
        return report

    all_picks = _attach_analysis_buckets(all_picks)
    flags = all_picks["is_experimental"].astype("string").str.strip().str.lower()
    all_picks["strategy"] = flags.map({
        "true": "experimental", "1": "experimental", "1.0": "experimental",
        "false": "validated", "0": "validated", "0.0": "validated",
    }).fillna("unclassified")
    traceable = (
        all_picks["model_version"].notna() & all_picks["quote_id"].notna()
        & all_picks["source"].eq("manual")
    )
    all_picks.loc[all_picks["strategy"].eq("validated") & ~traceable, "strategy"] = "unclassified"
    # Legacy files without flags remain measurable, but never labelled validated.
    scope = (
        "validated" if all_picks["strategy"].ne("unclassified").any() else "unclassified"
    )
    primary = all_picks[all_picks["strategy"] == scope]
    settled = primary[primary["status"] == "settled"]

    report = {
        "summary": _summarize_frame(primary),
        "summary_scope": scope,
        "by_strategy": _group_summary(all_picks, "strategy"),
        "by_model": _group_summary(all_picks.fillna({"model_version": "unclassified"}), "model_version"),
        "by_market": _group_summary(settled, "market"),
        "by_league": _group_summary(settled, "league"),
        "by_probability_bucket": _group_summary(settled, "probability_bucket"),
        "by_odd_bucket": _group_summary(settled, "odd_bucket"),
        "by_edge_bucket": _group_summary(settled, "edge_bucket"),
        "by_ev_bucket": _group_summary(settled, "ev_bucket"),
        "by_rating": _group_summary(settled, "rating"),
        "quality": _quality_summary(all_picks),
    }
    _write_report(report, settings.reports_dir / "performance_summary.json")
    return report


def format_performance_message(report: dict[str, object]) -> str:
    summary = report["summary"]
    if summary["total"] == 0 and not report.get("by_strategy"):
        return "Todavia no hay picks guardados para medir rendimiento."

    lines = [
        "Rendimiento de predicciones registradas (no acredita apuestas ni envios):",
        f"Alcance: {report.get('summary_scope', 'historico')}",
        (
            f"Total: {summary['total']} | Settled: {summary['settled']} | Pending: {summary['pending']} | "
            f"Wins: {summary['wins']} | Losses: {summary['losses']} | Hit rate: {summary['hit_rate']:.1%} | "
            f"Profit: {summary['total_profit']:+.2f}u | ROI: {summary['roi']:.1%} | Yield: {summary['yield']:.1%}"
        ),
    ]
    if "priced_settled" in summary:
        lines.append(
            f"Liquidaciones con cuota valida: {summary['priced_settled']} | "
            f"Sin cuota valida (fuera de ROI): {summary['unpriced_settled']}"
        )
    for strategy, values in report.get("by_strategy", {}).items():
        lines.append(
            f"Estrategia {strategy}: {values['total']} picks | "
            f"{values['wins']}-{values['losses']} | Pending: {values['pending']} | "
            f"ROI: {values['roi']:.1%}"
        )

    market_rows = _top_group_lines(report["by_market"])
    if market_rows:
        lines.append("Por mercado:")
        lines.extend(market_rows)

    league_rows = _top_group_lines(report["by_league"], limit=3)
    if league_rows:
        lines.append("Por liga:")
        lines.extend(league_rows)

    quality = report.get("quality", {})
    if quality:
        lines.append("Calidad de picks:")
        lines.append(
            f"Unicos: {quality['unique_picks']} | Duplicados ignorados: {quality['duplicate_rows']} | "
            f"Cuota media: {quality['avg_odd']:.2f} | Edge medio: {quality['avg_edge']:.1%} | EV medio: {quality['avg_expected_value']:.1%}"
        )
        lines.append(
            f"Con cuota: {quality['with_odds']} | Con value positivo: {quality['positive_ev']} | "
            f"Pendientes: {quality['pending']}"
        )

    return "\n".join(lines)


def _read_pick_file(file_path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(file_path)
    except EmptyDataError:
        return pd.DataFrame(columns=PICK_COLUMNS)
    for column in PICK_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame.copy()


def _load_all_pick_files() -> pd.DataFrame:
    settings = get_settings()
    pick_files = sorted(settings.predictions_dir.glob("picks_*.csv"))
    frames = [_read_pick_file(file_path) for file_path in pick_files]
    if not frames:
        return pd.DataFrame(columns=PICK_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _dedupe_picks(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    key_columns = [
        "match_date",
        "home_team",
        "away_team",
        "league",
        "market",
        "selection",
        "line",
    ]
    frame = frame.copy()
    for column in key_columns:
        if column not in frame.columns:
            frame[column] = None
    for column in ("model_version", "is_experimental"):
        if column in frame:
            key_columns.append(column)
    if "prediction_date" in frame:
        frame = frame.sort_values("prediction_date", kind="stable", na_position="last")
    return frame.drop_duplicates(subset=key_columns, keep="first").copy()


def _load_completed_results() -> pd.DataFrame:
    settings = get_settings()
    frames: list[pd.DataFrame] = []
    for file_path in settings.raw_dir.glob("*.csv"):
        frame = pd.read_csv(file_path)
        frame = frame.rename(
            columns={
                "Home": "HomeTeam",
                "Away": "AwayTeam",
                "HG": "FTHG",
                "AG": "FTAG",
            }
        )
        required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        if not required.issubset(frame.columns):
            continue
        league = file_path.stem.rsplit("_", 1)[0]
        completed = frame.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
        if completed.empty:
            continue
        completed["match_date"] = _parse_source_dates(completed["Date"]).dt.date
        completed["league"] = league
        completed = completed.dropna(subset=["match_date"])
        completed["home_team"] = completed["HomeTeam"].map(lambda x: canonical_team_name(league, x) or x)
        completed["away_team"] = completed["AwayTeam"].map(lambda x: canonical_team_name(league, x) or x)
        completed["result_home_goals"] = pd.to_numeric(completed["FTHG"], errors="coerce")
        completed["result_away_goals"] = pd.to_numeric(completed["FTAG"], errors="coerce")
        completed["result_home_corners"] = pd.to_numeric(completed.get("HC"), errors="coerce")
        completed["result_away_corners"] = pd.to_numeric(completed.get("AC"), errors="coerce")
        completed = completed.dropna(subset=["result_home_goals", "result_away_goals"])
        frames.append(
            completed[
                [
                    "match_date",
                    "league",
                    "home_team",
                    "away_team",
                    "result_home_goals",
                    "result_away_goals",
                    "result_home_corners",
                    "result_away_corners",
                ]
            ]
        )

    if not frames:
        return pd.DataFrame(
            columns=[
                "match_date",
                "league",
                "home_team",
                "away_team",
                "result_home_goals",
                "result_away_goals",
            ]
        )

    results = pd.concat(frames, ignore_index=True)
    return results.drop_duplicates(
        subset=["match_date", "league", "home_team", "away_team"], keep="last"
    )


def _settle_pick_frame(picks: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    frame = picks.copy()
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce").dt.date
    frame["_original_match_date"] = frame["match_date"]
    if "kickoff" in frame:
        utc_day = pd.to_datetime(frame["kickoff"], errors="coerce", utc=True, format="mixed")
        present = utc_day.notna()
        frame.loc[present, "match_date"] = utc_day.loc[present].dt.date
    frame = frame.dropna(subset=["match_date", "league", "home_team", "away_team"])
    if results.empty:
        frame["status"] = frame["status"].fillna("pending")
        frame["outcome"] = frame["outcome"].fillna("pending")
        frame["match_date"] = frame.pop("_original_match_date")
        return frame

    merged = frame.drop(columns=[
        "result_home_goals", "result_away_goals", "result_home_corners", "result_away_corners",
    ], errors="ignore").merge(
        results,
        on=["match_date", "league", "home_team", "away_team"],
        how="left",
    )

    merged["outcome"] = merged.apply(
        lambda row: _determine_outcome(
            row["market"],
            row["result_home_goals"],
            row["result_away_goals"],
            row.get("selection"),
            row.get("line"),
            row.get("result_home_corners"),
            row.get("result_away_corners"),
        ),
        axis=1,
    )
    merged["status"] = merged["outcome"].apply(
        lambda outcome: "settled" if outcome in {"won", "lost"} else "pending"
    )
    merged["match_date"] = merged.pop("_original_match_date")
    return merged


def _determine_outcome(
    market: str,
    home_goals: float | None,
    away_goals: float | None,
    selection: str | None = None,
    line: float | None = None,
    home_corners: float | None = None,
    away_corners: float | None = None,
) -> str:
    total_market = re.fullmatch(r"(Corners )?(Over|Under) (\d+(?:\.\d+)?)( Corners)?", str(market))
    if total_market:
        threshold = float(total_market[3])
        supplied_line = pd.to_numeric(line, errors="coerce")
        # Integer/quarter lines need push/half-win accounting, not a binary result.
        if threshold % 1 != 0.5 or (pd.notna(supplied_line) and supplied_line != threshold):
            return "pending"
        scores = (
            (home_corners, away_corners)
            if total_market[1] or total_market[4] else (home_goals, away_goals)
        )
        values = [pd.to_numeric(value, errors="coerce") for value in scores]
        if any(pd.isna(value) or not math.isfinite(value) or value < 0 for value in values):
            return "pending"
        total = sum(values)
        won = total > threshold if total_market[2] == "Over" else total < threshold
        return "won" if won else "lost"
    if pd.isna(home_goals) or pd.isna(away_goals):
        return "pending"
    if market == "BTTS":
        selected = "si" if pd.isna(selection) else str(selection).strip().lower()
        if selected not in {"si", "s\u00ed", "no"}:
            return "pending"
        both_scored = float(home_goals) > 0 and float(away_goals) > 0
        return "won" if both_scored == (selected != "no") else "lost"
    if market == "1X2":
        if float(home_goals) > float(away_goals):
            result = "Local"
        elif float(home_goals) < float(away_goals):
            result = "Visitante"
        else:
            result = "Empate"
        return "won" if selection == result else "lost"
    return "pending"


def _probability_bucket(probability: float) -> str:
    value = float(probability)
    if value >= 0.70:
        return "0.70+"
    if value >= 0.65:
        return "0.65-0.69"
    if value >= 0.60:
        return "0.60-0.64"
    return "<0.60"


def _odd_bucket(odd: float | None) -> str:
    value = pd.to_numeric(odd, errors="coerce")
    if pd.isna(value) or not math.isfinite(value) or value <= 1:
        return "sin cuota"
    value = float(value)
    if value < 1.50:
        return "<1.50"
    if value < 1.80:
        return "1.50-1.79"
    if value < 2.10:
        return "1.80-2.09"
    if value < 2.50:
        return "2.10-2.49"
    return "2.50+"


def _edge_bucket(edge: float | None) -> str:
    value = pd.to_numeric(edge, errors="coerce")
    if pd.isna(value):
        return "sin edge"
    value = float(value)
    if value < 0.02:
        return "<2%"
    if value < 0.05:
        return "2%-4.9%"
    if value < 0.10:
        return "5%-9.9%"
    return "10%+"


def _ev_bucket(expected_value: float | None) -> str:
    value = pd.to_numeric(expected_value, errors="coerce")
    if pd.isna(value):
        return "sin EV"
    value = float(value)
    if value < 0.0:
        return "negativo"
    if value < 0.03:
        return "0%-2.9%"
    if value < 0.08:
        return "3%-7.9%"
    return "8%+"


def _attach_analysis_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    working["probability"] = pd.to_numeric(working["probability"], errors="coerce").fillna(0.0)
    working["probability_bucket"] = working["probability"].apply(_probability_bucket)
    working["odd_bucket"] = working["odd"].apply(_odd_bucket)
    working["edge_bucket"] = working["edge"].apply(_edge_bucket)
    working["ev_bucket"] = working["expected_value"].apply(_ev_bucket)
    working["rating"] = working["rating"].fillna("sin rating")
    return working


def _summarize_frame(frame: pd.DataFrame) -> dict[str, object]:
    settled = frame[(frame["status"] == "settled") & frame["outcome"].isin(["won", "lost"])]
    wins = int((settled["outcome"] == "won").sum())
    losses = int((settled["outcome"] == "lost").sum())
    total_settled = wins + losses
    priced_settled = int((settled.apply(_risk_units, axis=1) > 0).sum()) if not settled.empty else 0
    total_profit = round(float(settled.apply(_profit_units, axis=1).sum()), 4) if not settled.empty else 0.0
    risked_units = round(float(settled.apply(_risk_units, axis=1).sum()), 4) if not settled.empty else 0.0
    return {
        "total": int(len(frame)),
        "settled": total_settled,
        "pending": int((frame["status"] == "pending").sum()),
        "wins": wins,
        "losses": losses,
        "hit_rate": round(wins / total_settled, 4) if total_settled else 0.0,
        "total_profit": total_profit,
        "risked_units": risked_units,
        "priced_settled": priced_settled,
        "unpriced_settled": total_settled - priced_settled,
        "roi": round(total_profit / risked_units, 4) if risked_units else 0.0,
        "yield": round(total_profit / risked_units, 4) if risked_units else 0.0,
    }


def _risk_units(row: pd.Series) -> float:
    odd = pd.to_numeric(row.get("odd"), errors="coerce")
    if pd.isna(odd) or not math.isfinite(odd) or odd <= 1:
        return 0.0
    stake = pd.to_numeric(row.get("stake_units"), errors="coerce")
    if pd.isna(stake) or not math.isfinite(stake) or float(stake) <= 0:
        return 1.0
    return float(stake)


def _profit_units(row: pd.Series) -> float:
    outcome = row.get("outcome")
    odd = pd.to_numeric(row.get("odd"), errors="coerce")
    stake = _risk_units(row)
    if not stake:
        return 0.0
    if outcome == "won":
        return stake * float(odd - 1.0)
    if outcome == "lost":
        return -stake
    return 0.0


def _quality_summary(frame: pd.DataFrame) -> dict[str, object]:
    raw = _load_all_pick_files()
    unique_count = int(len(frame))
    duplicate_rows = max(0, int(len(raw) - unique_count))
    odd = pd.to_numeric(frame["odd"], errors="coerce")
    odd = odd.where((odd > 1) & (odd < float("inf")))
    edge = pd.to_numeric(frame["edge"], errors="coerce")
    expected_value = pd.to_numeric(frame["expected_value"], errors="coerce")
    return {
        "total_rows": int(len(raw)),
        "unique_picks": unique_count,
        "duplicate_rows": duplicate_rows,
        "pending": int((frame["status"] == "pending").sum()),
        "settled": int((frame["status"] == "settled").sum()),
        "with_odds": int(odd.notna().sum()),
        "positive_ev": int((expected_value > 0).sum()),
        "avg_odd": round(float(odd.mean()), 4) if odd.notna().any() else 0.0,
        "avg_edge": round(float(edge.mean()), 4) if edge.notna().any() else 0.0,
        "avg_expected_value": round(float(expected_value.mean()), 4)
        if expected_value.notna().any()
        else 0.0,
    }


def _group_summary(frame: pd.DataFrame, column: str) -> dict[str, dict[str, object]]:
    if frame.empty:
        return {}

    grouped: dict[str, dict[str, object]] = {}
    for group_value, group_frame in frame.groupby(column):
        grouped[str(group_value)] = _summarize_frame(group_frame)
    return grouped


def _top_group_lines(groups: dict[str, dict[str, object]], limit: int = 5) -> list[str]:
    items = sorted(groups.items(), key=lambda item: item[1]["settled"], reverse=True)
    lines: list[str] = []
    for name, summary in items[:limit]:
        lines.append(
            f"{name}: {summary['wins']}-{summary['losses']} | {summary['hit_rate']:.1%} hit rate"
        )
    return lines


def _write_report(report: dict[str, object], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
