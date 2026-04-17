from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from sports_ai_bot.utils.config import get_settings


PICK_COLUMNS = [
    "prediction_date",
    "match_date",
    "home_team",
    "away_team",
    "match_label",
    "league",
    "market",
    "probability",
    "confidence",
    "model_name",
    "odd",
    "implied_probability",
    "edge",
    "expected_value",
    "stake_units",
    "rating",
    "factors",
    "status",
    "outcome",
    "result_home_goals",
    "result_away_goals",
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
    all_picks = _load_all_pick_files()
    if all_picks.empty:
        report = {
            "summary": {
                "total": 0,
                "settled": 0,
                "pending": 0,
                "wins": 0,
                "losses": 0,
                "hit_rate": 0.0,
            },
            "by_market": {},
            "by_league": {},
            "by_probability_bucket": {},
        }
        _write_report(report, settings.reports_dir / "performance_summary.json")
        return report

    settled = all_picks[all_picks["status"] == "settled"].copy()
    settled["probability_bucket"] = settled["probability"].apply(_probability_bucket)

    report = {
        "summary": _summarize_frame(all_picks),
        "by_market": _group_summary(settled, "market"),
        "by_league": _group_summary(settled, "league"),
        "by_probability_bucket": _group_summary(settled, "probability_bucket"),
    }
    _write_report(report, settings.reports_dir / "performance_summary.json")
    return report


def format_performance_message(report: dict[str, object]) -> str:
    summary = report["summary"]
    if summary["total"] == 0:
        return "Todavia no hay picks guardados para medir rendimiento."

    lines = [
        "Rendimiento historico:",
        (
            f"Total: {summary['total']} | Settled: {summary['settled']} | Pending: {summary['pending']} | "
            f"Wins: {summary['wins']} | Losses: {summary['losses']} | Hit rate: {summary['hit_rate']:.1%}"
        ),
    ]

    market_rows = _top_group_lines(report["by_market"])
    if market_rows:
        lines.append("Por mercado:")
        lines.extend(market_rows)

    league_rows = _top_group_lines(report["by_league"], limit=3)
    if league_rows:
        lines.append("Por liga:")
        lines.extend(league_rows)

    return "\n".join(lines)


def _read_pick_file(file_path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(file_path)
    except EmptyDataError:
        return pd.DataFrame(columns=PICK_COLUMNS)
    for column in PICK_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[PICK_COLUMNS].copy()


def _load_all_pick_files() -> pd.DataFrame:
    settings = get_settings()
    pick_files = sorted(settings.predictions_dir.glob("picks_*.csv"))
    frames = [_read_pick_file(file_path) for file_path in pick_files]
    if not frames:
        return pd.DataFrame(columns=PICK_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _load_completed_results() -> pd.DataFrame:
    settings = get_settings()
    frames: list[pd.DataFrame] = []
    for file_path in settings.raw_dir.glob("*.csv"):
        frame = pd.read_csv(file_path)
        required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        if not required.issubset(frame.columns):
            continue
        league = file_path.stem.rsplit("_", 1)[0]
        completed = frame.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
        if completed.empty:
            continue
        completed["match_date"] = pd.to_datetime(
            completed["Date"], dayfirst=True, errors="coerce"
        ).dt.date
        completed["league"] = league
        completed = completed.dropna(subset=["match_date"])
        completed["home_team"] = completed["HomeTeam"]
        completed["away_team"] = completed["AwayTeam"]
        completed["result_home_goals"] = pd.to_numeric(completed["FTHG"], errors="coerce")
        completed["result_away_goals"] = pd.to_numeric(completed["FTAG"], errors="coerce")
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
    frame = frame.dropna(subset=["match_date", "league", "home_team", "away_team"])
    if results.empty:
        frame["status"] = frame["status"].fillna("pending")
        frame["outcome"] = frame["outcome"].fillna("pending")
        return frame.reindex(columns=PICK_COLUMNS)

    merged = frame.drop(columns=["result_home_goals", "result_away_goals"], errors="ignore").merge(
        results,
        on=["match_date", "league", "home_team", "away_team"],
        how="left",
    )

    merged["outcome"] = merged.apply(
        lambda row: _determine_outcome(
            row["market"], row["result_home_goals"], row["result_away_goals"]
        ),
        axis=1,
    )
    merged["status"] = merged["outcome"].apply(
        lambda outcome: "settled" if outcome in {"won", "lost"} else "pending"
    )
    return merged.reindex(columns=PICK_COLUMNS)


def _determine_outcome(market: str, home_goals: float | None, away_goals: float | None) -> str:
    if pd.isna(home_goals) or pd.isna(away_goals):
        return "pending"
    if market == "Over 2.5":
        return "won" if float(home_goals) + float(away_goals) > 2.5 else "lost"
    if market == "BTTS":
        return "won" if float(home_goals) > 0 and float(away_goals) > 0 else "lost"
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


def _summarize_frame(frame: pd.DataFrame) -> dict[str, object]:
    settled = frame[frame["status"] == "settled"]
    wins = int((settled["outcome"] == "won").sum())
    losses = int((settled["outcome"] == "lost").sum())
    total_settled = wins + losses
    return {
        "total": int(len(frame)),
        "settled": total_settled,
        "pending": int((frame["status"] == "pending").sum()),
        "wins": wins,
        "losses": losses,
        "hit_rate": round(wins / total_settled, 4) if total_settled else 0.0,
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
