import json
from types import SimpleNamespace

import pandas as pd
import pytest

from sports_ai_bot.evaluate import performance


@pytest.fixture
def storage(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        predictions_dir=tmp_path / "predictions",
        raw_dir=tmp_path / "raw",
        reports_dir=tmp_path / "reports",
    )
    settings.predictions_dir.mkdir()
    settings.raw_dir.mkdir()
    monkeypatch.setattr(performance, "get_settings", lambda: settings)
    return settings


def pick(**changes):
    return {
        "prediction_date": "2026-09-10", "match_date": "2026-09-12",
        "home_team": "A", "away_team": "B", "league": "E0",
        "market": "BTTS", "selection": "Si", "line": None,
        "odd": 2.0, "stake_units": 2, "is_experimental": False,
        "status": "pending", "outcome": "pending", **changes,
    }


def save_results(storage, **changes):
    pd.DataFrame([{
        "Date": "12/09/2026", "HomeTeam": "A", "AwayTeam": "B",
        "FTHG": 2, "FTAG": 1, **changes,
    }]).to_csv(storage.raw_dir / "E0_2627.csv", index=False)


@pytest.mark.parametrize("odd", [None, "invalid", 0, 1, -2, float("inf")])
def test_missing_or_invalid_odds_do_not_create_profit_or_risk(storage, odd):
    pd.DataFrame([
        pick(odd=odd), pick(odd=odd, selection="No"),
        pick(market="Over 2.5", selection=None, odd=2.5),
    ]).to_csv(storage.predictions_dir / "picks_2026-09-10.csv", index=False)
    save_results(storage)
    summary = performance.build_performance_report()["summary"]
    assert summary["wins"] == 2
    assert summary["losses"] == 1
    assert summary["total_profit"] == 3
    assert summary["risked_units"] == 2
    assert summary["roi"] == summary["yield"] == 1.5
    assert summary["priced_settled"] == 1
    assert summary["unpriced_settled"] == 2


@pytest.mark.parametrize("selection,goals,outcome", [
    ("Si", (1, 1), "won"), ("S\u00ed", (1, 0), "lost"),
    ("No", (1, 0), "won"), ("No", (1, 1), "lost"),
    ("unknown", (1, 1), "pending"), (None, (1, 1), "won"),
])
def test_btts_selection(storage, selection, goals, outcome):
    path = storage.predictions_dir / "picks_btts.csv"
    pd.DataFrame([pick(selection=selection)]).to_csv(path, index=False)
    save_results(storage, FTHG=goals[0], FTAG=goals[1])
    performance.settle_picks()
    assert pd.read_csv(path).iloc[0]["outcome"] == outcome


def test_dedupe_across_days_keeps_first_quote_and_selection_line_identity(storage):
    first = [pick(), pick(selection="No"), pick(market="Under 2.5", line=2.5)]
    pd.DataFrame(first).to_csv(storage.predictions_dir / "picks_2026-09-10.csv", index=False)
    pd.DataFrame([
        pick(prediction_date="2026-09-11", odd=9),
        pick(market="Under 2.5", line=3.5),
    ]).to_csv(storage.predictions_dir / "picks_2026-09-11.csv", index=False)
    save_results(storage)
    report = performance.build_performance_report()
    assert report["summary"]["total"] == 4
    assert report["quality"]["duplicate_rows"] == 1
    unique = performance._dedupe_picks(performance._load_all_pick_files().iloc[::-1])
    assert unique[(unique.market == "BTTS") & (unique.selection == "Si")].iloc[0].odd == 2
    assert performance.build_performance_report() == report


@pytest.mark.parametrize("market,corners,outcome", [
    ("Over 9.5 Corners", {"HC": 6, "AC": 4}, "won"),
    ("Under 9.5 Corners", {"HC": 6, "AC": 4}, "lost"),
    ("Under 9.5 Corners", {"HC": 0, "AC": 0}, "won"),
    ("Over 9.5 Corners", {"HC": 6}, "pending"),
    ("Over 9.5 Corners", {}, "pending"),
    ("Under 2.5", {}, "lost"),
    ("Under 3.5", {}, "won"),
    ("Under 3", {}, "pending"),
    ("Under 3.25", {}, "pending"),
    ("unknown", {}, "pending"),
])
def test_totals_and_corners_settlement(storage, market, corners, outcome):
    path = storage.predictions_dir / "picks_totals.csv"
    pd.DataFrame([pick(market=market, selection=None)]).to_csv(path, index=False)
    save_results(storage, **corners)
    performance.settle_picks()
    row = pd.read_csv(path).iloc[0]
    assert row.outcome == outcome
    assert row.status == ("pending" if outcome == "pending" else "settled")


def test_strategy_report_separates_experiments_and_unknown_flags(storage):
    pd.DataFrame([
        pick(model_version="model-v2", quote_id="quote-1", source="manual"),
        pick(selection="No", is_experimental="True"),
        pick(market="Under 2.5", is_experimental=None),
    ]).to_csv(storage.predictions_dir / "picks_strategies.csv", index=False)
    save_results(storage)
    report = performance.build_performance_report()
    assert report["summary_scope"] == "validated"
    assert report["summary"]["total"] == 1
    assert report["summary"]["total_profit"] == 2
    assert report["by_market"]["BTTS"]["losses"] == 0
    assert report["by_strategy"]["experimental"]["losses"] == 1
    assert report["by_strategy"]["unclassified"]["losses"] == 1
    assert report["quality"]["duplicate_rows"] == 0
    assert json.loads((storage.reports_dir / "performance_summary.json").read_text()) == report
    message = performance.format_performance_message(report)
    assert "Estrategia experimental" in message
    assert "fuera de ROI" in message


def test_experimental_only_report_is_not_reported_as_empty(storage):
    pd.DataFrame([pick(is_experimental=True)]).to_csv(
        storage.predictions_dir / "picks_experimental.csv", index=False
    )
    report = performance.build_performance_report()
    assert report["summary"]["total"] == 0
    assert report["by_strategy"]["experimental"]["pending"] == 1
    assert "Estrategia experimental" in performance.format_performance_message(report)


def test_legacy_without_flag_is_unclassified(storage):
    pd.DataFrame([pick()]).drop(columns="is_experimental").to_csv(
        storage.predictions_dir / "picks_legacy.csv", index=False
    )
    save_results(storage)
    report = performance.build_performance_report()
    assert report["summary_scope"] == "unclassified"
    assert report["summary"]["wins"] == 1


def test_legacy_false_flag_does_not_imply_validation(storage):
    pd.DataFrame([pick(is_experimental=False)]).to_csv(
        storage.predictions_dir / "picks_old.csv", index=False
    )
    save_results(storage)
    report = performance.build_performance_report()
    assert report["summary_scope"] == "unclassified"
    assert "validated" not in report["by_strategy"]


def test_empty_report_public_contract(storage):
    assert performance.settle_picks() == {"files": 0, "settled": 0, "pending": 0}
    report = performance.build_performance_report()
    assert report["summary"]["total"] == 0
    assert "no hay picks" in performance.format_performance_message(report)


def test_first_missing_quote_is_not_replaced_by_later_quote(storage):
    pd.DataFrame([pick(odd=None)]).to_csv(
        storage.predictions_dir / "picks_2026-09-10.csv", index=False
    )
    pd.DataFrame([pick(prediction_date="2026-09-11", odd=5)]).to_csv(
        storage.predictions_dir / "picks_2026-09-11.csv", index=False
    )
    save_results(storage)
    report = performance.build_performance_report()
    assert report["summary"]["wins"] == 1
    assert report["summary"]["total_profit"] == 0
    assert report["summary"]["risked_units"] == 0
    assert report["summary"]["roi"] == 0
    assert report["summary"]["unpriced_settled"] == 1
    assert report["quality"]["with_odds"] == 0
