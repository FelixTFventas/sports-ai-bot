from sports_ai_bot.research import corners


def test_build_corners_research_report_summarizes_coverage(monkeypatch, tmp_path) -> None:
    settings = type(
        "Settings",
        (),
        {
            "the_odds_api_region": "eu",
            "the_odds_api_bookmakers_list": lambda self: ["bet365", "pinnacle"],
            "reports_dir": tmp_path,
        },
    )()

    monkeypatch.setattr(corners, "get_settings", lambda: settings)
    monkeypatch.setattr(
        corners,
        "get_events_for_league",
        lambda league_name, days_ahead=2: [
            {"id": f"{league_name}-1", "home_team": "A", "away_team": "B"},
            {"id": f"{league_name}-2", "home_team": "C", "away_team": "D"},
        ],
    )
    monkeypatch.setattr(
        corners,
        "get_event_markets_for_league",
        lambda league_name, event_id: {
            "bookmakers": [
                {
                    "title": "Pinnacle",
                    "markets": [
                        {"key": "alternate_totals_corners"} if event_id.endswith("1") else {"key": "totals"}
                    ],
                }
            ]
        },
    )
    monkeypatch.setattr(
        corners,
        "get_event_odds_for_league",
        lambda league_name, event_id, markets: {
            "bookmakers": [
                {
                    "markets": [
                        {
                            "key": markets,
                            "outcomes": [
                                {"point": 9.5, "price": 1.8},
                                {"point": 10.5, "price": 2.0},
                            ],
                        }
                    ]
                }
            ]
        },
    )

    report = corners.build_corners_research_report(days_ahead=2)

    assert report["days_ahead"] == 2
    assert len(report["leagues"]) == len(corners.TARGET_LEAGUES)
    first = report["leagues"][0]
    assert first["events_with_corners"] == 1
    assert first["coverage_pct"] == 0.5
    assert first["market_keys"] == ["alternate_totals_corners"]
    assert first["bookmakers"] == ["Pinnacle"]
    assert first["sample_points"] == [9.5, 10.5]


def test_format_corners_research_report_includes_summary() -> None:
    report = {
        "days_ahead": 2,
        "region": "eu",
        "configured_bookmakers": ["bet365", "pinnacle"],
        "leagues": [
            {
                "league": "premier_league",
                "event_count_48h": 4,
                "events_with_corners": 3,
                "coverage_pct": 0.75,
                "market_keys": ["alternate_totals_corners"],
                "bookmakers": ["Pinnacle"],
                "sample_event": "A vs B",
                "sample_points": [9.5, 10.5],
                "sample_prices": [1.8, 2.0],
            }
        ],
    }

    message = corners.format_corners_research_report(report)

    assert "premier_league: 3/4 con corners" in message
    assert "bet365, pinnacle" in message
    assert "alternate_totals_corners" in message
    assert "Pinnacle" in message


def test_build_corners_odds_preview_collects_lines(monkeypatch, tmp_path) -> None:
    settings = type(
        "Settings",
        (),
        {
            "the_odds_api_region": "eu",
            "the_odds_api_bookmakers_list": lambda self: ["bet365", "pinnacle"],
            "reports_dir": tmp_path,
        },
    )()

    monkeypatch.setattr(corners, "get_settings", lambda: settings)
    monkeypatch.setattr(
        corners,
        "get_events_for_league",
        lambda league_name, days_ahead=2: [{"id": f"{league_name}-1", "home_team": "A", "away_team": "B"}],
    )
    monkeypatch.setattr(
        corners,
        "get_event_markets_for_league",
        lambda league_name, event_id: {
            "bookmakers": [{"title": "Pinnacle", "markets": [{"key": "alternate_totals_corners"}]}]
        },
    )
    monkeypatch.setattr(
        corners,
        "get_event_odds_for_league",
        lambda league_name, event_id, markets: {
            "bookmakers": [
                {
                    "title": "Pinnacle",
                    "markets": [
                        {
                            "key": markets,
                            "outcomes": [
                                {"name": "Over", "point": 9.5, "price": 1.8},
                                {"name": "Under", "point": 9.5, "price": 2.0},
                            ],
                        }
                    ],
                }
            ]
        },
    )

    report = corners.build_corners_odds_preview(days_ahead=2, limit_per_league=1)

    assert report["limit_per_league"] == 1
    assert len(report["previews"]) == len(corners.TARGET_LEAGUES)
    assert report["previews"][0]["bookmaker"] == "Pinnacle"
    assert report["previews"][0]["lines"] == ["Over 9.5 @ 1.80", "Under 9.5 @ 2.00"]


def test_format_corners_odds_preview_includes_lines() -> None:
    report = {
        "days_ahead": 2,
        "limit_per_league": 1,
        "configured_bookmakers": ["bet365", "pinnacle"],
        "previews": [
            {
                "league": "premier_league",
                "event_label": "A vs B",
                "market_key": "alternate_totals_corners",
                "bookmaker": "Pinnacle",
                "lines": ["Over 9.5 @ 1.80", "Under 9.5 @ 2.00"],
            }
        ],
    }

    message = corners.format_corners_odds_preview(report)

    assert "Preview cuotas corners" in message
    assert "premier_league | A vs B" in message
    assert "Over 9.5 @ 1.80" in message


def test_build_target_corners_odds_preview_collects_target_line(monkeypatch, tmp_path) -> None:
    settings = type(
        "Settings",
        (),
        {
            "the_odds_api_region": "eu",
            "the_odds_api_bookmakers_list": lambda self: ["bet365", "pinnacle"],
            "reports_dir": tmp_path,
        },
    )()

    monkeypatch.setattr(corners, "get_settings", lambda: settings)
    monkeypatch.setattr(
        corners,
        "get_events_for_league",
        lambda league_name, days_ahead=2: [{"id": f"{league_name}-1", "home_team": "A", "away_team": "B"}],
    )
    monkeypatch.setattr(
        corners,
        "get_event_markets_for_league",
        lambda league_name, event_id: {
            "bookmakers": [{"title": "Pinnacle", "markets": [{"key": "alternate_totals_corners"}]}]
        },
    )
    monkeypatch.setattr(
        corners,
        "get_event_odds_for_league",
        lambda league_name, event_id, markets: {
            "bookmakers": [
                {
                    "title": "Pinnacle",
                    "markets": [
                        {
                            "key": markets,
                            "outcomes": [
                                {"name": "Over", "point": 9.5, "price": 1.8},
                                {"name": "Under", "point": 9.5, "price": 2.0},
                            ],
                        }
                    ],
                }
            ]
        },
    )

    report = corners.build_target_corners_odds_preview(target_point=9.5, selection="Over")

    assert len(report["previews"]) == len(corners.TARGET_LEAGUES)
    assert report["previews"][0]["selection"] == "Over"
    assert report["previews"][0]["point"] == 9.5
    assert report["previews"][0]["price"] == 1.8


def test_format_target_corners_odds_preview_includes_target_line() -> None:
    report = {
        "days_ahead": 2,
        "limit_per_league": 3,
        "target_point": 9.5,
        "selection": "Over",
        "configured_bookmakers": ["bet365", "pinnacle"],
        "previews": [
            {
                "league": "premier_league",
                "event_label": "A vs B",
                "market_key": "alternate_totals_corners",
                "bookmaker": "Pinnacle",
                "selection": "Over",
                "point": 9.5,
                "price": 1.8,
            }
        ],
    }

    message = corners.format_target_corners_odds_preview(report)

    assert "Preview corners objetivo" in message
    assert "Objetivo: Over 9.5" in message
    assert "premier_league | A vs B" in message
    assert "Over 9.5 @ 1.80" in message


def test_build_corners_picks_filters_by_price_and_formats_market(monkeypatch) -> None:
    monkeypatch.setattr(corners, "_load_corners_model", lambda target_point: object())
    monkeypatch.setattr(corners, "_load_corners_fixture_features", lambda: None)
    monkeypatch.setattr(
        corners,
        "_corners_model_probability",
        lambda fixture_features, model, league, match_date, home_team, away_team, selection: 0.7
        if home_team == "A"
        else 0.4,
    )
    monkeypatch.setattr(
        corners,
        "_corners_feature_summary",
        lambda fixture_features, preview: "Corners recientes local 6.00 | visitante 4.00",
    )
    monkeypatch.setattr(
        corners,
        "build_target_corners_odds_preview",
        lambda **kwargs: {
            "previews": [
                {
                    "league": "premier_league",
                    "match_date": "2026-04-20",
                    "home_team": "A",
                    "away_team": "B",
                    "event_label": "A vs B",
                    "market_key": "alternate_totals_corners",
                    "bookmaker": "Pinnacle",
                    "selection": "Over",
                    "point": 9.5,
                    "price": 1.8,
                },
                {
                    "league": "la_liga",
                    "match_date": "2026-04-20",
                    "home_team": "C",
                    "away_team": "D",
                    "event_label": "C vs D",
                    "market_key": "alternate_totals_corners",
                    "bookmaker": "Pinnacle",
                    "selection": "Over",
                    "point": 9.5,
                    "price": 2.4,
                },
            ]
        },
    )

    picks = corners.build_corners_picks(limit=6, target_point=9.5, selection="Over")

    assert len(picks) == 1
    assert picks[0].market == "Over 9.5 Corners"
    assert picks[0].odd == 1.8
    assert round(picks[0].edge or 0.0, 4) > 0
    assert picks[0].match_label == "A vs B"
    assert picks[0].match_date == "2026-04-20"
    assert picks[0].home_team == "A"
    assert picks[0].away_team == "B"
