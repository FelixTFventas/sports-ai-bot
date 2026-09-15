import sys

import pytest

from sports_ai_bot import main
from sports_ai_bot.utils.config import Settings


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, DATA_DIR=tmp_path, TELEGRAM_BOT_TOKEN="secret-test")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    return settings


@pytest.mark.parametrize("command,expected", [
    ("coverage", "integracion automatica colombiana NO verificada"),
    ("check-config", "quote_max_age_minutes"),
])
def test_cli_local_does_not_expose_tokens(isolated, monkeypatch, capsys, command, expected):
    monkeypatch.setattr(sys, "argv", ["sports-ai-bot", command])
    main.main()
    text = capsys.readouterr().out
    assert expected in text
    assert "secret-test" not in text


def test_cli_observation_is_explicit(isolated, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sports-ai-bot", "generate", "--observation"])
    calls = []
    monkeypatch.setattr(main, "generate_picks", lambda **kwargs: calls.append(kwargs) or [])
    monkeypatch.setattr(main, "format_picks", lambda picks: "observacion")
    main.main()
    assert calls == [{"observation": True}]
    assert "observacion" in capsys.readouterr().out
