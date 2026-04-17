from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")
    post_hour_local: str = Field(default="09:00", alias="POST_HOUR_LOCAL")
    bot_language: str = Field(default="es", alias="BOT_LANGUAGE")
    api_football_key: str = Field(default="", alias="API_FOOTBALL_KEY")
    api_football_base_url: str = Field(
        default="https://v3.football.api-sports.io", alias="API_FOOTBALL_BASE_URL"
    )
    api_football_host: str = Field(default="", alias="API_FOOTBALL_HOST")

    data_dir: Path = BASE_DIR / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    models_dir: Path = data_dir / "models"
    predictions_dir: Path = data_dir / "predictions"
    reports_dir: Path = data_dir / "reports"

    def missing_bot_env(self) -> list[str]:
        missing: list[str] = []
        if not self.telegram_bot_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not self.telegram_chat_id:
            missing.append("TELEGRAM_CHAT_ID")
        return missing

    def has_api_football(self) -> bool:
        return bool(self.api_football_key and self.api_football_base_url)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
