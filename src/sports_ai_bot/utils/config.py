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
    bot_timezone: str = Field(default="America/Bogota", alias="BOT_TIMEZONE")
    bot_language: str = Field(default="es", alias="BOT_LANGUAGE")
    the_odds_api_key: str = Field(default="", alias="THE_ODDS_API_KEY")
    the_odds_api_base_url: str = Field(
        default="https://api.the-odds-api.com/v4", alias="THE_ODDS_API_BASE_URL"
    )
    the_odds_api_region: str = Field(default="eu", alias="THE_ODDS_API_REGION")
    the_odds_api_bookmaker: str = Field(default="bet365", alias="THE_ODDS_API_BOOKMAKER")
    the_odds_api_extra_bookmakers: str = Field(
        default="pinnacle", alias="THE_ODDS_API_EXTRA_BOOKMAKERS"
    )
    corners_pick_selection: str = Field(default="Over", alias="CORNERS_PICK_SELECTION")
    corners_pick_point: float = Field(default=9.5, alias="CORNERS_PICK_POINT")
    corners_pick_min_price: float = Field(default=1.65, alias="CORNERS_PICK_MIN_PRICE")
    corners_pick_max_price: float = Field(default=2.15, alias="CORNERS_PICK_MAX_PRICE")
    data_dir: Path = Field(default=BASE_DIR / "data", alias="DATA_DIR")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def predictions_dir(self) -> Path:
        return self.data_dir / "predictions"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    def missing_bot_env(self) -> list[str]:
        missing: list[str] = []
        if not self.telegram_bot_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not self.telegram_chat_id:
            missing.append("TELEGRAM_CHAT_ID")
        return missing

    def has_the_odds_api(self) -> bool:
        return bool(self.the_odds_api_key and self.the_odds_api_base_url)

    def the_odds_api_bookmakers_list(self) -> list[str]:
        bookmakers: list[str] = []
        for value in [self.the_odds_api_bookmaker, self.the_odds_api_extra_bookmakers]:
            for bookmaker in value.split(","):
                normalized = bookmaker.strip()
                if normalized and normalized not in bookmakers:
                    bookmakers.append(normalized)
        return bookmakers

    def the_odds_api_bookmakers_query(self) -> str:
        return ",".join(self.the_odds_api_bookmakers_list())

    def corners_pick_market_label(self) -> str:
        return f"{self.corners_pick_selection} {self.corners_pick_point:g} Corners"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
