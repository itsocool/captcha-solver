from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
	model_config = SettingsConfigDict(
		env_file=ENV_FILE,
		env_file_encoding="utf-8",
		extra="ignore",
	)

	app_title: str = "Captcha Predictor"
	default_captcha_id: str = "supreme_court"
	web_host: str = "0.0.0.0"
	web_port: int = 5000
	web_debug: bool = False
	db_driver: str = "sqlite3"
	db_path: str = "./db/captchaSolver.sqlite3"
	database_url: str = "sqlite:///./db/captchaSolver.sqlite3"
	db_schema_path: str = "./db/schema.sql"

	@property
	def template_dir(self) -> Path:
		return BASE_DIR / "web" / "templates"

	@property
	def static_dir(self) -> Path:
		return BASE_DIR / "web" / "static"


@lru_cache
def get_settings() -> Settings:
	return Settings()
