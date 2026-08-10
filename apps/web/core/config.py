from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# WEB_DIR = apps/web, BASE_DIR = 저장소 루트 (.env / pyproject.toml / db 상대경로의 기준)
WEB_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = WEB_DIR.parents[1]
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
	model_config = SettingsConfigDict(
		env_file=ENV_FILE,
		env_file_encoding="utf-8",
		extra="ignore",
	)

	app_title: str = "Captcha Solver"
	default_captcha_id: str = "supreme_court"
	web_host: str = "0.0.0.0"
	web_port: int = 5000
	web_debug: bool = False
	db_driver: str = "sqlite3"
	db_path: str = "./db/captchaSolver.sqlite3"
	database_url: str = "sqlite:///./db/captchaSolver.sqlite3"
	db_schema_path: str = "./db/schema.sql"
	db_seed_path: str = "./db/seed_captcha_types.sql"

	@property
	def template_dir(self) -> Path:
		return WEB_DIR / "templates"

	@property
	def static_dir(self) -> Path:
		return WEB_DIR / "static"


@lru_cache
def get_settings() -> Settings:
	return Settings()
