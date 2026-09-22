from functools import lru_cache
import os
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# 소스 설치는 저장소 루트, wheel 설치는 작업 디렉터리를 데이터 기준으로 삼는다.
# WEB_DATA_DIR 로 .env / db / captcha_data 의 기준을 명시할 수 있다.
WEB_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = Path(os.environ.get(
	"WEB_DATA_DIR",
	WEB_DIR.parents[1] if (WEB_DIR / "pyproject.toml").is_file() else Path.cwd(),
)).resolve()
CAPTCHA_DATA_DIR = BASE_DIR / "captcha_data"
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
	model_config = SettingsConfigDict(
		env_file=ENV_FILE,
		env_file_encoding="utf-8",
		extra="ignore",
	)

	app_title: str = "Captcha Solver"
	# 표시용 앱 버전. .env 의 APP_VERSION 으로 이미지 재빌드 없이 올린다(비면 pyproject 폴백).
	app_version: str = ""
	default_captcha_id: str = "supreme_court"
	# 리버스 프록시 뒤에서 하위 경로(예: /captcha)에 물릴 때의 접두사. FastAPI root_path 와
	# 템플릿/JS 의 링크·fetch 경로에 붙는다. 비면 루트(/)에 뜬 것으로 본다.
	web_context_path: str = ""
	web_host: str = "0.0.0.0"
	web_port: int = 5000
	web_debug: bool = False
	db_driver: str = "sqlite3"
	db_path: str = "./db/captchaSolver.sqlite3"
	database_url: str = "sqlite:///./db/captchaSolver.sqlite3"
	db_schema_path: str = "./db/schema.sql"
	db_seed_path: str = "./db/seed_captcha_types.sql"

	@field_validator("web_context_path")
	@classmethod
	def _normalize_context_path(cls, value: str) -> str:
		# "captcha", "/captcha", "/captcha/" 어느 쪽으로 넣어도 "/captcha" 로 맞춘다.
		value = value.strip().strip("/")
		return f"/{value}" if value else ""

	@property
	def template_dir(self) -> Path:
		return WEB_DIR / "templates"

	@property
	def static_dir(self) -> Path:
		return WEB_DIR / "static"


@lru_cache
def get_settings() -> Settings:
	return Settings()
