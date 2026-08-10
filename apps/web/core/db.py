"""SQLite 접근 계층. 서비스 대상 캡차 설정을 읽는다."""
import sqlite3
from pathlib import Path

from web.core.config import BASE_DIR, get_settings


_SERVICE_CONFIG: dict | None = None


def _resolve(path: str) -> Path:
	candidate = Path(path)
	return candidate if candidate.is_absolute() else BASE_DIR / candidate


def connect() -> sqlite3.Connection:
	db_path = _resolve(get_settings().db_path)
	db_path.parent.mkdir(parents=True, exist_ok=True)
	conn = sqlite3.connect(db_path)
	conn.row_factory = sqlite3.Row
	return conn


def _add_missing_columns(conn: sqlite3.Connection) -> None:
	"""CREATE TABLE IF NOT EXISTS 로는 못 따라가는 컬럼 추가를 보정한다.

	SQLite 에는 ADD COLUMN IF NOT EXISTS 가 없어서 schema.sql 안에 둘 수 없다.
	"""
	added = [
		("train_data_configs", "characters", "TEXT NOT NULL DEFAULT ''"),
	]
	for table, column, decl in added:
		existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
		if not existing:
			continue  # 테이블 자체가 없으면 schema.sql 이 최신 정의로 만든다
		if column not in existing:
			conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
			print(f"[db] {table}.{column} 컬럼 추가")


def init_db() -> None:
	"""schema.sql 적용. IF NOT EXISTS / INSERT OR IGNORE 라 반복 실행해도 안전하다."""
	schema_path = _resolve(get_settings().db_schema_path)
	with connect() as conn:
		_add_missing_columns(conn)
		conn.executescript(schema_path.read_text(encoding="utf-8"))


def get_service_config(reload: bool = False) -> dict:
	"""서비스 대상 캡차 목록과 기본 캡차 ID.

	테이블이 비어 있거나 없으면 .env의 DEFAULT_CAPTCHA_ID만 서비스한다.
	반환: {"default_captcha_id": str, "serviced": [str], "source": "db" | "fallback"}
	"""
	global _SERVICE_CONFIG

	if _SERVICE_CONFIG is not None and not reload:
		return _SERVICE_CONFIG

	settings = get_settings()
	rows = []
	try:
		with connect() as conn:
			rows = conn.execute(
				"SELECT captcha_id, is_default FROM service_captchas"
				" WHERE enabled = 1 ORDER BY sort_order, captcha_id"
			).fetchall()
	except sqlite3.Error as e:
		print(f"[db] service_captchas 조회 실패, .env 기본값 사용: {e}")

	if rows:
		serviced = [row["captcha_id"] for row in rows]
		default = next((row["captcha_id"] for row in rows if row["is_default"]), serviced[0])
		_SERVICE_CONFIG = {"default_captcha_id": default, "serviced": serviced, "source": "db"}
	else:
		_SERVICE_CONFIG = {
			"default_captcha_id": settings.default_captcha_id,
			"serviced": [settings.default_captcha_id],
			"source": "fallback",
		}

	return _SERVICE_CONFIG
