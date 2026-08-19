"""SQLite 접근 계층. 서비스 대상 캡차 설정을 읽는다."""
import json
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
		("train_data_configs", "preprocess", "TEXT NOT NULL DEFAULT 'default'"),
		("train_data_configs", "crop", "TEXT"),
	]
	for table, column, decl in added:
		existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
		if not existing:
			continue  # 테이블 자체가 없으면 schema.sql 이 최신 정의로 만든다
		if column not in existing:
			conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
			print(f"[db] {table}.{column} 컬럼 추가")


def init_db() -> None:
	"""schema.sql 과 시드 SQL 을 순서대로 적용한다.

	둘 다 IF NOT EXISTS / INSERT OR IGNORE 라 반복 실행해도 안전하다.
	스키마 파일이 없으면 기동을 막지만, 시드는 없어도 건너뛴다.
	"""
	settings = get_settings()
	schema_path = _resolve(settings.db_schema_path)
	seed_path = _resolve(settings.db_seed_path)

	with connect() as conn:
		_add_missing_columns(conn)
		conn.executescript(schema_path.read_text(encoding="utf-8"))

		if seed_path.is_file():
			conn.executescript(seed_path.read_text(encoding="utf-8"))
		else:
			print(f"[db] 시드 파일이 없어 건너뜁니다: {seed_path}")


def get_service_config(reload: bool = False) -> dict:
	"""서비스 대상 캡차 목록과 기본 캡차 ID.

	테이블이 비어 있거나 없으면 Settings.default_captcha_id 만 서비스한다.
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


def get_train_params(captcha_id: str, rev: int) -> dict | None:
	"""Training 페이지에서 (캡차, 리비전)별로 마지막에 쓴 학습 파라미터. 없으면 None.

	DB 조회가 실패하거나 저장된 JSON 이 깨졌으면 None 을 돌려주고, 호출부가 기본값을 쓴다.
	"""
	try:
		with connect() as conn:
			row = conn.execute(
				"SELECT params FROM train_run_params WHERE captcha_id = ? AND rev = ?",
				(captcha_id, rev),
			).fetchone()
	except sqlite3.Error as e:
		print(f"[db] train_run_params 조회 실패: {e}")
		return None

	if not row:
		return None
	try:
		return json.loads(row["params"])
	except (ValueError, TypeError):
		return None


def save_train_config(captcha_id: str, rev: int, config: dict, backend: str = "pytorch") -> None:
	"""학습 시작 시 그 대상의 학습 정보(감지된 문자셋·크기·전처리)를 DB 에 반영한다.

	라벨을 고치면 감지 문자셋이 바뀌는데 train_data_configs 는 그대로라 기록이 낡는다.
	학습을 시작할 때 실제로 쓰는 값으로 맞춘다. 실패는 삼켜 학습을 막지 않는다.
	config 키: image_width, image_height, label_length, characters, threshold, preprocess, crop.
	"""
	try:
		with connect() as conn:
			conn.execute(
				"INSERT INTO train_data_configs"
				" (captcha_id, backend, rev, image_width, image_height, label_length,"
				"  characters, threshold, preprocess, crop, updated_at)"
				" VALUES (?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)"
				" ON CONFLICT(captcha_id, backend, rev) DO UPDATE SET"
				"  image_width=excluded.image_width, image_height=excluded.image_height,"
				"  label_length=excluded.label_length, characters=excluded.characters,"
				"  threshold=excluded.threshold, preprocess=excluded.preprocess,"
				"  crop=excluded.crop, updated_at=CURRENT_TIMESTAMP",
				(
					captcha_id, backend, rev,
					config["image_width"], config["image_height"], config["label_length"],
					config["characters"], config["threshold"], config["preprocess"],
					json.dumps(config["crop"]) if config.get("crop") else None,
				),
			)
	except (sqlite3.Error, KeyError) as e:
		print(f"[db] train_data_configs 갱신 실패: {e}")


def save_train_params(captcha_id: str, rev: int, params: dict) -> None:
	"""학습 파라미터를 (캡차, 리비전) 키로 upsert 한다. 실패해도 학습은 계속되게 삼킨다."""
	try:
		with connect() as conn:
			conn.execute(
				"INSERT INTO train_run_params(captcha_id, rev, params, updated_at)"
				" VALUES (?, ?, ?, CURRENT_TIMESTAMP)"
				" ON CONFLICT(captcha_id, rev) DO UPDATE SET"
				" params = excluded.params, updated_at = CURRENT_TIMESTAMP",
				(captcha_id, rev, json.dumps(params, ensure_ascii=False)),
			)
	except sqlite3.Error as e:
		print(f"[db] train_run_params 저장 실패: {e}")


def get_data_source_params(captcha_id: str, rev: int) -> dict | None:
	"""Data Source 페이지에서 (캡차, 리비전)별로 마지막에 쓴 수집 입력값. 없으면 None.

	train_run_params 와 같은 계약: 조회 실패나 깨진 JSON 은 None 이고 호출부가 기본값을 쓴다.
	"""
	try:
		with connect() as conn:
			row = conn.execute(
				"SELECT params FROM data_source_params WHERE captcha_id = ? AND rev = ?",
				(captcha_id, rev),
			).fetchone()
	except sqlite3.Error as e:
		print(f"[db] data_source_params 조회 실패: {e}")
		return None

	if not row:
		return None
	try:
		return json.loads(row["params"])
	except (ValueError, TypeError):
		return None


def save_data_source_params(captcha_id: str, rev: int, params: dict) -> None:
	"""수집 입력값을 (캡차, 리비전) 키로 upsert 한다. 실패해도 수집은 계속되게 삼킨다."""
	try:
		with connect() as conn:
			conn.execute(
				"INSERT INTO data_source_params(captcha_id, rev, params, updated_at)"
				" VALUES (?, ?, ?, CURRENT_TIMESTAMP)"
				" ON CONFLICT(captcha_id, rev) DO UPDATE SET"
				" params = excluded.params, updated_at = CURRENT_TIMESTAMP",
				(captcha_id, rev, json.dumps(params, ensure_ascii=False)),
			)
	except sqlite3.Error as e:
		print(f"[db] data_source_params 저장 실패: {e}")
