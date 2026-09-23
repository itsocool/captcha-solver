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
		("captcha_types", "seq", "INTEGER NOT NULL DEFAULT 0"),
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


def get_captcha_seq() -> dict[str, int]:
	"""captcha_types.seq — 화면에 보이는 캡차 순서 (captcha_id → seq). 실패하면 빈 dict.

	호출부(services/captcha.ordered_captcha_ids)가 빈 dict 면 레지스트리 순서로 폴백한다.
	"""
	try:
		with connect() as conn:
			rows = conn.execute("SELECT captcha_id, seq FROM captcha_types").fetchall()
	except sqlite3.Error as e:
		print(f"[db] captcha_types.seq 조회 실패: {e}")
		return {}
	return {row["captcha_id"]: int(row["seq"]) for row in rows}


def get_data_source_predictions(captcha_id: str, rev: int) -> dict[str, dict]:
	"""신뢰도 조회 실패를 빈 결과로 숨기지 않고 호출부로 전달한다."""
	from contextlib import closing

	with closing(connect()) as conn:
		rows = conn.execute(
			"SELECT name, prediction, confidence, image_size, image_mtime_ns"
			" FROM data_source_predictions WHERE captcha_id = ? AND rev = ?",
			(captcha_id, rev),
		).fetchall()
	return {row["name"]: dict(row) for row in rows}


def save_data_source_prediction(captcha_id: str, rev: int, name: str,
                               prediction: str, confidence: float, image_stat) -> None:
	"""예측을 커밋한 뒤 반환한다. 저장 실패 시 성공 이벤트를 보내지 않는다."""
	from contextlib import closing

	if not (0 <= confidence <= 1):
		raise ValueError("예측 신뢰도는 0 ~ 1 범위여야 합니다")
	with closing(connect()) as conn, conn:
		conn.execute(
			"INSERT INTO data_source_predictions"
			" (captcha_id, rev, name, prediction, confidence, image_size, image_mtime_ns)"
			" VALUES (?, ?, ?, ?, ?, ?, ?)"
			" ON CONFLICT(captcha_id, rev, name) DO UPDATE SET"
			" prediction=excluded.prediction, confidence=excluded.confidence,"
			" image_size=excluded.image_size, image_mtime_ns=excluded.image_mtime_ns,"
			" updated_at=CURRENT_TIMESTAMP",
			(captcha_id, rev, name, prediction, confidence, image_stat.st_size, image_stat.st_mtime_ns),
		)


def move_data_source_prediction(conn: sqlite3.Connection, captcha_id: str, rev: int,
                                name: str, new_name: str, image_stat) -> float | None:
	"""파일명 변경 트랜잭션에서 유효한 예측만 새 이름으로 연결한다."""
	row = conn.execute(
		"SELECT confidence FROM data_source_predictions"
		" WHERE captcha_id = ? AND rev = ? AND name = ?"
		" AND image_size = ? AND image_mtime_ns = ?",
		(captcha_id, rev, name, image_stat.st_size, image_stat.st_mtime_ns),
	).fetchone()
	conn.execute(
		"DELETE FROM data_source_predictions WHERE captcha_id = ? AND rev = ? AND name = ?",
		(captcha_id, rev, new_name),
	)
	if row is None:
		conn.execute(
			"DELETE FROM data_source_predictions WHERE captcha_id = ? AND rev = ? AND name = ?",
			(captcha_id, rev, name),
		)
		return None
	conn.execute(
		"UPDATE data_source_predictions SET name = ? WHERE captcha_id = ? AND rev = ? AND name = ?",
		(new_name, captcha_id, rev, name),
	)
	return float(row["confidence"])


def get_data_source_label_edits(captcha_id: str, rev: int) -> dict[str, dict]:
	"""현재 파일에 연결된 마지막 수동 편집 기록을 조회한다."""
	from contextlib import closing

	with closing(connect()) as conn:
		rows = conn.execute(
			"SELECT current_name AS name, image_size, image_mtime_ns, edited_at"
			" FROM data_source_label_edits WHERE id IN ("
			" SELECT MAX(id) FROM data_source_label_edits"
			" WHERE captcha_id = ? AND rev = ? AND current_name IS NOT NULL"
			" GROUP BY current_name)",
			(captcha_id, rev),
		).fetchall()
	return {row["name"]: dict(row) for row in rows}


def move_data_source_label_edits(conn: sqlite3.Connection, captcha_id: str, rev: int,
                                 name: str, new_name: str, image_stat,
                                 manual_edit: bool) -> str | None:
	"""같은 트랜잭션 안에서 이력의 파일 연결을 옮기고 수동 편집만 추가한다."""
	if name != new_name:
		conn.execute(
			"UPDATE data_source_label_edits SET current_name = NULL"
			" WHERE captcha_id = ? AND rev = ? AND current_name = ?",
			(captcha_id, rev, new_name),
		)
	conn.execute(
		"UPDATE data_source_label_edits SET current_name ="
		" CASE WHEN image_size = ? AND image_mtime_ns = ? THEN ? ELSE NULL END"
		" WHERE captcha_id = ? AND rev = ? AND current_name = ?",
		(image_stat.st_size, image_stat.st_mtime_ns, new_name, captcha_id, rev, name),
	)
	if manual_edit:
		conn.execute(
			"INSERT INTO data_source_label_edits"
			" (captcha_id, rev, previous_name, new_name, current_name, image_size, image_mtime_ns)"
			" VALUES (?, ?, ?, ?, ?, ?, ?)",
			(captcha_id, rev, name, new_name, new_name, image_stat.st_size, image_stat.st_mtime_ns),
		)
	row = conn.execute(
		"SELECT edited_at FROM data_source_label_edits"
		" WHERE captcha_id = ? AND rev = ? AND current_name = ? ORDER BY id DESC LIMIT 1",
		(captcha_id, rev, new_name),
	).fetchone()
	return row["edited_at"] if row else None
