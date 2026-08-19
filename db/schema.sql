PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS captcha_types (
	captcha_id TEXT PRIMARY KEY,
	name TEXT NOT NULL DEFAULT '',
	description TEXT NOT NULL DEFAULT '',
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
	updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS train_data_configs (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	captcha_id TEXT NOT NULL,
	backend TEXT NOT NULL DEFAULT 'pytorch',
	rev INTEGER NOT NULL DEFAULT 1,
	train_data_base_dir TEXT NOT NULL DEFAULT './captcha_data',
	image_width INTEGER NOT NULL DEFAULT 200 CHECK (image_width > 0),
	image_height INTEGER NOT NULL DEFAULT 50 CHECK (image_height > 0),
	label_length INTEGER NOT NULL DEFAULT 6 CHECK (label_length > 0),
	-- 선언 문자 집합. 빈 문자열이면 images/train 파일명에서 자동 감지한다
	-- (TrainData.characters 와 같은 의미).
	characters TEXT NOT NULL DEFAULT '',
	threshold INTEGER NOT NULL DEFAULT 255 CHECK (threshold BETWEEN 0 AND 255),
	-- 전처리 종류 (TrainData.preprocess). meta.json 에 그대로 실려 추론 클라이언트가
	-- 같은 전처리를 재현한다. 'default' | 'supreme_court' | 'iptime'.
	preprocess TEXT NOT NULL DEFAULT 'default',
	-- 전처리 크롭 박스 JSON [left, top, right, bottom] (PIL 규약). NULL 이면 크롭 없음.
	-- 크롭 전 좌표계는 image_width/image_height 다.
	crop TEXT,
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
	updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
	FOREIGN KEY (captcha_id) REFERENCES captcha_types(captcha_id) ON DELETE CASCADE,
	UNIQUE (captcha_id, backend, rev)
);

CREATE INDEX IF NOT EXISTS idx_train_data_configs_captcha_id
	ON train_data_configs(captcha_id);

CREATE TABLE IF NOT EXISTS train_data_characters (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	train_data_config_id INTEGER NOT NULL,
	character TEXT NOT NULL CHECK (length(character) = 1),
	sort_order INTEGER NOT NULL CHECK (sort_order >= 0),
	FOREIGN KEY (train_data_config_id) REFERENCES train_data_configs(id) ON DELETE CASCADE,
	UNIQUE (train_data_config_id, character),
	UNIQUE (train_data_config_id, sort_order)
);

CREATE TABLE IF NOT EXISTS train_info_cache (
	train_data_config_id INTEGER PRIMARY KEY,
	image_width INTEGER NOT NULL CHECK (image_width > 0),
	image_height INTEGER NOT NULL CHECK (image_height > 0),
	label_length INTEGER NOT NULL CHECK (label_length > 0),
	characters TEXT NOT NULL DEFAULT '',
	threshold INTEGER NOT NULL DEFAULT 255 CHECK (threshold BETWEEN 0 AND 255),
	detected_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
	FOREIGN KEY (train_data_config_id) REFERENCES train_data_configs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS schema_migrations (
	version INTEGER PRIMARY KEY,
	name TEXT NOT NULL,
	applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO schema_migrations(version, name)
VALUES (1, 'initial_dataclass_schema');

-- 서비스 정보: 어떤 캡차를 서비스할지, 그중 기본값은 무엇인지
CREATE TABLE IF NOT EXISTS service_captchas (
	captcha_id TEXT PRIMARY KEY,
	enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
	is_default INTEGER NOT NULL DEFAULT 0 CHECK (is_default IN (0, 1)),
	sort_order INTEGER NOT NULL DEFAULT 0,
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
	updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 기본 캡차는 하나만 존재할 수 있다
CREATE UNIQUE INDEX IF NOT EXISTS idx_service_captchas_default
	ON service_captchas(is_default) WHERE is_default = 1;

INSERT OR IGNORE INTO service_captchas(captcha_id, enabled, is_default, sort_order) VALUES
	('supreme_court', 1, 1, 0),
	('gov24', 1, 0, 1),
	('wetax', 1, 0, 2),
	('iptime', 1, 0, 3);

INSERT OR IGNORE INTO schema_migrations(version, name)
VALUES (2, 'service_captchas');

-- 이름 붙은 문자 집합 상수 (원래 저장소 루트 data.json 의 constants 였고, 이제 여기가 유일한 출처)
CREATE TABLE IF NOT EXISTS character_sets (
	name TEXT PRIMARY KEY,
	characters TEXT NOT NULL CHECK (length(characters) > 0),
	created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
	updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO character_sets(name, characters) VALUES
	('DIGITS',            '0123456789'),
	('LOWER_CASE',        'abcdefghijklmnopqrstuvwxyz'),
	('UPPER_CASE',        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
	('ALPHABET',          'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'),
	('ALPHA_NUMERIC',     '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'),
	('CAPTCHA_CHAR_SETS', '2345678bcdefgmnpwxy'),
	('DEV_CHAR_SETS',     '2345678ABCDEFGHKLMNPRSTUVWYZabcdefhklmnoprstuvwyz');

INSERT OR IGNORE INTO schema_migrations(version, name)
VALUES (3, 'train_data_configs_characters'), (4, 'character_sets');

INSERT OR IGNORE INTO schema_migrations(version, name)
VALUES (6, 'train_data_configs_preprocess_crop');

-- Training 페이지에서 (캡차, 리비전)별로 마지막에 쓴 학습 파라미터를 기억한다.
-- 대상을 바꾸면 저장된 값을 불러와 폼을 채운다. 값은 services/train.PARAM_SPEC 키 +
-- loss_type/use_amp 를 담은 JSON 이다 (shuffle 은 되돌릴 수 없는 1회성 동작이라 제외).
-- 코어 설정(train_data_configs)이 아니라 UI 편의값이라 타입 컬럼 대신 JSON 으로 둔다.
CREATE TABLE IF NOT EXISTS train_run_params (
	captcha_id TEXT NOT NULL,
	rev INTEGER NOT NULL DEFAULT 1,
	params TEXT NOT NULL DEFAULT '{}',
	updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (captcha_id, rev),
	FOREIGN KEY (captcha_id) REFERENCES captcha_types(captcha_id) ON DELETE CASCADE
);

INSERT OR IGNORE INTO schema_migrations(version, name)
VALUES (7, 'train_run_params');
