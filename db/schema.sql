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
	rev INTEGER NOT NULL DEFAULT 0,
	train_data_base_dir TEXT NOT NULL DEFAULT './captcha_data',
	image_width INTEGER NOT NULL DEFAULT 200 CHECK (image_width > 0),
	image_height INTEGER NOT NULL DEFAULT 50 CHECK (image_height > 0),
	label_length INTEGER NOT NULL DEFAULT 6 CHECK (label_length > 0),
	-- 선언 문자 집합. 빈 문자열이면 images/train 파일명에서 자동 감지한다
	-- (TrainData.characters 와 같은 의미).
	characters TEXT NOT NULL DEFAULT '',
	threshold INTEGER NOT NULL DEFAULT 255 CHECK (threshold BETWEEN 0 AND 255),
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
	('kshop', 1, 0, 3);

INSERT OR IGNORE INTO schema_migrations(version, name)
VALUES (2, 'service_captchas');

-- 이름 붙은 문자 집합 상수 (출처: 저장소 루트 data.json 의 constants)
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
