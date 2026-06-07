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
