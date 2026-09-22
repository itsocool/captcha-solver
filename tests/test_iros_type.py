from pathlib import Path

import pytest

from web.core import db
from web.core.config import BASE_DIR, get_settings


def test_iros_registry_uses_independent_data_with_wetax_preprocessing(captcha_data_dir):
	from web.core.engine import get_captcha_type_list

	base = captcha_data_dir("wetax", size=(200, 60))
	captcha_data_dir("iros", size=(200, 60))
	registry = get_captcha_type_list(train_data_base_dir=base)
	iros = registry["iros"]
	assert iros.name == "인터넷등기소"
	assert iros.train_data.captcha_id == "iros"
	assert Path(iros.train_data.get_model_path()) == Path(base) / "iros/1/model/model.pth"
	assert iros.train_data.threshold == registry["wetax"].train_data.threshold == 255
	assert iros.train_data.image_height == registry["wetax"].train_data.image_height == 60
	assert iros.train_data.preprocess == "default"
	assert (iros.train_data.detected_image_width, iros.train_data.detected_image_height) == (200, 60)
	assert iros.train_data.detected_label_length == 6


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
	monkeypatch.setenv("DB_PATH", str(tmp_path / "test.sqlite3"))
	monkeypatch.setenv("DB_SCHEMA_PATH", str(BASE_DIR / "db/schema.sql"))
	monkeypatch.setenv("DB_SEED_PATH", str(BASE_DIR / "db/seed_captcha_types.sql"))
	monkeypatch.setattr(db, "_SERVICE_CONFIG", None)
	get_settings.cache_clear()
	yield
	get_settings.cache_clear()


def test_iros_is_seeded_and_existing_settings_survive_reinitialization(isolated_db):
	db.init_db()
	config = db.get_service_config(reload=True)
	assert "iros" in config["serviced"]
	assert config["default_captcha_id"] == "supreme_court"
	with db.connect() as conn:
		row = conn.execute("SELECT * FROM captcha_types WHERE captcha_id = 'iros'").fetchone()
		assert row["name"] == "인터넷등기소"
		assert row["seq"] == 5
		row = conn.execute("SELECT * FROM train_data_configs WHERE captcha_id = 'iros'").fetchone()
		assert (row["rev"], row["image_width"], row["image_height"], row["label_length"], row["threshold"]) == (1, 200, 60, 6, 255)
		conn.execute("UPDATE service_captchas SET enabled = 0 WHERE captcha_id = 'iros'")
		conn.execute("UPDATE train_data_configs SET threshold = 75 WHERE captcha_id = 'iros'")
	db.init_db()
	assert "iros" not in db.get_service_config(reload=True)["serviced"]
	with db.connect() as conn:
		rows = conn.execute("SELECT threshold FROM train_data_configs WHERE captcha_id = 'iros'").fetchall()
		assert [row["threshold"] for row in rows] == [75]
