"""Standalone web project contracts (also run against the built wheel)."""

from importlib.metadata import version

from fastapi.testclient import TestClient

from web.core.config import BASE_DIR, get_settings
from web.core.version import get_app_version


def test_version_uses_web_distribution(monkeypatch):
	monkeypatch.setenv("APP_VERSION", "")
	get_settings.cache_clear()
	try:
		assert get_app_version() == version("web")
	finally:
		get_settings.cache_clear()


def test_registered_data_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
	from web.core import engine
	from web.services import batch_predict, captcha, data_source, train

	calls = []

	def registry(train_data_base_dir="./captcha_data"):
		calls.append(str(train_data_base_dir))
		return {}

	monkeypatch.setattr(engine, "get_captcha_type_list", registry)
	monkeypatch.chdir(tmp_path)
	assert captcha.list_captcha_types(serviced_only=False) == []
	assert batch_predict.list_targets() == []
	assert data_source.list_targets() == []
	assert train.list_targets() == []
	assert calls == [str(BASE_DIR / "captcha_data")] * 4


def test_packaged_assets_and_routes(monkeypatch):
	from web import app as app_module
	from web.frontend import router as frontend_router

	monkeypatch.setattr(frontend_router, "get_service_config", lambda: {"default_captcha_id": "wetax"})
	monkeypatch.setattr(frontend_router, "list_captcha_types", lambda: [("wetax", "WETAX")])
	monkeypatch.setattr(frontend_router, "device_options", lambda: [])
	client = TestClient(app_module.create_app())
	assert client.get("/ping").json() == {"ping": "pong"}
	assert client.get("/").status_code == 200
	assert client.get("/static/js/theme.js").status_code == 200
	assert client.get("/static/vendor/tailwind.browser.js").status_code == 200
	assert client.get("/static/favicon.ico").status_code == 200
	assert "/api/v1/predictImage" in client.get("/openapi.json").json()["paths"]
