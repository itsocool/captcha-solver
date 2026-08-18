import pytest
from fastapi.testclient import TestClient

from web.core.config import Settings


@pytest.mark.parametrize("raw, expected", [
	("", ""),
	("/", ""),
	("captcha", "/captcha"),
	("/captcha", "/captcha"),
	("/captcha/", "/captcha"),
	(" /captcha ", "/captcha"),
])
def test_context_path_is_normalized(raw, expected):
	assert Settings(web_context_path=raw).web_context_path == expected


def test_prefixed_and_stripped_paths_both_route(monkeypatch):
	"""프록시가 접두사를 떼든(/health) 그대로 넘기든(/captcha/health) 같은 라우트에 닿아야 한다."""
	monkeypatch.setenv("WEB_CONTEXT_PATH", "/captcha")
	from web.core.config import get_settings
	get_settings.cache_clear()
	from web import app as app_module
	client = TestClient(app_module.create_app())
	try:
		assert client.get("/captcha/ping").json() == {"ping": "pong"}
		assert client.get("/ping").json() == {"ping": "pong"}
		# StaticFiles(Mount)는 접두사가 빠지면 하위 root_path 를 못 잘라 404 였다 (itsocool.tech 사례)
		assert client.get("/captcha/static/js/theme.js").status_code == 200
		assert client.get("/static/js/theme.js").status_code == 200
		# 템플릿·JS 가 쓰는 접두사가 HTML 로 내려온다
		html = client.get("/captcha/").text
		assert 'data-context-path="/captcha"' in html
		assert 'href="/captcha/static/favicon.ico"' in html
		assert 'action="/captcha/api/v1/predictImage"' in html
		# Swagger 가 접두사 붙은 openapi.json 을 찾도록 servers 에 root_path 가 실린다
		assert client.get("/captcha/openapi.json").json()["servers"] == [{"url": "/captcha"}]
	finally:
		get_settings.cache_clear()
