from datetime import datetime

from web.core.config import BASE_DIR

PACKAGES_DIR = BASE_DIR / "packages"

# 표시명 · glob 패턴. 패턴이 None 이면 다운로드 파일이 없는 항목(web 자신 등)이다.
_APPS = [
	{"key": "web", "name": "Web", "subtitle": "FastAPI 웹 서비스 · Docker 로 배포", "pattern": None},
	{"key": "springBoot", "name": "Spring Boot", "subtitle": "추론 전용 서버 (JVM)",
		"pattern": "springBoot/captchaSolver-*.jar"},
	{"key": "cli", "name": "CLI", "subtitle": "커맨드라인 도구 (Windows)",
		"pattern": "cli/captcha-cli-*-win64-setup.exe"},
	{"key": "WinConsoleApp", "name": "Windows Console App", "subtitle": "콘솔 앱 (Windows)",
		"pattern": "WinConsoleApp/captcha-solver-*-win64.exe"},
	{"key": "ChromeExtensions", "name": "Chrome Extension", "subtitle": "ipTIME 로그인 캡차 자동 인식",
		"pattern": "iptime-captcha-*.zip"},
]


def _latest_match(pattern: str):
	matches = sorted(PACKAGES_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
	return matches[0] if matches else None


def list_downloads() -> list[dict]:
	"""/download 페이지용. 앱별 packages/ 최신 산출물을 찾아 카드 데이터로 만든다."""
	items = []
	for app in _APPS:
		item = {"key": app["key"], "name": app["name"], "subtitle": app["subtitle"], "available": False}
		if app["pattern"]:
			match = _latest_match(app["pattern"])
			if match:
				stat = match.stat()
				item.update(
					available=True,
					filename=match.name,
					size_mb=round(stat.st_size / 1024 / 1024, 1),
					mtime=datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
				)
		items.append(item)
	return items


def resolve_download(key: str):
	"""주어진 key 의 최신 산출물 경로. 없으면 None."""
	app = next((a for a in _APPS if a["key"] == key), None)
	if app is None or not app["pattern"]:
		return None
	return _latest_match(app["pattern"])
