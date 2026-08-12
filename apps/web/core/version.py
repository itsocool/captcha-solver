import tomllib
from importlib.metadata import PackageNotFoundError, version as package_version

from web.core.config import BASE_DIR, get_settings


PROJECT_NAME = "captcha-solver"


def get_app_version() -> str:
	# 런타임 env(APP_VERSION) 우선: 이미지 재빌드 없이 .env 만 고쳐 버전을 올린다.
	env_version = get_settings().app_version.strip()
	if env_version:
		return env_version

	# 폴백: pyproject.toml (editable 설치의 메타데이터는 오래된 버전을 들 수 있어 이걸 먼저 본다).
	try:
		with (BASE_DIR / "pyproject.toml").open("rb") as pyproject_file:
			return tomllib.load(pyproject_file)["project"]["version"]
	except (OSError, tomllib.TOMLDecodeError, KeyError):
		pass

	try:
		return package_version(PROJECT_NAME)
	except PackageNotFoundError:
		return "unknown"
