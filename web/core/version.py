import tomllib
from importlib.metadata import PackageNotFoundError, version as package_version

from web.core.config import BASE_DIR


PROJECT_NAME = "captcha-solver"


def get_app_version() -> str:
	# pyproject.toml 우선: editable 설치의 메타데이터는 오래된 버전을 들고 있을 수 있습니다.
	try:
		with (BASE_DIR / "pyproject.toml").open("rb") as pyproject_file:
			return tomllib.load(pyproject_file)["project"]["version"]
	except (OSError, tomllib.TOMLDecodeError, KeyError):
		pass

	try:
		return package_version(PROJECT_NAME)
	except PackageNotFoundError:
		return "unknown"
