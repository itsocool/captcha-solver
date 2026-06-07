import tomllib
from importlib.metadata import PackageNotFoundError, version as package_version

from web.core.config import BASE_DIR


PROJECT_NAME = "captcha-solver"


def get_app_version() -> str:
	try:
		return package_version(PROJECT_NAME)
	except PackageNotFoundError:
		pass

	try:
		with (BASE_DIR / "pyproject.toml").open("rb") as pyproject_file:
			pyproject = tomllib.load(pyproject_file)
	except (OSError, tomllib.TOMLDecodeError):
		return "unknown"

	return pyproject.get("project", {}).get("version", "unknown")
