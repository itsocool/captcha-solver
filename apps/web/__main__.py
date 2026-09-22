"""Run the installed FastAPI application with ``web`` or ``python -m web``."""


def main() -> None:
	import uvicorn

	from web.core.config import WEB_DIR, get_settings

	settings = get_settings()
	uvicorn.run(
		"web.app:app",
		host=settings.web_host,
		port=settings.web_port,
		reload=settings.web_debug,
		reload_dirs=[str(WEB_DIR)] if settings.web_debug else None,
	)


if __name__ == "__main__":
	main()
