from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.api.system import router as system_router
from web.api.v1.router import router as api_v1_router
from web.core.config import get_settings
from web.frontend.router import create_router as create_frontend_router


def create_app() -> FastAPI:
	settings = get_settings()
	fastapi_app = FastAPI(title=settings.app_title)
	fastapi_app.mount(
		"/static",
		StaticFiles(directory=str(settings.static_dir)),
		name="static",
	)
	templates = Jinja2Templates(directory=str(settings.template_dir))
	fastapi_app.include_router(create_frontend_router(templates))
	fastapi_app.include_router(system_router)
	fastapi_app.include_router(api_v1_router, prefix="/api/v1")
	return fastapi_app


settings = get_settings()
app = create_app()
APP_HOST = settings.web_host
APP_PORT = settings.web_port
APP_DEBUG = settings.web_debug

if __name__ == '__main__':
	# 개발용 실행: uvicorn으로 실행됩니다.
	import uvicorn
	uvicorn.run(app, host=APP_HOST, port=APP_PORT, reload=APP_DEBUG)
