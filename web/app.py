from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.api.system import router as system_router
from web.api.v1.router import router as api_v1_router
from web.core.config import get_settings
from web.core.db import get_service_config, init_db
from web.frontend.router import create_router as create_frontend_router
from web.services.captcha import preload_models


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
	# 서비스 정보(기본/서비스 대상 캡차)를 DB에서 읽고, 그 대상만 로드/워밍업한다.
	init_db()
	config = get_service_config(reload=True)
	print(f"[service] default={config['default_captcha_id']} serviced={config['serviced']} ({config['source']})")
	for captcha_id, state in preload_models().items():
		print(f"[preload] {captcha_id}: {state}")
	yield


def create_app() -> FastAPI:
	settings = get_settings()
	fastapi_app = FastAPI(title=settings.app_title, lifespan=lifespan)
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
