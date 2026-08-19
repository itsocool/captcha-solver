from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send
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


class PrependRootPath:
	"""프록시가 접두사(root_path)를 떼고 넘기면 path 앞에 도로 붙인다.

	ASGI 스펙은 path 가 root_path 를 포함한다고 본다. 라우트 매칭은 접두사가 없어도
	알아서 넘어가지만 StaticFiles 같은 Mount 는 하위 root_path(/captcha/static)를 못
	잘라내 404 가 난다. 이미 붙어 있으면(안 떼는 프록시) 손대지 않는다.
	"""

	def __init__(self, app: ASGIApp, root_path: str):
		self.app = app
		self.root_path = root_path

	async def __call__(self, scope: Scope, receive: Receive, send: Send):
		if self.root_path and scope["type"] in ("http", "websocket"):
			path = scope["path"]
			if path != self.root_path and not path.startswith(self.root_path + "/"):
				scope = {**scope, "path": self.root_path + path,
				         "raw_path": self.root_path.encode() + scope.get("raw_path", path.encode())}
		await self.app(scope, receive, send)


def create_app() -> FastAPI:
	settings = get_settings()
	# root_path: 프록시가 접두사를 떼든 말든 라우팅되고, /docs 가 접두사 붙은 openapi.json 을 찾는다.
	fastapi_app = FastAPI(title=settings.app_title, root_path=settings.web_context_path, lifespan=lifespan)
	fastapi_app.add_middleware(PrependRootPath, root_path=settings.web_context_path)
	fastapi_app.mount(
		"/static",
		StaticFiles(directory=str(settings.static_dir)),
		name="static",
	)
	templates = Jinja2Templates(directory=str(settings.template_dir))
	# 템플릿의 링크·정적 파일 경로는 절대 경로라 root_path 가 자동으로 안 붙는다. 전역으로 넘긴다.
	templates.env.globals["context_path"] = settings.web_context_path

	def static_url(path: str) -> str:
		"""정적 파일 URL. 파일 mtime 을 ?v= 로 붙여 배포·수정 뒤 브라우저가 옛 JS 를 캐시에서
		꺼내 쓰지 못하게 한다 (템플릿은 바뀌었는데 JS 만 낡으면 DOM 참조가 어긋나 조용히 깨진다).
		StaticFiles 는 Cache-Control 을 안 보내 브라우저가 휴리스틱으로 재검증 없이 캐시할 수 있다."""
		target = settings.static_dir / path
		try:
			version = int(target.stat().st_mtime)
		except OSError:
			version = 0
		return f"{settings.web_context_path}/static/{path}?v={version}"

	templates.env.globals["static_url"] = static_url
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
