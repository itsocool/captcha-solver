from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from web.core.config import get_settings
from web.core.version import get_app_version


def create_router(templates: Jinja2Templates) -> APIRouter:
	router = APIRouter()

	@router.get("/", response_class=HTMLResponse)
	async def index(request: Request):
		settings = get_settings()
		return templates.TemplateResponse(
			request=request,
			name="index.html",
			context={
				"default_captcha_id": settings.default_captcha_id,
				"app_version": get_app_version(),
				"predict_image_url": "/api/v1/predictImage",
				"predict_json_url": "/api/v1/predictJson",
			},
		)

	return router
