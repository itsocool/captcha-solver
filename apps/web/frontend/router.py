from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from web.core.db import get_service_config
from web.core.version import get_app_version
from web.services.captcha import list_captcha_types, model_status


def create_router(templates: Jinja2Templates) -> APIRouter:
	router = APIRouter()

	@router.get("/", response_class=HTMLResponse)
	async def index(request: Request):
		config = get_service_config()
		return templates.TemplateResponse(
			request=request,
			name="index.html",
			context={
				"active_nav": "predict",
				"default_captcha_id": config["default_captcha_id"],
				"captcha_types": list_captcha_types(),
				"app_version": get_app_version(),
				"predict_image_url": "/api/v1/predictImage",
			},
		)

	@router.get("/status", response_class=HTMLResponse)
	async def status(request: Request):
		config = get_service_config()
		models = model_status()
		devices = {model["device"] for model in models if model["loaded"]}
		return templates.TemplateResponse(
			request=request,
			name="status.html",
			context={
				"active_nav": "status",
				"models": models,
				"device": ", ".join(sorted(devices)) if devices else "-",
				"default_captcha_id": config["default_captcha_id"],
				"config_source": config["source"],
				"app_version": get_app_version(),
			},
		)

	return router
